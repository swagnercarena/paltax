# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for normalizing flow dark matter substructure inference."""

import copy
import functools
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from clu import metric_writers
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import optax
from optax import GradientTransformation, EmptyState
import wandb

from paltax import input_pipeline
from paltax import models
from paltax import maf_flow
from paltax import train


def initialized(
    rng: Sequence[int], image_size: int, parameter_dim: int, model: Any
) -> Tuple[Any, Any]:
    """Initialize the model parameters

    Args:
        rng: jax PRNG key.
        image_size: Size of the input image.
        parameter_dim: Size of parameter dimension.
        model: Model class to initialize.

    Returns:
        Initialized model parameters and batch stats.
    """
    context_shape = (1, image_size, image_size, 1)
    y_shape = (1, parameter_dim)
    @jax.jit
    def init(*args):
        return model.init(*args)
    variables = init(
        {'params': rng}, jnp.ones(y_shape),
        jnp.ones(context_shape)
    )
    return variables['params'], variables['batch_stats']


def get_optimizer(
    optimizer: str,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float],
    params: Mapping[str, jnp.ndarray], update_embedding: Optional[bool] = True
) -> Any:
    """Create the optax optimizer instance with masking for int parameters.

    Args:
        optimizer: Optimizer to use.
        learning_rate_schedule: Learning rate schedule.
        params: Parameters of the initialzied model. Used to extract the dtype
            of the parameters.
        update_embedding: If False, set the gradients for the embedding module
            to zero.

    Returns:
        Optimizer instance and the optimizer mask.
    """
    base_optimizer = train.get_optimizer(optimizer, learning_rate_schedule)

    # Add gradient clipping
    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        base_optimizer
    )

    # Map out the int parameters and tell the optimizer it can freeze them.
    def _find_int(param):
        if (param.dtype == jnp.int32 or param.dtype == jnp.int64):
            return 'freeze'
        return 'train'

    opt_mask = jax.tree_map(_find_int, params)

    if update_embedding is False:
        # For all the embedding module parameters, set the gradient to zero.
        opt_mask['embedding_module'] = jax.tree_map(
            lambda x: 'embedding', opt_mask['embedding_module']
        )

    # Create a custom update function for our integer parameters.
    def _init_empty_state(params) -> EmptyState:
        del params
        return EmptyState()

    def set_to_zero() -> GradientTransformation:
        def update_fn(updates, state, params=None):
            del params  # Unused by the zero transform.
            return (
                jax.tree_util.tree_map(
                    functools.partial(jnp.zeros_like, dtype=int), updates
                ), # Force int dtype to avoid type errors from the jitted func.
                state
            )

        return GradientTransformation(_init_empty_state, update_fn)

    # Map between the two optimizers depending on the parameter.
    optimizer = optax.multi_transform(
        {
            'train': base_optimizer, 'freeze': set_to_zero(),
            'embedding': optax.set_to_zero()
        },
        param_labels=opt_mask
    )

    return optimizer, jax.tree_map(lambda x: x == 'freeze', opt_mask)


class TrainState(train.TrainState):
    """Training state class for models with optimizer mask."""
    opt_mask: Any


def create_train_state_nf(
    rng: Sequence[int], config: ml_collections.ConfigDict,
    model: Any, image_size: int, parameter_dim: int,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> TrainState:
    """Create initial training state for flow model.

    Args:
        rng: jax PRNG key.
        config: Configuration with optimizer specification.
        model: Instance of model architecture.
        image_size: Dimension of square image.
        learning_rate_schedule: Learning rate schedule.

    Returns:
        Initialized TrainState for model.
    """
    params, batch_stats = initialized(rng, image_size, parameter_dim, model)
    optimizer = config.get('optimizer', 'adam')
    update_embedding = config.get('update_embedding', True)
    tx, opt_mask = get_optimizer(
        optimizer, learning_rate_schedule, params,
        update_embedding=update_embedding
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        opt_mask=opt_mask
    )
    return state


def draw_sample(
    rng: Sequence[int],
    context: jnp.ndarray,
    flow_params: Mapping[str, Mapping[str, jnp.ndarray]],
    flow_weight: float,
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    flow_module: models.ModuleDef,
    batch_size: int,
    input_config: Mapping[str, Mapping[str, jnp.ndarray]],
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]],
    sample_norm_max: Optional[float] = 4.0
) -> jnp.ndarray:
    """Draw from a mixture of the input configuration and flow.

    Args:
        rng: jax PRNG key.
        context: Embedded context to condition MAF Flow with.
        all_models: Tuple of model classes to consider for each component.
        kwargs_psf: Keyword arguments defining the point spread function. The
            psf is applied in the supersampled space, so the size of pixels
            should be defined with respect to the supersampled space.
        flow_params: Parameters of the flow to use for sampling.
        flow_weight: Proportion of samples between 0 and 1 that should be
            assigned to the flow. Remaining samples will be asigned to the
            lensing_config.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        flow_module: Module defining the flow.
        batch_size: Size of the batch to sample.
        input_config: Configuration used to generate lensing images.
        normalize_config: Seperate config that specifying the lensing
            parameter distirbutions to use when normalizing the model outputs.
        sample_norm_max: Maximum norm to allow for flow samples.

    Returns:
        Mixture of samples from the lensing_config and the flow. Also returns
        the nan fraction given by the flow for logging.
    """
    # Split the rng between our two tasks.
    rng_params, rng_flow = jax.random.split(rng)

    # Draw a batch of parameters from the lensing_config.
    rng_params = jax.random.split(rng_params, batch_size)
    truth_from_config = jax.vmap(
        functools.partial(
            input_pipeline.draw_truth,
            rotation_angle=0.0,
            all_models=input_config['all_models'],
            kwargs_psf=input_config['kwargs_psf'],
            truth_parameters=input_config['truth_parameters'],
            normalize_config=normalize_config
        ),
        in_axes=[None, None, 0]
    )(
        input_config['lensing_config'], cosmology_params, rng_params
    )

    # Draw a batch of parameters from the flow.
    sample_shape = (batch_size,)
    truth_from_flow = flow_module.apply(
        flow_params, rng_flow, context, sample_shape=sample_shape,
        method='sample'
    )

    # Determine the ratio based off of the flow weight.
    flow_mask = jnp.expand_dims(
        jnp.linspace(0, 1, batch_size) <= flow_weight, axis=-1
    )
    # Eliminate nans that can be produced in early rounds of training.
    flow_mask *= ~jnp.isnan(truth_from_flow)
    flow_mask *= jnp.abs(truth_from_flow) < sample_norm_max

    truth = jnp.nan_to_num(truth_from_flow) * flow_mask
    truth += truth_from_config * (~flow_mask)

    return truth, jnp.mean(jnp.isnan(truth_from_flow))


def extract_flow_context(
    state: TrainState, target_image: jnp.ndarray
) -> Tuple[Mapping[str, Mapping[str, jnp.ndarray]], jnp.ndarray]:
    """Extract flow parameters and the target image context.

    Args:
        state: Current TrainState object for the model.
        target_image: Target image for context.

    Returns:
        Flow parameters and encoded context.
    """
    # Insert image into the training batch.
    image_batch = jnp.expand_dims(target_image, axis=0)

    # Extract the flow parameters.
    flow_params = {
        'params': state.params['flow_module']
    }
    # Extract the context from the model.
    context, _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        image_batch, mutable=['batch_stats'],
        method='embed_context', train=False
    )

    return flow_params, context[0] # Remove batch dimension.


def get_flow_weight_schedule(
    config: ml_collections.ConfigDict
) -> Callable[[Union[int, jnp.ndarray]], float]:
    """Return the flow weight schedule function.

    Args:
        config: Training configuration.

    Returns:
        Mapping from the step to the flow weight.
    """
    constant_fn = optax.constant_schedule(0.0)
    flow_weight_schedule_type = config.get(
        'flow_weight_schedule_type', 'linear'
    )

    # Select the flow function to apply after the initial training.
    if flow_weight_schedule_type == 'linear':
        schedule_fn = optax.linear_schedule(
            0.0, 1.0,
            config.num_train_steps - config.num_initial_train_steps
        )
    elif flow_weight_schedule_type == 'power':
        schedule_fn = optax.polynomial_schedule(
            0.0, 1.0, config.flow_weight_schedule_power,
            config.num_train_steps - config.num_initial_train_steps
        )
    elif flow_weight_schedule_type == 'constant':
        schedule_fn = optax.constant_schedule(config.flow_weight_constant)
    else:
        raise ValueError(
            f'{flow_weight_schedule_type} is not a valid learning ' +
            'rate schedule valid type.'
        )

    # Join two schedules
    schedule_fn = optax.join_schedules(
        schedules=[constant_fn, schedule_fn],
        boundaries=[config.num_initial_train_steps]
    )
    return schedule_fn


def get_refinement_step_list(
    config: ml_collections.ConfigDict
) -> Tuple[Sequence[int], int]:
    """Get the sequential refinement step list from the config.

    Args:
        config: Configuration specifying training and model parameters.

    Returns:
        Sequential refinment step list and total number of steps.
    """
    num_initial_steps = config.num_initial_train_steps
    num_steps_per_refinement = config.num_steps_per_refinement

    # Keep adding refinements until we're out of steps.
    cur_num_steps = num_initial_steps

    refinement_step_list = []
    while cur_num_steps < config.num_train_steps:
        refinement_step_list.append(
            cur_num_steps
        )
        cur_num_steps += num_steps_per_refinement

    return refinement_step_list


def gaussian_log_prob(
    mean: jnp.ndarray, prec: jnp.ndarray, truth: jnp.ndarray
) -> jnp.ndarray:
    """Gaussian log probability calculated on mean, covariance, and truth.

    Args:
        mean: Mean of Gaussian.
        prec: Precision matrix of Gaussian.
        truth: True value of the parameters.

    Returns:
        Gaussian loss including only terms that depend on truth (i.e. dropping
        determinant of the covariance matrix.)

    Notes:
        Does not inlcude terms that are constant in the truth.
    """
    error = truth - mean
    log_prob = -0.5 * jnp.einsum('...n,nm,...m->...', error, prec, error)

    return log_prob


def apt_loss(log_posterior: jnp.ndarray, log_prior: jnp.ndarray) -> jnp.ndarray:
    """APT loss with Gaussian prior.

    Args:
        log_posterior: Log posterior output by the model. Has shape (batch_size,
            n_atoms).
        log_prior: Log prior for truth values Has shape (batch_size, n_atoms).

    Returns:
        APT loss for given posterior and prior.
    """
    # Fundamental quantity is ratio of posterior to prior.
    log_prop_posterior_full = log_posterior - log_prior

    # Normalize each batch by values on remaining samples.
    log_prop_posterior = (
        log_prop_posterior_full[:, 0] -
        jax.scipy.special.logsumexp(log_prop_posterior_full, axis=1)
    )
    return -jnp.mean(log_prop_posterior)


def apt_get_atoms(
    rng: Sequence[int], truth: jnp.ndarray, n_atoms: int
) -> jnp.ndarray:
    """Return atoms for each truth in the batch.

    Args:
        rng: jax PRNG key.
        truth: Truth values to sample for atoms.
        n_atoms: Number of atoms for each truth.

    Returns:
        Atoms with shape (batch_size, n_atoms). The first atom is always the
        true value at that index.
    """
    # Different random permutation for each truth
    rng_perm = jax.random.split(rng, len(truth))

    # Select the contrastive indices for each truth.
    choice_vmap = jax.vmap(
        functools.partial(
            jax.random.choice, shape=(n_atoms - 1,), replace=False
        ),
        in_axes=[0, None]
    )
    # One less than length since we can't select the truth for contrastive.
    cont_indices = choice_vmap(rng_perm, len(truth) - 1)

    # Shift indices >= the true index for each batch to ensure the true index is
    # never chosen.
    shift_mask = cont_indices < jnp.arange(len(truth))[:, None]
    cont_indices = cont_indices * shift_mask + (cont_indices + 1) * ~shift_mask

    return jnp.concatenate([truth[:, None], truth[cont_indices]], axis=1)


def train_step(
    rng: Sequence[int], state: TrainState, batch: Mapping[str, jnp.ndarray],
    mu_prior: jnp.ndarray, prec_prior: jnp.ndarray,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float],
    n_atoms: int, opt_mask: Mapping[str, jnp.ndarray]
) -> Tuple[TrainState, Mapping[str, Any]]:
    """Perform a single training step.

    Args:
        state: Current TrainState object for the model.
        batch: Dictionary of images and truths to be used for training.
        mu_prior: Mean of the prior distribution.
        prec_prior: Precision matrix for the prior distribution.
        learning_rate_schedule: Learning rate schedule to apply.

    Returns:
        Updated TrainState object and metrics for training step.
    """
    truth = batch['truth']
    image = batch['image']

    # Get atoms for each evaluation.
    truth_apt = apt_get_atoms(rng, truth, n_atoms)
    log_prior = gaussian_log_prob(mu_prior, prec_prior, truth_apt)

    # Select the thetas we will use for the apt_loss
    def loss_fn(params):
        """Loss function for training."""
        log_posterior, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            truth_apt, image, mutable=['batch_stats'], method='call_apt'
        )
        loss = apt_loss(log_posterior, log_prior)
        return loss, new_model_state

    # Extract learning rate for current step.
    step = state.step
    lr = learning_rate_schedule(step)

    # Extract gradients for weight updates and current model state / loss.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (loss, new_model_state), grads = grad_fn(state.params)

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    def _pmean_if_not_freeze(grad, freeze_grad):
        # Apply pmean only if it is not a frozen gradient.
        if freeze_grad:
            return grad
        return jax.lax.pmean(grad, axis_name='batch')
    grads = jax.tree_map(_pmean_if_not_freeze, grads, opt_mask)

    metrics = {'learning_rate' : lr, 'loss': loss}

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats']
    )

    return new_state, metrics


def train_and_evaluate_nf(
    config: ml_collections.ConfigDict, input_config: dict, workdir: str,
    rng: Sequence[int], target_image: jnp.ndarray,
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None,
    log_prob_batches: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None
):
    """Train model specified by configuration files.

    Args:
        config: Configuration specifying training and model parameters.
        input_config: Configuration used to generate lensing images.
        workddir: Directory to which model checkpoints will be saved.
        rng: jax PRNG key.
        target_image: Target image / observation for sequential inference.
        normalize_config: Optional seperate config that specifying the lensing
            parameter distirbutions to use when normalizing the model outputs.
        log_prob_batches: Labeled batches of truth and images. If provided, the
            log probability of each batch will be evaluated and logged at the
            end of each epoch.

    Returns:
        Training state after training has completed.
    """

    if normalize_config is None:
        normalize_config = copy.deepcopy(input_config['lensing_config'])
    if log_prob_batches is None:
        log_prob_batches = {}

    # Pull parameters from config files.
    image_size = input_config['kwargs_detector']['n_x']
    learning_rate = config.learning_rate
    base_learning_rate = learning_rate * config.batch_size / 256.

    # Set up local logging and wandb logging.
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )
    wandb.init(
        config=config.copy_and_resolve_references(),
        project=config.get('wandb_project', None),
        name=config.get('wandb_run_name', None),
        mode=config.get('wandb_mode', 'disabled')
    )

    # Log the input config being used.
    artifact = wandb.Artifact("input_config", type="code")
    artifact.add_file(config.input_config_path)
    wandb.log_artifact(artifact)

    steps_per_epoch = config.steps_per_epoch
    refinement_step_list = get_refinement_step_list(config)

    # Set up the normalizing flow model.
    num_outputs = len(input_config['truth_parameters'][0])
    embedding_module = getattr(models, config.embedding_model)(
        num_outputs=config.embedding_dim
    )
    flow_module = maf_flow.MAF(
        num_outputs, config.n_maf_layer, config.hidden_dims, config.activation
    )
    model = maf_flow.EmbeddedFlow(embedding_module, flow_module)

    # Get the learning rate schedule and the schedule for the ratio of draws
    # from the prior and the previous flow.
    learning_rate_schedule = train.get_learning_rate_schedule(
        config, base_learning_rate
    )
    flow_weight_schedule = get_flow_weight_schedule(config)

    # Load the initial state for the model.
    rng, rng_state = jax.random.split(rng)
    state = create_train_state_nf(
        rng_state, config, model, image_size, num_outputs,
        learning_rate_schedule
    )
    opt_mask = state.opt_mask
    state = checkpoints.restore_checkpoint(workdir, state)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    # Get the prior distribution that will be enforced.
    mu_prior = config.mu_prior
    prec_prior = config.prec_prior

    # Don't pmap over the prior parameters.
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            learning_rate_schedule=learning_rate_schedule,
            n_atoms=config.n_atoms,
            opt_mask=opt_mask
        ),
        axis_name='batch',
        in_axes=(0, 0, 0, None, None)
    )

    # Create the lookup tables used to speed up derivative and function
    # calculations.
    lookup_tables = input_pipeline.initialize_lookup_tables(input_config)
    draw_image_pmap = jax.pmap(jax.jit(jax.vmap(
        functools.partial(
            input_pipeline.draw_image,
            all_models=input_config['all_models'],
            principal_model_indices=input_config['principal_model_indices'],
            kwargs_simulation=input_config['kwargs_simulation'],
            kwargs_detector=input_config['kwargs_detector'],
            kwargs_psf=input_config['kwargs_psf'],
            truth_parameters=input_config['truth_parameters'],
            normalize_config=normalize_config,
            lookup_tables=lookup_tables),
        in_axes=(None, 0, None, None, None, 0))),
        in_axes=(None, 0, None, None, None, 0)
    )

    # Set the cosmology prameters and the simulation grid.
    rng, rng_cosmo = jax.random.split(rng)
    cosmology_params = input_pipeline.initialize_cosmology_params(
        input_config, rng_cosmo
    )
    grid_x, grid_y = input_pipeline.generate_grids(input_config)

    # Set up code for sampling truth values and evaluating log probability.
    draw_sample_pmap = jax.pmap(
        functools.partial(
            draw_sample, batch_size=config.batch_size,
            input_config=input_config, normalize_config=normalize_config,
            flow_module=flow_module
        ),
        in_axes=(0, 0, 0, None, None)
    )
    log_prob_pmap = jax.pmap(functools.partial(state.apply_fn, train=False))

    # Jit compile the function for extracting the flow parrameters and context.
    extract_flow_context_pmap = jax.pmap(extract_flow_context)
    target_image = jax_utils.replicate(jnp.expand_dims(target_image, axis=-1))

    train_metrics = []
    train_metrics_last_t = time.time()

    print('Initial compilation, this might take some minutes...')

    # This restart isn't perfect, but we're not going to load weights again
    # so we'll just use whatever model we have now.
    flow_params, context = extract_flow_context_pmap(
        state, target_image
    )

    for step in range(step_offset, config.num_train_steps):

        # Check if it's time for a refinement.
        if step in refinement_step_list:
            print(f'Saving new flow weights for sampling at step {step}')
            flow_params, context = extract_flow_context_pmap(
                state, target_image
            )

        # Generate truths and images
        rng, rng_images, rng_truth, rng_atoms = jax.random.split(rng, 4)

        rng_images = jax.random.split(
            rng_images, num=jax.device_count() * config.batch_size).reshape(
                (jax.device_count(), config.batch_size, -1)
        )
        rng_truth = jax.random.split(rng_truth, num=jax.device_count())

        if step % input_config.get('cosmology_update_frequency', 1) == 0:
            print(f'Updating cosmology parameters at step {step}')
            # Update the cosmology parameters for each batch using the source.
            for source_model in input_config['all_models']['all_source_models']:
                cosmology_params = (
                    source_model.modify_cosmology_params(cosmology_params)
                )

        truth, nan_fraction = draw_sample_pmap(
            rng_truth, context, flow_params, flow_weight_schedule(step),
            cosmology_params
        )
        image = draw_image_pmap(
            input_config['lensing_config'], truth, cosmology_params, grid_x,
            grid_y, rng_images
        )
        image = jnp.expand_dims(image, axis=-1)

        # Set our batch and do one step of training.
        batch = {'image': image, 'truth': truth}
        rng_atoms = jax.random.split(rng_atoms, num=jax.device_count())
        state, metrics = p_train_step(
            rng_atoms, state, batch, mu_prior, prec_prior
        )
        metrics['nan_fraction'] = nan_fraction

        # Log training metrics.
        train_metrics.append(metrics)
        wandb.log(jax.tree_map(lambda x: jnp.mean(x), metrics), step=step + 1)
        wandb.log({'flow_weight': flow_weight_schedule(step)}, step=step + 1)

        # Log additional metrics every epoch.
        if (step + 1) % steps_per_epoch == 0:
            train_metrics = common_utils.get_metrics(train_metrics)
            summary = {
                f'epoch_{k}': v
                for k, v in jax.tree_util.tree_map(
                    lambda x: x.mean(), train_metrics
                ).items()
            }
            summary['epoch_steps_per_second'] = steps_per_epoch / (
                time.time() - train_metrics_last_t)

            # Log the performance on any provided batches.
            for batch_name, logging_batch in log_prob_batches.items():
                log_prob_batch_val = log_prob_pmap(
                    {'params': state.params, 'batch_stats': state.batch_stats},
                    jax_utils.replicate(logging_batch['truth']),
                    jax_utils.replicate(
                        jnp.expand_dims(logging_batch['image'], axis=-1)
                    )
                )
                summary[f'epoch_{batch_name}_log_prob'] = float(
                    jnp.mean(log_prob_batch_val)
                )

            writer.write_scalars(step + 1, summary)
            wandb.log(summary, step=step + 1)
            train_metrics = []
            train_metrics_last_t = time.time()

        # Save the checkpoint every epoch.
        if (
            (step + 1) % steps_per_epoch == 0 or
            step + 1 == config.num_train_steps
        ):
            state = train.sync_batch_stats(state)
            train.save_checkpoint(state, workdir, config.keep_every_n_steps)


    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    wandb.finish()

    return state
