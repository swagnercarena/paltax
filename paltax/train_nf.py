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

import bisect
import copy
import functools
import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import optax

from paltax import input_pipeline
from paltax import models
from paltax import maf_flow
from paltax import train, train_snpe


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


def create_train_state_nf(
    rng: Sequence[int], config: ml_collections.ConfigDict,
    model: Any, image_size: int, parameter_dim: int,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> train.TrainState:
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
    tx = train.get_optimizer(optimizer, learning_rate_schedule)
    state = train.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats)
    return state


def draw_sample(
    rng: Sequence[int],
    context: jnp.ndarray,
    flow_params: Mapping[str, Mapping[str, jnp.ndarray]],
    flow_module: models.ModuleDef,
    flow_weight: float,
    batch_size: int,
    input_config: Mapping[str, Mapping[str, jnp.ndarray]],
    cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]]
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
        flow_module: Module defining the flow.
        batch_size: Size of the batch to sample.
        flow_weight: Proportion of samples between 0 and 1 that should be
            assigned to the flow. Remaining samples will be asigned to the
            lensing_config.
        input_config: Configuration used to generate lensing images.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        normalize_config: Seperate config that specifying the lensing
            parameter distirbutions to use when normalizing the model outputs.

    Returns:
        Mixture of samples from the lensing_config and the flow.
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
        jnp.linspace(0, 1, batch_size) < flow_weight, axis=-1
    )
    truth = truth_from_flow * flow_mask + truth_from_config * (~flow_mask)

    return truth


def extract_flow_context(
    state: train.TrainState, target_image: jnp.ndarray, image_batch: jnp.ndarray
) -> Tuple[Mapping[str, Mapping[str, jnp.ndarray]], jnp.ndarray]:
    """Extract flow parameters and the target image context.

    Args:
        state: Current TrainState object for the model.
        target_image: Target image for context.
        image_batch: Batch of images for batch normalization purposes.

    Returns:
        Flow parameters and encoded context.
    """
    # Insert image into the training batch.
    image_batch = image_batch.at[0].set(target_image)
    # Extract the flow parameters.
    flow_params = {
        'params': train.cross_replica_mean(state.params['flow_module'])
    }
    # Extract the context from the model.
    context, _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        image_batch, mutable=['batch_stats'], method='embed_context'
    )
    # We only want the context for the target image.
    context = train.cross_replica_mean(context)[0]
    return flow_params, context


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
    if flow_weight_schedule_type:
        schedule_fn = optax.linear_schedule(
            0.0, 1.0,
            config.num_train_steps - config.num_initial_train_steps
        )
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


def train_step(
    state: train.TrainState, batch: Mapping[str, jnp.ndarray],
    mu_prior: jnp.ndarray, prec_prior: jnp.ndarray,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> Tuple[train.TrainState, Mapping[str, Any]]:
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

    raise NotImplementedError

    #  # Define loss function seperately for use with jax.value_and_grad.
    # def loss_fn(params):
    #     """loss function used for training."""
    #     outputs, new_model_state = state.apply_fn(
    #         {'params': params, 'batch_stats': state.batch_stats},
    #         batch['image'], mutable=['batch_stats']
    #     )
    #     loss = train.snpe_c_loss(
    #         outputs, batch['truth'], prop_encoding, mu_prior, prec_prior
    #     )

    #     return loss, (new_model_state, outputs)

    # # Extract learning rate for current step.
    # step = state.step
    # lr = learning_rate_schedule(step)

    # # Extract gradients for weight updates and current model state and outputs
    # # for both weight updates and metrics.
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # aux, grads = grad_fn(state.params)
    # # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    # grads = lax.pmean(grads, axis_name='batch')
    # new_model_state, outputs = aux[1]
    # metrics = compute_metrics(
    #     outputs, batch['truth'], prop_encoding, mu_prior, prec_prior
    # )
    # metrics['learning_rate'] = lr

    # new_state = state.apply_gradients(
    #     grads=grads, batch_stats=new_model_state['batch_stats']
    # )

    # return new_state, metrics


def train_and_evaluate_nf(
    config: ml_collections.ConfigDict, input_config: dict, workdir: str,
    rng: Union[Iterator[Sequence[int]], Sequence[int]],
    target_image: jnp.ndarray,
    normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None
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

    Returns:
        Training state after training has completed.
    """

    if normalize_config is None:
        normalize_config = copy.deepcopy(input_config['lensing_config'])

    # Pull parameters from config files.
    image_size = input_config['kwargs_detector']['n_x']
    learning_rate = config.learning_rate
    base_learning_rate = learning_rate * config.batch_size / 256.

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    steps_per_epoch = config.steps_per_epoch
    refinement_step_list, num_steps = train_snpe._get_refinement_step_list(
        config
    )

    # Set up the normalizing flow model.
    num_outputs = len(input_config['truth_parameters'][0])
    embedding_module = getattr(models, config.embedding_model)(
        num_outputs=config.embedding_dim
    )
    flow_module = maf_flow.MAF(
        num_outputs, config.n_maf_layer, config.hidden_dims, config.activation
    )
    model = maf_flow.EmbeddedFlow(embedding_module, flow_module)

    learning_rate_schedule = train_snpe.get_learning_rate_schedule(
        config, base_learning_rate
    )
    flow_weight_schedule = get_flow_weight_schedule(config)

    # Load the initial state for the model.
    rng, rng_state = jax.random.split(rng)
    state = create_train_state_nf(
        rng_state, config, model, image_size, num_outputs,
        learning_rate_schedule
    )
    state = checkpoints.restore_checkpoint(workdir, state)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    # Get the prior distribution that will be enforced.
    mu_prior = config.mu_prior
    prec_prior = config.prec_prior

    # Don't pmap over the sequential distributions.
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            learning_rate_schedule=learning_rate_schedule
        ),
        axis_name='batch',
        in_axes=(0, 0, None, None, None)
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
        in_axes=(None, None, None, None, 0, None))),
        in_axes=(None, None, None, None, 0, None)
    )

    # Set the cosmology prameters and the simulation grid.
    rng, rng_cosmo = jax.random.split(rng)
    cosmology_params = input_pipeline.initialize_cosmology_params(
        input_config, rng_cosmo
    )
    grid_x, grid_y = input_pipeline.generate_grids(input_config)

    # Set up code for sampling truth values.
    draw_sample_pmap = jax.pmap(
        functools.partial(
            draw_sample, batch_size=config.batch_size,
            input_config=input_config, cosmology_params=cosmology_params,
            normalize_config=normalize_config
        ),
        axis_name='batch'
    )

    # Jit compile the function for extracting the flow parrameters and context.
    extract_flow_context_jit = jax.jit(extract_flow_context)
    # Create an artificial batch of initial images in case we are restarting the
    # model.
    image = jax_utils.replicate(jnp.tile(
        jnp.expand_dims(target_image, axis=-1),
        (config.batch_size, 1, 1, 1)
    ))

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()

    print('Initial compilation, this might take some minutes...')

    if bisect.bisect_left(refinement_step_list, step_offset) > 0:
        # This restart isn't perfect, but we're not going to load weights again
        # so we'll just use whatever model we have now.
        flow_params, context = extract_flow_context_jit(
            state,
            jnp.expand_dims(target_image, axis=-1),
            image
        )

    for step in range(step_offset, num_steps):

        # Check if it's time for a refinement.
        if step in refinement_step_list:
            flow_params, context = extract_flow_context_jit(
                state,
                jnp.expand_dims(target_image, axis=-1),
                image
            )

        # Generate truths and images
        rng, rng_images = jax.random.split(rng)
        # Rotations will break sequential refinement (since refinement proposals
        # are not rotation invariant).
        rotation_angle = 0.0

        rng_images = jax.random.split(
            rng_images, num=jax.device_count() * config.batch_size).reshape(
                (jax.device_count(), config.batch_size, -1)
        )
        truth = draw_sample_pmap(
            rng_images[0], context, flow_params, flow_module,
            flow_weight_schedule(step)
        )
        image = draw_image_pmap(
            input_config['lensing_config'], truth, cosmology_params, grid_x,
            grid_y, rng_images, rotation_angle
        )
        image = jnp.expand_dims(image, axis=-1)

        batch = {'image': image, 'truth': truth}
        state, metrics = p_train_step(
            state, batch, mu_prior, prec_prior
        )
        for h in hooks:
            h(step)
        if step == step_offset:
            print('Initial compilation completed.')

        train_metrics.append(metrics)
        if (step + 1) % steps_per_epoch == 0:
            train_metrics = common_utils.get_metrics(train_metrics)
            summary = {
                f'train_{k}': v
                for k, v in jax.tree_util.tree_map(
                    lambda x: x.mean(), train_metrics
                ).items()
            }
            summary['steps_per_second'] = steps_per_epoch / (
                time.time() - train_metrics_last_t)
            writer.write_scalars(step + 1, summary)
            print(summary)
            train_metrics = []
            train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0 or step + 1 == num_steps:
            state = train.sync_batch_stats(state)
            train.save_checkpoint(state, workdir, config.keep_every_n_steps)


    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
