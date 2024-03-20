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
"""Training script for sequential dark matter substructure inference."""

import bisect
import copy
import functools
import time
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

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
from paltax import train


def compute_metrics(
        outputs: jnp.ndarray, truth: jnp.ndarray, prop_encoding: jnp.ndarray,
        mu_prior: jnp.ndarray, prec_prior: jnp.ndarray
    ) -> Mapping[str, jnp.ndarray]:
    """Compute the performance metrics of the output.

    Args:
        outputs: Values outputted by the model.
        truth: True value of the parameters.
        prop_encoding: Encoding of the proposition mixture.
        mu_prior: Mean of prior distribution.
        prec_prior: Precision matrix of prior distribution.

    Returns:
        Value of each of the metrics.
    """
    loss = train.snpe_c_loss(outputs, truth, prop_encoding, mu_prior,
                             prec_prior)
    mean, _ = jnp.split(outputs, 2, axis=-1)
    rmse = jnp.sqrt(jnp.mean(jnp.square(mean - truth)))
    metrics = {
        'loss': loss,
        'rmse': rmse,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def get_learning_rate_schedule(
        config: ml_collections.ConfigDict,
        base_learning_rate: float) -> Any:
    """Return the learning rate schedule function.

    Args:
        config: Training configuration.
        base_learning_rate: Base learning rate for the schedule.

    Returns:
        Mapping from step to learning rate according to the schedule.
    """

    # Cosine decay with linear warmup up until refinement begins.
    warmup_fn = optax.linear_schedule(init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_steps)
    cosine_steps = max(config.num_initial_train_steps - config.warmup_steps,
                        1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
        decay_steps=cosine_steps)

    # After refinment, cosine decay with the base value multiplier.
    post_steps = max(config.num_train_steps -
                     config.num_initial_train_steps, 1)
    cosine_fn_post = optax.cosine_decay_schedule(
        init_value=base_learning_rate * config.refinement_base_value_multiplier,
        decay_steps=post_steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn, cosine_fn_post],
        boundaries=[config.warmup_steps, config.num_initial_train_steps])

    return schedule_fn


def train_step(
    state: train.TrainState, batch: Mapping[str, jnp.ndarray],
    prop_encoding: jnp.ndarray, mu_prior: jnp.ndarray, prec_prior: jnp.ndarray,
    learning_rate_schedule: Mapping[int, float]
) -> Tuple[train.TrainState, Mapping[str, Any]]:
    """Perform a single training step.

    Args:
        state: Current TrainState object for the model.
        batch: Dictionairy of images and truths to be used for training.
        prop_encoding: Proposal encoding.
        mu_prior: Mean of the prior distribution.
        prec_prior: Precision matrix for the prior distribution.
        learning_rate_schedule: Learning rate schedule to apply.

    Returns:
        Updated TrainState object and metrics for training step."""

     # Define loss function seperately for use with jax.value_and_grad.
    def loss_fn(params):
        """loss function used for training."""
        outputs, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats'])
        loss = train.snpe_c_loss(outputs, batch['truth'], prop_encoding,
                                 mu_prior, prec_prior)

        return loss, (new_model_state, outputs)

    # Extract learning rate for current step.
    step = state.step
    lr = learning_rate_schedule(step)

    # Extract gradients for weight updates and current model state and outputs
    # for both weight updates and metrics.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, outputs = aux[1]
    metrics = compute_metrics(outputs, batch['truth'], prop_encoding,
                              mu_prior, prec_prior)
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])

    return new_state, metrics


def proposal_distribution_update(
        current_posterior: jnp.ndarray, mean_norm: jnp.ndarray,
        std_norm: jnp.ndarray, mu_prop_init: jnp.ndarray,
        prec_prop_init: jnp.ndarray, prop_encoding: jnp.ndarray,
        prop_decay_factor: float, input_config: dict
    ) -> jnp.ndarray:
    """Update the input configuration and return new proposl distribution.

    Args:
        current_posterior: The outputs from the network representing the
            current network posterior for the observed image.
        mean_norm: The mean of each distribution before normalization.
        std_norm: The standard deviation of each distribution before
            normalization.
        mu_prop_init: Initial mean for the proposal distribution.
        prec_prop_init: Initial precision matrix for the porposal distribution.
        prop_encoding: Previous proposal encoding to be updated.
        prop_decay_factor: Decay factor to apply to previous proposal. A
            negative value indicates to use proposal averaging.
        input_config: Configuration used to generate simulations specific
            in seperate configuration file.

    Returns:
        New proposal encoding.

    Notes:
        prop_encoding_init are used to overwrite proposals that are wider than
        the initial proposal. Variable input_config is changed in place.
    """
    # Get the new proposal distribution.
    mu_prop, log_var_prop = jnp.split(current_posterior, 2, axis=-1)

    # Any parameter that has larger variance than the initial proposal is
    # overwritten to the initial proposal distirbution.
    log_var_bound = jnp.log(1 / jnp.diag(prec_prop_init))
    overwrite_prop_bool = log_var_prop > log_var_bound
    mu_prop = (mu_prop_init * overwrite_prop_bool +
               mu_prop * jnp.logical_not(overwrite_prop_bool))
    log_var_prop = (log_var_bound * overwrite_prop_bool +
                    log_var_prop * jnp.logical_not(overwrite_prop_bool))
    std_prop = jnp.exp(log_var_prop/2)

    # Modify the proposal encoding
    if prop_decay_factor >= 0.0:
        add_normal_to_encoding_vmap = jax.vmap(
            input_pipeline.add_normal_to_encoding, in_axes=[0, 0, 0, None]
        )
        prop_encoding = add_normal_to_encoding_vmap(
            prop_encoding, mu_prop, std_prop, prop_decay_factor
        )
    else:
        average_normal_to_encoding_vmap = jax.vmap(
            input_pipeline.average_normal_to_encoding, in_axes=[0, 0, 0]
        )
        prop_encoding = average_normal_to_encoding_vmap(
            prop_encoding, mu_prop, std_prop
        )

    # Modify the input config.
    mu_prop_unormalized = mu_prop * std_norm + mean_norm
    std_prop_unormalized = std_prop * std_norm

    for truth_i in range(len(input_config['truth_parameters'][0])):
        rewrite_object = input_config['truth_parameters'][0][truth_i]
        rewrite_key = input_config['truth_parameters'][1][truth_i]
        input_config['lensing_config'][rewrite_object][rewrite_key] = (
            input_pipeline.add_normal_to_encoding(
                input_config['lensing_config'][rewrite_object][rewrite_key],
                mu_prop_unormalized[truth_i], std_prop_unormalized[truth_i],
                prop_decay_factor)
        )

    return prop_encoding

def _get_refinement_step_list(config: ml_collections.ConfigDict
) -> Tuple[Sequence[int], int]:
    """Get the sequential refinement step list from the config.

    Args:
        config: Configuration specifying training and model parameters.

    Returns:
        Sequential refinment step list and total number of steps.
    """
    num_initial_steps = config.num_initial_train_steps
    num_steps_per_refinement = config.num_steps_per_refinement
    num_steps = num_initial_steps
    num_steps += num_steps_per_refinement * config.num_refinements

    refinement_step_list = []
    for refinement in range(config.num_refinements):
        refinement_step_list.append(
            num_initial_steps + num_steps_per_refinement * refinement
        )
    return refinement_step_list, num_steps

def train_and_evaluate_snpe(
        config: ml_collections.ConfigDict,
        input_config: dict,
        workdir: str,
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
        logdir=workdir, just_logging=jax.process_index() != 0)

    steps_per_epoch = config.steps_per_epoch
    refinement_step_list, num_steps = _get_refinement_step_list(config)

    model_cls = getattr(models, config.model)
    num_outputs = len(input_config['truth_parameters'][0]) * 2
    model = model_cls(num_outputs=num_outputs, dtype=jnp.float32)

    learning_rate_schedule = get_learning_rate_schedule(config,
        base_learning_rate)

    # Load the initial state for the model.
    rng, rng_state = jax.random.split(rng)
    state = train.create_train_state(rng_state, config, model, image_size,
        learning_rate_schedule)
    state = checkpoints.restore_checkpoint(state, workdir)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    # Get the distributions for sequential inference.
    mu_prior = config.mu_prior
    prec_prior = config.prec_prior
    mu_prop_init = config.mu_prop_init
    prec_prop_init = config.prec_prop_init
    std_prop_init = jnp.power(jnp.diag(prec_prop_init), -0.5)
    prop_decay_factor = config.prop_decay_factor

    # Don't pmap over the sequential distributions.
    p_train_step = jax.pmap(functools.partial(train_step,
        learning_rate_schedule=learning_rate_schedule),
        axis_name='batch', in_axes=(0, 0, None, None, None))
    p_get_outputs = jax.pmap(train.get_outputs, axis_name='batch')

    draw_image_and_truth_pmap = jax.pmap(jax.jit(jax.vmap(
        functools.partial(
            input_pipeline.draw_image_and_truth,
            all_models=input_config['all_models'],
            principal_model_indices=input_config['principal_model_indices'],
            kwargs_simulation=input_config['kwargs_simulation'],
            kwargs_detector=input_config['kwargs_detector'],
            kwargs_psf=input_config['kwargs_psf'],
            truth_parameters=input_config['truth_parameters'],
            normalize_config=normalize_config),
        in_axes=(None, None, None, None, 0, None))),
        in_axes=(None, None, None, None, 0, None)
    )

    # Set the cosmology prameters and the simulation grid.
    rng, rng_cosmo = jax.random.split(rng)
    cosmology_params = input_pipeline.initialize_cosmology_params(input_config,
                                                                 rng_cosmo)
    grid_x, grid_y = input_pipeline.generate_grids(input_config)

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()

    print('Initial compilation, this might take some minutes...')

    # Get our initial proposal. May be in the midst of refinement here. Second
    # vmap is over the processes.
    prop_encoding = jax.vmap(input_pipeline.encode_normal)(
        mu_prop_init, std_prop_init
    )
    std_norm = jnp.array([0.15, 0.1, 0.16, 0.16, 0.1, 0.1, 0.05, 0.05, 0.16,
                          0.16, 1.1e-3])
    mean_norm = jnp.array([1.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           2e-3])

    # A repeated pattern that needs access to the compiled functions. Need to
    # do this for functions on which partial is called because every call to
    # partial returns a new hash (i.e. jit will forget that it already
    # compiled the function).
    def _get_new_prop_encoding(prop_encoding):
        """Use the posterior to encode the new proposal distribution."""
        # We need to encode the target image in a draw from the current
        # distribution so that batch normalization behaves as intended.
        rng_images = jax.random.split(
            jax.random.PRNGKey(0),
            num=jax.device_count() * config.batch_size).reshape(
                (jax.device_count(), config.batch_size, -1)
            )
        image, _ = draw_image_and_truth_pmap(
            input_config['lensing_config'], cosmology_params, grid_x, grid_y,
            rng_images, 0.0
        )
        target_batch = {
            'image': jnp.expand_dims(image.at[0,0].set(target_image), axis=-1)
        }

        # Grab posterior on image of interest (remove pmap and batch indices).
        current_posterior = p_get_outputs(state, target_batch)[0][0,0]

        # Encode the new proposal and return the new encoding.
        prop_encoding = proposal_distribution_update(
            current_posterior, mean_norm, std_norm, mu_prop_init,
            prec_prop_init, prop_encoding, prop_decay_factor,
            input_config
        )
        return prop_encoding


    if bisect.bisect_left(refinement_step_list, step_offset) > 0:
        # This restart isn't perfect, but we're not going to load weights again
        # so we'll just use whatever model we have now.
        print(f'Restarting refinement stage at step {step_offset}')
        prop_encoding = _get_new_prop_encoding(prop_encoding)

    for step in range(step_offset, num_steps):

        # Check if it's time for a refinement.
        if step in refinement_step_list:
            print(f'Starting refinement stage at step {step}')
            prop_encoding = _get_new_prop_encoding(prop_encoding)

        # Generate truths and images
        rng, rng_images = jax.random.split(rng)
        # Rotations will break sequential refinement (since refinement proposals
        # are not rotation invariant).
        rotation_angle = 0.0

        rng_images = jax.random.split(
            rng_images, num=jax.device_count() * config.batch_size).reshape(
                (jax.device_count(), config.batch_size, -1)
            )
        image, truth = draw_image_and_truth_pmap(input_config['lensing_config'],
                                                 cosmology_params, grid_x,
                                                 grid_y, rng_images,
                                                 rotation_angle)
        image = jnp.expand_dims(image, axis=-1)

        batch = {'image': image, 'truth': truth}
        state, metrics = p_train_step(state, batch, prop_encoding, mu_prior,
                                      prec_prior)
        for h in hooks:
            h(step)
        if step == step_offset:
            print('Initial compilation completed.')

        train_metrics.append(metrics)
        if (step + 1) % steps_per_epoch == 0:
            train_metrics = common_utils.get_metrics(train_metrics)
            summary = {
                f'train_{k}': v
                for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
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
