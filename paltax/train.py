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
"""Training script for dark matter substructure inference."""

import copy
import functools
from importlib import import_module
import os
import sys
import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import optax

from paltax import input_pipeline
from paltax import models


def initialized(
    rng: Sequence[int], image_size: int, model: Any
) -> Tuple[Any, Any]:
    """Initialize the model parameters

    Args:
        rng: jax PRNG key.
        image_size: Size of the input image.
        model: Model class to initialize.

    Returns:
        Initialized model parameters and batch stats.
    """
    input_shape = (1, image_size, image_size, 1)
    @jax.jit
    def init(*args):
        return model.init(*args)
    variables = init({'params': rng}, jnp.ones(input_shape, model.dtype))
    return variables['params'], variables['batch_stats']


def gaussian_loss(outputs: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
    """Gaussian loss calculated on the outputs and truth values.

    Args:
        outputs: Values outputted by the model.
        truth: True value of the parameters.

    Returns:
        Gaussian loss.

    Notes:
        Loss does not inlcude constant factor of 1 / (2 * pi) ^ (d/2)
    """
    mean, log_var = jnp.split(outputs, 2, axis=-1)
    loss = 0.5 * jnp.sum(
        jnp.multiply(jnp.square(mean - truth), jnp.exp(-log_var)), axis = -1)
    loss += 0.5*jnp.sum(log_var, axis = -1)
    return jnp.mean(loss)


def _calculate_outputs_comb(
    mu_post: jnp.ndarray, prec_post: jnp.ndarray, mu_prop: jnp.ndarray,
    prec_prop: jnp.ndarray, mu_prior: jnp.ndarray, prec_prior: jnp.ndarray
) -> jnp.ndarray:
    """Returns the outputs representing the product distribution.

    Args:
        mu_post: Mean of the posterior distribution.
        prec_post: Precision matrix of the posterior distribution.
        mu_prop: Mean of the proposal distribution.
        prec_prop: Precision matrix of the proposal distribution.
        mu_prior: Mean of the prior distribution.
        prec_prior: Precision matrix of the prior distribution.

    Returns:
        Mean and log variance of the product distribution posterior * prior
            / proposal. The mean and log variance are concatenated together
            to serve as an input to the gaussian_loss function.
    """
    prec_comb = prec_post + prec_prop - prec_prior
    cov_comb = jnp.linalg.inv(prec_comb)
    eta_comb = jax.vmap(jnp.dot)(prec_post, mu_post)
    eta_comb += jnp.dot(prec_prop, mu_prop)
    eta_comb -= jnp.dot(prec_prior, mu_prior)
    mu_comb = jax.vmap(jnp.dot)(cov_comb, eta_comb)

    # For now our loss function only accepts the log variance and not a full
    # covariance matrix.
    log_var_comb = -jnp.log(jax.vmap(jnp.diag)(prec_comb))
    outputs_comb = jnp.concatenate([mu_comb, log_var_comb], axis=-1)

    return outputs_comb


def snpe_c_loss(
    outputs: jnp.ndarray, truth: jnp.ndarray, prop_encoding:jnp.ndarray,
    mu_prior: jnp.ndarray, prec_prior: jnp.ndarray
) -> jnp.ndarray:
    """Gaussian loss weighted by ratio of proposal to prior for SNPE type c.

    Args:
        outputs: Values outputted by the model.
        truth: True value of the parameters.
        prop_encoding: Encoding of the current proposal function.
        mu_prior: Mean of the prior distribution.
        prec_prior: Precision matrix for the prior distribution. For a infinite
            uniform prior this can be a matrix of zeros. While not techinically
            a well-defined prior, it is equivalent to a Gaussian with infinite
            variance in each variable.

    Returns:
        Gaussian loss multiplied by ratio of proposal to prior and normalized
            to be a well-defined pdf.

    Notes:
        Loss does not inlcude constant factor of 1 / (2 * pi) ^ (d/2)
    """
    # Break out the mean and log variance predictions from our network
    # posterior.
    mu_post, log_var_post = jnp.split(outputs, 2, axis=-1)
    prec_post = jax.vmap(jnp.diag)(jnp.exp(-log_var_post))

    # Extract the distributions from the encoding.
    mu_prop_vmap = jax.vmap(input_pipeline._get_normal_mean)(prop_encoding).T
    prec_prop_vmap = jax.vmap(jnp.diag)(
        1 / jnp.square(jax.vmap(input_pipeline._get_normal_std)(
            prop_encoding).T + 1e-9
        )
    )
    weights_vmap = jax.vmap(input_pipeline._get_normal_weights)(prop_encoding)
    weights_vmap = weights_vmap[0]

    # Calculate the loss term for each proposal component.
    outputs_vmap = jax.vmap(
        _calculate_outputs_comb, in_axes=[None, None, 0, 0, None, None])(
            mu_post, prec_post, mu_prop_vmap, prec_prop_vmap, mu_prior,
            prec_prior
    )
    gaussian_loss_vmap = jax.vmap(gaussian_loss, in_axes=[0, None])(
        outputs_vmap, truth
    )
    # Sum the weighted components in exponential space.
    return jax.scipy.special.logsumexp(gaussian_loss_vmap, b=weights_vmap)


def compute_metrics(
    outputs: jnp.ndarray, truth: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Compute the performance metrics of the output.

    Args:
        outputs: Values outputted by the model.
        truth: True value of the parameters.

    Returns:
        Value of each of the metrics.
    """
    loss = gaussian_loss(outputs, truth)
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
    base_learning_rate: float
) -> Callable[[Union[int, jnp.ndarray]], float]:
    """Return the learning rate schedule function.

    Args:
        config: Training configuration.
        base_learning_rate: Base learning rate for the schedule.

    Returns:
        Mapping from step to learning rate according to the schedule.
    """
    schedule_function_type = config.schedule_function_type

    if schedule_function_type == 'cosine':
        # Cosine decay with linear warmup.
        warmup_fn = optax.linear_schedule(init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=config.warmup_steps)
        cosine_steps = max(config.num_train_steps - config.warmup_steps, 1)
        cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
            decay_steps=cosine_steps)
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],
            boundaries=[config.warmup_steps])
    elif schedule_function_type == 'constant':
        # Constant learning rate.
        schedule_fn = optax.constant_schedule(base_learning_rate)
    elif schedule_function_type == 'linear':
        # Constant learning rate.
        schedule_fn = optax.linear_schedule(
            base_learning_rate,
            base_learning_rate * config.end_value_multiplier,
            config.num_train_steps)
    elif schedule_function_type == 'exp_decay':
        # Exponential decay learning rate.
        schedule_fn = optax.exponential_decay(base_learning_rate,
                                              config.steps_per_epoch,
                                              config.decay_rate)
    else:
        raise ValueError(f'{schedule_function_type} is not a valid learning ' +
                         'rate schedule valid type.')
    return schedule_fn


class TrainState(train_state.TrainState):
    """Training state class for models."""
    batch_stats: Any


def get_outputs(
    state: TrainState, batch: Mapping[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
    """Get the outputs for a batch.

    Args:
        state: Current model state.
        batch: Batch of images and truths.

    Returns:
        Model output and updated batch stats.
    """
    return state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['image'], mutable=['batch_stats']
    )


def train_step(
    state: TrainState, batch: Mapping[str, jnp.ndarray],
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> Tuple[TrainState, Mapping[str, Any]]:
    """Perform a single training step.

    Args:
        state: Current TrainState object for the model.
        batch: Dictionairy of images and truths to be used for training.
        learning_rate_schedule: Learning rate schedule to apply.

    Returns:
        Updated TrainState object and metrics for training step."""

    # Define loss function seperately for use with jax.value_and_grad.
    def loss_fn(params):
        """Loss function used for training."""
        outputs, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats'])
        loss = gaussian_loss(outputs, batch['truth'])

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
    metrics = compute_metrics(outputs, batch['truth'])
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])

    return new_state, metrics


def save_checkpoint(
    state: TrainState, workdir: str,  keep_every_n_steps: Optional[int] = None
):
    """Save the current training state as a checkpoint.

    Args:
        state: State that will be saved to checkpoint file.
        workdir: Directory to save checkpoints to.
        keep_every_n_steps: Checkpoints at this cadence will be preserved no
            matter how old.

    Notes:
        Checkpoint name will be automatically determined by current step.

    """
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3,
            keep_every_n_steps=keep_every_n_steps)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state: TrainState) -> TrainState:
    """Sync the batch statistics across replicas.

    Args:
        state: State to sync.

    Returns:
        Synchronized training state.
    """
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def get_optimizer(
    optimizer: str,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> Any:
    """Create the optax optimizer instance.

    Args:
        optimizer: Optimizer to use.
        learning_rate_schedule: Learning rate schedule.

    Returns:
        Optimizer instance.
    """
    if optimizer == 'adam':
        tx = optax.adam(
            learning_rate=learning_rate_schedule
        )
    elif optimizer == 'sgd':
        tx = optax.sgd(
            learning_rate=learning_rate_schedule
        )
    else:
        raise ValueError(f'Optimizer {optimizer} is not an option.')
    return tx


def create_train_state(
    rng: Sequence[int], config: ml_collections.ConfigDict,
    model: Any, image_size: int,
    learning_rate_schedule: Callable[[Union[int, jnp.ndarray]], float]
) -> TrainState:
    """Create initial training state.

    Args:
        rng: jax PRNG key.
        config: Configuration with optimizer specification.
        model: Instance of model architecture.
        image_size: Dimension of square image.
        learning_rate_schedule: Learning rate schedule.

    Returns:
        Initialized TrainState for model.
    """
    params, batch_stats = initialized(rng, image_size, model)
    optimizer = config.get('optimizer', 'adam')
    tx = get_optimizer(optimizer, learning_rate_schedule)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats)
    return state


def _get_config(config_path: str) -> Any:
    """Return config from provided path.

    Args:
        config_path: Path to configuration file.

    Returns:
        Loaded configuration file.
    """
    # Get the dictionary from the .py file.
    config_dir, config_file = os.path.split(os.path.abspath(config_path))
    sys.path.insert(0, config_dir)
    config_name, _ = os.path.splitext(config_file)
    config_module = import_module(config_name)
    return config_module.get_config()


def train_and_evaluate(
        config: ml_collections.ConfigDict,
        input_config: dict,
        workdir: str,
        rng: Union[Iterator[Sequence[int]], Sequence[int]],
        normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None
) -> TrainState:
    """Train model specified by configuration files.

    Args:
        config: Configuration specifying training and model parameters.
        input_config: Configuration used to generate lensing images.
        workddir: Directory to which model checkpoints will be saved.
        rng: jax PRNG key.
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

    if config.batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices'
        )

    steps_per_epoch = config.steps_per_epoch
    num_steps = config.num_train_steps

    model_cls = getattr(models, config.model)
    num_outputs = len(input_config['truth_parameters'][0]) * 2
    model = model_cls(num_outputs=num_outputs, dtype=jnp.float32)

    learning_rate_schedule = get_learning_rate_schedule(config,
        base_learning_rate)

    if isinstance(rng, Iterator):
        # Create the rng key we'll use to always insert a new random
        # rotation.
        rng_state, rng_rotation_seed = jax.random.split(next(rng), 2)
    else:
        rng, rng_state = jax.random.split(rng)
    state = create_train_state(rng_state, config, model, image_size,
        learning_rate_schedule)
    state = checkpoints.restore_checkpoint(workdir, state)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    # Create the lookup tables used to speed up derivative and function
    # calculations.
    lookup_tables = input_pipeline.initialize_lookup_tables(input_config)

    p_train_step = jax.pmap(functools.partial(train_step,
        learning_rate_schedule=learning_rate_schedule),
        axis_name='batch')
    draw_image_and_truth_pmap = jax.pmap(jax.jit(jax.vmap(
        functools.partial(
            input_pipeline.draw_image_and_truth,
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
    if isinstance(rng, Iterator):
        rng_cosmo = next(rng)
    else:
        rng, rng_cosmo = jax.random.split(rng)
    cosmology_params = input_pipeline.initialize_cosmology_params(
        input_config, rng_cosmo
    )
    grid_x, grid_y = input_pipeline.generate_grids(input_config)

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    print('Initial compilation, this might take some minutes...')
    for step in range(step_offset, num_steps):

        # Generate truths and images
        if isinstance(rng, Iterator):
            rng_images = next(rng)
            rng_rotation, rng_rotation_seed = jax.random.split(
                rng_rotation_seed)
            # If we're cycling over a fixed set, we should also include
            # a random rotation.
            rotation_angle = jax.random.uniform(rng_rotation) * 2 * jnp.pi
        else:
            rng, rng_images = jax.random.split(rng)
            # If we're always drawing new images, we don't need an aditional
            # rotation.
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
        state, metrics = p_train_step(state, batch)
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
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir, config.keep_every_n_steps)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
