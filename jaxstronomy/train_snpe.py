# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for dark matter substructure inference."""

import bisect
import copy
import functools
import time
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from absl import app
from absl import flags
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import common_utils
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import optax

from jaxstronomy import input_pipeline
from jaxstronomy import models
from jaxstronomy import train
from jaxstronomy import utils


FLAGS = flags.FLAGS
# Most flags come from train.
flags.DEFINE_string('target_image_path', None, 'path to the target image.')


def compute_metrics(
        outputs: jnp.ndarray, truth: jnp.ndarray, mu_prop, prec_prop,
        mu_prior, prec_prior) -> Mapping[str, jnp.ndarray]:
    """Compute the performance metrics of the output.

    Args:
        outputs: Values outputted by the model.
        truth: True value of the parameters.

    Returns:
        Value of each of the metrics.
    """
    loss = train.snpe_c_loss(outputs, truth, mu_prop, prec_prop, mu_prior,
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


def train_step(state, batch, mu_prop, prec_prop, mu_prior, prec_prior,
               learning_rate_schedule):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        outputs, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats'])
        loss = train.snpe_c_loss(outputs, batch['truth'], mu_prop, prec_prop,
                                 mu_prior, prec_prior)

        return loss, (new_model_state, outputs)

    step = state.step
    lr = learning_rate_schedule(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, outputs = aux[1]
    metrics = compute_metrics(outputs, batch['truth'], mu_prop, prec_prop,
                              mu_prior, prec_prior)
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])

    return new_state, metrics


def train_and_evaluate_snpe(
        config: ml_collections.ConfigDict,
        input_config: dict,
        workdir: str,
        rng: Union[Iterator[Sequence[int]], Sequence[int]],
        image_size: int, learning_rate: float,
        target_image: jnp.ndarray,
        normalize_config: Optional[Mapping[str, Mapping[str, jnp.ndarray]]] = None
):
    """
    """

    if normalize_config is None:
        normalize_config = copy.deepcopy(input_config['lensing_config'])

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    steps_per_epoch = config.steps_per_epoch
    num_initial_steps = config.num_initial_train_steps
    num_steps_per_refinement = config.num_steps_per_refinement
    num_steps = num_initial_steps
    num_steps += num_steps_per_refinement * config.num_refinements

    refinement_step_list = []
    for refinement in range(config.num_refinements):
        refinement_step_list.append(
            num_initial_steps + num_steps_per_refinement * refinement
        )

    base_learning_rate = learning_rate * config.batch_size / 256.

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
    state = train.create_train_state(rng_state, config, model, image_size,
        learning_rate_schedule)
    state = train.restore_checkpoint(state, workdir)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    mu_prior = jax_utils.replicate(config.mu_prior)
    prec_prior = jax_utils.replicate(config.prec_prior)
    mu_prop_init = jax_utils.replicate(config.mu_prop_init)
    prec_prop_init = jax_utils.replicate(config.prec_prop_init)

    p_train_step = jax.pmap(functools.partial(train_step,
        learning_rate_schedule=learning_rate_schedule),
        axis_name='batch')
    p_get_outputs = jax.pmap(train.get_outputs, axis_name='batch')

    draw_image_and_truth_pmap = jax.pmap(jax.jit(jax.vmap(
        functools.partial(
            input_pipeline.draw_image_and_truth,
            all_models=input_config['all_models'],
            principal_md_index=input_config['principal_md_index'],
            principal_source_index=input_config['principal_source_index'],
            kwargs_simulation=input_config['kwargs_simulation'],
            kwargs_detector=input_config['kwargs_detector'],
            kwargs_psf=input_config['kwargs_psf'],
            truth_parameters=input_config['truth_parameters'],
            normalize_config=normalize_config),
        in_axes=(None, None, None, None, 0, None))),
        in_axes=(None, None, None, None, 0, None)
    )
    if isinstance(rng, Iterator):
        rng_cosmo = next(rng)
    else:
        rng, rng_cosmo = jax.random.split(rng)
    cosmology_params = input_pipeline.intialize_cosmology_params(input_config,
                                                                 rng_cosmo)
    grid_x, grid_y = input_pipeline.generate_grids(input_config)

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()

    print('Initial compilation, this might take some minutes...')

    # Get our initial proposal. May be in the midst of refinement here.
    mu_prop = mu_prop_init
    prec_prop = prec_prop_init
    target_batch = jax_utils.replicate({'image': target_image})
    std_norm = jnp.array([0.15, 0.1, 0.16, 0.16, 0.1, 0.1, 0.05, 0.05, 0.16,
                          0.16, 1.1e-3])
    mean_norm = jnp.array([1.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           2e-3])

    if bisect.bisect_left(refinement_step_list, step_offset) > 0:
        # This restart isn't perfect, but we're not going to load weights again
        # so we'll just use whatever model we have now.
        # TODO This should be a seperate function. Need to change this.
        current_posterior = train.cross_replica_mean(
            p_get_outputs(state, target_batch)[0])[0]

        # Get the new proposal distribution.
        mu_prop, log_var_prop = jnp.split(current_posterior[0], 2, axis=-1)
        prec_prop = jnp.diag(jnp.exp(-log_var_prop))

        # Modify the input config.
        # TODO this is hard coded and needs to be fixed.
        mu_prop_unormalized = mu_prop * std_norm + mean_norm
        std_prop_unormalized = jnp.exp(log_var_prop/2) * std_norm

        for truth_i in range(len(input_config['truth_parameters'][0])):
            rewrite_object = input_config['truth_parameters'][0][truth_i]
            rewrite_key = input_config['truth_parameters'][1][truth_i]
            input_config['lensing_config'][rewrite_object][rewrite_key] = (
                input_pipeline.encode_normal(
                    mean=mu_prop_unormalized[truth_i],
                    std=std_prop_unormalized[truth_i])
            )

        mu_prop = jax_utils.replicate(mu_prop)
        prec_prop = jax_utils.replicate(prec_prop)

    for step in range(step_offset, num_steps):

        # Check if it's time for a refinement.
        if step in refinement_step_list:
            print(f'Starting refinement stage at step {step}')

            # Get the predictions for the derised image and turn it into a new
            # proposal distribution
            current_posterior = train.cross_replica_mean(
                p_get_outputs(state, target_batch)[0])[0]

            # Get the new proposal distribution.
            mu_prop, log_var_prop = jnp.split(current_posterior[0], 2, axis=-1)
            prec_prop = jnp.diag(jnp.exp(-log_var_prop))

            # Modify the input config.
            # TODO this is hard coded and needs to be fixed.
            mu_prop_unormalized = mu_prop * std_norm + mean_norm
            std_prop_unormalized = jnp.exp(log_var_prop/2) * std_norm

            for truth_i in range(len(input_config['truth_parameters'][0])):
                rewrite_object = input_config['truth_parameters'][0][truth_i]
                rewrite_key = input_config['truth_parameters'][1][truth_i]
                input_config['lensing_config'][rewrite_object][rewrite_key] = (
                    input_pipeline.encode_normal(
                        mean=mu_prop_unormalized[truth_i],
                        std=std_prop_unormalized[truth_i])
                )

            mu_prop = jax_utils.replicate(mu_prop)
            prec_prop = jax_utils.replicate(prec_prop)

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
        state, metrics = p_train_step(state, batch, mu_prop,
                                      prec_prop, mu_prior, prec_prior)
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


def main(_):
    """Train neural network model with configuration defined by flags."""
    train_config = train._get_config(FLAGS.train_config_path)
    input_config = train._get_config(FLAGS.input_config_path)
    image_size = input_config['kwargs_detector']['n_x']
    target_image = jnp.load(FLAGS.target_image_path)
    target_image = jnp.expand_dims(target_image, axis=(0,-1))

    rng = jax.random.PRNGKey(0)
    if FLAGS.num_unique_batches > 0:
        rng_list = jax.random.split(rng, FLAGS.num_unique_batches)
        rng = utils.random_permutation_iterator(rng_list, rng)
    train_and_evaluate_snpe(train_config, input_config, FLAGS.workdir, rng,
                       image_size, FLAGS.learning_rate, target_image)


if __name__ == '__main__':
    app.run(main)
