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
"""Tests for train_snpe.py."""

import functools
import pathlib
from tempfile import TemporaryDirectory

from absl.testing import parameterized
import chex
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from paltax import input_pipeline
from paltax import train
from paltax import train_snpe
from paltax import train_test


def _create_test_config():
    """Return a test configuration."""
    config = ml_collections.ConfigDict()
    config.rng_key = 0
    config.train_type = 'SNPE'

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/InputConfigs/input_config_test.py'

    # As defined in the `models` module.
    config.model = 'ResNet18VerySmall'

    config.momentum = 0.9
    config.batch_size = 32

    config.cache = False
    config.half_precision = False

    # One step of training and one step of refinement.
    steps_per_epoch = ml_collections.config_dict.FieldReference(1)
    config.steps_per_epoch = steps_per_epoch
    config.num_initial_train_steps = steps_per_epoch * 1
    config.num_steps_per_refinement = steps_per_epoch * 1
    config.num_train_steps = steps_per_epoch * 2
    config.num_refinements = ((
        config.get_ref('num_train_steps') -
         config.get_ref('num_initial_train_steps')) //
        config.get_ref('num_steps_per_refinement'))

    # Decide how often to save the model in checkpoints.
    config.keep_every_n_steps = steps_per_epoch

    # Parameters of the learning rate schedule
    config.learning_rate = 0.01
    config.warmup_steps = 1 * steps_per_epoch
    config.refinement_base_value_multiplier = 0.5

    config.mu_prior = jnp.zeros(5)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape)) / 25
    config.mu_prop_init = jnp.zeros(5)
    config.prec_prop_init = jnp.diag(jnp.ones(config.mu_prop_init.shape))
    config.prop_decay_factor = 0.0

    config.std_norm = jnp.array(
        [0.15, 0.1, 0.16, 0.16, 0.1]
    )
    config.mean_norm = jnp.array(
        [1.1, 2.0, 0.0, 0.0, 0.0]
    )

    return config


class TrainSNPETests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants
    def test_compute_metrics(self):
        # Test the the computed metrics match expectations.
        batch_size = int(1e3)
        rng = jax.random.PRNGKey(0)
        outputs = jnp.stack([jax.random.normal(rng, (batch_size,)),
            jnp.zeros((batch_size,))], axis=-1)
        truth = jnp.zeros((batch_size,1))
        mu_prop_init = jnp.ones(1)
        std_prop_init = jnp.ones(1)
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            mu_prop_init, std_prop_init
        )
        mu_prior = jnp.ones(1)
        prec_prior = jnp.array([[1]])

        loss = train.snpe_c_loss(outputs, truth, prop_encoding, mu_prior,
                                 prec_prior)
        rmse = jnp.sqrt(jnp.mean(jnp.square(outputs[:,0] - truth)))
        outputs = jnp.expand_dims(outputs, axis=0)
        truth = jnp.expand_dims(truth, axis=0)

        compute_metrics = jax.pmap(self.variant(train.compute_metrics),
                                   axis_name='batch')
        metrics = compute_metrics(outputs, truth)

        self.assertAlmostEqual(metrics['rmse'], rmse, places=4)
        self.assertAlmostEqual(metrics['loss'], loss, places=4)

    def test_get_learning_rate_schedule(self):
        # Test the learning rate schedule generated.
        config = _create_test_config()
        config.steps_per_epoch = 10
        base_learning_rate = config.learning_rate
        lr_schedule = train_snpe.get_learning_rate_schedule(
            config, base_learning_rate
        )

        steps = np.array([0, 5, 10, 12, 19])
        expected = np.array([
            0.0, base_learning_rate/2.0,
            config.refinement_base_value_multiplier * base_learning_rate,
            config.refinement_base_value_multiplier *
                (base_learning_rate / 2) * (1.0 + np.cos(2 / 10 * np.pi)),
            config.refinement_base_value_multiplier *
                (base_learning_rate / 2) * (1.0 + np.cos(9 / 10 * np.pi))]
        )
        np.testing.assert_array_almost_equal(lr_schedule(steps), expected)

    def test_train_step(self):
        # Test that an individual train step executes.
        train_state = train_test._create_train_state()
        batch = {'image': jnp.ones((5, 4, 4, 1)), 'truth': jnp.ones((5, 5))}
        config = _create_test_config()
        base_learning_rate = config.learning_rate
        lr_schedule = train_snpe.get_learning_rate_schedule(
            config, base_learning_rate
        )

        p_train_step = jax.pmap(functools.partial(train_snpe.train_step,
            learning_rate_schedule=lr_schedule),
            axis_name='batch', in_axes=(0, 0, None, None, None))

        # Need to generate the proposal encoding.
        std_prop_init = jnp.power(jnp.diag(config.prec_prop_init), -0.5)
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            config.mu_prop_init, std_prop_init
        )
        new_state, metrics = p_train_step(
            jax_utils.replicate(train_state), jax_utils.replicate(batch),
            prop_encoding, config.mu_prior, config.prec_prior
        )

        self.assertEqual(new_state.step, 1)
        self.assertTrue('loss' in metrics)

    def test_proposal_distribution_update(self):
        # Test that the current posterior is correctly overwritten.
        config = _create_test_config()
        input_config = train._get_config(config.input_config_path)
        current_posterior = jnp.array(
            [0.1, 0.2, 0.0, -0.2, 1.0,
             jnp.log(4), 0.0, jnp.log(0.25), 0.0, 0.0]
        )
        mean_norm = jnp.array([1.1, 2.0, 0.0, 0.0, 2e-3])
        std_norm = jnp.array([0.15, 0.1, 0.16, 0.16, 1.1e-3])
        mu_prop_init = jnp.zeros(5)
        prec_prop_init = jnp.diag(jnp.ones(5))
        std_prop_init = jnp.ones(5)
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            mu_prop_init, std_prop_init
        )
        prop_decay_factor = 0.5

        new_prop_encoding = train_snpe.proposal_distribution_update(
            current_posterior, mean_norm, std_norm, mu_prop_init,
            prec_prop_init, prop_encoding, prop_decay_factor,
            input_config)
        lensing_config = input_config['lensing_config']

        # Check that the input config matches
        for i in range(len(input_config['truth_parameters'][0])):
            lensing_object = input_config['truth_parameters'][0][i]
            lensing_key = input_config['truth_parameters'][1][i]
            object_distribution = lensing_config[lensing_object][lensing_key]
            object_prop_encoding = new_prop_encoding[i]
            mask = input_pipeline._get_normal_weights(object_prop_encoding) > 0
            np.testing.assert_array_almost_equal(
                (input_pipeline._get_normal_mean(object_prop_encoding)
                    * std_norm[i] + mean_norm[i]) * mask,
                input_pipeline._get_normal_mean(object_distribution))
            np.testing.assert_array_almost_equal(
                (input_pipeline._get_normal_std(object_prop_encoding)
                    * std_norm[i]) * mask,
                input_pipeline._get_normal_std(object_distribution))

            # Also test the weights
            weights = input_pipeline._get_normal_weights(object_prop_encoding)
            np.testing.assert_array_almost_equal(
                weights[0],1 - prop_decay_factor
            )
            np.testing.assert_array_almost_equal(weights[1], prop_decay_factor)
            self.assertAlmostEqual(jnp.sum(weights[2:]), 0.0)

        # Check that the values match the posterior with the bounds applied.
        mean_vmap = jax.vmap(input_pipeline._get_normal_mean)(new_prop_encoding)
        np.testing.assert_array_almost_equal(
            mu_prop_init, mean_vmap[:, 1])
        np.testing.assert_array_almost_equal(
            jnp.array([0.0, 0.2, 0.0, -0.2, 1.0]), mean_vmap[:, 0])
        std_vmap = jax.vmap(input_pipeline._get_normal_std)(new_prop_encoding)
        np.testing.assert_array_almost_equal(
            std_prop_init, std_vmap[:, 1])
        np.testing.assert_array_almost_equal(
            jnp.array([1.0, 1.0, 0.5, 1.0, 1.0]), std_vmap[:, 0])

    def test_proposal_distribution_update_avg(self):
        # Test that the current posterior is correctly overwritten when
        # averaging is requested.
        config = _create_test_config()
        input_config = train._get_config(config.input_config_path)
        current_posterior = jnp.array(
            [0.1, 0.2, 0.0, -0.2, 1.0,
             jnp.log(4), 0.0, jnp.log(0.25), 0.0, 0.0]
        )
        mean_norm = jnp.array([1.1, 2.0, 0.0, 0.0, 2e-3])
        std_norm = jnp.array([0.15, 0.1, 0.16, 0.16, 1.1e-3])
        mu_prop_init = jnp.zeros(5)
        prec_prop_init = jnp.diag(jnp.ones(5))
        std_prop_init = jnp.ones(5)
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            mu_prop_init, std_prop_init
        )
        prop_decay_factor = -1.0

        new_prop_encoding = train_snpe.proposal_distribution_update(
            current_posterior, mean_norm, std_norm, mu_prop_init,
            prec_prop_init, prop_encoding, prop_decay_factor,
            input_config)
        # Only need to check that the correct weight updating scheme
        # was used.
        for i in range(len(input_config['truth_parameters'][0])):
            object_prop_encoding = new_prop_encoding[i]
            weights = input_pipeline._get_normal_weights(object_prop_encoding)
            np.testing.assert_array_almost_equal(weights[0], weights[1])
            self.assertAlmostEqual(jnp.sum(weights[2:]), 0.0)

    def test_train_and_evaluate_snpe(self):
        # Test that train and evaluation works given a configuration
        # file.
        config = _create_test_config()
        input_config = train._get_config(config.input_config_path)
        rng = jax.random.PRNGKey(2)
        target_image = jnp.ones((4,4))

        with TemporaryDirectory() as tmp_dir_name:
            state = train_snpe.train_and_evaluate_snpe(
                config, input_config, tmp_dir_name, rng, target_image
            )
            self.assertEqual(state.step, 2)
