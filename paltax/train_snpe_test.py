# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for train.py."""

import os

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

from paltax import input_pipeline
from paltax import train
from paltax import train_snpe

TEST_INPUT_CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                                      'InputConfigs/input_config_test.py')

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

    # def test_get_learning_rate_schedule(self):
    #     self.assertTrue(False)

    # def test_train_step(self):
    #     self.assertTrue(False)

    def test_proposal_distribution_update(self):
        # Test that the current posterior is correctly overwritten.
        input_config = train._get_config(TEST_INPUT_CONFIG_PATH)
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
        input_config = train._get_config(TEST_INPUT_CONFIG_PATH)
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