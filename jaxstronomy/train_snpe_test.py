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
import jax.numpy as jnp
import numpy as np

from jaxstronomy import train
from jaxstronomy import train_snpe

TEST_INPUT_CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                                      'InputConfigs/input_config_test.py')

class TrainSNPETests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    def test_compute_metrics(self):
        self.assertTrue(False)

    def test_get_learning_rate_schedule(self):
        self.assertTrue(False)

    def test_train_step(self):
        self.assertTrue(False)

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

        mu_prop, prec_prop = train_snpe.proposal_distribution_update(
            current_posterior, mean_norm, std_norm, mu_prop_init,
            prec_prop_init, input_config)
        lensing_config = input_config['lensing_config']

        # Check that the input config matches
        for i in range(len(input_config['truth_parameters'][0])):
            # TODO This will break if I ever change the encoding. Not a great
            # test in that sense.
            lensing_object = input_config['truth_parameters'][0][i]
            lensing_key = input_config['truth_parameters'][1][i]
            object_distribution = lensing_config[lensing_object][lensing_key]
            self.assertAlmostEqual(mu_prop[i] * std_norm[i] + mean_norm[i],
                                   object_distribution[4])
            self.assertAlmostEqual(1/jnp.sqrt(prec_prop[i,i]) * std_norm[i],
                                   object_distribution[5])

        # Check that the values math the posterior with the bounds applied.
        np.testing.assert_array_almost_equal(
            jnp.array([0.0, 0.2, 0.0, -0.2, 1.0]), mu_prop)
        np.testing.assert_array_almost_equal(
            jnp.diag(jnp.array([1.0, 1.0, 1/0.25, 1.0, 1.0])), prec_prop)
