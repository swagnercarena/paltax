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

import functools

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxstronomy import models
from jaxstronomy import train

class TrainTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    # def test_initialized(self):
    #     # Test the the model includes initialized weights and biases and
    #     # can be applied.
    #     rng = jax.random.PRNGKey(0)
    #     model = models.ResNet50(num_outputs=2, dtype=jnp.float32)
    #     image_size = 128
    #     params, batch_stats = train.initialized(rng, image_size, model)

    #     self.assertTupleEqual(params['conv_init']['kernel'].shape,
    #         (7, 7, 1, 64))
    #     self.assertTupleEqual(params['bn_init']['scale'].shape,
    #         (64,))
    #     self.assertTupleEqual(batch_stats['bn_init']['mean'].shape,
    #         (64,))

    #     output, _ = model.apply({'params':params, 'batch_stats':batch_stats},
    #         jax.random.normal(rng, shape=(3, image_size, image_size, 1)),
    #         mutable=['batch_stats'])
    #     self.assertTupleEqual(output.shape, (3, 2))
    #     np.testing.assert_array_less(jnp.zeros(2), jnp.std(output, axis=0))

    @chex.all_variants
    def test_gaussian_loss(self):
        # Test that the loss function is equivalent to applying a Gaussian.
        batch_size = int(1e7)
        rng = jax.random.PRNGKey(0)
        outputs = jnp.stack([jax.random.normal(rng, (batch_size,)),
            jnp.zeros((batch_size,))], axis=-1)
        truth = jnp.zeros((batch_size,1))
        gaussian_loss = self.variant(train.gaussian_loss)

        # With variance 1 and dropping the factor of 2 pi, the negative entropy
        # should simply be 0.5
        self.assertAlmostEqual(gaussian_loss(outputs, truth), 0.5, places=2)

        outputs = jnp.stack([
            jax.random.normal(rng, (batch_size,)) * jnp.sqrt(jnp.e),
            jnp.ones((batch_size,))], axis=-1)
        self.assertAlmostEqual(gaussian_loss(outputs, truth), 1.0, places=2)
