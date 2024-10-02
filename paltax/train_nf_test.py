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
"""Tests for train.py."""

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
from scipy.stats import multivariate_normal

from paltax import input_pipeline
from paltax import maf_flow
from paltax import models
from paltax import train_nf
from paltax import train_test


def _create_train_state():
    """Create generic train state for testing."""
    rng = jax.random.PRNGKey(0)
    config = ml_collections.ConfigDict()
    embedding_module = models.ResNet18VerySmall(
        num_outputs=2, dtype=jnp.float32
    )
    maf_module = maf_flow.MAF(
        n_dim=2, n_maf_layers=2, hidden_dims=[4,4], activation='gelu'
    )
    model = maf_flow.EmbeddedFlow(embedding_module, maf_module)
    image_size = 4
    parameter_dim = 2
    learning_rate_schedule = train_test._get_learning_rate()

    train_state = train_nf.create_train_state_nf(
        rng, config, model, image_size, parameter_dim,
        learning_rate_schedule
    )

    return train_state


class TrainNFTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of training functions."""

    def test_initialized(self):
        # Test the the model includes initialized weights and biases and
        # can be applied.
        rng = jax.random.PRNGKey(0)
        embedding_module = models.ResNet18VerySmall(
            num_outputs=2, dtype=jnp.float32
        )
        maf_module = maf_flow.MAF(
            n_dim=2, n_maf_layers=2, hidden_dims=[4,4], activation='gelu'
        )
        model = maf_flow.EmbeddedFlow(embedding_module, maf_module)
        image_size = 128
        params, batch_stats = train_nf.initialized(rng, image_size, 2, model)

        self.assertTupleEqual(
            params['embedding_module']['conv_init']['kernel'].shape,
            (7, 7, 1, 8)
        )
        self.assertTupleEqual(
            batch_stats['embedding_module']['bn_init']['mean'].shape,
            (8,)
        )

        output, _ = model.apply(
            {'params':params, 'batch_stats':batch_stats},
            jax.random.normal(rng, shape=(3, 2)),
            jax.random.normal(rng, shape=(3, image_size, image_size, 1)),
            mutable=['batch_stats']
        )
        self.assertTupleEqual(output.shape, (3,))

    def test_create_state_nf(self):
        # Test that the train state is created without issue.
        train_state = _create_train_state()
        self.assertEqual(train_state.step, 0)
