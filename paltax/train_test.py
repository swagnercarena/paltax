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

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scipy.stats import multivariate_normal

from paltax import input_pipeline
from paltax import models
from paltax import train

class TrainTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of training functions."""

    def test_initialized(self):
        # Test the the model includes initialized weights and biases and
        # can be applied.
        rng = jax.random.PRNGKey(0)
        model = models.ResNet50(num_outputs=2, dtype=jnp.float32)
        image_size = 128
        params, batch_stats = train.initialized(rng, image_size, model)

        self.assertTupleEqual(params['conv_init']['kernel'].shape,
            (7, 7, 1, 64))
        self.assertTupleEqual(params['bn_init']['scale'].shape,
            (64,))
        self.assertTupleEqual(batch_stats['bn_init']['mean'].shape,
            (64,))

        output, _ = model.apply({'params':params, 'batch_stats':batch_stats},
            jax.random.normal(rng, shape=(3, image_size, image_size, 1)),
            mutable=['batch_stats'])
        self.assertTupleEqual(output.shape, (3, 2))
        np.testing.assert_array_less(jnp.zeros(2), jnp.std(output, axis=0))

    def test_initialized_resnetd(self):
        # Test the initialization of the ResNetD architecture works.
        rng = jax.random.PRNGKey(0)
        model = models.ResNetD50(num_outputs=2, dtype=jnp.float32)
        image_size = 128
        params, batch_stats = train.initialized(rng, image_size, model)

        self.assertTupleEqual(params['conv_init_1']['kernel'].shape,
            (3, 3, 1, 32))
        self.assertTupleEqual(params['bn_init_1']['scale'].shape,
            (32,))
        self.assertTupleEqual(batch_stats['bn_init_1']['mean'].shape,
            (32,))

        output, _ = model.apply({'params':params, 'batch_stats':batch_stats},
            jax.random.normal(rng, shape=(3, image_size, image_size, 1)),
            mutable=['batch_stats'])
        self.assertTupleEqual(output.shape, (3, 2))
        np.testing.assert_array_less(jnp.zeros(2), jnp.std(output, axis=0))

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

        # We can repeat this experiment with a two dimensional output.
        outputs = jnp.concatenate([jax.random.normal(rng, (batch_size, 2)),
            jnp.zeros((batch_size, 2))], axis=-1)
        truth = jnp.zeros((batch_size, 2))
        self.assertAlmostEqual(gaussian_loss(outputs, truth), 1.0, places=2)


    @chex.all_variants
    def test_snpe_c_loss(self):
        # Test a few edge cases of the snpe_c loss to make sure it behaves.
        snpe_c_loss = self.variant(train.snpe_c_loss)

        # Start with prior and proposal being equal and require it gives
        # Gaussian loss.
        dim = 2
        mu_prior = jnp.zeros(dim)
        prec_prior = jnp.diag(jnp.ones(dim) * 4)
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            mu_prior, jnp.ones(dim) / 2
        )

        batch_size = int(1e4)
        rng = jax.random.PRNGKey(0)
        rng_out, rng_test, rng = jax.random.split(rng, 3)
        outputs = jnp.concatenate(
            [jax.random.normal(rng_out, (batch_size, dim)),
             jnp.zeros((batch_size, dim))], axis=-1
        )
        truth = jax.random.normal(rng_test, (batch_size, dim))

        self.assertAlmostEqual(train.gaussian_loss(outputs, truth),
                               snpe_c_loss(outputs, truth, prop_encoding,
                                           mu_prior, prec_prior))

        # We can easily test more complicated configurations so long as we don't
        # demand that our analytical solution have the right normalization.
        # Just make sure the covraiance of the prior precision matrix is larger
        # than the proposal precision matrix.

        rng_prior, rng_prop = jax.random.split(rng)
        mu_prior = jax.random.normal(rng_prior, (dim,))
        mu_prop = jax.random.normal(rng_prop, (dim,))
        prec_prop = jnp.diag(jax.random.uniform(rng_prior, (dim,)) + 0.2)
        std_prop = jnp.sqrt(1 / jnp.diag(prec_prop))
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            mu_prop, std_prop
        )
        prec_prior = prec_prop / 4

        def analytical(outputs, truth, mu_prop, prec_prop, mu_prior,
                       prec_prior):
            mu_post, log_var_post = jnp.split(outputs, 2, axis=-1)
            prec_post = jnp.diag(jnp.exp(-log_var_post[0]))
            unormed_pdf = multivariate_normal(
                mu_prop, jnp.linalg.inv(prec_prop)).logpdf(truth)
            unormed_pdf += multivariate_normal(
                mu_post[0], jnp.linalg.inv(prec_post)).logpdf(truth)
            unormed_pdf -= multivariate_normal(
                mu_prior, jnp.linalg.inv(prec_prior)).logpdf(truth)
            return -unormed_pdf

        # Use the loss on the first truth value as a reference value and
        # test the first 10 draws.
        tr_i = jnp.array([0])
        out_i = jnp.array([0])
        loss_zero = snpe_c_loss(outputs[out_i], truth[tr_i], prop_encoding,
                                mu_prior, prec_prior)
        loss_zero_unorm = analytical(
            outputs[out_i], truth[tr_i], mu_prop, prec_prop, mu_prior,
            prec_prior
        )
        for i in range(10):
            tr_i = jnp.array([i])
            loss_ratio = snpe_c_loss(outputs[out_i], truth[tr_i], prop_encoding,
                                     mu_prior, prec_prior)
            loss_ratio -= loss_zero
            loss_ratio_unorm = analytical(outputs[out_i], truth[tr_i], mu_prop,
                                          prec_prop, mu_prior, prec_prior)
            loss_ratio_unorm -= loss_zero_unorm
            self.assertAlmostEqual(loss_ratio, loss_ratio_unorm, places=4)

        # Now test a mixture of two Gausisan proposals.
        mu_prior = jnp.zeros(dim)
        prec_prior = jnp.diag(jnp.ones(dim) * 4)
        prop_encoding = jax.vmap(input_pipeline.encode_normal)(
            mu_prior, jnp.ones(dim) / 2
        )
        decay_factor = 0.2
        prop_encoding = jax.vmap(
            input_pipeline.add_normal_to_encoding, in_axes=[0, 0, 0, None])(
            prop_encoding, mu_prior, jnp.ones(dim) / 2, decay_factor
        )
        prop_encoding = jax.vmap(
            input_pipeline.add_normal_to_encoding, in_axes=[0, 0, 0, None])(
            prop_encoding, mu_prior, jnp.ones(dim) / 2, decay_factor
        )
        self.assertAlmostEqual(train.gaussian_loss(outputs, truth),
                               snpe_c_loss(outputs, truth, prop_encoding,
                                           mu_prior, prec_prior))


    @chex.all_variants
    def test_compute_metrics(self):
        # Test the the computed metrics match expectations.
        batch_size = int(1e3)
        rng = jax.random.PRNGKey(0)
        outputs = jnp.stack([jax.random.normal(rng, (batch_size,)),
            jnp.zeros((batch_size,))], axis=-1)
        truth = jnp.zeros((batch_size,1))
        gaussian_loss = train.gaussian_loss(outputs, truth)
        rmse = jnp.sqrt(jnp.mean(jnp.square(outputs[:,0] - truth)))

        outputs = jnp.expand_dims(outputs, axis=0)
        truth = jnp.expand_dims(truth, axis=0)

        compute_metrics = jax.pmap(self.variant(train.compute_metrics),
                                   axis_name='batch')
        metrics = compute_metrics(outputs, truth)

        self.assertAlmostEqual(metrics['rmse'], rmse, places=4)
        self.assertAlmostEqual(metrics['loss'], gaussian_loss, places=4)


    def test_get_learning_rate_schedule(self):
        # Test that the correct schedule is returned given the configuration
        # file.
        config = ml_collections.ConfigDict()

        # Start with cosine function.
        config.schedule_function_type = 'cosine'
        config.warmup_steps = 10
        config.num_train_steps = 100
        base_learning_rate = 0.1
        lr_schedule = train.get_learning_rate_schedule(
            config, base_learning_rate
        )
        steps = np.array([0, 5, 10, 50, 100])
        expected = np.array([
            0.0, base_learning_rate/2.0, base_learning_rate,
            (base_learning_rate / 2) * (1.0 + np.cos(40 / 90 * np.pi)),
            (base_learning_rate / 2) * (1.0 + np.cos(90 / 90 * np.pi))])
        np.testing.assert_array_almost_equal(lr_schedule(steps), expected)

        # Constant function.
        config.schedule_function_type = 'constant'
        lr_schedule = train.get_learning_rate_schedule(
            config, base_learning_rate
        )
        expected = np.ones(len(steps)) * base_learning_rate
        np.testing.assert_array_almost_equal(lr_schedule(steps), expected)

        # Linear function.
        config.schedule_function_type = 'linear'
        config.end_value_multiplier = 0.01
        lr_schedule = train.get_learning_rate_schedule(
            config, base_learning_rate
        )
        expected = [
            base_learning_rate * ((1 - step / config.num_train_steps) +
                                  config.end_value_multiplier * step /
                                  config.num_train_steps) for step in steps
        ]
        np.testing.assert_array_almost_equal(lr_schedule(steps), expected)

        # Exponential decay function.
        config.schedule_function_type = 'exp_decay'
        config.steps_per_epoch = 10
        config.decay_rate = 0.9
        lr_schedule = train.get_learning_rate_schedule(
            config, base_learning_rate
        )
        expected = [
            base_learning_rate *
            config.decay_rate ** (step / config.steps_per_epoch)
            for step in steps
        ]
        np.testing.assert_array_almost_equal(lr_schedule(steps), expected)

        # Test ValueError
        with self.assertRaises(ValueError):
            config.schedule_function_type = 'other'
            lr_schedule = train.get_learning_rate_schedule(
                config, base_learning_rate
            )