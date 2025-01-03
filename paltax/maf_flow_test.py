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
"""Tests for maf_flow.py."""

from typing import List

from absl.testing import absltest
import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from paltax import maf_flow
from paltax import models


class UtilTest(chex.TestCase):
    """Runs tests of MAF utility functions."""

    def test___made_degrees(self):
        # Test that the correct degrees are returned for some test
        # configurations.
        input_size = 3
        hidden_dims = jnp.array([4, 7, 5])

        degrees = maf_flow._made_degrees(input_size, hidden_dims)
        expected = [
            jnp.array([1, 2, 3]), jnp.array([1, 2, 1, 2]),
            jnp.array([1, 2, 1, 2, 1, 2, 1]), jnp.array([1, 2, 1, 2, 1])
        ]
        for deg, exp in zip(degrees, expected):
            np.testing.assert_array_almost_equal(deg, exp)

        # Repeat the test with context
        n_context = 2
        expected = [
            jnp.array([1, 2, 3]), jnp.array([2, 2, 2, 2]),
            jnp.array([2, 2, 2, 2, 2, 2, 2]), jnp.array([2, 2, 2, 2, 2])
        ]
        degrees = maf_flow._made_degrees(input_size, hidden_dims, n_context)
        for deg, exp in zip(degrees, expected):
            np.testing.assert_array_almost_equal(deg, exp)

    def test__made_masks(self):
        # Test that the masks have the desired shape and boolean entries.
        input_size = 4
        hidden_dims = [5]
        n_context = 2
        degrees = maf_flow._made_degrees(input_size, hidden_dims, n_context)

        masks = maf_flow._made_masks(degrees, n_context)
        expected = [
            jnp.array([
                [ True,  True,  True,  True,  True],
                [True, True, True,  True,  True],
                [False, True, False, True, False],
                [False, False, False, False, False]
            ]),
            jnp.array([
                [True,  True], [False,  True], [True,  True], [False,  True],
                [True,  True]
            ])
        ]
        for mask, exp in zip(masks, expected):
            np.testing.assert_array_almost_equal(mask, exp)

        # Make sure the product mask behaves as expected
        product = jnp.array(
            [[True, True], [True, True], [False, True], [False, False]]
        )
        np.testing.assert_array_almost_equal(
            product, jnp.matmul(masks[0], masks[1])
        )

    def test__made_dense_autoregressive_masks(self):
        # Test that the masks match the output from _made_masks except for the
        # final mask.
        n_params = 2
        n_total_cond = 3
        n_context = 3
        hidden_dims = [5]
        masks = maf_flow._made_dense_autoregressive_masks(
            n_params, n_total_cond, n_context, hidden_dims
        )

        # Compare to _made_masks
        degrees = maf_flow._made_degrees(n_total_cond, hidden_dims, n_context)
        masks_expected = maf_flow._made_masks(degrees, n_context)
        for mask, mask_exp in zip(masks[:-1], masks_expected[:-1]):
            np.testing.assert_array_almost_equal(mask, mask_exp)

        # Check the the columns are duplicated.
        for i in range(masks_expected[-1].shape[-1]):
            np.testing.assert_array_almost_equal(
                masks[-1][:, 2 * i], masks_expected[-1][:, i]
            )
            np.testing.assert_array_almost_equal(
                masks[-1][:, 2 * i + 1], masks_expected[-1][:, i]
            )


class MaskedDenseTest(chex.TestCase):
    """Run tests on the MaskedDense Layer"""

    @chex.all_variants
    def test_call(self):
        # Test that the masking behaves as expected.
        rng = jax.random.PRNGKey(0)
        n_params = 2
        n_total_cond = 4
        n_context = 1
        hidden_dims = [5]
        masks = maf_flow._made_dense_autoregressive_masks(
            n_params, n_total_cond, n_context, hidden_dims
        )

        md_layer = [
            maf_flow.MaskedDense(features=masks[0].shape[-1], mask=masks[0]),
            maf_flow.MaskedDense(features=masks[1].shape[-1], mask=masks[1])
        ]
        params = [
            md_layer[0].init(rng, jnp.ones((1, n_total_cond))),
            md_layer[1].init(rng, jnp.ones((1, hidden_dims[0])))
        ]
        apply_func = [self.variant(layer.apply) for layer in md_layer]

        # First, test that the output has the desired shape
        batch_size = 2
        x = jax.random.normal(rng, ((batch_size, n_total_cond)))
        x = apply_func[0](params[0], x)
        x = apply_func[1](params[1], x)
        self.assertTupleEqual(
            x.shape, (batch_size, n_params * (n_total_cond - n_context))
        )

        # Now check that the autoregressive property is respected.
        y = jax.random.normal(rng, ((batch_size, n_total_cond)))
        y = y.at[:,2].set(0.0)
        y = apply_func[0](params[0], y)
        y = apply_func[1](params[1], y)
        np.testing.assert_array_almost_equal(x[:, :4], y[:, :4])
        np.testing.assert_array_less(
            jnp.zeros_like(x[:, 4:]), jnp.abs(x[:, 4:] - y[:, 4:])
        )


class MADETest(chex.TestCase):
    """Run tests on the MADE module."""

    @chex.all_variants
    def test_call(self):
        # Test that the MADE is autoregressive and calls without error.
        in_dim = 3
        cond_dim = 1
        hidden_dims = [5]
        rng = jax.random.PRNGKey(0)

        made_layer = maf_flow.MADE(hidden_dims)
        apply_fn = self.variant(made_layer.apply)
        params = made_layer.init(
            rng, jnp.ones((1, in_dim)), jnp.ones((1, cond_dim))
        )
        batch_size = 2
        x = jax.random.normal(rng, ((batch_size, in_dim)))
        context = jax.random.normal(rng, ((batch_size, cond_dim)))
        y = made_layer.apply(params, x, context)

        # Check the shape
        self.assertTupleEqual(y.shape, (batch_size, in_dim, 2))

        # Check the autoregressive property of the MADE
        context_mod = context.at[:, 0].set(0.0)
        y_mod = apply_fn(params, x, context_mod)
        np.testing.assert_array_less(
            jnp.zeros_like(y), jnp.abs(y - y_mod)
        )

        x_mod = x.at[:, 1].set(0.0)
        y_mod = apply_fn(params, x_mod, context)
        np.testing.assert_array_almost_equal(y[:, :2], y_mod[:, :2])
        np.testing.assert_array_less(
            jnp.zeros_like(y[:, 2:]), jnp.abs(y[:, 2:] - y_mod[:, 2:])
        )


class _MAFLayerWrapper(nn.Module):
    hidden_dims: List[int]
    activation: str

    def setup(self):
        self.made_layer = maf_flow.MADE(self.hidden_dims, self.activation)
        self.maf_layer = maf_flow._MAFLayer(self.made_layer)

class MAFLayerTest(chex.TestCase):
    """Run tests on the MAFLayer module."""

    @chex.all_variants
    def test_forward_inverse(self):
        # Test the forward and inverse pass of the MAFLayer
        class MAFLayerWrapper(_MAFLayerWrapper):
            def __call__(self, x, y, context):
                return (
                    self.maf_layer.forward(x, context),
                    self.maf_layer.inverse(y, context)
                )

        # Initialize our MAFLayer wrapper.
        hidden_dims = [5]
        activation = 'tanh'
        maf_wrap = MAFLayerWrapper(hidden_dims, activation)
        apply_fn = self.variant(maf_wrap.apply)
        in_dim = 3
        n_context = 2
        rng = jax.random.PRNGKey(0)
        params = maf_wrap.init(
            rng,
            jnp.ones((1, in_dim)),
            jnp.ones((1, in_dim)),
            jnp.ones((1, n_context))
        )

        # Test the dimensionality of the forward and inverse pass.
        batch_size = 2
        rng_x, rng_y = jax.random.split(rng)
        x = jax.random.normal(rng_x, (batch_size, in_dim))
        y = jax.random.normal(rng_y, (batch_size, in_dim))
        context = jnp.ones((batch_size, n_context))
        y_maf, x_maf = apply_fn(params, x, y, context)
        self.assertTupleEqual(y_maf.shape, (batch_size, in_dim))
        self.assertTupleEqual(x_maf.shape, (batch_size, in_dim))

        # Test that the inverse maps back to the original x and vice versa.
        y_inv, x_inv = apply_fn(params, x_maf, y_maf, context)
        np.testing.assert_array_almost_equal(
            x, x_inv
        )
        np.testing.assert_array_almost_equal(
            y, y_inv
        )

    @chex.all_variants
    def test_log_det(self):
        # Test the log determinent of the MAF layer.
        class MAFLayerWrapper(_MAFLayerWrapper):
            def __call__(self, x, y, context):
                return (
                    self.maf_layer.forward(x, context),
                    self.maf_layer.forward_log_det_jacobian(x, context),
                    self.maf_layer.inverse_log_det_jacobian(y, context)
                )

        # Initialize our MAFLayer wrapper.
        hidden_dims = [5]
        activation = 'tanh'
        maf_wrap = MAFLayerWrapper(hidden_dims, activation)
        apply_fn = self.variant(maf_wrap.apply)
        in_dim = 3
        n_context = 2
        rng = jax.random.PRNGKey(0)
        params = maf_wrap.init(
            rng,
            jnp.ones((1, in_dim)),
            jnp.ones((1, in_dim)),
            jnp.ones((1, n_context))
        )

        # Test that the jacobian is non-trivial and correctly inverted.
        batch_size = 2
        rng_x, rng_y = jax.random.split(rng)
        x = jax.random.normal(rng_x, (batch_size, in_dim))
        y = jax.random.normal(rng_y, (batch_size, in_dim))
        context = jnp.ones((batch_size, n_context))
        y_maf, log_jac_fow, _ = apply_fn(params, x, y, context)
        _, _, log_jac_inv = apply_fn(params, x, y_maf, context)
        self.assertTupleEqual(log_jac_inv.shape, (batch_size,))
        np.testing.assert_array_less(
            jnp.zeros_like(log_jac_fow), jnp.abs(log_jac_fow)
        )
        np.testing.assert_array_less(
            jnp.zeros_like(log_jac_inv), jnp.abs(log_jac_inv)
        )
        np.testing.assert_array_almost_equal(
            log_jac_fow, -log_jac_inv
        )


class PermuteLayerTest(chex.TestCase):
    """Run tests on the PermuteLayer module."""

    @chex.all_variants
    def test_call(self):
        # Test that the permutation is fixed and calls without errors.
        permute_layer = maf_flow._PermuteLayer()
        rng = jax.random.PRNGKey(0)
        in_dim = 4
        params = permute_layer.init(rng, jnp.ones((1, in_dim)))
        apply_fn = self.variant(permute_layer.apply)

        batch_size = 2
        x = jax.random.normal(rng, (batch_size, in_dim))
        x_perm = apply_fn(params, x)
        # Check that the permutation does not return the same array
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_almost_equal, x, x_perm
        )
        # Check that the sum is preserved along the permuted axis.
        np.testing.assert_array_almost_equal(
            jnp.sum(x, axis=-1), jnp.sum(x_perm, axis=-1)
        )

        # Check that a second call returns the same permutation.
        x_perm_second = apply_fn(params, x)
        np.testing.assert_array_almost_equal(x_perm, x_perm_second)

        # Check that the inverse bool works as desired.
        x_recon = apply_fn(params, x_perm, True)
        np.testing.assert_array_almost_equal(x, x_recon)

class PermuteTest(chex.TestCase):
    """Run tests on the Permute bijection."""

    @chex.all_variants
    def test_forward_inverse(self):
        # Test the permutation layer permutes and inverts the permutation.
        class _PermuteWrapper(nn.Module):
            def setup(self):
                self.permute_layer = maf_flow._PermuteLayer()
                self.permute = maf_flow.Permute(self.permute_layer)

            def __call__(self, x, y):
                return (
                    self.permute.forward_and_log_det(x, None) +
                    self.permute.inverse_and_log_det(y, None)
                )

        perm_wrap = _PermuteWrapper()
        apply_fn = self.variant(perm_wrap.apply)
        in_dim = 3
        rng = jax.random.PRNGKey(0)
        params = perm_wrap.init(
            rng, jnp.ones((1, in_dim)), jnp.ones((1, in_dim))
        )

        # Test that the jacobians are 0 and that the permutation can be
        # inverted.
        batch_size = 2
        rng_x, rng_y = jax.random.split(rng)
        x = jax.random.normal(rng_x, (batch_size, in_dim))
        y = jax.random.normal(rng_y, (batch_size, in_dim))
        y_perm, log_det, x_perm, log_det_inv = apply_fn(params, x, y)

        self.assertTupleEqual(log_det.shape, (batch_size,))
        self.assertTupleEqual(x_perm.shape, (batch_size, in_dim))
        self.assertTupleEqual(y_perm.shape, (batch_size, in_dim))
        np.testing.assert_array_almost_equal(
            jnp.sum(x, axis=-1), jnp.sum(y_perm, axis=-1)
        )
        np.testing.assert_array_almost_equal(
            jnp.sum(x_perm, axis=-1), jnp.sum(y, axis=-1)
        )
        np.testing.assert_array_almost_equal(log_det, jnp.zeros_like(log_det))
        np.testing.assert_array_almost_equal(log_det_inv, log_det)

        y_recon, _, x_recon, _ = apply_fn(params, x_perm, y_perm)
        np.testing.assert_array_almost_equal(x, x_recon)
        np.testing.assert_array_almost_equal(y, y_recon)


class ChainConditionalTest(chex.TestCase):
    """Run tests on the ChainConditional bijection."""

    @chex.all_variants
    def test_forward_inverse(self):
        # Test that the forward and inverse operations behave as expected
        # when combining a permutation with a MAF.
        class _CCWrapper(nn.Module):
            hidden_dims: List[int]
            activation: str

            def setup(self):
                self.permute_layer = maf_flow._PermuteLayer()
                self.permute = maf_flow.Permute(self.permute_layer)
                self.made_layer = maf_flow.MADE(
                    self.hidden_dims, self.activation
                )
                self.maf_layer = maf_flow._MAFLayer(self.made_layer)
                self.cc_layer = maf_flow.ChainConditional(
                    [self.permute, self.maf_layer]
                )

            def __call__(self, x, y, context):
                return (
                    self.cc_layer.forward_and_log_det(x, context) +
                    self.cc_layer.inverse_and_log_det(y, context)
                )

        hidden_dims = [5]
        activation = 'tanh'
        in_dim = 3
        n_context = 2
        rng = jax.random.PRNGKey(0)
        cc_wrap = _CCWrapper(hidden_dims, activation)
        apply_fn = self.variant(cc_wrap.apply)
        params = cc_wrap.init(
            rng, jnp.ones((1, in_dim)), jnp.ones((1, in_dim)),
            jnp.ones((1, n_context))
        )

        batch_size = 2
        rng_x, rng_y = jax.random.split(rng)
        x = jax.random.normal(rng_x, (batch_size, in_dim))
        y = jax.random.normal(rng_y, (batch_size, in_dim))
        context = jax.random.normal(rng_y, (batch_size, n_context))
        y_cc, log_det, x_cc, _ = apply_fn(params, x, y, context)
        y_recon, _, x_recon, log_det_inv = apply_fn(
            params, x_cc, y_cc, context
        )
        np.testing.assert_array_almost_equal(x, x_recon)
        np.testing.assert_array_almost_equal(y, y_recon)
        np.testing.assert_array_almost_equal(log_det, -log_det_inv)


class _TransformedConditionalWrapper(nn.Module):
    hidden_dims: List[int]
    activation: str
    in_dim: int

    def setup(self):
        self.made_layer = maf_flow.MADE(
            self.hidden_dims, self.activation
        )
        self.maf_layer = maf_flow._MAFLayer(self.made_layer)
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.in_dim))
        self.flow = maf_flow.TransformedConditional(
            self.base_dist, self.maf_layer
        )


class TransformedConditionalTest(chex.TestCase):
    """Run tests on the TransformedConditional class."""

    @chex.all_variants
    def test_forward(self):
        # Test that we draw samples with an arbitrary batch dimension.
        class _TC_Wrapper(_TransformedConditionalWrapper):

            def __call__(self, y, context):
                return self.flow.log_prob(y, context)

            def sample(self, rng, context, sample_shape):
                return self.flow.sample(rng, context, sample_shape)

            def sample_and_log_prob(self, rng, context, sample_shape):
                return self.flow.sample_and_log_prob(rng, context, sample_shape)

            def inverse(self, y, context):
                x, log_det_inv = jax.vmap(
                    self.maf_layer.inverse_and_log_det, in_axes=[0, None]
                )(y.reshape((-1, y.shape[-1])), context)
                return x.reshape(y.shape), log_det_inv.reshape(y.shape[:-1])

        hidden_dims = [5]
        activation = 'tanh'
        in_dim = 3
        n_context = 2
        rng = jax.random.PRNGKey(0)
        tc_wrapper = _TC_Wrapper(hidden_dims, activation, in_dim)
        apply_fn = self.variant(
            tc_wrapper.apply,
            static_argnames = ['method', 'sample_shape']
        )

        params = tc_wrapper.init(
            rng, jnp.ones((1, in_dim)), jnp.ones((1, n_context))
        )

        batch_size = 5
        y = jax.random.normal(rng, (batch_size, in_dim))
        context = jax.random.normal(rng, (batch_size, n_context))

        # Calculate the log probability and check its shape.
        log_prob = apply_fn(params, y, context)
        self.assertTupleEqual(log_prob.shape, (batch_size,))

        # Draw samples and check their shape
        sample_shape = (batch_size * 2, 2)
        samples = apply_fn(
            params, rng, context[0], method='sample', sample_shape=sample_shape
        )
        self.assertTupleEqual(samples.shape, sample_shape + (in_dim,))

        # Finally check that the log probabilities are self consistent.
        samples, log_prob = apply_fn(
            params, rng, context[0], method='sample_and_log_prob',
            sample_shape=(batch_size * 2, 2)
        )
        x_samples, log_det_inv = apply_fn(
            params, samples, context[0], method='inverse'
        )
        log_prob_manual = distrax.MultivariateNormalDiag(
            jnp.zeros(x_samples.shape[-1])
        ).log_prob(x_samples)
        np.testing.assert_array_almost_equal(
            log_prob, log_prob_manual + log_det_inv
        )


class MAFTest(chex.TestCase):
    """Run tests on the MAF module."""

    @chex.all_variants
    def test_apply(self):
        # Test the the MAF module initializes and returns transformed
        # variables.
        n_dim = 8
        n_context = 4
        n_maf_layers = 3
        hidden_dims = [16]
        activation = 'gelu'
        maf = maf_flow.MAF(n_dim, n_maf_layers, hidden_dims, activation)

        rng = jax.random.PRNGKey(0)
        params = maf.init(rng, jnp.ones((1, n_dim)), jnp.ones((1, n_context)))
        apply_fn = self.variant(
            maf.apply, static_argnames=['method', 'sample_shape']
        )

        batch_size = 4
        y = jax.random.normal(rng, (batch_size, n_dim))
        context = jax.random.normal(rng, (batch_size, n_context))

        # Test call function.
        log_prob = apply_fn(params, y, context)
        self.assertTupleEqual(log_prob.shape, (batch_size,))

        # Test sampling functions.
        sample_shape = (2, 2)
        samples = apply_fn(
            params, rng, context[0], sample_shape=sample_shape, method='sample'
        )
        self.assertTupleEqual(samples.shape, sample_shape + (n_dim,))
        samples, log_prob = apply_fn(
            params, rng, context[0], sample_shape=sample_shape,
            method='sample_and_log_prob'
        )
        self.assertTupleEqual(samples.shape, sample_shape + (n_dim,))
        self.assertTupleEqual(log_prob.shape, sample_shape)


class EmbeddedFlowTest(chex.TestCase):
    """Run tests on the EmbeddedFlow module."""

    @chex.all_variants
    def test_apply(self):
        # Test the the MAF module initializes and returns transformed
        # variables.
        n_dim = 8
        embedding_dim = 4
        image_size = 64
        n_maf_layers = 3
        hidden_dims = [16]
        activation = 'gelu'
        embedding_module = models.ResNet18VerySmall(num_outputs=embedding_dim)
        maf_model = maf_flow.MAF(n_dim, n_maf_layers, hidden_dims, activation)
        maf = maf_flow.EmbeddedFlow(embedding_module, maf_model)

        rng = jax.random.PRNGKey(0)
        params = maf.init(
            rng, jnp.ones((1, n_dim)), jnp.ones((1, image_size, image_size, 1))
        )
        apply_fn = self.variant(
            maf.apply,
            static_argnames=['method', 'sample_shape', 'mutable', 'train']
        )

        batch_size = 4
        y = jax.random.normal(rng, (batch_size, n_dim))
        context = jax.random.normal(
            rng, (batch_size, image_size, image_size, 1)
        )

        # Test call function.
        log_prob, _ = apply_fn(params, y, context, mutable=('batch_stats',))
        self.assertTupleEqual(log_prob.shape, (batch_size,))

        # Test the apt call function.
        n_atoms = 2
        log_prob_apt, _ = apply_fn(
            params, jnp.repeat(y[:, None], n_atoms, axis=1), context,
            method='call_apt', mutable=('batch_stats',)
        )
        self.assertTupleEqual(log_prob_apt.shape, (batch_size, n_atoms))
        for i in range(n_atoms):
            np.testing.assert_array_almost_equal(
                log_prob_apt[:, i], log_prob, decimal=5
            )

        # Test embed_context function.
        embed_context, _ = apply_fn(
            params, context, method='embed_context', mutable=('batch_stats',),
            train=True
        )

        # Test sampling functions.
        sample_shape = (2, 2)
        samples, _ = apply_fn(
            params, rng, context, sample_shape=sample_shape, method='sample',
            mutable=('batch_stats',)
        )
        self.assertTupleEqual(
            samples.shape, (batch_size,) + sample_shape + (n_dim,)
        )
        (samples, log_prob), _ = apply_fn(
            params, rng, context, sample_shape=sample_shape,
            method='sample_and_log_prob', mutable=('batch_stats',)
        )
        self.assertTupleEqual(
            samples.shape, (batch_size,) + sample_shape + (n_dim,)
        )
        self.assertTupleEqual(log_prob.shape, (batch_size,) + sample_shape)

        # Test that using the output of embed_context returns the same samples.
        maf_apply_fn = self.variant(
            maf_model.apply, static_argnames=['method', 'sample_shape']
        )
        rng_sample = jax.random.split(rng, len(context))
        maf_samples = maf_apply_fn(
            {'params': params['params']['flow_module']}, rng_sample[0],
            embed_context[0], sample_shape=sample_shape, method='sample'
        )
        np.testing.assert_array_almost_equal(
            samples[0], maf_samples, decimal=5
        )


if __name__ == '__main__':
    absltest.main()
