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

"""Implementation of MAF Normalizing flow

Implementation follows jax-conditional-flow
(https://github.com/smsharma/jax-conditional-flow) closely but is reimplemented
here for simplicity.

"""

import abc
from typing import List, Tuple, Optional, Union

import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact

from paltax.models import ModuleDef


def _made_degrees(
    input_size: int, hidden_dims: Union[List[int], None]
) -> List[jnp.ndarray]:
    """Return degrees to build masks for MADE fully connected networks.

    Args:
        input_size: Number of inputs to the fully connected network.
        hidden_dims: Hidden layer dimensions of the fully connected network. If
            None no hidden layers.

    Returns:
        Degree for each input and node in the hidden layers.

    Notes:
        Assume left-to-right input ordering (i.e. random perumetations must be
        applied before MADE operation) and distributed the hidden units roughly
        equally among all of the possible degrees.
    """
    # Add degrees for the inputs (simple arange for left-to-right ordering).
    degrees = [jnp.arange(1, input_size + 1)]

    # Add degrees for each hidden dimension.
    if hidden_dims is None:
        hidden_dims = []

    for dimension in hidden_dims:
        degrees.append(
            jnp.ceil(
                jnp.arange(1, dimension + 1) * (input_size - 1.0) /
                (dimension + 1.0)
            ).astype(int)
        )

    return degrees


def _made_masks(
    degrees: List[jnp.ndarray], n_context: int
) -> List[jnp.ndarray]:
    """Create boolean masks to apply to weight matrices from degrees of nodes.

    Args:
        degrees: Degree for each input and node in the hidden layers.
        n_context: Number of parameters that are context and should therefore
            not be included in the outputs. All outputs will be conditioned on
            the full context.

    Returns:
        Binary mask matrices for autoregressivity.
    """
    # First N-1 masks just specify which outputs have degree at least as
    # large as the input.
    masks = [
        inp[:, jnp.newaxis] <= output
        for inp, output in zip(degrees[:-1], degrees[1:])
    ]
    # Final mask drops context variables, ensuring that all variables are
    # connected to the context.
    masks.append(
        degrees[-1][:, jnp.newaxis] < degrees[0][n_context:]
    )
    return masks


def _made_dense_autoregressive_masks(
    n_params: int, n_total_cond: int, n_context: int,
    hidden_dims: Union[List[int], None]
) -> jnp.ndarray:
    """Create masks for MADE fully connected network.

    Args:
        n_params: Number of parameters per dimension.
        n_total_cond: Total number of parameters that can be conditioned on.
        n_context: Number of parameters that are context and should therefore
            not be included in the outputs. All outputs will be conditioned on
            the full context.
        hidden_dims: Hidden layer dimensions of the fully connected network. If
            None no hidden layers.
    """
    degrees = _made_degrees(n_total_cond, hidden_dims)
    masks = _made_masks(degrees, n_context)

    # Final mask needs to be tiled to account for more than one output per
    # variable (in the MADE case mean and log variance).
    masks[-1] = jnp.reshape(
        jnp.tile(masks[-1][:, :, jnp.newaxis], (1, 1, n_params)),
        (masks[-1].shape[0], masks[-1].shape[1] * n_params)
    )

    return masks


class MaskedDense(nn.Dense):
    """Linear transformation with masking.

    Args:
        mask: Mask to apply on the weights.
    """
    mask: jnp.ndarray = None

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies the linear transformation used masked weights.

        Args:
            inputs: Array to be transformed.

        Returns:
            Transformed input.
        """
        kernel = self.param(
            'kernel', self.kernel_init, (jnp.shape(inputs)[-1], self.features),
            self.param_dtype
        )
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(
            inputs, kernel, bias, dtype=self.dtype
        )
        kernel = self.mask * kernel

        y = jax.lax.dot_general(
            inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        return y


class MADE(nn.Module):
    """Masked autoencoder for distribution estimation.

    Implementation of https://arxiv.org/abs/1705.07057v4 that follows the code
    in https://github.com/smsharma/jax-conditional-flow.

    Args:
        hidden_dims: Hidden dimensions of the fully connected network.
        activation: Activation function of the fully connected network.
    """
    hidden_dims: Union[List[int], None]
    activation: Optional[str] = 'tanh'

    @compact
    def __call__(
        self, x: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Call for model.

        Args:
            x: Current parameters.
            context: Context for MADE transformation. If None no context.
        """
        # Save the shape of the current parameters.
        x_shape = x.shape
        in_dim = x.shape[-1]
        cond_dim = 0

        # Append context if it is provided.
        if context is not None:
            # Stack context on left to ensure that all outputs are conditioned
            # on context.
            x = jnp.hstack([context, x])
            cond_dim = context.shape[-1]

        n_total_cond = in_dim + cond_dim
        masks = _made_dense_autoregressive_masks(
            2, n_total_cond, cond_dim, self.hidden_dims
        )

        for mask in masks[:-1]:
            x = MaskedDense(features=mask.shape[-1], mask=mask)(x)
            x = getattr(jax.nn, self.activation)(x)
        x = MaskedDense(features=masks[-1].shape[-1], mask=masks[-1])(x)

        # Reshape to extract the parameters for each input.
        return x.reshape(x_shape + (2,))


class _ConidtionalBijector(distrax.Bijector):
    """Base for conditional bijectors."""
    # pylint: disable=arguments-differ

    def forward(
        self, x:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Compute y = f(x).

        Args:
            x: Untransformed random variables.
            context: Context for transformation.

        Returns:
            Transformed random variables.
        """
        y, _ = self.forward_and_log_det(x, context)
        return y

    def inverse(
        self, y:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Compute x = f^{-1}(y).

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Untransformed random variables.
        """
        x, _ = self.inverse_and_log_det(y, context)
        return x

    def forward_log_det_jacobian(
        self, x:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Compute log det of y = f(x) transformation.

        Args:
            x: Untransformed random variables.
            context: Context for transformation.

        Returns:
            Log determinant of transformation.
        """
        _, log_det = self.forward_and_log_det(x, context)
        return log_det

    def inverse_log_det_jacobian(
        self, y:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Compute log det of x = f^{-1}(y) transformation.

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Log determinant of inverse transformation.
        """
        _, log_det = self.inverse_and_log_det(y, context)
        return log_det

    @abc.abstractmethod
    def forward_and_log_det(
        self, x: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the forward pass and compute the log determinent

        Args:
            x: Untransformed random variables.
            context: Context for transformation.

        Returns:
            Transformed random variables and log determinant.
        """

    @abc.abstractmethod
    def inverse_and_log_det(
        self, y:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the inverse pass and compute the log determinent

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Untransformed random variables and log determinant.
        """


class _MAFLayer(_ConidtionalBijector):
    """Masked autoregressive flow layer implementation.

    Implementation of https://arxiv.org/abs/1705.07057v4 that follows the code
    in https://github.com/smsharma/jax-conditional-flow.

    Args:
        made_layer: Instance of MADE that will be used as foundation of MAF
            layer.
    """

    def __init__(self, made_layer: nn.Module):
        super().__init__(event_ndims_in=1)
        # For the flax modle to play nice, the made_layer has to be initialized
        # within another module (i.e. not inside a distrx.Bijector).
        self.made_layer = made_layer

    def forward_and_log_det(
        self, x: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the forward pass and compute the log determinent

        Args:
            x: Untransformed random variables.
            context: Context for transformation.

        Returns:
            Transformed random variables and log determinant.
        """
        in_dim = x.shape[-1]

        # Initialize y and the log_det
        y = jnp.zeros_like(x)
        log_det = None

        # On each loop, one of the y dimensions will reach it's final state.
        # For example, on the first loop only the first dimension of y will have
        # the correct value because all the other ith dimensions will have a
        # mean and scale calculated with the incorrect 0,...,i-1th dimensions.
        # On the next pass the 2nd dimensions will be calculated using the
        # correct y[...,0] value and will therefore have the correct value.
        # TODO: flax.linen.scan might improve compile times.
        for _ in range(in_dim):
            gaussian_params = self.made_layer(y, context)
            y, log_det = (
                distrax.ScalarAffine(
                    shift=gaussian_params[..., 0],
                    log_scale=gaussian_params[..., 1]
                ).forward_and_log_det(x)
            )

        return y, jnp.sum(log_det, axis=-1)

    def inverse_and_log_det(
        self, y:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the inverse pass and compute the log determinent

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Untransformed random variables and log determinant.
        """
        gaussian_params = self.made_layer(y, context)
        x, log_det = distrax.ScalarAffine(
            shift=gaussian_params[..., 0], log_scale=gaussian_params[..., 1]
        ).inverse_and_log_det(y)

        return x, jnp.sum(log_det, axis=-1)


class _PermuteLayer(nn.Module):
    """Permutation module for features."""

    @compact
    def __call__(
        self, x:jnp.ndarray, inverse: Optional[bool]=False
    ) -> jnp.ndarray:
        """Perform random permutation.

        Args:
            x: Unpermuted parameters.
            inverse: Boolean controlling whether the permutation will be applied
                (False) or if it will be inverted (True).

        Returns:
            Permuted parameters.

        Notes:
            Permutation is fixed by initial seed.
        """
        permutation = self.param(
            'permutation', self._init_permutation, x.shape[-1]
        )
        permutation = nn.cond(
            inverse, lambda mdl, x: jnp.argsort(x), lambda mdl, x: x,
            self, permutation
        )
        x = jnp.moveaxis(x, -1, 0) # simplify indexing
        x = x[permutation]
        return jnp.moveaxis(x, 0, -1)

    def _init_permutation(self, rng: List[int], in_dim: int):
        """Initialize a permutation from a given random key and dimensionality.

        Args:
            rng: Jax PRNG key
            in_dim: Size of dimension for permutation.

        Returns:
            Random permutation for dimension.
        """
        return jax.random.permutation(rng, in_dim)


class Permute(_ConidtionalBijector):
    """Permutation layer implementation.

    Args:
        permutation_layer: _PermuteLayer to use for bijector.
    """

    def __init__(self, permutation_layer: nn.Module):
        super().__init__(event_ndims_in=1)
        # Must be initialized within a nn.Module.
        self.permutation_layer = permutation_layer

    def forward_and_log_det(
        self, x:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Conduct the forward pass and compute the log determinent

        Args:
            x: Untransformed random variables.

        Returns:
            Transformed random variables and log determinant.
        """
        y = self.permutation_layer(x)
        return y, jnp.zeros(x.shape[:-1])

    def inverse_and_log_det(
        self, y:jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Conduct the inverse pass and compute the log determinent

        Args:
            y: Transformed random variables.

        Returns:
            Untransformed random variables and log determinant.
        """
        # True bool indicated to invert permutation.
        x = self.permutation_layer(y, True)
        return x, jnp.zeros(y.shape[:-1])


class ChainConditional(_ConidtionalBijector):
    """Chain of bijectors that allow for context.

    Args:
        bijectors: Bijectors to be composed into one final bijection.
    """

    def __init__(self, bijectors: List[distrax.BijectorLike]):
        super().__init__(event_ndims_in=1)
        self.bijectors = bijectors

    def forward_and_log_det(
        self, x: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the forward pass and compute the log determinent

        Args:
            x: Untransformed random variables.
            context: Context for transformation.

        Returns:
            Transformed random variables and log determinant.
        """
        log_det = jnp.zeros(x.shape[:-1])
        for bijector in self.bijectors:
            x, ld = bijector.forward_and_log_det(x, context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(
        self, y: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conduct the inverse pass and compute the log determinent

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Untransformed random variables and log determinant.
        """
        log_det = jnp.zeros(y.shape[:-1])
        for bijector in reversed(self.bijectors):
            y, ld = bijector.inverse_and_log_det(y, context)
            log_det += ld
        return y, log_det


class TransformedConditional():
    """Distribution of random variable after bijective transformation.

    Args:
        distribution: Base distribution that is transformed by bijector.
        bijector: Bijector the transforms the distribution.
    """

    def __init__(
        self, distribution: distrax.Distribution,
        bijector: _ConidtionalBijector
    ):
        self.distribution = distribution
        self.bijector = bijector

    def sample(
        self, rng: List[int], context: Union[jnp.ndarray, None],
        sample_shape: List[int]
    ) -> jnp.ndarray:
        """Draw samples from transformed distribution.

        Args:
            rng: Jax PRNG key.
            context: Single context for transformation.
            sample_shape: Desired output shape of samples.

        Returns:
            Samples of transformed variables.

        Notes:
            Assumes you are only providing one context for which you want to
            draw many samples. If that is not the case, you will need to vmap.
        """
        x = self.distribution.sample(seed=rng, sample_shape=sample_shape)
        y = jax.vmap(self.bijector.forward, in_axes=[0, None])(
            x.reshape((-1, x.shape[-1])), context
        )
        return y.reshape(x.shape)

    def log_prob(
        self, y: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Calculate log probability of samples in transformed space.

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Log probability of transformed variables.
        """
        x, log_det = self.bijector.inverse_and_log_det(y, context)
        log_prob = self.distribution.log_prob(x) + log_det
        return log_prob

    def sample_and_log_prob(
        self, rng: List[int], context: Union[jnp.ndarray, None],
        sample_shape: List[int]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Draw samples and corresponding log probability of the sample.

        Args:
            rng: Jax PRNG key.
            context: Single context for transformation.
            sample_shape: Desired output shape of samples.

        Returns:
            Samples of transformed variables and log probability.

        Notes:
            Assumes you are only providing one context for which you want to
            draw many samples. If that is not the case, you will need to vmap.
        """
        x, lp_x = self.distribution.sample_and_log_prob(
            seed=rng, sample_shape=sample_shape)

        # Use vmap with reshapes to respect sample_shape.
        y, log_det_forw = jax.vmap(
            self.bijector.forward_and_log_det, in_axes=[0, None]
        )(x.reshape((-1, x.shape[-1])), context)

        log_prob = jax.vmap(jnp.subtract)(
            lp_x.reshape((-1,)), log_det_forw
        )

        return y.reshape(x.shape), log_prob.reshape(sample_shape)


class MAF(nn.Module):
    """Masked autoregressive flow implementation.

    Implementation of https://arxiv.org/abs/1705.07057v4 that follows the code
    in https://github.com/smsharma/jax-conditional-flow.

    Args:
        n_dim: Dimensionality of random variables.
        n_maf_layers: Number of MAF layers to construct.
        hidden_dims: Hidden dimensions of each MADE layer underlying each MAF
            layer.
        activation: Activation function to use in MADE layers.
    """
    n_dim: int
    n_maf_layers: int
    hidden_dims: List[int]
    activation: str

    def setup(self):
        """Setup the MaskedAutoregressiveFlow module."""
        # Create a list of bijectors. For each transformation, we want to first
        # permute the dimensions and then apply a MAF layer.
        self.made_layers = [
            MADE(self.hidden_dims, self.activation) for _ in range(
                self.n_maf_layers
            )
        ]
        self.permute_layers = [
            _PermuteLayer() for _ in range(self.n_maf_layers)
        ]
        bijectors = []

        for maf_i in range(self.n_maf_layers):
            # Create permutation and append to bijectors.
            permute = Permute(self.permute_layers[maf_i])
            bijectors.append(permute)

            # Create MAF and append to bijectors.
            maf_layer = _MAFLayer(self.made_layers[maf_i])
            bijectors.append(maf_layer)

        # Create one chain conditional bijector from all of our bijectors.
        self.bijector = ChainConditional(bijectors)

        # Use a diagonal multivariate Gaussian as the base distribution.
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.n_dim))

        # Construct the flow
        self.flow = TransformedConditional(self.base_dist, self.bijector)

    def __call__(
        self, y: jnp.ndarray, context: Union[jnp.ndarray, None]
    ) -> jnp.ndarray:
        """Return log probability of transformed random variables.

        Args:
            y: Transformed random variables.
            context: Context for transformation.

        Returns:
            Log probability of transformed random variables.
        """
        return self.flow.log_prob(y, context)

    def sample(
        self, rng: List[int], context: jnp.ndarray, sample_shape: List[int]
    ) -> jnp.ndarray:
        """Draw samples from transformed distribution.

        Args:
            rng: Jax PRNG key.
            context: Single context for transformation.
            sample_shape: Desired output shape of samples.

        Returns:
            Samples of transformed variables.

        Notes:
            Assumes you are only providing one context for which you want to
            draw many samples. If that is not the case, you will need to vmap.
        """
        return self.flow.sample(rng, context, sample_shape)

    def sample_and_log_prob(
        self, rng: List[int], context: jnp.ndarray, sample_shape: List[int]
    ) -> jnp.ndarray:
        """Draw samples and corresponding log probability of the sample.

        Args:
            rng: Jax PRNG key.
            context: Single context for transformation.
            sample_shape: Desired output shape of samples.

        Returns:
            Samples of transformed variables and log probability.

        Notes:
            Assumes you are only providing one context for which you want to
            draw many samples. If that is not the case, you will need to vmap.
        """
        return self.flow.sample_and_log_prob(rng, context, sample_shape)


class EmbeddedFlow(nn.Module):
    """Flow with a specific model for embeddings.

    Args:
        embedding_module: Module used to map from conditional data to embedding.
        flow_module: Module that will ingest context and define flow.
    """
    embedding_module: ModuleDef
    flow_module: ModuleDef

    def __call__(
        self, y: jnp.ndarray, unembedded_context: jnp.ndarray
    ) -> jnp.ndarray:
        """Return log probability of transformed random variables.

        Args:
            y: Transformed random variables.
            unembedded_context: Context for flow transformation before
                embedding. This should be the raw data.

        Returns:
            Log probability of transformed random variables.
        """
        context = self.embed_context(unembedded_context)
        return self.flow_module(y, context)

    def embed_context(self, unembedded_context: jnp.ndarray) -> jnp.ndarray:
        """Embed context using embedding network.

        Args:
            unembedded_context: Context for flow transformation before
                embedding. This should be the raw data.

        Returns:
            Context after embedding.
        """
        return self.embedding_module(unembedded_context)

    def sample(
        self, rng: List[int], unembedded_context: jnp.ndarray,
        sample_shape: List[int]
    ) -> jnp.ndarray:
        """Draw samples from transformed distribution.

        Args:
            rng: Jax PRNG key.
            unembedded_context: Context for flow transformation before
                embedding. This should be the raw data.
            sample_shape: Desired output shape of samples.

        Returns:
            Samples of transformed variables.

        Notes:
            The embedding model may have a batch-size dependent transformation
            in it (i.e. batch norm). Passing in only one point of context may
            may lead to unexpected behavior.
        """
        context = self.embed_context(unembedded_context)
        rng_sample = jax.random.split(rng, len(context))
        return jax.vmap(self.flow_module.sample, in_axes=[0, 0, None])(
            rng_sample, context, sample_shape
        )

    def sample_and_log_prob(
        self, rng: List[int], unembedded_context: jnp.ndarray,
        sample_shape: List[int]
    ) -> jnp.ndarray:
        """Draw samples and corresponding log probability of the sample.

        Args:
            rng: Jax PRNG key.
            unembedded_context: Context for flow transformation before
                embedding. This should be the raw data.
            sample_shape: Desired output shape of samples.

        Returns:
            Samples of transformed variables and log probability.

        Notes:
            The embedding model may have a batch-size dependent transformation
            in it (i.e. batch norm). Passing in only one point of context may
            may lead to unexpected behavior.
        """
        context = self.embed_context(unembedded_context)
        rng_sample = jax.random.split(rng, len(context))
        return jax.vmap(
            self.flow_module.sample_and_log_prob, in_axes=[0, 0, None]
        )(rng_sample, context, sample_shape)
