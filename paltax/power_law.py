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
"""Power law integration and draws in jax.

This module includes functions to integrate a power law and draw from a power
law distribution.
"""

from typing import Sequence

import jax
import jax.numpy as jnp


def power_law_integrate(p_min: float, p_max: float, slope: float) -> float:
    """Integrate a power law

    Args:
        p_min: Lower bound of the power law
        p_max: Upper bound of the power law
        slope: Slope of the power law

    Returns:
        Integral of the power law x^slope from p_min to p_max
    """
    upper_bound = 1/(slope+1)*p_max**(slope+1)
    lower_bound = 1/(slope+1)*p_min**(slope+1)
    return upper_bound-lower_bound


def power_law_draw(p_min: float, p_max: float, slope: float, norm:float,
    rng: Sequence[int], pad_length: int) -> jnp.ndarray:
    """Sample from a power law

    Args:
        p_min: Lower bound of the power law
        p_max: Upper bound of the power law
        slope: Slope of the power law
        norm: Normalization of the power law
        rng: jax PRNG key used for noise realization.
        max_subhalos: The length to pad the draw to. Must be a static argument
            for jax.jit compilation. If more subhalos are drawn than the
            pad_length, they will be dropped.

    Returns:
        Values drawn from the power law. Padded values are set to 0.
    """
    # Get the expected number of draws
    n_expected = norm * power_law_integrate(p_min,p_max,slope)

    # Split our random key for the number of draws and the cdf draws
    rng_draws, rng_cdf = jax.random.split(rng)

    # Draw the number of objects as a poisson process
    n_draws = jax.random.poisson(rng_draws, n_expected)

    # Get the positions in the cdf we want to draw from a uniform and then
    # convert that to values in the pdf
    cdf = jax.random.uniform(rng_cdf, shape=(pad_length,))
    s_one = slope + 1
    draws = (cdf * (p_max ** s_one - p_min ** s_one) +
        p_min ** s_one) ** (1 / s_one)

    # Pad the draws
    indices = jnp.arange(pad_length)
    is_not_pad = indices < n_draws

    return draws * is_not_pad
