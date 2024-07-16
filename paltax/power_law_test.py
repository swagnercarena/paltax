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
"""Tests for power_law.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from scipy.integrate import quad

from paltax import power_law


class PowerLawTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_p_min_{p_min}_p_max_{p_max}_slope_{slope}', p_min, p_max, slope)
        for p_min, p_max, slope in zip([3, 6, 4], [7, 9, 4], [-1.9, -1.7, 5])
    ])
    def test_power_law_integrate(self, p_min, p_max, slope):
        # Check that analytic integral agrees with numerical results.
        def p_func(x, slope):
            return x**slope

        pl_integrate =  self.variant(power_law.power_law_integrate)

        self.assertAlmostEqual(pl_integrate(p_min, p_max, slope),
            quad(p_func, p_min, p_max, args=(slope))[0])

    @chex.all_variants
    def test_power_law_draw(self):
        # Check that draws roughly follow the power law and the norm
        # we expect
        p_min = 1e6
        p_max = 1e9
        slope = -1.9
        desired_count = 100
        norm = desired_count / power_law.power_law_integrate(p_min, p_max,
            slope)

        pl_draw = self.variant(functools.partial(power_law.power_law_draw,
            pad_length = 200))

        total_subs = 0
        rng = jax.random.PRNGKey(0)
        n_loops = 500
        for _ in range(n_loops):
            rng_draw, rng = jax.random.split(rng)
            draws = pl_draw(p_min, p_max, slope, norm, rng_draw)
            total_subs += jnp.sum(draws > 0)
        self.assertEqual(jnp.round(total_subs/n_loops) , desired_count)

        # Check for a specific draw that it follows the cdf
        desired_count = 1e6
        pad_length = int(2e6)
        norm = desired_count / power_law.power_law_integrate(p_min, p_max,
            slope)
        pl_draw = self.variant(functools.partial(power_law.power_law_draw,
            pad_length = pad_length))
        draws = pl_draw(p_min, p_max, slope, norm, rng)

        # Check the cdf at a handful of test points
        test_points = jnp.logspace(6, 9, 20)
        for test_point in test_points:
            self.assertAlmostEqual(
                jnp.sum(jnp.logical_and(draws < test_point , draws > 0)) /
                jnp.sum(draws > 0),
                power_law.power_law_integrate(p_min, test_point, slope) /
                power_law.power_law_integrate(p_min, p_max, slope), places=2
            )

    @chex.all_variants
    def test_suppressed_power_law_draw(self):
        # Check that draws roughly follow the power law and the norm
        # we expect
        p_min = 1e6
        p_max = 1e9
        p_supp = 1e8
        slope = -1.9
        desired_count = 10000
        norm = desired_count / power_law.power_law_integrate(p_min, p_max,
            slope)

        supp_pl_draw = self.variant(
            functools.partial(
                power_law.suppressed_power_law_draw,
                pad_length = 20000
            )
        )

        # Check that the ratio between the 10^7 and 10^8 draws meet our
        # expectations.
        total_m_7 = 0
        total_m_8 = 0
        rng = jax.random.PRNGKey(0)
        n_loops = 500
        eps = 1e6
        for _ in range(n_loops):
            rng_draw, rng = jax.random.split(rng)
            draws = supp_pl_draw(p_supp, p_min, p_max, slope, norm, rng_draw)
            total_m_7 += jnp.sum(
                jnp.logical_and(draws > 1e7 - eps, draws < 1e7 + eps)
            )
            total_m_8 += jnp.sum(
                jnp.logical_and(draws > 1e8 - eps, draws < 1e8 + eps)
            )

        correct_ratio = (
            ((1e7 ** slope) * (1 + p_supp / 1e7) ** (-1.5)) /
            ((1e8 ** slope) * (1 + p_supp / 1e8) ** (-1.5))
        )
        self.assertAlmostEqual(total_m_7 / total_m_8, correct_ratio, places=1)


if __name__ == '__main__':
    absltest.main()
