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
"""Tests for nfw_functions.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np

from jaxstronomy import cosmology_utils
from jaxstronomy import nfw_functions

COSMOLOGY_PARAMS_INIT = immutabledict({
    'omega_m_zero': 0.3089,
    'omega_b_zero': 0.0486,
    'omega_de_zero': 0.6910088292453472,
    'omega_rad_zero': 9.117075466e-5,
    'temp_cmb_zero': 2.7255,
    'hubble_constant': 67.74,
    'n_s': 0.9667,
    'sigma_eight': 0.8159,
})


def _prepare_cosmology_params(cosmology_params_init, z_lookup_max, dz,
    r_min=1e-4, r_max=1e3, n_r_bins=2):
  # Only generate a lookup table for values we need.
  # When 0,0 is specified for the two z values, need to select a small non-zero
  # values to generate a non-empty table.
  z_lookup_max = max(z_lookup_max, 1e-7)
  dz = max(dz, 1e-7)
  return cosmology_utils.add_lookup_tables_to_cosmology_params(
      dict(cosmology_params_init), z_lookup_max, dz, r_min, r_max, n_r_bins)


class NfwFuntionsTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_m_{np.log10(m):.0f}_z_{z}', m, z, expected) for m, z, expected in
        zip([1e7, 1e8, 1e9], [0.1, 0.2, 0.3], [4.39736, 9.14635, 18.98417])
    ])
    def test_r_two_hund_from_m(self, m, z, expected):
        # Test that for scalars and arrays the correct overdensity radius is
        # returned.
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
            z)

        r_two_hund_from_m =  self.variant(nfw_functions.r_two_hund_from_m)

        self.assertAlmostEqual(r_two_hund_from_m(cosmology_params, m, z),
            expected)

        np.testing.assert_array_almost_equal(
            r_two_hund_from_m(cosmology_params,jnp.full(10, m), z),
            jnp.full(10, expected))


if __name__ == '__main__':
    absltest.main()
