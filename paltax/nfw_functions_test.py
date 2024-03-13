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
"""Tests for nfw_functions.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from paltax import cosmology_utils
from paltax import nfw_functions


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


def _prepare_main_deflector_params():
    main_deflector_params = {'mass': 1e13, 'z_lens': 0.5, 'theta_e': 2.38,
        'center_x': 0.0, 'center_y': 0.0}
    return main_deflector_params


def _prepare_substructure_params():
    substructure_params = {'c_zero': 18, 'conc_zeta': -0.2, 'conc_beta': 0.8,
        'conc_m_ref': 1e8, 'conc_dex_scatter': 0.0}
    return substructure_params


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
            expected, places=4)

        np.testing.assert_array_almost_equal(
            r_two_hund_from_m(cosmology_params,jnp.full(10, m), z),
            jnp.full(10, expected), decimal=4)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_c_{c}_z_{z}', c, z, expected) for c, z, expected in
        zip([8, 12, 14], [0.1, 0.2, 0.3],
        [3662396.563509418, 10945906.187043915, 17983344.644410152])
    ])
    def test_rho_nfw_from_c(self, c, z, expected):
        # Test that for scalars and arrays the correct normalization is
        # returned.
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
            z)

        rho_nfw_from_c =  self.variant(nfw_functions.rho_nfw_from_c)

        # Divide by the order of magntiude since testing is in terms of decimal.
        self.assertAlmostEqual(rho_nfw_from_c(cosmology_params, c, z) / 100000,
            expected / 100000, places=4)

        np.testing.assert_array_almost_equal(
            rho_nfw_from_c(cosmology_params,jnp.full(10, c), z) / 100000,
            jnp.full(10, expected) / 100000, decimal=4)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_rt_{rt}_rho_{rho}_rs_{rs}', rt, rho, rs) for rt, rho, rs in
        zip([0.1, 1.5, 12.2], [1, 10, 100], [2, 4, 8])
    ])
    def test__cored_nfw_integral(self, rt, rho, rs):
        # Compare analytic answer to numerical integral
        r_upper = jnp.linspace(0.1, 4, 100)

        _cored_nfw_integral =  self.variant(nfw_functions._cored_nfw_integral)

        def cored_nfw_func(r):
            if r<rt:
                x_tidal = rt / rs
                return 4 * np.pi * rho / (x_tidal * (1 + x_tidal) ** 2) * r ** 2
            else:
                x = r / rs
                return 4 * np.pi * r ** 2 * rho / (x * (1 + x) ** 2)

        analytic_values = _cored_nfw_integral(rt, rho, rs, r_upper)

        for i, r in enumerate(r_upper):
            self.assertAlmostEqual(1.0,
                quad(cored_nfw_func, 0, r, epsabs=1.49e-11,
				    epsrel=1.49e-11)[0] / analytic_values[i], places=4)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_rt_{rt}_rho_{rho}_rs_{rs}', rt, rho, rs) for rt, rho, rs in
        zip([0.1, 1.5, 12.2], [1, 10, 100], [2, 4, 8])
    ])
    def test_cored_nfw_draws(self, rt, rho, rs):
        # Compare to expected distributions.
        r_max = 4
        n_draws = int(2e5)
        rng = jax.random.PRNGKey(0)

        cored_nfw_draws =  self.variant(functools.partial(
            nfw_functions.cored_nfw_draws, n_draws=n_draws))
        r_draws = cored_nfw_draws(rt, rho, rs, r_max, rng)

        n_test_points = 100
        r_test = jnp.linspace(0, r_max, n_test_points)
        analytic_values = nfw_functions._cored_nfw_integral(rt, rho, rs, r_test)

        for i, r in enumerate(r_test):
            self.assertAlmostEqual(jnp.mean(r_draws < r),
                analytic_values[i] / jnp.max(analytic_values), places=2)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_rs_{rs}_rho_{rho}', rs, rho, eo, et) for rs, rho, eo, et in
        zip([0.1, 1.5, 12.2], [1e13, 2e13, 3e13],
            [0.015902785701266995, 0.23854178551900487, 1.940139855554573],
            [6.503977420097886, 2926.7898390440478, 290415.5997622107])
    ])
    def test_convert_to_lensing_nfw(self, rs, rho, eo, et):
        # Compare to expected distributions.
        z = 0.5
        z_source = 1.0
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, 1.0,
            0.5)

        convert_to_lensing_nfw =  self.variant(
            nfw_functions.convert_to_lensing_nfw)

        r_s_ang, alpha_rs = convert_to_lensing_nfw(cosmology_params, rs, z, rho,
            z_source)
        self.assertAlmostEqual(r_s_ang, eo, places=4)
        self.assertAlmostEqual(1.0, et / alpha_rs, places=4)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_rt_{rt}', rs, rho, rt, expected) for rs, rho, rt, expected in
        zip([0.1, 1.5, 12.2], [1e13, 2e13, 3e13], [0.1, 1.0, 1.5],
            [0.015902785701266995, 0.15902785701266992, 0.23854178551900487])
    ])
    def test_convert_to_lensing_tnfw(self, rs, rho, rt, expected):
        # Compare to expected distributions.
        z = 0.5
        z_source = 1.0
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, 1.0,
            0.5)

        convert_to_lensing_tnfw =  self.variant(
            nfw_functions.convert_to_lensing_tnfw)

        _, _, rt_ang = convert_to_lensing_tnfw(cosmology_params, rs, z, rho, rt,
            z_source)
        self.assertAlmostEqual(rt_ang, expected, places=4)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters([
        (f'_m_{m}_z_{z}', m, z, expected) for m, z, expected in zip(
            [1e9, 1e10], [0.2, 0.3], [13.7, 10.8])
    ])
    def test_mass_concentration(self, m, z, expected):
        # Test that the mass draws follow the desired distribution.
        main_deflector_params = _prepare_main_deflector_params()
        subhalo_params = _prepare_substructure_params()
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            main_deflector_params['z_lens'], main_deflector_params['z_lens'])
        h = cosmology_params['hubble_constant'] / 100
        r_min = cosmology_utils.lagrangian_radius(cosmology_params,
            subhalo_params['conc_m_ref'] * h)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m * h)
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            main_deflector_params['z_lens'], main_deflector_params['z_lens'],
            r_min, r_max)

        mass_concentration = self.variant(nfw_functions.mass_concentration)

        rng = jax.random.PRNGKey(0)
        subhalo_params['conc_dex_scatter'] = 0.0
        conc = mass_concentration(subhalo_params, cosmology_params,
            jnp.full((10,), m), z, rng)
        # Peak height is the limit on the precision of this comparison.
        np.testing.assert_array_almost_equal(conc, jnp.full((10,), expected),
            decimal=0)

        # Check the scatter does something
        subhalo_params['conc_dex_scatter'] = 0.1
        conc = mass_concentration(subhalo_params, cosmology_params,
            jnp.full((10000,), m), z, rng)
        scatter = jnp.log10(conc) - expected
        self.assertAlmostEqual(jnp.std(scatter), 0.1, places=2)

if __name__ == '__main__':
    absltest.main()
