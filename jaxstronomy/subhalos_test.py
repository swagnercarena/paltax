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
from jaxstronomy import power_law
from jaxstronomy import subhalos


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


def _prepare_subhalo_params():
    subhalo_params = {'sigma_sub': 5e-4, 'shmf_plaw_index': -2.0,
        'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'k_one': 0.0, 'k_two': 0.0,
        'c_zero': 18, 'conc_zeta': -0.2, 'conc_beta': 0.8, 'conc_m_ref': 1e8,
        'conc_dex_scatter': 0.0}
    return subhalo_params


class SubhalosTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_m_{m}_z_{z}_ko_{ko}_kt_{kt}', m, z, ko, kt, expected) for
            m, z, ko, kt, expected in zip([1e13, 2e13, 1e13], [0.1, 0.2, 0.3],
                [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [1.0, 1.183215956619923, 0.8])
    ])
    def test_host_scaling_function(self, m, z, ko, kt, expected):
        # Test the scaling function for different parameter values.

        host_scaling_function =  self.variant(subhalos.host_scaling_function)

        self.assertAlmostEqual(host_scaling_function(m, z, ko, kt),
            expected, places=4)

    @chex.all_variants
    def test_draw_nfw_masses(self):
        # Test that the mass draws follow the desired distribution.
        main_deflector_params = _prepare_main_deflector_params()
        subhalo_params = _prepare_subhalo_params()
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            main_deflector_params['z_lens'], main_deflector_params['z_lens'])

        pad_length = 1000
        draw_nfw_masses = self.variant(functools.partial(
            subhalos.draw_nfw_masses, pad_length=pad_length))

        # Calculate the norm by hand and make sure the statistics agree
        kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params,
            main_deflector_params['z_lens'])
        area_elem = jnp.pi * (3 * kpa * main_deflector_params['theta_e']) ** 2
        f_host = subhalos.host_scaling_function(main_deflector_params['mass'],
            main_deflector_params['z_lens'], subhalo_params['k_one'],
            subhalo_params['k_two'])

        e_counts =  power_law.power_law_integrate(subhalo_params['m_min'],
            subhalo_params['m_max'],subhalo_params['shmf_plaw_index'])
        norm = area_elem * f_host*(subhalo_params['m_pivot']**(-
                subhalo_params['shmf_plaw_index']-1))
        e_total = e_counts * norm * subhalo_params['sigma_sub']

        rng = jax.random.PRNGKey(0)
        n_loops = 100
        total = 0
        for _ in range(n_loops):
            rng_draw, rng = jax.random.split(rng)
            total += jnp.sum(draw_nfw_masses(main_deflector_params,
                subhalo_params, cosmology_params, rng_draw) > 0)
        self.assertAlmostEqual(total / n_loops, e_total, places = 0)

    @chex.all_variants
    def test_rejection_sampling(self):
        # Test that the distributions are respected and that the keep
        # calculation is correct
        rng = jax.random.PRNGKey(0)
        radial_coord = jax.random.uniform(rng, (int(1e6),))
        r_two_hund = 0.5
        r_bound = 0.7

        rejection_sampling = self.variant(subhalos.rejection_sampling)

        is_inside, cart_pos = rejection_sampling(radial_coord, r_two_hund,
            r_bound, rng)

        np.testing.assert_array_almost_equal(jnp.sqrt(jnp.sum(cart_pos ** 2,
            axis = -1)), radial_coord)
        phi = jnp.arccos(cart_pos[:, 2] / radial_coord)
        theta = jnp.arctan(cart_pos[:, 1] / cart_pos[:, 0])
        self.assertAlmostEqual(jnp.mean(phi), jnp.pi/2, places=2)
        self.assertAlmostEqual(jnp.mean(theta), 0.0, places=2)

        # Manually confirm bounds
        self.assertEqual(
            jnp.sum(is_inside[radial_coord * jnp.sin(phi) > r_bound]), 0.0)
        self.assertEqual(
            jnp.sum(is_inside[radial_coord * jnp.cos(phi) > r_two_hund]), 0.0)
