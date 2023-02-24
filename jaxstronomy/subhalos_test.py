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
"""Tests for subhalos.py."""

import functools

from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np

from jaxstronomy import cosmology_utils
from jaxstronomy import nfw_functions
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
        'center_x': jnp.array([0.01]), 'center_y': jnp.array([-0.03])}
    return main_deflector_params


def _prepare_subhalo_params():
    subhalo_params = {'sigma_sub': 5e-4, 'shmf_plaw_index': -2.0,
        'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'k_one': 0.0, 'k_two': 0.0,
        'c_zero': 18, 'conc_zeta': -0.2, 'conc_beta': 0.8, 'conc_m_ref': 1e8,
        'conc_dex_scatter': 0.0}
    return subhalo_params


def _prepare_test_truncation_radius_expected():
    return jnp.array([0.22223615, 0.74981341, 1.4133784, 2.32006694,
        3.57303672,5.30341642,7.68458502,10.94766194,15.40115448, 21.45666411])


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

    @chex.all_variants(without_device=False)
    def test_sample_cored_nfw(self):
        # Test that for large enough padding the bounds are met and the
        # distribution of points is correct.
        main_deflector_params = _prepare_main_deflector_params()
        z_lens = main_deflector_params['z_lens']
        subhalo_params = _prepare_subhalo_params()

        # Initialize our cosmology dictionary
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_lens, z_lens)
        dr = 1e-6
        lag_r = cosmology_utils.lagrangian_radius(cosmology_params,
            main_deflector_params['mass'])
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_lens, z_lens, lag_r - dr * 2, lag_r + dr * 2, 5)

        n_subs = 1000
        sampling_pad_length = 200000
        sample_cored_nfw = self.variant(functools.partial(
            subhalos.sample_cored_nfw, n_subs=n_subs,
            sampling_pad_length=sampling_pad_length))

        rng = jax.random.PRNGKey(0)
        cart_pos = sample_cored_nfw(main_deflector_params, subhalo_params,
            cosmology_params, rng)

        # Check that all the positions are within the desired volume.
        radial_coord = jnp.sqrt(jnp.sum(jnp.square(cart_pos), axis=-1))
        phi = jnp.arccos(cart_pos[:, 2] / radial_coord)
        theta = jnp.arctan(cart_pos[:, 1] / cart_pos[:, 0])
        kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
        r_bound = main_deflector_params['theta_e'] * 3 * kpa
        r_two_hund = nfw_functions.r_two_hund_from_m(cosmology_params,
            main_deflector_params['mass'], z_lens)
        self.assertAlmostEqual(jnp.mean(phi), jnp.pi/2, places=1)
        self.assertAlmostEqual(jnp.mean(theta), 0.0, places=1)
        self.assertEqual(jnp.mean(radial_coord * jnp.sin(phi) > r_bound), 0.0)
        self.assertEqual(jnp.mean(radial_coord * jnp.cos(phi) > r_two_hund),
            0.0)

    @chex.all_variants
    def test_truncation_radius(self):
        # Run some precomputed values to check agreement.
        m_two_hund = jnp.logspace(6, 9, 10)
        radii = jnp.linspace(10, 300, 10)
        expected = _prepare_test_truncation_radius_expected()

        get_truncation_radius = self.variant(subhalos.get_truncation_radius)

        np.testing.assert_array_almost_equal(get_truncation_radius(m_two_hund,
            radii), expected, decimal=5)

        # Make sure a truncation radius of 0 cannot be returned.
        self.assertGreater(get_truncation_radius(0.0, 1e-4), 0.0)
        self.assertGreater(get_truncation_radius(1e-4, 0.0), 0.0)

    @chex.all_variants(without_device=False)
    def test_convert_to_lensing(self):
        main_deflector_params = _prepare_main_deflector_params()
        z_lens = main_deflector_params['z_lens']
        z_source = 1.5
        source_params = {'z_source': z_source}
        subhalo_params = _prepare_subhalo_params()
        rng = jax.random.PRNGKey(0)

        # Initialize our cosmology dictionary
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_source, 0.01)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, 1e7)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params,
            main_deflector_params['mass'] * 10)
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_source, 0.01, r_min, r_max, 1000)

        subhalo_masses = jnp.logspace(8, 10, 100)
        subhalo_masses *= jnp.arange(0, len(subhalo_masses)) > 2
        subhalo_cart_pos = jax.random.normal(rng, shape=(100, 3))

        convert_to_lensing = self.variant(subhalos.convert_to_lensing)

        subhalos_z, subhalos_kwargs = convert_to_lensing(
            main_deflector_params, source_params, subhalo_params,
            cosmology_params, subhalo_masses, subhalo_cart_pos, rng)

        np.testing.assert_array_almost_equal(subhalos_z,
            jnp.ones(subhalos_z.shape) * z_lens)

        # Test that the lensing quantities follow manual calculations.
        kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
        np.testing.assert_array_almost_equal(subhalos_kwargs['center_x'],
            subhalo_cart_pos[:,0] / kpa + main_deflector_params['center_x'])
        np.testing.assert_array_almost_equal(subhalos_kwargs['center_y'],
            subhalo_cart_pos[:,1] / kpa + main_deflector_params['center_y'])
        # Check that the model indexing worked correctly.
        self.assertAlmostEqual(jnp.mean(
            subhalos_kwargs['model_index'][subhalo_masses > 0]), 0.0)
        self.assertAlmostEqual(jnp.mean(
            subhalos_kwargs['model_index'][subhalo_masses == 0]), -1.0)

        self.assertTrue('alpha_rs' in subhalos_kwargs)
        self.assertTrue('scale_radius' in subhalos_kwargs)
        self.assertTrue('trunc_radius' in subhalos_kwargs)

    @chex.all_variants(without_device=False)
    def test_draw_subhalos(self):
        main_deflector_params = _prepare_main_deflector_params()
        z_lens = main_deflector_params['z_lens']
        z_source = 1.5
        source_params = {'z_source': z_source}
        subhalo_params = _prepare_subhalo_params()
        rng = jax.random.PRNGKey(0)

        # Initialize our cosmology dictionary
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_source, 0.01)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, 1e7)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params,
            main_deflector_params['mass'] * 10)
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_source, 0.01, r_min, r_max, 1000)

        subhalos_pad_length = 100
        sampling_pad_length = 10000
        draw_subhalos = self.variant(functools.partial(subhalos.draw_subhalos,
            subhalos_pad_length=subhalos_pad_length,
            sampling_pad_length=sampling_pad_length))

        subhalos_z, subhalos_kwargs = draw_subhalos(main_deflector_params,
            source_params, subhalo_params, cosmology_params, rng)

        np.testing.assert_array_almost_equal(subhalos_z,
            jnp.ones(subhalos_z.shape) * z_lens)
        self.assertTrue('alpha_rs' in subhalos_kwargs)
        self.assertTrue('scale_radius' in subhalos_kwargs)
        self.assertTrue('trunc_radius' in subhalos_kwargs)
        self.assertTrue('model_index' in subhalos_kwargs)
        self.assertTrue('center_y' in subhalos_kwargs)
        self.assertTrue('center_x' in subhalos_kwargs)
