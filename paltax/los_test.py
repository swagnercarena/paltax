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
"""Tests for los.py."""

import functools

from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np

from paltax import cosmology_utils
from paltax import los
from paltax import power_law


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


def _prepare_cosmology_params_los(los_params, cosmology_params_init,
    z_lookup_max, dz, r_min=1e-4, r_max=1e3, n_r_bins=2):
    cosmology_params = _prepare_cosmology_params(cosmology_params_init,
        z_lookup_max, dz, r_min, r_max, n_r_bins)
    return los.add_los_lookup_tables_to_cosmology_params(los_params,
        cosmology_params, z_lookup_max)


def _prepare_main_deflector_params():
    main_deflector_params = {'mass': 1e13, 'z_lens': 0.5, 'theta_e': 2.38,
        'center_x': 0.0, 'center_y': 0.0}
    return main_deflector_params


def _prepare_source_params():
    main_deflector_params = {'z_source': 1.5}
    return main_deflector_params


def _prepare_los_params():
    los_params = {'delta_los': 1.1, 'r_min':0.5, 'r_max':10.0, 'm_min': 1e6,
        'm_max': 1e10, 'cone_angle': 8.0, 'angle_buffer': 0.8,
        'c_zero': 18, 'conc_zeta': -0.2, 'conc_beta': 0.8, 'conc_m_ref': 1e8,
        'conc_dex_scatter': 0.0}
    return los_params


def _prepare_nu_function_expected():
    return jnp.array([1.53490537e-02, 2.89663922e-02, 5.63195644e-02,
        1.16683673e-01, 2.56467729e-01, 1.53451398e-01, 1.22804756e-15,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00])


def _prepare_mass_function_exact_expected(z):
    if z == 0.1:
        return jnp.array([5.76805863e-13, 1.38454033e-20])
    elif z == 0.2:
        return jnp.array([7.70313721e-13, 1.84294010e-20])
    elif z == 0.3:
        return jnp.array([1.00724901e-12, 2.40028314e-20])
    else:
        raise ValueError(f'{z} is not a valid input redshift.')


class LosTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants
    def test_nu_function(self):
        # Test the nu function for precomputed values.
        nu = jnp.logspace(-3, 3, 10)
        expected = _prepare_nu_function_expected()

        nu_function =  self.variant(los.nu_function)

        np.testing.assert_array_almost_equal(nu_function(nu), expected)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.1, 0.2, 0.3]])
    def test_mass_function_exact(self, z):
        # Test that the mass function gives the correct exact evaluation.
        masses = jnp.array([1e6, 1e6])
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
            z)
        dr = 1e-6
        lag_r = cosmology_utils.lagrangian_radius(cosmology_params, masses[0])
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
                z, lag_r - dr * 2, lag_r + dr * 2, 5)

        # First expected value is for 1e-6. Second expected value is for 1e-10.
        expected = _prepare_mass_function_exact_expected(z)

        mass_function = self.variant(los.mass_function_exact)

        self.assertAlmostEqual(mass_function(cosmology_params, masses, z)[0],
            expected[0])

        masses = jnp.array([1e10, 1e10])
        lag_r = cosmology_utils.lagrangian_radius(cosmology_params, masses[0])
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
                z, lag_r - dr * 2, lag_r + dr * 2, 5)

        self.assertAlmostEqual(mass_function(cosmology_params, masses, z)[0],
            expected[1])

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.1, 0.2, 0.3]])
    def test__mass_function_power_law_numerical(self, z):
        # Make sure that the power law fit is in decent agreement with the los
        # values.
        m_min = 1e6
        m_max = 1e10
        masses = jnp.logspace(jnp.log10(m_min) , jnp.log10(m_max), 100)
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
            z)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
                z, r_min, r_max, 1000)
        mass_function_exact = los.mass_function_exact(cosmology_params, masses,
            z)

        slope, norm = self.variant(los._mass_function_power_law_numerical)(
            cosmology_params, z, m_min, m_max)
        estimate = norm * masses ** slope

        np.testing.assert_array_almost_equal(estimate / estimate,
            mass_function_exact / estimate, decimal = 1)

    def test_add_los_lookup_tables_to_cosmology_params(self):
        # Make sure that the power law fit is in decent agreement with the los
        # values.
        los_params = _prepare_los_params()
        m_min = los_params['m_min']
        m_max = los_params['m_max']
        z_lookup_max = 0.5
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_lookup_max, los_params['dz'])
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params_los(los_params,
            COSMOLOGY_PARAMS_INIT, z_lookup_max, los_params['dz'], r_min, r_max,
            1000)

        # Manually test two values
        i = 2
        slope_expected, norm_expected = los._mass_function_power_law_numerical(
            cosmology_params, los_params['dz'] / 2 * i, los_params['m_min'],
            los_params['m_max']
        )
        self.assertAlmostEqual(
            cosmology_params['mass_function_slope_lookup_table'][i],
            slope_expected, places=3
        )
        self.assertAlmostEqual(
            cosmology_params['mass_function_norm_lookup_table'][i],
            norm_expected, places=3
        )

        i = 4
        slope_expected, norm_expected = los._mass_function_power_law_numerical(
            cosmology_params, los_params['dz'] / 2 * i, los_params['m_min'],
            los_params['m_max']
        )
        self.assertAlmostEqual(
            cosmology_params['mass_function_slope_lookup_table'][i],
            slope_expected, places=3
        )
        self.assertAlmostEqual(
            cosmology_params['mass_function_norm_lookup_table'][i],
            norm_expected, places=3
        )

        # Test global properties.
        self.assertAlmostEqual(cosmology_params['los_z_lookup_max'], 1.12)
        self.assertAlmostEqual(cosmology_params['los_num_z_bins'], 10)


    @chex.all_variants(without_device=False)
    def test_mass_function_power_law(self):
        # Check that it agrees with the numerical function.
        los_params = _prepare_los_params()
        m_min = los_params['m_min']
        m_max = los_params['m_max']
        z_lookup_max = 0.5
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            z_lookup_max, los_params['dz'])
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params_los(los_params,
            COSMOLOGY_PARAMS_INIT, z_lookup_max, los_params['dz'], r_min, r_max,
            1000)

        z = 0.1
        mass_function_power_law = self.variant(los.mass_function_power_law)
        test_slope, test_norm = mass_function_power_law(cosmology_params, z)
        expected_slope, expected_norm = los._mass_function_power_law_numerical(
            cosmology_params, z, m_min, m_max)
        self.assertAlmostEqual(test_slope, expected_slope, places=3)
        self.assertAlmostEqual(test_norm, expected_norm, places=3)

        z = 0.2-1e-6
        test_slope, test_norm = mass_function_power_law(cosmology_params, z)
        expected_slope, expected_norm = los._mass_function_power_law_numerical(
            cosmology_params, z, m_min, m_max)
        self.assertAlmostEqual(test_slope, expected_slope, places=3)
        self.assertAlmostEqual(test_norm, expected_norm, places=3)


    @chex.all_variants(without_device=False)
    def test_cone_angle_to_radius(self):
        # Test that the conditional statements in jax behave as one would
        # expect given a manual calculation in python.
        main_deflector_params = _prepare_main_deflector_params()
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, 1.5,
            0.25)

        cone_angle_to_radius = self.variant(los.cone_angle_to_radius)

        for z in [0.25, 0.5]:
            hand_calc = cosmology_utils.kpc_per_arcsecond(cosmology_params, z)
            hand_calc *= los_params['cone_angle'] * 0.5
            self.assertAlmostEqual(hand_calc, cone_angle_to_radius(
                    main_deflector_params, source_params, los_params,
                    cosmology_params, z))

        for z in [0.75, 1.0, 1.25, 1.5]:
            # These should be shrinking from the main deflector redshift
            hand_calc = los_params['angle_buffer']
            hand_calc *= (
                cosmology_utils.comoving_distance(cosmology_params, 0.0, 1.5) /
                cosmology_utils.comoving_distance(cosmology_params, 0.5, 1.5))
            hand_calc *= (
                cosmology_utils.comoving_distance(cosmology_params, 0.5, z) /
                cosmology_utils.comoving_distance(cosmology_params, 0.0, z))
            hand_calc = 1-hand_calc
            hand_calc *= cosmology_utils.kpc_per_arcsecond(cosmology_params,
                z)
            hand_calc *= los_params['cone_angle'] * 0.5
            self.assertAlmostEqual(hand_calc, cone_angle_to_radius(
                    main_deflector_params, source_params, los_params,
                    cosmology_params, z), places=5)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.1, 0.4, 0.8]])
    def test_volume_element(self, z):
        # Test that the volume element
        main_deflector_params = _prepare_main_deflector_params()
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            source_params['z_source'], los_params['dz'] / 2)

        volume_element = self.variant(los.volume_element)
        los_radius = los.cone_angle_to_radius(main_deflector_params,
            source_params, los_params, cosmology_params,
            z + los_params['dz'] / 2)
        dz_in_kpc = cosmology_utils.comoving_distance(cosmology_params, z,
            z + los_params['dz'])
        dz_in_kpc /= (1 + z + los_params['dz'] / 2) / 1e3
        hand_calc = los_radius ** 2 * np.pi * dz_in_kpc

        self.assertAlmostEqual(volume_element(main_deflector_params,
            source_params, los_params, cosmology_params, z) / hand_calc,
            hand_calc/hand_calc, places = 5)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.1, 0.4, 0.8]])
    def test_expected_num_halos(self, z):
        # Check that the expected halo counts matches a calculation by hand.
        main_deflector_params = _prepare_main_deflector_params()
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        m_min = los_params['m_min']
        m_max = los_params['m_max']
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            source_params['z_source'], los_params['dz'] / 2)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params_los(los_params,
            COSMOLOGY_PARAMS_INIT, source_params['z_source'],
            los_params['dz'] / 2, r_min, r_max, 1000)

        slope, norm = los.mass_function_power_law(cosmology_params, z)
        norm *= los.volume_element(main_deflector_params, source_params,
            los_params, cosmology_params, z)
        norm *= power_law.power_law_integrate(m_min, m_max, slope)
        hand_calc = norm * los_params['delta_los']

        expected_num_halos = self.variant(los.expected_num_halos)

        # Low precision for jitted call from float32.
        self.assertAlmostEqual(expected_num_halos(main_deflector_params,
            source_params, los_params, cosmology_params, z) / hand_calc,
            hand_calc / hand_calc, places=3)

        # Negative normalization should be treated like 0.0
        los_params['delta_los'] = -1.0
        self.assertAlmostEqual(
            expected_num_halos(main_deflector_params, source_params,
                los_params, cosmology_params, z),
            0.0)

    @chex.all_variants(without_device=False)
    def test_draw_redshifts(self):
        # Check that the ratio of drawn redshifts approximates the expected
        # number of los halos in each slice.
        z_min = 0.3
        z_max = 0.5
        main_deflector_params = _prepare_main_deflector_params()
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        m_min = los_params['m_min']
        m_max = los_params['m_max']
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            source_params['z_source'], los_params['dz'] / 2)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params_los(los_params,
            COSMOLOGY_PARAMS_INIT, source_params['z_source'],
            los_params['dz'] / 2, r_min, r_max, 1000)
        num_expected_min = los.expected_num_halos(main_deflector_params,
            source_params, los_params, cosmology_params, z_min)
        num_expected_max = los.expected_num_halos(main_deflector_params,
            source_params, los_params, cosmology_params,
            z_max - los_params['dz'])

        num_z_bins = 1000
        pad_length = 3000
        draw_redshifts = self.variant(functools.partial(los.draw_redshifts,
            num_z_bins=num_z_bins, pad_length=pad_length))

        rng = jax.random.PRNGKey(0)
        z_draws = draw_redshifts(main_deflector_params, source_params,
            los_params, cosmology_params, z_min, z_max, rng)

        # Doing many draws is expensive, so we won't be picky.
        self.assertAlmostEqual(
            num_expected_max / (num_expected_max + num_expected_min),
            jnp.sum(z_draws > 0.4) / jnp.sum(z_draws > 0.3), places = 1)
        self.assertAlmostEqual(1.0,
            jnp.sum(z_draws > 0.3) / (num_expected_max + num_expected_min),
            places = 1)

    @chex.all_variants(without_device=False)
    def test_draw_masses(self):
        # Testing that the masses are roughly drawn from the desired power law.
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        m_min = los_params['m_min']
        m_max = los_params['m_max']
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            source_params['z_source'], los_params['dz'] / 2)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params_los(los_params,
            COSMOLOGY_PARAMS_INIT, source_params['z_source'],
            los_params['dz'] / 2, r_min, r_max, 1000)

        rng = jax.random.PRNGKey(0)
        z = 0.3
        z_values = jnp.full((1000,), z)

        draw_masses = self.variant(los.draw_masses)

        slope, _ = los.mass_function_power_law(cosmology_params, z)
        integrate_below = power_law.power_law_integrate(m_min, 5e6, slope)
        integrate_above = power_law.power_law_integrate(5e6, m_max, slope)

        masses = draw_masses(los_params, cosmology_params, z_values, rng)
        # Draws are expensive in the non-jit case, so don't be picky.
        self.assertAlmostEqual(jnp.sum(masses > 5e6) / jnp.sum(masses < 5e6),
            integrate_above / integrate_below, places=1)

    @chex.all_variants(without_device=False)
    @parameterized.named_parameters([(f'_z_{z}', z) for z in [0.1, 0.4, 0.8]])
    def test_draw_positions(self, z):
        # Test that the positions are uniformly distributed within a disk.
        main_deflector_params = _prepare_main_deflector_params()
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT, z,
            z)

        rng = jax.random.PRNGKey(0)
        z_values = jnp.full((10000,), z)
        draw_positions = self.variant(los.draw_positions)

        los_pos = draw_positions(main_deflector_params, source_params,
            los_params, cosmology_params, z_values, rng)

        # Check the angle distribution
        angles = jnp.arctan(los_pos[:,1] / los_pos[:,0])
        self.assertAlmostEqual(np.mean(angles<0),np.mean(angles>0),places=1)

        # Make sure that within a square they are uniformly distributed
        bound = jnp.sqrt(2) * los.cone_angle_to_radius(main_deflector_params,
            source_params, los_params, cosmology_params, z)
        mask = jnp.logical_and(jnp.abs(los_pos[:,0]) < bound / 2,
            jnp.abs(los_pos[:,1]) < bound / 2)
        los_pos = los_pos[mask]

        self.assertAlmostEqual(2 * jnp.mean(los_pos[:,0] < -bound / 4),
            jnp.mean(los_pos[:,0] > 0), places=1)
        self.assertAlmostEqual(2 * jnp.mean(los_pos[:,1] < -bound / 4),
            jnp.mean(los_pos[:,1] > 0), places=1)

    @chex.all_variants(without_device=False)
    def test_draw_los(self):
        # Some basic tests that the final output of draw_los is sensible.
        main_deflector_params = _prepare_main_deflector_params()
        source_params = _prepare_source_params()
        los_params = _prepare_los_params()
        # Don't want to draw too many halos
        los_params['m_min'] = 1e8
        m_min = los_params['m_min']
        m_max = los_params['m_max']
        cosmology_params = _prepare_cosmology_params(COSMOLOGY_PARAMS_INIT,
            source_params['z_source'], los_params['dz'] / 2)
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, m_min / 10)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, m_max * 10)
        cosmology_params = _prepare_cosmology_params_los(los_params,
            COSMOLOGY_PARAMS_INIT, source_params['z_source'],
            los_params['dz'] / 2, r_min, r_max, 1000)

        rng = jax.random.PRNGKey(0)
        num_z_bins = 1000
        los_pad_length = 100
        draw_los = self.variant(functools.partial(los.draw_los,
            num_z_bins=num_z_bins, los_pad_length=los_pad_length))

        los_tuple = draw_los(main_deflector_params, source_params, los_params,
            cosmology_params, rng)

        # Simple tests on the redshifts. TODO add more tests on the parameters.
        los_before_z = los_tuple[0][0]
        los_after_z = los_tuple[1][0]

        # Check bounds on redshifts that are drawn.
        self.assertEqual(jnp.sum(los_before_z >= 0.5), 0.0)
        self.assertEqual(jnp.sum(los_before_z < 0.0),
            jnp.sum(los_before_z == -1.0))
        self.assertEqual(jnp.sum(los_after_z >= 1.5), 0.0)
        self.assertEqual(jnp.sum(los_after_z < 0.5),
            jnp.sum(los_after_z == -1.0))
