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
"""Tests for lens_models.py.

Expected values are drawn from lenstronomy:
https://github.com/lenstronomy/lenstronomy.
"""

import functools
import inspect

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

from paltax import lens_models
from paltax import utils


def _prepare_epl_parameters():
    return {
            'theta_e': 1.1,
            'slope': 2.0,
            'axis_ratio': 0.5,
            'angle': np.pi / 4,
            'center_x': 0.2,
            'center_y': -0.2,
    }


def _prepare_epl_ellip_parameters():
    return {
            'theta_e': 1.3,
            'slope': 2.1,
            'ellip_x': 0.2,
            'ellip_xy': -0.3,
            'center_x': 0.1,
            'center_y': -0.1,
    }

def _prepare_nfw_parameters():
    return {
            'scale_radius': 1.2,
            'alpha_rs': 4.0,
            'center_x': 0.2,
            'center_y': -0.2,
    }


def _prepare_shear_parameters():
    return {
            'gamma_ext': 1.2,
            'angle': np.pi / 7,
            'zero_x': 0.2,
            'zero_y': -0.2,
    }


def _prepare_tnfw_parameters():
    tnfw_parameters = _prepare_nfw_parameters()
    tnfw_parameters['trunc_radius'] = 2.0
    return tnfw_parameters


def _prepare_x_y():
    rng = jax.random.PRNGKey(0)
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.normal(rng_x, shape=(3,))
    y = jax.random.normal(rng_y, shape=(3,))
    return x, y


def _prepare_reduced_radii():
    rng = jax.random.PRNGKey(0)
    rng_rr, rng_trr = jax.random.split(rng)
    reduced_radius = jax.random.uniform(rng_rr, shape=(3,))
    truncated_reduced_radius = jax.random.uniform(rng_trr, shape=(3,))
    return reduced_radius, truncated_reduced_radius


def _prepare__polar_to_cartesian_expected(gamma_ext, angle):
    if gamma_ext == 0.1 and angle == np.pi / 6:
        return jnp.array([0.05, 0.0866025404])
    elif gamma_ext == 2.0 and angle == -np.pi / 3:
        return jnp.array([-1.0, -1.732050808])
    elif gamma_ext == 20.3 and angle == np.pi / 22:
        return jnp.array([19.4777074, 5.719170904])
    else:
        raise ValueError(
                f'gamma_ext={gamma_ext} angle={angle} are not a supported '
                ' parameter combination.')


class AllTest(chex.TestCase):
    """Runs tests of __all__ property of lens_models module."""

    def test_all(self):
        all_present = sorted(lens_models.__all__)
        all_required = []
        ignore_list = ['_LensModelBase', 'Any']
        for name, value in inspect.getmembers(lens_models):
            if inspect.isclass(value) and name not in ignore_list:
                all_required.append(name)

        self.assertListEqual(all_present, sorted(all_required))


class LensModelBaseTest(chex.TestCase):
    """Runs tests of __LensModelBase functions."""

    def test_modify_cosmology_params(self):
        # Make sure the dict is modified by default.
        input_dict = {'a': 20}
        new_dict = lens_models._LensModelBase().modify_cosmology_params(
            input_dict)
        self.assertDictEqual(input_dict, new_dict)

    @chex.all_variants
    def test_convert_to_angular(self):
        input_dict = {'a': 1.0, 'b': 12.0, 'c': 2}
        cosmology_params = {}

        def call_convert_to_angular(input_dict, cosmology_params, cls_to_call):
            return cls_to_call.convert_to_angular(input_dict, cosmology_params)

        convert_to_angular = self.variant(
            functools.partial(
                call_convert_to_angular,
                cls_to_call=lens_models._LensModelBase()
            )
        )

        self.assertDictEqual(
            input_dict, convert_to_angular(input_dict, cosmology_params))

    def test_add_lookup_tables(self):
        # Test that the dictionary isn't modified.
        lookup_tables = {}
        lookup_tables = lens_models._LensModelBase.add_lookup_tables(
            lookup_tables
        )
        self.assertEmpty(lookup_tables)


class EPLTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of EPL derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(lens_models.EPL.parameters)
        correct_parameters = sorted(_prepare_epl_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_derivatives(self):
        x, y = _prepare_x_y()
        parameters = _prepare_epl_parameters()
        expected = jnp.array([[-0.28093671, 0.50107465, -0.92706789],
                              [1.07278208, -1.03556666, -0.31412716]])

        derivatives = self.variant(lens_models.EPL.derivatives)

        np.testing.assert_allclose(jnp.asarray(derivatives(x, y, **parameters)),
                                   expected, rtol=1e-6)

    @chex.all_variants
    def test__complex_derivative(self):
        x, y = _prepare_x_y()
        scale_length = 2.0
        parameters = _prepare_epl_parameters()
        expected = jnp.array([
                0.24284143 + 3.02660008j,
                0.78504067 - 2.88261008j,
                -1.42493992 - 2.47468632j,
        ])

        complex_derivative = self.variant(lens_models.EPL._complex_derivative)

        np.testing.assert_allclose(
                complex_derivative(x, y, scale_length, parameters['axis_ratio'],
                                                     parameters['slope']),
                expected,
                rtol=1e-6)

    @chex.all_variants
    def test__hypergeometric_series(self):
        ellip_angle = jax.random.normal(jax.random.PRNGKey(0), shape=(3,))
        ellip_angle *= np.pi
        parameters = _prepare_epl_parameters()
        expected = jnp.array([
                0.83773771 - 0.4450688j,
                0.0816033 - 1.1360697j,
                0.60248699 + 0.86020766j,
        ])

        hypergeometric_series = self.variant(
            lens_models.EPL._hypergeometric_series
        )

        np.testing.assert_allclose(
            hypergeometric_series(
                ellip_angle, parameters['slope'],
                parameters['axis_ratio']
            ), expected
        )


class EPLEllipTest(chex.TestCase):
    """Runs tests of EPLEllipTest derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(lens_models.EPLEllip.parameters)
        correct_parameters = sorted(_prepare_epl_ellip_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_derivatives(self):
        # Ensure that the results are the same as EPL after conversion.
        x, y = _prepare_x_y()
        epl_ellip_parameters = _prepare_epl_ellip_parameters()
        epl_parameters = _prepare_epl_ellip_parameters()
        axis_ratio, angle = utils.ellip_to_angle(
            epl_parameters.pop('ellip_x'), epl_parameters.pop('ellip_xy'))
        epl_parameters['axis_ratio'] = axis_ratio
        epl_parameters['angle'] = angle

        derivatives = self.variant(lens_models.EPLEllip.derivatives)

        np.testing.assert_allclose(
            jnp.asarray(derivatives(x, y, **epl_ellip_parameters)),
            jnp.asarray(lens_models.EPL.derivatives(x, y, **epl_parameters)),
            rtol=1e-6)


class NFWTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of NFW derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(lens_models.NFW.parameters)
        correct_parameters = sorted(_prepare_nfw_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_derivatives(self):
        x, y = _prepare_x_y()
        nfw_parameters = _prepare_nfw_parameters()
        expected = jnp.array([[-0.18378359, 1.1456927, -3.3447392],
                              [4.0263114, -3.7813387, -1.7911061]])

        derivatives = self.variant(lens_models.NFW.derivatives)

        np.testing.assert_allclose(
            jnp.array(derivatives(x, y, **nfw_parameters)), expected, rtol=1e-4
        )

    @chex.all_variants
    @parameterized.named_parameters([
            (f'_ars_{ars}_sr_{sr}', ars, sr, expected)
                for ars, sr, expected in zip(
                    [0.5, 1.2], [0.1, 1.5],
                    [40.736141915886606, 0.43451884710279054])
    ])
    def test__alpha_to_rho(self, ars, sr, expected):
        self.assertAlmostEqual(
                self.variant(lens_models.NFW._alpha_to_rho)(ars, sr), expected)

    @chex.all_variants
    def test__nfw_derivatives(self):
        x, y = _prepare_x_y()
        radius = jnp.sqrt(x**2 + y**2)
        rho_input = 2.0
        nfw_parameters = _prepare_nfw_parameters()
        expected = jnp.array(
            [[0.42659888, 1.3704396, -2.2209157],
            [3.4939308, -3.2851558, -2.4732482]]
        )

        nfw_derivatives = self.variant(lens_models.NFW._nfw_derivatives)

        np.testing.assert_allclose(
            jnp.array(nfw_derivatives(
                radius, nfw_parameters['scale_radius'], rho_input, x, y
            )), expected, rtol=1e-4
        )

    @chex.all_variants
    def test__nfw_integral(self):
        reduced_radius = jax.random.uniform(jax.random.PRNGKey(0), shape=(3,))
        expected = jnp.array([0.2952541, 0.07187331, 0.18085146])

        integral = self.variant(lens_models.NFW._nfw_integral)

        np.testing.assert_allclose(
                jnp.asarray(integral(reduced_radius)), expected, rtol=1e-5)


class ShearTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of Shear and ShearCart derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(lens_models.Shear.parameters)
        correct_parameters = sorted(_prepare_shear_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_derivatives(self):
        x, y = _prepare_x_y()
        parameters = _prepare_shear_parameters()
        expected = jnp.array([[1.2095041, -0.7264168, -0.9143843],
                              [-1.0582784, 1.0540832, -0.393031]])

        derivatives = self.variant(lens_models.Shear.derivatives)

        np.testing.assert_allclose(
                jnp.asarray(derivatives(x, y, **parameters)), expected,
                rtol=1e-5)

    @chex.all_variants
    @parameterized.named_parameters([
            (f'_ge_{ge}_ang_{ang}', ge, ang)
            for ge, ang in zip([0.1, 2.0, 20.3],
                               [np.pi / 6, -np.pi / 3, np.pi / 22])
    ])
    def test__polar_to_cartesian(self, gamma_ext, angle):
        expected = _prepare__polar_to_cartesian_expected(gamma_ext, angle)
        polar_to_cartesian = self.variant(lens_models.Shear._polar_to_cartesian)

        np.testing.assert_allclose(
                jnp.array(polar_to_cartesian(gamma_ext, angle)), expected,
                rtol=1e-5)


class TNFWTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of TNFW derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(lens_models.TNFW.parameters)
        correct_parameters = sorted(_prepare_tnfw_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_derivatives(self):
        x, y = _prepare_x_y()
        tnfw_parameters = _prepare_tnfw_parameters()
        expected = jnp.array(
            [[-0.13036814, 0.8646701, -2.6658154],
            [2.8560917, -2.8538284, -1.4275427]]
        )

        derivatives = self.variant(lens_models.TNFW.derivatives)

        np.testing.assert_allclose(
            jnp.asarray(derivatives(x, y, **tnfw_parameters)), expected,
            rtol=1e-5
        )

    @chex.all_variants
    def test__tnfw_derivatives(self):
        x, y = _prepare_x_y()
        r = jnp.sqrt(x**2 + y**2)
        tnfw_parameters = _prepare_tnfw_parameters()
        rho_input = 2.0
        expected = jnp.array(
            [[0.3161475, 0.97571015, -1.7840269],
            [2.5893118, -2.3389282, -1.9868112]]
        )

        tnfw_derivatives = self.variant(lens_models.TNFW._tnfw_derivatives)

        np.testing.assert_allclose(
            jnp.asarray(
                tnfw_derivatives(r, tnfw_parameters['scale_radius'], rho_input,
                tnfw_parameters['trunc_radius'], x, y)
            ), expected, rtol=1e-5
        )

    @chex.all_variants
    def test__tnfw_integral(self):
        reduced_radius, truncated_reduced_radius = _prepare_reduced_radii()
        expected = jnp.array([0.10208127, 0.01536825, 0.02873538])

        tnfw_integral = self.variant(lens_models.TNFW._tnfw_integral)

        np.testing.assert_allclose(
                tnfw_integral(reduced_radius, truncated_reduced_radius),
                expected,
                rtol=1e-5)

    @chex.all_variants
    def test__tnfw_log(self):
        reduced_radius, truncated_reduced_radius = _prepare_reduced_radii()
        expected = jnp.array([-1.2336285, -0.15917292, -0.8316463])

        tnfw_log = self.variant(lens_models.TNFW._tnfw_log)

        np.testing.assert_allclose(
            tnfw_log(reduced_radius, truncated_reduced_radius), expected,
            rtol=1e-5
        )

    @chex.all_variants
    def test__nfw_function(self):
        reduced_radius = jnp.asarray([0.2, 1.0, 1.5])
        expected = jnp.array([2.33970328, 1.0, 0.75227469])

        nfw_function = self.variant(lens_models.TNFW._nfw_function)

        np.testing.assert_allclose(
                nfw_function(reduced_radius), expected, rtol=1e-5)

    def test_add_lookup_tables(self):
        # Check that the lookup table adds the desired nfw_function lookup.
        lookup_tables = {}
        lookup_tables = lens_models.TNFW.add_lookup_tables(lookup_tables)

        self.assertAlmostEqual(lookup_tables['tnfw_lookup_dr'], 0.001)
        self.assertAlmostEqual(lookup_tables['tnfw_lookup_log_min_radius'], -3)
        np.testing.assert_array_almost_equal(
            lookup_tables['tnfw_lookup_nfw_func'],
            jnp.log(lens_models.TNFW._nfw_function(jnp.logspace(-3, 3, 6001)))
        )

    @chex.all_variants
    def test_derivatives_lookup(self):
        # Test that the derivatives return the same answer with and without
        # the use of the lookup tables.
        x, y = [jnp.linspace(-100, 100, 100)] * 2
        tnfw_parameters = _prepare_tnfw_parameters()

        lookup_tables = lens_models.TNFW.add_lookup_tables({})
        derivatives_lookup = self.variant(
            functools.partial(
                lens_models.TNFW.derivatives, lookup_tables=lookup_tables
            )
        )
        np.testing.assert_allclose(
            jnp.asarray(derivatives_lookup(x, y, **tnfw_parameters)),
            jnp.asarray(lens_models.TNFW.derivatives(x, y, **tnfw_parameters)),
            rtol=4e-4
        )

        # Test that the lookup tables are being used.
        lookup_tables['tnfw_lookup_nfw_func'] = jnp.ones(5) * 10
        derivatives_lookup = self.variant(
            functools.partial(
                lens_models.TNFW.derivatives, lookup_tables=lookup_tables
            )
        )
        np.testing.assert_array_less(
            jnp.abs(lens_models.TNFW.derivatives(x, y, **tnfw_parameters)[0]),
            jnp.abs(derivatives_lookup(x, y, **tnfw_parameters)[0])
        )
