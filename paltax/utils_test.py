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
"""Tests for utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from paltax import lens_models
from paltax import utils

COSMOLOGY_DICT = {
        'omega_m_zero': 0.30966,
        'omega_de_zero': 0.688846306,
        'omega_rad_zero': 0.0014937,
        'hubble_constant': 67.66,
}


def _prepare_x_y_angular():
    rng = jax.random.PRNGKey(3)
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.normal(rng_x, shape=(3,))
    y = jax.random.normal(rng_y, shape=(3,))
    return x, y


def _prepare_kwargs_detector():
    return {'n_x': 2, 'n_y': 2, 'pixel_width': 0.04, 'supersampling_factor': 2}


def _prepare_ellip_to_angle_expected():
    angle = jnp.array([
        -0.78539816, -0.77996359, -0.77377235, -0.76665695, -0.75839741,
        -0.7486995 , -0.73716128, -0.72322067, -0.70607053, -0.68451699,
        -0.65673631, -0.61985013, -0.56919428, -0.49721055, -0.39269908,
        -0.24497866, -0.0621775 ,  0.11554533,  0.25354925,  0.34994643,
        0.41649063,  0.46364761,  0.49824575,  0.52448102,  0.54495452,
        0.56132568,  0.57468856,  0.58578778,  0.59514497,  0.60313516,
        0.61003419,  0.61604912,  0.62133822,  0.62602438,  0.63020446,
        0.63395573,  0.63734059,  0.64040996,  0.64320578,  0.6457629 ,
        0.64811052,  0.65027329,  0.65227214,  0.65412496,  0.65584715,
        0.65745202,  0.65895113,  0.66035458,  0.66167121,  0.66290883])
    axis_ratio = jnp.array([
        0.66666667, 0.68383307, 0.70132021, 0.71912865, 0.7372559 ,
        0.7556948 , 0.77443124, 0.79344031, 0.81268003, 0.8320804 ,
        0.85152379, 0.87080793, 0.88957318, 0.90715754, 0.92232629,
        0.93293886, 0.93628242, 0.93117052, 0.91934347, 0.90350591,
        0.88558669, 0.86666667, 0.8473246 , 0.82787707, 0.80850352,
        0.78930892, 0.77035595, 0.75168218, 0.73330975, 0.71525102,
        0.69751196, 0.68009431, 0.66299695, 0.64621684, 0.62974962,
        0.61359003, 0.59773223, 0.58216998, 0.56689685, 0.55190624,
        0.53719156, 0.52274621, 0.50856366, 0.49463746, 0.4809613 ,
        0.46752897, 0.45433443, 0.44137175, 0.42863519, 0.41611913])

    return axis_ratio, angle


class UtilsTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of utility functions."""

    def test_coordinates_evaluate(self):
        parameters = _prepare_kwargs_detector()
        expected_x = jnp.array([
                -0.03, -0.01, 0.01, 0.03, -0.03, -0.01, 0.01, 0.03, -0.03,
                -0.01, 0.01, 0.03, -0.03, -0.01, 0.01, 0.03
        ])
        expected_y = jnp.array([
                -0.03, -0.03, -0.03, -0.03, -0.01, -0.01, -0.01, -0.01, 0.01,
                0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03
        ])

        np.testing.assert_allclose(
                jnp.array(utils.coordinates_evaluate(**parameters)),
                jnp.array([expected_x, expected_y]))

    @chex.all_variants
    def test_unpack_parameters_xy(self):
        x, y = _prepare_x_y_angular()
        kwargs_lens = {
                'alpha_rs': 1.0,
                'scale_radius': 1.0,
                'center_x': 0.0,
                'center_y': 0.0,
                'fake_param': 19.2
        }
        expected = jnp.array([[-0.90657, -0.29612964, 0.22304466],
                              [0.44380534, -0.9504099, 0.97678715]])

        unpack_parameters_derivatives = self.variant(
                utils.unpack_parameters_xy(lens_models.NFW.derivatives,
                                           lens_models.NFW.parameters))

        np.testing.assert_allclose(
                unpack_parameters_derivatives(x, y, kwargs_lens), expected,
                rtol=1e-5)

    def test_downsampling(self):
        downsample = utils.downsample

        image = jnp.ones((12, 12))
        np.testing.assert_allclose(downsample(image, 3), jnp.ones((4, 4)) * 9)
        np.testing.assert_allclose(downsample(image, 4), jnp.ones((3, 3)) * 16)

        image = jax.random.normal(jax.random.PRNGKey(0), shape=(4, 4))
        expected = jnp.array([[0.37571156, 0.7770451],
                              [-0.67193794, 0.014301866]]) * 4
        np.testing.assert_allclose(downsample(image, 2), expected)

    @chex.all_variants
    @parameterized.named_parameters([
            (f'_mag_{mag}_mzp_{mzp}', mag, mzp, expected)
            for mag, mzp, expected in zip(
            [20.35, -2, 15], [18.0, 5.6, 8.8],
            [0.11481536214968811, 1096.4781961431852, 0.0033113112148259144])
    ])
    def test_magnitude_to_cps(self, mag, mzp, expected):
        magnitude_to_cps = self.variant(utils.magnitude_to_cps)

        self.assertAlmostEqual(magnitude_to_cps(mag, mzp), expected, places=5)

    @chex.all_variants
    @parameterized.named_parameters([(f'z_{z}', z) for z in [0.5, 0.2134]])
    def test_get_k_correction(self, z):
        get_k_correction = self.variant(utils.get_k_correction)
        self.assertAlmostEqual(get_k_correction(z), 2.5 * jnp.log(1 + z))

    @chex.all_variants
    def test_rotate_coordinates(self):
        rotate_coordinates = self.variant(utils.rotate_coordinates)
        grid_x = jnp.ones(10)
        grid_y = jnp.zeros(10)
        angle = jnp.pi/4
        grid_x, grid_y = rotate_coordinates(grid_x, grid_y, angle)

        # Make sure rotation is counterclockwise.
        np.testing.assert_array_almost_equal(grid_x, np.ones(10) / np.sqrt(2))
        np.testing.assert_array_almost_equal(grid_y, np.ones(10) / np.sqrt(2))

    def test_random_permutation_iterator(self):
        # Test that the iterator cycles over all values and returns a random
        # permutation.
        array_to_cycle = jnp.linspace(0, 10, 11)
        rng = jax.random.PRNGKey(0)
        generator = utils.random_permutation_iterator(array_to_cycle, rng)

        cycle_one = []
        cycle_two = []
        for cycle in [cycle_one, cycle_two]:
            for _ in range(len(array_to_cycle)):
                cycle.append(float(next(generator)))

        self.assertNotEqual(cycle_one, cycle_two)
        self.assertEqual(set(cycle_one), set(cycle_two))

    @chex.all_variants
    def test_ellip_to_angle(self):
        # Compare conversion values to lenstornomy
        ellip_x = np.linspace(0.0, 0.1)
        ellip_xy = np.linspace(-0.2, 0.4)
        ellip_to_angle = self.variant(utils.ellip_to_angle)

        axis_ratio, angle = ellip_to_angle(ellip_x, ellip_xy)
        axis_ratio_expected, angle_expected = _prepare_ellip_to_angle_expected()

        np.testing.assert_array_almost_equal(axis_ratio, axis_ratio_expected)
        np.testing.assert_array_almost_equal(angle, angle_expected)


if __name__ == '__main__':
    absltest.main()
