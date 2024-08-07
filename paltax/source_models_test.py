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
"""Tests for source_models.py.

Expected values are drawn from lenstronomy:
https://github.com/lenstronomy/lenstronomy.
"""

import functools
import inspect
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from paltax import cosmology_utils
from paltax import source_models
from paltax import utils

# NOTE: The tests are done assuming that the test
#       file contains 5 images

COSMOS_TEST_PATH = (
    str(pathlib.Path(__file__).parent) +
    '/test_files/cosmos_catalog_test.h5'
)


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


def catalog_weights_from_hdf5(parameter):
    # Extract catalog weights from hdf5 file corresponding to parameter
    with h5py.File(COSMOS_TEST_PATH, "r") as hdf5_file:
        catalog_weights = hdf5_file[parameter + '_weights']
        total_num_galaxies = len(hdf5_file['images'])
    return catalog_weights, total_num_galaxies

def compute_catalog_weights_cdf(catalog_weights, start_val, images_per_chunk):
    return (jnp.cumsum(
        catalog_weights[start_val:start_val + images_per_chunk]) / 
        jnp.sum(catalog_weights[start_val:start_val + images_per_chunk])
    )

def _prepare_cosmology_params(
        cosmology_params_init, z_lookup_max, dz, r_min=1e-4, r_max=1e3,
        n_r_bins=2):
    # Only generate a lookup table for values we need.
    # When 0,0 is specified for the two z values, need to select a small
    # non-zero values to generate a non-empty table.
    z_lookup_max = max(z_lookup_max, 1e-7)
    dz = max(dz, 1e-7)
    return cosmology_utils.add_lookup_tables_to_cosmology_params(
            dict(cosmology_params_init), z_lookup_max, dz, r_min, r_max,
            n_r_bins)


def _prepare_x_y():
    rng = jax.random.PRNGKey(0)
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.normal(rng_x, shape=(3,))
    y = jax.random.normal(rng_y, shape=(3,))
    return x, y


def _prepare_image():
    rng = jax.random.PRNGKey(0)
    return jax.random.normal(rng, shape=(62, 64))


def _prepare__brightness_expected(sersic_radius, n_sersic):
    if sersic_radius == 2.2 and n_sersic == 1.0:
        return jnp.array([2.227407, 1.9476248, 2.9091272])
    elif sersic_radius == 2.3 and n_sersic == 2.0:
        return jnp.array([2.9430487, 2.4276965, 4.5396194])
    elif sersic_radius == 3.5 and n_sersic == 3.0:
        return jnp.array([5.8227673, 4.809319, 9.121668])
    else:
        raise ValueError(
                f'sersic_radius={sersic_radius} n_sersic={n_sersic} are not a '
                'supported parameter combination.')


def _prepare_interpol_parameters():
    return {
            'image': _prepare_image(),
            'amp': 1.4,
            'center_x': 0.2,
            'center_y': -0.2,
            'angle': -np.pi / 6,
            'scale': 1.5
    }


def _prepare_cosmos_parameters():
    return {
            'galaxy_index': 0.01,
            'z_source': 1.5,
            'amp': 1.0,
            'output_ab_zeropoint': 23.5,
            'catalog_ab_zeropoint': 25.6,
            'center_x': 0.2,
            'center_y': -0.2,
            'angle': -np.pi / 6,
    }


def _prepare_sersic_elliptic_parameters():
    return {
            'amp': 1.4,
            'sersic_radius': 1.0,
            'n_sersic': 2.0,
            'axis_ratio': 0.2,
            'angle': np.pi / 6,
            'center_x': 0.2,
            'center_y': -0.2,
    }


class AllTest(absltest.TestCase):
    """Runs tests of __all__ property of source_models module."""

    def test_all(self):
        all_present = sorted(source_models.__all__)
        all_required = []
        ignore_list = ['_SourceModelBase', 'Any']
        for name, value in inspect.getmembers(source_models):
            if inspect.isclass(value) and name not in ignore_list:
                all_required.append(name)

        self.assertListEqual(all_present, sorted(all_required))


class SourceModelBaseTest(chex.TestCase):
    """Runs tests of _SourceModelBase functions."""

    def test_modify_cosmology_params(self):
        # Make sure the dict is modified by default.
        input_dict = {'a': 20}
        new_dict = source_models._SourceModelBase().modify_cosmology_params(
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
                cls_to_call=source_models._SourceModelBase()
            )
        )

        self.assertDictEqual(
            input_dict, convert_to_angular(input_dict, cosmology_params))


class InterpolTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of Interpol brightness functions."""

    def test_parameters(self):
        annotated_parameters = sorted(source_models.Interpol.parameters)
        correct_parameters = sorted(_prepare_interpol_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_function(self):
        x, y = _prepare_x_y()
        parameters = _prepare_interpol_parameters()
        expected = jnp.array([1.71894064, 0.4886814, 2.13953358])

        function = self.variant(source_models.Interpol.function)

        np.testing.assert_allclose(
                function(x, y, **parameters), expected, rtol=1e-5)

    @chex.all_variants
    def test__image_interpolation(self):
        x, y = _prepare_x_y()
        image = _prepare_image()
        expected = jnp.array([0.31192497, 0.64870896, 1.48134785])

        image_interpolation = self.variant(
                source_models.Interpol._image_interpolation)

        np.testing.assert_allclose(
                image_interpolation(x, y, image), expected, rtol=1e-5)

    @chex.all_variants
    def test__coord_to_image_pixels(self):
        x, y = _prepare_x_y()
        parameters = _prepare_interpol_parameters()
        expected = jnp.array([[-0.48121729, 0.51891287, -0.29162392],
                              [0.75206837, -0.48633681, -0.46977397]])

        coord_to_image_pixels = self.variant(
                source_models.Interpol._coord_to_image_pixels)

        np.testing.assert_allclose(
                jnp.asarray(
                        coord_to_image_pixels(x, y, parameters['center_x'],
                                              parameters['center_y'],
                                              parameters['angle'],
                                              parameters['scale'])),
                expected,
                rtol=1e-6)


class SersicEllipticTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of elliptical Sersic brightness functions."""

    def test_parameters(self):
        annotated_parameters = sorted(source_models.SersicElliptic.parameters)
        correct_parameters = sorted(
            _prepare_sersic_elliptic_parameters().keys()
        )
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_function(self):
        x, y = _prepare_x_y()
        parameters = _prepare_sersic_elliptic_parameters()
        expected = jnp.array([0.13602875, 0.20377299, 5.802394])

        function = self.variant(source_models.SersicElliptic.function)

        np.testing.assert_allclose(
                function(x, y, **parameters), expected, rtol=1e-5)

    @chex.all_variants
    def test__get_distance_from_center(self):
        x, y = _prepare_x_y()
        parameters = _prepare_sersic_elliptic_parameters()
        expected = jnp.array([2.6733015, 2.3254495, 0.37543342])

        get_distance_from_center = self.variant(
                source_models.SersicElliptic._get_distance_from_center)

        np.testing.assert_allclose(
                get_distance_from_center(x, y, parameters['axis_ratio'],
                                         parameters['angle'],
                                         parameters['center_x'],
                                         parameters['center_y']),
                expected,
                rtol=1e-5)

    @chex.all_variants
    @parameterized.named_parameters([
            (f'_sr_{sr}_ns_{ns}', sr, ns)
            for sr, ns in zip([2.2, 2.3, 3.5], [1.0, 2.0, 3.0])
    ])
    def test__brightness(self, sersic_radius, n_sersic):
        x, y = _prepare_x_y()
        radius = jnp.sqrt(x**2 + y**2)
        expected = _prepare__brightness_expected(sersic_radius, n_sersic)

        brightness = self.variant(source_models.SersicElliptic._brightness)

        np.testing.assert_allclose(
                brightness(radius, sersic_radius, n_sersic), expected,
                rtol=1e-5)

    @chex.all_variants
    @parameterized.named_parameters([
            (f'_n_s_{n_sersic}', n_sersic, expected)
            for n_sersic, expected in zip([1., 2., 3.],
                                          [1.6721, 3.6713, 5.6705])
    ])
    def test__b_n(self, n_sersic, expected):
        b_n = self.variant(source_models.SersicElliptic._b_n)

        np.testing.assert_allclose(b_n(n_sersic), expected, rtol=1e-5)


class CosmosCatalogTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of CosmosCatalog functions."""

    def test__init__(self):
        # Test that the intialization saves the path.
        cosmos_catalog = source_models.CosmosCatalog(COSMOS_TEST_PATH)
        self.assertEqual(cosmos_catalog.cosmos_path, COSMOS_TEST_PATH)

    def test_modify_cosmology_params(self):
        cosmos_catalog = source_models.CosmosCatalog(COSMOS_TEST_PATH)
        cosmology_params = {}
        cosmology_params = cosmos_catalog.modify_cosmology_params(
            cosmology_params
        )

        self.assertEqual(cosmology_params['cosmos_images'].shape,
                         (2, 256, 256))
        self.assertTupleEqual(cosmology_params['cosmos_redshifts'].shape, (2,))
        self.assertTupleEqual(cosmology_params['cosmos_pixel_sizes'].shape,
                              (2,))
        self.assertEqual(cosmology_params['cosmos_n_images'], 2)

    @chex.all_variants
    def test_convert_to_angular(self):
        # Test that it returns the parameters we need for the interpolation
        # function.
        cosmos_catalog = source_models.CosmosCatalog(COSMOS_TEST_PATH)


        # Start with the redshifts and zeropoints being equal.
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_INIT, 1.0, 0.01
        )
        cosmology_params = cosmos_catalog.modify_cosmology_params(
            cosmology_params
        )
        all_kwargs = {
            'galaxy_index': 0.1,
            'amp': 1.0,
            'z_source': cosmology_params['cosmos_redshifts'][0],
            'output_ab_zeropoint': 23.5,
            'catalog_ab_zeropoint': 23.5
        }

        convert_to_angular = self.variant(cosmos_catalog.convert_to_angular)
        angular_kwargs = convert_to_angular(all_kwargs, cosmology_params)

        np.testing.assert_array_almost_equal(
            angular_kwargs['image'],
            (cosmology_params['cosmos_images'][0] /
                cosmology_params['cosmos_pixel_sizes'][0] ** 2),
            decimal=4)
        self.assertAlmostEqual(angular_kwargs['amp'], 1.0)
        self.assertAlmostEqual(angular_kwargs['scale'],
                               cosmology_params['cosmos_pixel_sizes'][0])

        # Test that moving the image farther in redshift changes the pixel
        # size and image
        all_kwargs = {
            'galaxy_index': 0.1,
            'amp': 1.0,
            'z_source': 0.9,
            'output_ab_zeropoint': 23.5,
            'catalog_ab_zeropoint': 23.5
        }
        angular_kwargs = convert_to_angular(all_kwargs, cosmology_params)

        np.testing.assert_array_almost_equal(
            angular_kwargs['image'],
            (cosmology_params['cosmos_images'][0] /
                cosmology_params['cosmos_pixel_sizes'][0] ** 2),
            decimal=4)
        self.assertLess(angular_kwargs['amp'], 1.0)
        self.assertLess(angular_kwargs['scale'],
                        cosmology_params['cosmos_pixel_sizes'][0])

        # Introduce only an offset in zeropoints.
        all_kwargs = {
            'galaxy_index': 0.1,
            'amp': 1.0,
            'z_source': cosmology_params['cosmos_redshifts'][0],
            'output_ab_zeropoint': 26.5,
            'catalog_ab_zeropoint': 23.5
        }
        angular_kwargs = convert_to_angular(all_kwargs, cosmology_params)

        np.testing.assert_array_almost_equal(
            angular_kwargs['image'],
            (cosmology_params['cosmos_images'][0] /
                cosmology_params['cosmos_pixel_sizes'][0] ** 2),
            decimal=4)
        self.assertGreater(angular_kwargs['amp'], 1.0)
        self.assertAlmostEqual(angular_kwargs['scale'],
                               cosmology_params['cosmos_pixel_sizes'][0])


    @chex.all_variants
    def test_function(self):
        # The function here is inherited from interpolate, so the real test is
        # just to make sure that it doesn't crash when provided the angular
        # kwargs from convert_to_angular.
        parameters = _prepare_cosmos_parameters()
        cosmos_catalog = source_models.CosmosCatalog(COSMOS_TEST_PATH)
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_INIT, 2.0, 0.1
        )
        cosmology_params = cosmos_catalog.modify_cosmology_params(
            cosmology_params
        )
        angular_parameters = cosmos_catalog.convert_to_angular(parameters,
                                                               cosmology_params)
        for param in cosmos_catalog.physical_parameters:
            angular_parameters.pop(param)
        x, y = _prepare_x_y()

        function = self.variant(cosmos_catalog.function)

        self.assertTupleEqual(function(x, y, **angular_parameters).shape, (3,))


    @chex.all_variants
    @parameterized.named_parameters(
        (f'z_old_{z_old}_z_new_{z_new}', z_old, z_new)
        for z_old, z_new in zip([0.5, 0.0, 0.2], [0.5, 0.8, 0.7])
    )
    def test_z_scale_factor(self, z_old, z_new):
        # Test that the k-correction factor agrees with a manual calculation.
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_INIT, z_new, 0.1
        )
        z_scale_factor = self.variant(
            source_models.CosmosCatalog.z_scale_factor
        )
        expected = cosmology_utils.angular_diameter_distance(cosmology_params,
                                                             z_old)
        expected /= cosmology_utils.angular_diameter_distance(cosmology_params,
                                                              z_new)

        self.assertAlmostEqual(
            expected, z_scale_factor(z_old, z_new, cosmology_params), places=6
        )


    @chex.all_variants
    @parameterized.named_parameters(
        (f'z_old_{z_old}_z_new_{z_new}', z_old, z_new)
        for z_old, z_new in zip([0.5, 0.0, 0.2], [0.5, 0.8, 0.7])
    )
    def test_k_correct_image(self, z_old, z_new):
        # Test that the k-correction factor agrees with a manual calculation.
        mag_correction = utils.get_k_correction(z_new)
        mag_correction -= utils.get_k_correction(z_old)
        expected = 10 ** (- mag_correction / 2.5)
        k_correct_image = self.variant(
            source_models.CosmosCatalog.k_correct_image
        )

        self.assertAlmostEqual(
            expected, k_correct_image(z_old, z_new), places=6
        )


class WeightedCatalogTest(chex.TestCase):
    """Runs tests of WeightedCatalog functions."""

    def tests_setup(self):
        self.parameter = 'gini'
        self.images_per_chunk = 2
        self.weighted_catalog = source_models.WeightedCatalog(
            COSMOS_TEST_PATH, self.parameter, self.images_per_chunk
        )
        self.catalog_weights, self.total_num_galaxies = catalog_weights_from_hdf5(self.parameter)

    def test__init__(self):
        # Test that the intialization saves the hdf5 file and the correct file is opened
        self.assertEqual(self.weighted_catalog.hdf5_file.filename, COSMOS_TEST_PATH)

        # Test that the correct total number of galaxies is being stored (should be 5)
        assert self.weighted_catalog.total_num_galaxies == self.total_num_galaxies
        assert self.weighted_catalog.total_num_galaxies == 5

        # Test that the WeightedCatalog stores the images per chunk and starts at chunk 0
        assert self.weighted_catalog.images_per_chunk == self.images_per_chunk
        assert self.weighted_catalog.chunk_number == 0
        

    def test_modify_cosmology_params(self):
        # Test that the weights cdf are correctly calculated chunk-wise,
        # saved to the cosmology params, and that chunk number is incremented.
        cosmology_params = {}
        cosmology_params = self.weighted_catalog.modify_cosmology_params(
            cosmology_params
        )
        catalog_weights_cdf = compute_catalog_weights_cdf(self.catalog_weights, 0, 2)
        np.testing.assert_array_almost_equal(
            cosmology_params['catalog_weights_cdf'], catalog_weights_cdf
        )
        assert self.weighted_catalog.chunk_number == 1

        # Test that the correct number of images are being loaded in a chunk
        self.assertEqual(cosmology_params['cosmos_n_images'], 2)

        # Test that the weights cdf for the second chunk are correctly
        # calculated, saved, and that chunk number is incremented
        catalog_weights_cdf = compute_catalog_weights_cdf(self.catalog_weights, 2, 2)
        cosmology_params = self.weighted_catalog.modify_cosmology_params(
            cosmology_params
        )
        np.testing.assert_array_almost_equal(
            cosmology_params['catalog_weights_cdf'], catalog_weights_cdf
        )
        assert self.weighted_catalog.chunk_number == 2

        # Test that the weights cdf for the third chunk are correctly
        # calculated and saved
        catalog_weights_cdf = compute_catalog_weights_cdf(self.catalog_weights, 4, 2)
        cosmology_params = self.weighted_catalog.modify_cosmology_params(
            cosmology_params
        )
        np.testing.assert_array_almost_equal(
            cosmology_params['catalog_weights_cdf'], catalog_weights_cdf
        )
        # Since the third chunk is just one number, the cumsum should just be 1
        assert cosmology_params['catalog_weights_cdf'].size == 1
        assert cosmology_params['catalog_weights_cdf'][0] == 1

        # Test that when modify cosmology params is called a fourth time
        # it returns to the first chunk
        assert self.weighted_catalog.chunk_number == 0
        cosmology_params = self.weighted_catalog.modify_cosmology_params(
            cosmology_params
        )
        catalog_weights_cdf = compute_catalog_weights_cdf(self.catalog_weights, 0, 2)
        np.testing.assert_array_almost_equal(
            cosmology_params['catalog_weights_cdf'], catalog_weights_cdf
        )
        assert self.weighted_catalog.chunk_number == 1

    @chex.all_variants
    def test_convert_to_angular(self):
        # Test that we sample accoding to the weights
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_INIT, 1.0, 0.01
        )
        cosmology_params = self.weighted_catalog.modify_cosmology_params(
            cosmology_params
        )
        all_kwargs = {
            'galaxy_index': 0.87,
            'amp': 1.0,
            'z_source': cosmology_params['cosmos_redshifts'][0],
            'output_ab_zeropoint': 23.5,
            'catalog_ab_zeropoint': 23.5
        }
        convert_to_angular = self.variant(self.weighted_catalog.convert_to_angular)

        # Makes sure that the first image in the chunk is returned when the
        # galaxy index is <= 0.87 (in this example it is the second chunk)
        angular_kwargs = convert_to_angular(all_kwargs, cosmology_params)
        np.testing.assert_array_almost_equal(
            angular_kwargs['image'],
            (cosmology_params['cosmos_images'][0] /
                cosmology_params['cosmos_pixel_sizes'][0] ** 2),
            decimal=4)

        # Makes sure that the second image is returned when the galaxy index
        # is >= 0.88
        all_kwargs['galaxy_index'] = 0.88
        angular_kwargs = convert_to_angular(all_kwargs, cosmology_params)
        np.testing.assert_array_almost_equal(
            angular_kwargs['image'],
            (cosmology_params['cosmos_images'][1] /
                cosmology_params['cosmos_pixel_sizes'][1] ** 2),
            decimal=4)


if __name__ == '__main__':
    absltest.main()
