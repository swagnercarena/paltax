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
"""Tests for psf_models.py."""

import inspect
import pathlib

from absl.testing import absltest
import chex
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
from scipy import signal
from paltax import psf_models


KERNEL_PATH = (
    str(pathlib.Path(__file__).parent.parent) +
    '/datasets/hst_psf/emp_psf_f814w_2x.npy'
)


def _prepare_image():
    return jnp.load('test_files/psf_models_image_test.npy')


def _prepare_gaussian_parameters():
    return {'fwhm': 0.03, 'pixel_width': 0.04}


def _prepare_pixel_parameters():
    x = jnp.arange(-2, 2, 0.05)
    kernel = jnp.outer(jnp.exp(-x**2), jnp.exp(-x**2))
    return {'kernel_point_source': kernel}


def _prepare_pixel_catalog_parameters():
    return {'kernel_index': 0.31}


class AllTest(absltest.TestCase):
    """Runs tests of __all__ property of psf_models module."""

    def test_all(self):
        all_present = sorted(psf_models.__all__)
        all_required = []
        ignore_list = ['_PSFModelBase', '_Pixel', 'Any']
        for name, value in inspect.getmembers(psf_models):
            if inspect.isclass(value) and name not in ignore_list:
                all_required.append(name)

        self.assertListEqual(all_present, sorted(all_required))


class PSFModelBaseTest(chex.TestCase):
    """Runs tests of _PSFModelBase functions."""

    def test_modify_cosmology_params(self):
        # Make sure the dict is modified by default.
        input_dict = {'a': 20}
        new_dict = psf_models._PSFModelBase().modify_cosmology_params(
                input_dict)
        self.assertDictEqual(input_dict, new_dict)

    def test_add_lookup_tables(self):
        # Test that the dictionary isn't modified.
        lookup_tables = {}
        lookup_tables = psf_models._PSFModelBase.add_lookup_tables(
            lookup_tables
        )
        self.assertEmpty(lookup_tables)


class GaussianTest(chex.TestCase):
    """Runs tests of Gaussian derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(psf_models.Gaussian.parameters)
        correct_parameters = sorted(_prepare_gaussian_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_convolve(self):
        image = _prepare_image()
        parameters = _prepare_gaussian_parameters()
        cosmology_params = {}

        convolve = self.variant(psf_models.Gaussian.convolve)

        # Pulled from lenstronomy
        sigma_expected = 0.3184956751

        # As an additional consistency check, use a different channel axis for test
        # call to dm_pix.gaussian_blur.
        np.testing.assert_allclose(
                convolve(image, parameters, cosmology_params),
                dm_pix.gaussian_blur(
                        jnp.expand_dims(image, axis=0),
                        sigma_expected,
                        kernel_size=30,
                        channel_axis=0)[0],
                rtol=1e-5)


class PixelTest(chex.TestCase):
    """Runs tests of Pixel derivative functions."""

    def test_parameters(self):
        annotated_parameters = sorted(psf_models._Pixel.parameters)
        correct_parameters = sorted(_prepare_pixel_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_convolve(self):
        image = _prepare_image()
        parameters = _prepare_pixel_parameters()
        cosmology_params = {}

        convolve = self.variant(psf_models._Pixel.convolve)

        # Not much to test here other than parameters being passed through
        # correctly / matching non-jax scipy.
        np.testing.assert_allclose(
            convolve(image, parameters, cosmology_params),
            signal.convolve(
                image,
                parameters['kernel_point_source'] /
                    jnp.sum(parameters['kernel_point_source']),
                mode='same'),
            rtol=1e-5)


class PixelCatalogTest(chex.TestCase):
    """Runs tests of PixelCatalog derivative functions."""

    def test__init__(self):
        # Test that the path is saved
        pixel_catalog = psf_models.PixelCatalog(KERNEL_PATH)
        self.assertEqual(pixel_catalog.kernel_path, KERNEL_PATH)

    def test_parameters(self):
        annotated_parameters = sorted(psf_models.PixelCatalog.parameters)
        correct_parameters = sorted(_prepare_pixel_catalog_parameters().keys())
        self.assertListEqual(annotated_parameters, correct_parameters)

    def test_modify_cosmology_params(self):
        pixel_catalog = psf_models.PixelCatalog(KERNEL_PATH)
        cosmology_params = {}
        cosmology_params = pixel_catalog.modify_cosmology_params(
            cosmology_params
        )
        self.assertEqual(cosmology_params['kernel_images'].shape,
                         (56, 51, 51))
        self.assertEqual(cosmology_params['kernels_n_images'], 56)

    @chex.all_variants
    def test_convolve(self):
        # Test the convolution catalog works as expected.
        image = _prepare_image()
        parameters = _prepare_pixel_catalog_parameters()
        cosmology_params = {}
        pixel_catalog = psf_models.PixelCatalog(KERNEL_PATH)
        cosmology_params = pixel_catalog.modify_cosmology_params(
            cosmology_params
        )

        convolve = self.variant(psf_models.PixelCatalog.convolve)

        # Not much to test here other than parameters being passed through
        # correctly / matching non-jax scipy.
        np.testing.assert_allclose(
            convolve(image, parameters, cosmology_params),
            signal.convolve(
                image,
                cosmology_params['kernel_images'][17] /
                    jnp.sum(cosmology_params['kernel_images'][17]),
                mode='same'),
            atol=1e-5)


if __name__ == '__main__':
    absltest.main()
