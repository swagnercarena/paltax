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
"""Implementations of psf models for lensing.

Implementation differs significantly from lenstronomy, but nomenclature is kept
identical: https://github.com/lenstronomy/lenstronomy.
"""

from typing import Dict, Mapping, Union

import dm_pix
import jax
import jax.numpy as jnp
import numpy as np

from paltax import utils

__all__ = ['Gaussian', 'PixelCatalog']


class _PSFModelBase():
    """Base source model.

    Provides identity implementation of convert_to_angular for all source
    models.
    """

    parameters = ()

    def modify_cosmology_params(
        self,
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> Dict[str, Union[float, int, jnp.ndarray]]:
        """Modify cosmology params to include information required by model.

        Args:
            cosmology_params: Cosmological parameters that define the
                universe's expansion.

        Returns:
            Modified cosmology parameters.
        """
        return cosmology_params

    @staticmethod
    def add_lookup_tables(
        lookup_tables: Dict[str, Union[float, jnp.ndarray]]
    ) ->  Dict[str, jnp.ndarray]:
        """Add lookup tables used for psf calculations.

        Args:
            lookup_tables: Potentially empty dictionary of current lookup
                tables. Will be modified.

        Return:
            Modified lookup tables.
        """
        return lookup_tables


class Gaussian(_PSFModelBase):
    """Implementation of Gaussian point spread function."""

    parameters = ('fwhm', 'pixel_width')

    @staticmethod
    def convolve(
        image: jnp.ndarray,
        kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> jnp.ndarray:
        """Convolve an image with the Gaussian point spread function.

        Args:
            image: Image to convolve
            kwargs_psf: Keyword arguments defining the point spread function.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Convolved image.
        """
        # Cosmology parameters aren't used, but add this call to avoid upsetting
        # the linter.
        _ = cosmology_params

        sigma_angular = kwargs_psf['fwhm'] / (2 * jnp.sqrt(2 * jnp.log(2)))
        sigma_pixel = sigma_angular / kwargs_psf['pixel_width']

        # dm_pix expects channel dimensions.
        return dm_pix.gaussian_blur(
                jnp.expand_dims(image, axis=-1), sigma_pixel, kernel_size=30
            )[:, :, 0]


class _Pixel(_PSFModelBase):
    """Implementation of pixel point spread function."""

    parameters = ('kernel_point_source',)

    @staticmethod
    def convolve(
        image: jnp.ndarray,
        kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> jnp.ndarray:
        """Convolve an image with the Gaussian point spread function.

        Args:
            image: Image to convolve
            kwargs_psf: Keyword arguments defining the point spread function.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Convolved image.
        """
        # Cosmology parameters aren't used, but add this call to avoid upsetting
        # the linter.
        _ = cosmology_params

        # Always normalize kernel to 1 to avoid user error.
        kernel = (
                kwargs_psf['kernel_point_source'] /
                jnp.sum(kwargs_psf['kernel_point_source']))
        return jax.scipy.signal.convolve(
                image, kernel, mode='same')


class PixelCatalog(_PSFModelBase):
    """Extension of _Pixel psf to allow for a catalog of PSFs."""

    parameters = ('kernel_index',)

    def __init__(self, kernel_path: str):
        """Initialize the path to the kernels.

        Args:
            kernel_path: Path to the npz file containing the kernel images
        """
        # Save the kernel image path.
        self.kernel_path = kernel_path

    def modify_cosmology_params(
        self,
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> Dict[str, Union[float, int, jnp.ndarray]]:
        """Modify cosmology params to include information required by model.

        Args:
            cosmology_params: Cosmological parameters that define the
                universe's expansion.

        Returns:
            Modified cosmology parameters.
        """
        # Load the kernel images from disk.
        kernel_images = np.load(self.kernel_path)
        cosmology_params['kernels_n_images'] = len(kernel_images)

        # Convert attributes we need later to jax arrays.
        cosmology_params['kernel_images'] = jnp.asarray(kernel_images)

        return cosmology_params

    @staticmethod
    def convolve(
        image: jnp.ndarray,
        kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> jnp.ndarray:
        """Convolve an image with the Gaussian point spread function.

        Args:
            image: Image to convolve
            kwargs_psf: Keyword arguments defining the point spread function.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Convolved image.
        """
        # Extract the kernel and pass it to the Pixel class.
        kernel_index = jnp.floor(
            kwargs_psf['kernel_index'] * cosmology_params['kernels_n_images']
        ).astype(int)
        kernel_point_source = cosmology_params['kernel_images'][kernel_index]

        return _Pixel.convolve(
            image, {'kernel_point_source': kernel_point_source},
            cosmology_params
        )
