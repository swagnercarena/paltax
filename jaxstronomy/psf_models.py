# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementations of psf models for lensing.

Implementation differs significantly from lenstronomy, but nomenclature is kept
identical: https://github.com/lenstronomy/lenstronomy.
"""

from typing import Any, Mapping, Union

import dm_pix
import jax
import jax.numpy as jnp

__all__ = ['Gaussian', 'Pixel']


class _PSFModelBase():
    """Base source model.

    Provides identity implementation of convert_to_angular for all source
    models.
    """

    parameters = ()

    def modify_cosmology_params(
            self: Any,
            cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]]
        ) -> Mapping[str, Union[float, int, jnp.ndarray]]:
        """Modify cosmology params to include information required by model.

        Args:
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Modified cosmology parameters.
        """
        return cosmology_params


class Gaussian(_PSFModelBase):
  """Implementation of Gaussian point spread function."""

  parameters = ('fwhm', 'pixel_width')

  @staticmethod
  def convolve(
      image,
      kwargs_psf):
    """Convolve an image with the Gaussian point spread function.

    Args:
      image: Image to convolve
      kwargs_psf: Keyword arguments defining the point spread function.

    Returns:
      Convolved image.
    """
    sigma_angular = kwargs_psf['fwhm'] / (2 * jnp.sqrt(2 * jnp.log(2)))
    sigma_pixel = sigma_angular / kwargs_psf['pixel_width']

    # dm_pix expects channel dimensions.
    return dm_pix.gaussian_blur(
        jnp.expand_dims(image, axis=-1), sigma_pixel, kernel_size=30)[:, :, 0]


class Pixel(_PSFModelBase):
  """Implementation of Pixel point spread function."""

  parameters = ('kernel_point_source',)

  @staticmethod
  def convolve(
      image,
      kwargs_psf):
    """Convolve an image with the pixel point spread function.

    Args:
      image: Image to convolve
      kwargs_psf: Keyword arguments defining the point spread function.

    Returns:
      Convolved image.
    """
    # Always normalize kernel to 1 to avoid user error.
    kernel = (
        kwargs_psf['kernel_point_source'] /
        jnp.sum(kwargs_psf['kernel_point_source']))
    return jax.scipy.signal.convolve(
        image, kernel, mode='same')
