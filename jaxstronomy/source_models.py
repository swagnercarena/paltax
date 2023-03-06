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
"""Implementations of light profiles for lensing.

Implementation of light profiles for lensing closely following implementations
in lenstronomy: https://github.com/lenstronomy/lenstronomy.
"""

import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
from skimage.transform import downscale_local_mean
from tqdm import tqdm

from jaxstronomy import cosmology_utils

__all__ = ['Interpol', 'SersicElliptic', 'PaltasGalaxyCatalog']


class Interpol():
  """Interpolated light profile.

  Interpolated light profile functions, with calculation following those in
  Lenstronomy.
  """

  parameters = ('image', 'amp', 'center_x', 'center_y', 'angle', 'scale')

  @staticmethod
  def function(x, y, image, amp,
               center_x, center_y, angle,
               scale):
    """Calculate the brightness for the interpolated light profile.

    Args:
      x: X-coordinates at which to evaluate the profile.
      y: Y-coordinates at which to evaluate the profile.
      image: Source image as base for interpolation.
      amp: Normalization to source image.
      center_x: X-coordinate center of the light profile.
      center_y: Y-coordinate cetner of the light profile.
      angle: Clockwise rotation angle of simulated image with respect to
        simulation grid.
      scale: Pixel scale of the simulated image.

    Returns:
      Surface brightness at each coordinate.
    """
    x_image, y_image = Interpol._coord_to_image_pixels(x, y, center_x, center_y,
                                                       angle, scale)
    return amp * Interpol._image_interpolation(x_image, y_image, image)

  @staticmethod
  def _image_interpolation(x_image, y_image,
                           image):
    """Map coordinates to interpolated image brightness.

    Args:
      x_image: X-coordinates in the image plane.
      y_image: Y-coordinates in the image plane.
      image: Source image as base for interpolation.

    Returns:
      Interpolated image brightness.
    """
    # Interpolation in dm-pix expects (0,0) to be the upper left-hand corner,
    # whereas lensing calculation treat the image center as (0,0). Additionally,
    # interpolation treats x as rows and y as columns, whereas lensing does the
    # opposite. Finally, interpolation considers going down the rows as
    # increasing in the x-coordinate, whereas that's decreasing the
    # y-coordinate in lensing. We account for this with the offset, by
    # switching x and y in the interpolation input, and by negating y.
    offset = jnp.array([image.shape[0] / 2 - 0.5, image.shape[1] / 2 - 0.5])
    coordinates = jnp.concatenate(
        [jnp.expand_dims(coord, axis=0) for coord in [-y_image, x_image]],
        axis=0)
    coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1))
    return dm_pix.flat_nd_linear_interpolate_constant(
        image, coordinates, cval=0.0)

  @staticmethod
  def _coord_to_image_pixels(x, y, center_x, center_y, angle, scale):
    """Map from simulation coordinates to image coordinates.

    Args:
      x: X-coordinates at which to evaluate the profile.
      y: Y-coordinates at which to evaluate the profile.
      center_x: X-coordinate center of the light profile.
      center_y: Y-coordinate cetner of the light profile.
      angle: Clockwise rotation angle of simulated image with respect to
        simulation grid.
      scale: Pixel scale (in angular units) of the simulated image.

    Returns:
      X- and y-coordinates in the image plane.
    """
    x_image = (x - center_x) / scale
    y_image = (y - center_y) / scale
    # Lenstronomy uses clockwise rotation so we will stay consistent.
    complex_coords = jnp.exp(-1j * angle) * (x_image + 1j * y_image)

    return complex_coords.real, complex_coords.imag


class SersicElliptic():
  """Sersic light profile.

  Sersic light profile functions, with implementation closely following the
  Sersic class in Lenstronomy.
  """

  parameters = (
      'amp', 'sersic_radius', 'n_sersic', 'axis_ratio', 'angle', 'center_x',
      'center_y'
  )

  @staticmethod
  def function(x, y, amp, sersic_radius,
               n_sersic, axis_ratio, angle,
               center_x, center_y):
    """"Calculate the brightness for the elliptical Sersic light profile.

    Args:
      x: X-coordinates at which to evaluate the brightness.
      y: Y-coordinates at which to evaluate the derivative.
      amp: Amplitude of Sersic light profile.
      sersic_radius: Sersic radius.
      n_sersic: Sersic index.
      axis_ratio: Axis ratio of the major and minor axis of ellipticity.
      angle: Clockwise angle of orientation of major axis.
      center_x: X-coordinate center of the Sersic profile.
      center_y: Y-coordinate center of the Sersic profile.

    Returns:
      Brightness from elliptical Sersic profile.
    """
    radius = SersicElliptic._get_distance_from_center(x, y, axis_ratio, angle,
                                                      center_x, center_y)
    return amp * SersicElliptic._brightness(radius, sersic_radius, n_sersic)

  @staticmethod
  def _get_distance_from_center(x, y,
                                axis_ratio, angle,
                                center_x,
                                center_y):
    """Calculate the distance from the Sersic center, accounting for axis ratio.

    Args:
      x: X-coordinates at which to evaluate the brightness.
      y: Y-coordinates at which to evaluate the derivative
      axis_ratio: Axis ratio of the major and minor axis of ellipticity.
      angle: Clockwise angle of orientation of major axis.
      center_x: X-coordinate center of the Sersic profile.
      center_y: Y-coordinate center of the Sersic profile.

    Returns:
      Distance from Sersic center.
    """
    x_centered = x - center_x
    y_centered = y - center_y
    complex_coords = jnp.exp(-1j * angle) * (x_centered + 1j * y_centered)
    return jnp.sqrt((complex_coords.real * jnp.sqrt(axis_ratio))**2 +
                    (complex_coords.imag / jnp.sqrt(axis_ratio))**2)

  @staticmethod
  def _brightness(radius, sersic_radius,
                  n_sersic):
    """Return the sersic brightness.

    Args:
      radius: Radii at which to evaluate the brightness.
      sersic_radius: Sersic radius.
      n_sersic: Sersic index.

    Returns:
      Brightness values.
    """
    b_n = SersicElliptic._b_n(n_sersic)
    reduced_radius = radius / sersic_radius
    return jnp.nan_to_num(jnp.exp(-b_n * (reduced_radius**(1 / n_sersic) - 1.)))

  @staticmethod
  def _b_n(n_sersic):
    """Return approximation for Sersic b(n_sersic).

    Args:
      n_sersic: Sersic index.

    Returns:
      Approximate b(n_sersic).
    """
    return 1.9992 * n_sersic - 0.3271


class PaltasGalaxyGatalog:
  """Light profiles of real galaxies, using paltas to access a catalog
  """

  parameters = (
    'galaxy_index', 'z_source',
    'amp', 'center_x', 'center_y', 'angle',
  )

  def __init__(self, 
               cosmology_params=None,
               paltas_class=None, 
               maximum_size_in_pixels=256, 
               **source_parameters):
    if paltas_class is None:
      # No paltas class specified -- this source model shouldn't be used
      # (and will crash if you try it anyway)
      return
    assert cosmology_params is not None
    self.cosmology_parameters = cosmology_params
    self.paltas_catalog = paltas_class(
      # We only use the paltas class for extracting raw images, and do
      # redshift and magnitude scaling ourselves in jax.
      # Thus we can load an arbitrary cosmology here.
      cosmology_parameters='planck18',
      # We don't need all parameters, but paltas won't let us initialize
      # without a complete set
      source_parameters=(
        source_parameters
        | dict(center_x=np.nan, center_y=np.nan, 
               z_source=np.nan, 
               # rotations are done by jaxstronomy, not paltas
               random_rotation=False)))

    passes_cuts = self.paltas_catalog._passes_cuts()
    catalog_indices = np.where(passes_cuts)[0]
    self.n_images = passes_cuts.sum()

    # Allocate memory (in main RAM, not on the GPU) 
    # for all the galaxy images that pass the cuts
    size = maximum_size_in_pixels
    images = np.zeros((self.n_images, size, size))
    pixel_sizes = np.zeros(self.n_images)
    redshifts = np.zeros(self.n_images)

    for galaxy_i, catalog_i in tqdm(
        zip(np.arange(self.n_images), catalog_indices),
        desc='Slurping galaxies into RAM...'):
      img, meta = self.paltas_catalog.image_and_metadata(catalog_i)
      pixel_sizes[galaxy_i] = meta['pixel_width']
      redshifts[galaxy_i] = meta['z']
      
      # Check if the image is too large, if so, downsample
      img_size = max(img.shape[0], img.shape[1])
      if img_size > size:
        # Image is too large: downsample it and adjust the pixel size.
        # Since images are in electrons/sec/pixel, we have to use sum,
        # not mean, to downsample.
        downsample_factor =int(np.ceil(img_size / size))
        assert downsample_factor > 1
        img = (
          downscale_local_mean(img, (downsample_factor, downsample_factor))
          * downsample_factor**2)
        pixel_sizes[galaxy_i] *= downsample_factor
        # Recompute image size
        img_size = max(img.shape[0], img.shape[1])

      # Check if the image is too small, if so, pad with zeros
      if img_size < size:
        images[galaxy_i] = pad_image(img, size, size)

    # Convert attributes we need later to jax arrays
    self.pixel_sizes = jnp.asarray(pixel_sizes)
    self.redshifts = jnp.asarray(redshifts)
    # Place the giant image array in main RAM, not GPU memory
    with jax.default_device(jax.devices("cpu")[0]):
      self.images = jnp.asarray(images)

  def function(self, x, y, galaxy_index, z_source, amp, center_x, center_y, angle):
    # Conver from uniform[0,1] into a discrete index
    galaxy_index = jnp.floor(galaxy_index * self.n_images).astype(int)
    img = self.images[galaxy_index]

    # Convert to from electrons/sec/pixel to electrons/sec/arcsec
    pixel_size = self.pixel_sizes[galaxy_index]
    img = img / pixel_size**2

    # Take into account the difference in the magnitude zeropoints
    # of the input survey and the output survey. Note this doesn't
    # take into account the color of the object!
    img *= 10**((
        self.paltas_catalog.source_parameters['output_ab_zeropoint']
        - self.paltas_catalog.ab_zeropoint
      ) / 2.5)

    pixel_size *= self.z_scale_factor(self.redshifts[galaxy_index], z_source)

    # TODO: don't be lazy, implement k-corrections
    # Apply the k correction to the image from the redshifting
    # self.k_correct_image(img,metadata['z'],z_new)

    return Interpol.function(x, y, img, amp, center_x, center_y, angle, pixel_size)

  def z_scale_factor(self, z_old, z_new):
    """Return multiplication factor for object/pixel size for moving its
    redshift from z_old to z_new.

    Args:
      z_old (float): The original redshift of the object.
      z_new (float): The redshift the object will be placed at.

    Returns:
      (float): The multiplicative pixel size.
    """
    # Pixel length ~ angular diameter distance
    # (colossus uses funny /h units, but for ratios it
    #  fortunately doesn't matter)
    return (
      cosmology_utils.angular_diameter_distance(self.cosmology_parameters, z_old)
      / cosmology_utils.angular_diameter_distance(self.cosmology_parameters, z_new))


def pad_image(img, nx, ny):
  """Returns img with zeros padded on both sides so shape is (nx, ny)"""
  old_nx, old_ny = img.shape
  result = np.zeros((nx, ny), dtype=img.dtype)
  x_center = (nx - old_nx) // 2
  y_center = (ny - old_ny) // 2
  result[
    x_center:x_center + old_nx, 
    y_center:y_center + old_ny] = img
  return result
