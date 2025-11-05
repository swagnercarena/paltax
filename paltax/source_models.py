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
"""Implementations of light profiles for lensing.

Implementation of light profiles for lensing closely following implementations
in lenstronomy: https://github.com/lenstronomy/lenstronomy.
"""

from typing import Dict, Optional, Tuple, Union

import atexit
import dm_pix
import h5py
import jax.numpy as jnp

from paltax import cosmology_utils
from paltax import utils

__all__ = [
    'Interpol', 'SersicElliptic', 'DoubleSersicElliptic', 'CosmosCatalog',
    'WeightedCatalog'
]


class _SourceModelBase():
    """Base source model.

    Provides identity implementation of convert_to_angular for all source
    models.
    """

    physical_parameters = ()
    parameters = ()

    def modify_cosmology_params(
            self,
            cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
        ) -> Dict[str, Union[float, int, jnp.ndarray]]:
        """Modify cosmology params to include information required by model.

        Args:
            cosmology_params: Cosmological parameters that define the universe's
                expansion. Must be mutable.

        Returns:
            Modified cosmology parameters.
        """
        return cosmology_params

    @staticmethod
    def convert_to_angular(
            all_kwargs:  Dict[str, jnp.ndarray],
            cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
        ) -> Dict[str, jnp.ndarray]:
        """Convert any parameters in physical units to angular units.

        Args:
            all_kwargs: All of the arguments, possibly including some in
                physical units.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Arguments with any physical units parameters converted to angular
                units.
        """
        # Don't get yelled at by the linter. This will not slow down evaluation
        # after jit compilation.
        _ = cosmology_params
        return all_kwargs

    @staticmethod
    def add_lookup_tables(
        lookup_tables: Dict[str, Union[float, jnp.ndarray]]
    ) ->  Dict[str, jnp.ndarray]:
        """Add lookup tables used for function calculations.

        Args:
            lookup_tables: Potentially empty dictionary of current lookup
                tables. Will be modified.

        Return:
            Modified lookup tables.
        """
        return lookup_tables


class Interpol(_SourceModelBase):
    """Interpolated light profile.

    Interpolated light profile functions, with calculation following those in
    Lenstronomy.
    """

    parameters = ('image', 'amp', 'center_x', 'center_y', 'angle', 'scale')

    @staticmethod
    def function(
        x: jnp.ndarray, y: jnp.ndarray, image: jnp.ndarray, amp: float,
        center_x: float, center_y: float, angle: float, scale: float,
        lookup_tables: Optional[Dict[str, Union[float, jnp.ndarray]]] = None
    ) -> jnp.ndarray:
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
            lookup_tables: Optional lookup tables initialized with class.

        Returns:
            Surface brightness at each coordinate.
        """
        x_image, y_image = Interpol._coord_to_image_pixels(
            x, y, center_x, center_y, angle, scale)
        return amp * Interpol._image_interpolation(x_image, y_image, image)

    @staticmethod
    def _image_interpolation(x_image: jnp.ndarray, y_image: jnp.ndarray,
                             image: jnp.ndarray) -> jnp.ndarray:
        """Map coordinates to interpolated image brightness.

        Args:
            x_image: X-coordinates in the image plane.
            y_image: Y-coordinates in the image plane.
            image: Source image as base for interpolation.

        Returns:
            Interpolated image brightness.
        """
        # Interpolation in dm-pix expects (0,0) to be the upper left-hand
        # corner, whereas lensing calculation treat the image center as (0,0).
        # Additionally, interpolation treats x as rows and y as columns, whereas
        # lensing does the opposite. Finally, interpolation considers going down
        # the rows as increasing in the x-coordinate, whereas that's decreasing
        # the y-coordinate in lensing. We account for this with the offset, by
        # switching x and y in the interpolation input, and by negating y.
        offset = jnp.array([image.shape[0] / 2 - 0.5, image.shape[1] / 2 - 0.5])
        coordinates = jnp.concatenate(
                [jnp.expand_dims(coord, axis=0) for coord in [-y_image, x_image]],
                axis=0)
        coordinates += offset[..., None]
        return dm_pix.flat_nd_linear_interpolate_constant(
                image, coordinates, cval=0.0)

    @staticmethod
    def _coord_to_image_pixels(
        x: jnp.ndarray, y: jnp.ndarray, center_x: float, center_y: float,
        angle: float, scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


class SersicElliptic(_SourceModelBase):
    """Sersic light profile.

    Sersic light profile functions, with implementation closely following the
    Sersic class in Lenstronomy.
    """

    parameters = (
        'amp', 'sersic_radius', 'n_sersic', 'axis_ratio', 'angle', 'center_x',
        'center_y'
    )

    @staticmethod
    def function(
        x: jnp.ndarray, y: jnp.ndarray, amp: float, sersic_radius: float,
        n_sersic: float, axis_ratio: float, angle: float, center_x: float,
        center_y: float,
        lookup_tables: Optional[Dict[str, Union[float, jnp.ndarray]]] = None
    ) -> jnp.ndarray:
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
            lookup_tables: Optional lookup tables initialized with class.

        Returns:
            Brightness from elliptical Sersic profile.
        """
        radius = SersicElliptic._get_distance_from_center(
            x, y, axis_ratio, angle,center_x, center_y
        )
        return amp * SersicElliptic._brightness(radius, sersic_radius, n_sersic)

    @staticmethod
    def _get_distance_from_center(
        x: jnp.ndarray, y: jnp.ndarray, axis_ratio: float, angle: float,
        center_x: float, center_y: float) -> jnp.ndarray:
        """Calculate the distance from the Sersic center.

        Calculate distance accounting for axis ratio.

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
    def _brightness(radius: jnp.ndarray, sersic_radius: float,
                    n_sersic: float) -> jnp.ndarray:
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
    def _b_n(n_sersic: float) -> float:
        """Return approximation for Sersic b(n_sersic).

        Args:
            n_sersic: Sersic index.

        Returns:
            Approximate b(n_sersic).
        """
        return 1.9992 * n_sersic - 0.3271


class DoubleSersicElliptic(_SourceModelBase):
    """Double Sersic light profile with coupled parameters.

    Double Sersic light profile with two Sersic components that have
    coupled ellipticities and centers controlled by an epsilon parameter.
    """

    parameters = (
        'amp_1', 'sersic_radius_1', 'n_sersic_1',
        'amp_2', 'sersic_radius_2', 'n_sersic_2',
        'axis_ratio', 'angle', 'epsilon_angle', 'epsilon_ratio', 'center_x',
        'center_y', 'epsilon_center_x', 'epsilon_center_y'
    )

    @staticmethod
    def function(
        x: jnp.ndarray, y: jnp.ndarray,
        amp_1: float, sersic_radius_1: float, n_sersic_1: float,
        amp_2: float, sersic_radius_2: float, n_sersic_2: float,
        axis_ratio: float, angle: float, epsilon_angle: float,
        epsilon_ratio: float, center_x: float, center_y: float,
        epsilon_center_x: float, epsilon_center_y: float,
        lookup_tables: Optional[Dict[str, Union[float, jnp.ndarray]]] = None
    ) -> jnp.ndarray:
        """Calculate the brightness for the double elliptical Sersic light profile.

        Args:
            x: X-coordinates at which to evaluate the brightness.
            y: Y-coordinates at which to evaluate the derivative.
            amp_1: Amplitude of first Sersic component.
            sersic_radius_1: Sersic radius of first component.
            n_sersic_1: Sersic index of first component.
            amp_2: Amplitude of second Sersic component.
            sersic_radius_2: Sersic radius of second component.
            n_sersic_2: Sersic index of second component.
            axis_ratio: Base axis ratio of the major and minor axis of
                ellipticity.
            angle: Base clockwise angle of orientation of major axis.
            epsilon_angle: Epsilon offset between the two sersic angles of the
                two components.
            epsilon_ratio: Epsilon offset between the axis ratios of the two
                components.
            center_x: X-coordinate center of the Sersic profile.
            center_y: Y-coordinate center of the Sersic profile.
            epsilon_center_x: Epsilon offset between the x-coordinate of the
                sersic centers.
            epsilon_center_y: Epsilon offset between the y-coordinate of the
                sersic centers.
            lookup_tables: Optional lookup tables initialized with class.

        Returns:
            Brightness from double elliptical Sersic profile.

        Notes:
            The epsilon parameters are used to slightly offset the parameter
            values of the two components from the base values. The first Sersic
            will be given a positive epsilon/2 offset and the second will be
            given a negative epsilon/2 offset. The axis ratios for both
            components are bounded to the range [0.02, 1.0] after applying the
             epsilon offsets, to ensure physical validity.
        """
        # Calculate brightness from first component

        # Get the offset centers, axis ratios, and angles.
        center_x_1 = center_x + epsilon_center_x / 2.0
        center_y_1 = center_y + epsilon_center_y / 2.0
        center_x_2 = center_x - epsilon_center_x / 2.0
        center_y_2 = center_y - epsilon_center_y / 2.0
        # Enforce axis ratio bounds (0 < axis_ratio <= 1.0) after epsilon
        # offsets
        axis_ratio_1 = jnp.clip(axis_ratio + epsilon_ratio / 2.0, 2e-2, 1.0)
        axis_ratio_2 = jnp.clip(axis_ratio - epsilon_ratio / 2.0, 2e-2, 1.0)
        angle_1 = angle + epsilon_angle / 2.0
        angle_2 = angle - epsilon_angle / 2.0

        # Brightness from the first component.
        radius_1 = SersicElliptic._get_distance_from_center(
            x, y, axis_ratio_1, angle_1, center_x_1, center_y_1
        )
        brightness_1 = amp_1 * SersicElliptic._brightness(
            radius_1, sersic_radius_1, n_sersic_1
        )

        # Brightness from the second component.
        radius_2 = SersicElliptic._get_distance_from_center(
            x, y, axis_ratio_2, angle_2, center_x_2, center_y_2
        )
        brightness_2 = amp_2 * SersicElliptic._brightness(
            radius_2, sersic_radius_2, n_sersic_2
        )

        return brightness_1 + brightness_2


class CosmosCatalog(Interpol):
    """Light profiles of real galaxies from COSMOS.
    """

    physical_parameters = (
        'galaxy_index', 'z_source', 'output_ab_zeropoint',
        'catalog_ab_zeropoint'
    )
    parameters = ('image', 'amp', 'center_x', 'center_y', 'angle', 'scale')

    def __init__(
        self, cosmos_path: str, images_per_chunk: Optional[int] = 2048,
        default_redshift: Optional[float] = None,
        default_pixel_scale: Optional[float] = None,
        total_num_galaxies: Optional[int] = None
    ):
        """Initialize the path to the COSMOS galaxies.

        Args:
            cosmos_path: Path to the npz file containing the cosmos images,
                redshift array, and pixel sizes.
            images_per_chunk: Number of images to load into memory at a time.
            default_redshift: Default redshift to use for all galaxies.
            default_pixel_scale: Default pixel scale to use for all galaxies.
            total_num_galaxies: Total number of galaxies to consider in the
                catalog. If None, will use all the galaxies in the catalog.
        """

        # Opens the hdf5 file and extract the total number of images
        self.hdf5_file = h5py.File(cosmos_path, 'r')
        self.total_num_galaxies = (
            total_num_galaxies or len(self.hdf5_file['images'])
        )

        # Only load one chunk of the hdf5 file into memory at a time
        self.images_per_chunk = images_per_chunk
        self.chunk_number = 0

        # Set the default redshift and pixel scale if provided.
        self.default_redshift = default_redshift
        self.default_pixel_scale = default_pixel_scale

        # Close the hdf5 file whenever all code is finished executing
        atexit.register(self.cleanup)


    def modify_cosmology_params(
        self,
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> Dict[str, Union[float, int, jnp.ndarray]]:
        """Modify cosmology params to include information required by model.

        Args:
            cosmology_params: Cosmological parameters that define the universe's
                expansion. Must be mutable.

        Returns:
            Modified cosmology parameters.
        """

        # Sets up the chunk of images to be loaded
        start_val = self.chunk_number * self.images_per_chunk
        end_val = min(
            start_val + self.images_per_chunk, self.total_num_galaxies
        )

        # Load the chunk of images.
        images = self.hdf5_file['images'][start_val:end_val]
        if self.default_redshift is not None:
            redshifts = self.default_redshift * jnp.ones(len(images))
        else:
            redshifts = self.hdf5_file['redshifts'][start_val:end_val]
        if self.default_pixel_scale is not None:
            pixel_sizes = self.default_pixel_scale * jnp.ones(len(images))
        else:
            pixel_sizes = self.hdf5_file['pixel_sizes'][start_val:end_val]
        cosmology_params['cosmos_n_images'] = len(images)


        # Convert attributes we need later to jax arrays
        cosmology_params['cosmos_pixel_sizes'] = jnp.array(pixel_sizes)
        cosmology_params['cosmos_redshifts'] = jnp.array(redshifts)
        cosmology_params['cosmos_images'] = jnp.array(images)

        # Increment the chunk counter; reset to 0 if we've already loaded all
        # chunks.
        self.chunk_number = int(
            jnp.where(
                end_val >= self.total_num_galaxies, 0, self.chunk_number + 1
            )
        )

        return cosmology_params

    @staticmethod
    def convert_to_angular(
        all_kwargs: Dict[str, jnp.ndarray],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> Dict[str, jnp.ndarray]:
        """Convert any parameters in physical units to angular units.

        Args:
            all_kwargs: All of the arguments, possibly including some in
                physical units.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Arguments with any physical units parameters converted to angular
                units.
        """
        # Select the galaxy incdex from the uniform distribution.
        galaxy_index = jnp.floor(
            all_kwargs['galaxy_index'] * cosmology_params['cosmos_n_images']
        ).astype(int)

        return CosmosCatalog._convert_to_angular(
            all_kwargs, cosmology_params, galaxy_index
        )

    @staticmethod
    def _convert_to_angular(
        all_kwargs: Dict[str, jnp.ndarray],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]],
        galaxy_index: int
    ) -> Dict[str, jnp.ndarray]:
        """Convert any parameters in physical units to angular units.

        Args:
            all_kwargs: All of the arguments, possibly including some in
                physical units.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Arguments with any physical units parameters converted to angular
                units.

        Notes:
            Galaxy index must have already been converted to an index.
        """
        # Read the catalog values directly from the stored arrays.
        z_catalog = cosmology_params['cosmos_redshifts'][galaxy_index]
        pixel_scale_catalog = (
            cosmology_params['cosmos_pixel_sizes'][galaxy_index]
        )
        image = (
            cosmology_params['cosmos_images'][galaxy_index]
            / pixel_scale_catalog ** 2
        )

        # Take into account the difference in the magnitude zeropoints
        # of the input survey and the output survey. Note this doesn't
        # take into account the color of the object.
        amp = all_kwargs['amp']
        amp *= 10 ** (
            (
                all_kwargs['output_ab_zeropoint'] -
                all_kwargs['catalog_ab_zeropoint']
            ) / 2.5
        )

        # Calculate the new pixel scale and amplitude at the given redshift.
        pixel_scale = pixel_scale_catalog * CosmosCatalog.z_scale_factor(
            z_catalog, all_kwargs['z_source'], cosmology_params
        )
        amp *= CosmosCatalog.k_correct_image(z_catalog, all_kwargs['z_source'])

        # Add the new keywords to the original dictionary.
        all_kwargs['image'] = image
        all_kwargs['amp'] = amp
        all_kwargs['scale'] = pixel_scale

        return all_kwargs

    @staticmethod
    def z_scale_factor(
        z_old: float, z_new: float,
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> float:
        """Return scaling of pixel size from moving to a new redshift.

        Args:
            z_old: Original redshift of the object.
            z_new: Redshift the object will be placed at.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Multiplicative factor for pixel size.
        """
        # Pixel length ~ angular diameter distance
        return (
            cosmology_utils.angular_diameter_distance(cosmology_params, z_old)
            / cosmology_utils.angular_diameter_distance(cosmology_params, z_new)
        )

    @staticmethod
    def k_correct_image(z_old: float, z_new: float) -> float:
        """Return the amplitude rescaling from k-correction of the source.

        Args:
            z_old: Original redshift of the object.
            z_new: Redshift the object will be placed at.

        Returns:
            Amplitude of the k-correction.
        """
        mag_k_correction = utils.get_k_correction(z_new)
        mag_k_correction -= utils.get_k_correction(z_old)
        return 10 ** (-mag_k_correction / 2.5)

    def cleanup(self) -> None:
        """Closes the hdf5 file"""
        self.hdf5_file.close()


class WeightedCatalog(CosmosCatalog):
    """Light profiles from catalog with custom weights
    """

    def __init__(
        self, cosmos_path: str, parameter: str,
        images_per_chunk: Optional[int] = 2048,
        total_num_galaxies: Optional[int] = None
    ):
        """Initialize the path to the catalog galaxies and catalog weights.

        Args:
            cosmos_path: Path to the npz file containing the cosmos images,
                redshift array, and pixel sizes.
            parameter: Name of the parameter to use for the weights.
            images_per_chunk: Number of images to load into memory at a time.
            total_num_galaxies: Total number of galaxies to consider in the
                catalog. If None, will use all the galaxies in the catalog.
        """

        # Saves the cosmos path, chunk size, and opens the hdf5 file
        super().__init__(
            cosmos_path=cosmos_path,
            images_per_chunk=images_per_chunk,
            total_num_galaxies=total_num_galaxies
        )
        self.parameter = parameter

    def modify_cosmology_params(
        self,
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> Dict[str, Union[float, int, jnp.ndarray]]:
        """Modify cosmology params to include information required by model.

        Args:
            cosmology_params: Cosmological parameters that define the universe's
                expansion. Must be mutable.

        Returns:
            Modified cosmology parameters.
        """

        # Sets up the chunk of weights to be loaded
        start_val = self.chunk_number * self.images_per_chunk
        end_val = min(
            start_val + self.images_per_chunk, self.total_num_galaxies
        )

        catalog_weights = (
            self.hdf5_file[self.parameter + '_weights'][start_val:end_val]
        )
        catalog_weights_cdf = (
            jnp.cumsum(catalog_weights) / jnp.sum(catalog_weights)
        )

        cosmology_params['catalog_weights_cdf'] = catalog_weights_cdf

        cosmology_params = super().modify_cosmology_params(
            cosmology_params=cosmology_params
        )

        return cosmology_params

    @staticmethod
    def convert_to_angular(
        all_kwargs: Dict[str, jnp.ndarray],
        cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
    ) -> Dict[str, jnp.ndarray]:
        """Convert any parameters in physical units to angular units.

        Args:
            all_kwargs: All of the arguments, possibly including some in
                physical units.
            cosmology_params: Cosmological parameters that define the universe's
                expansion.

        Returns:
            Arguments with any physical units parameters converted to angular
                units.
        """
        # Select the galaxy index using the weighted distribution
        galaxy_index = jnp.searchsorted(
            cosmology_params['catalog_weights_cdf'], all_kwargs['galaxy_index']
        )

        return CosmosCatalog._convert_to_angular(
            all_kwargs, cosmology_params, galaxy_index
        )
