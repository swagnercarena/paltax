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
"""Strong lensing utility functions."""

import functools
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp


def coordinates_evaluate(
        n_x: int, n_y: int, pixel_width: float, supersampling_factor: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the coordinate grid in the observer plane in angular units.

    Args:
        n_x: Number of pixels in x direction
        n_y: Number of pixels in y direction
        pixel_width: Size of a pixel in angular units. Pixels are assumed to be
            square.
        supersampling_factor: Factor by which to supersample light rays. Number
            of light rays will scale like supersampling_factor**2.

    Returns:
        X- and y-coordinates at which to raytrace.
    """
    # Center the x- and y-coordinates at 0.
    x_span = (
            jnp.arange(n_x * supersampling_factor) / supersampling_factor -
            (n_x - 1 / supersampling_factor) / 2.)
    y_span = (
            jnp.arange(n_y * supersampling_factor) / supersampling_factor -
            (n_y - 1 / supersampling_factor) / 2.)
    pix_xy = jnp.stack(
            [coord.flatten() for coord in jnp.meshgrid(x_span, y_span)], axis=0)
    radec_xy = pix_xy * pixel_width
    return radec_xy[0], radec_xy[1]


def unpack_parameters_xy(
        func: Callable[[Any], Any], parameters:Sequence[str],
) -> Callable[[Any], Any]:
    """Returns function wrapper that unpacks parameters for grid functions.

    Returns function wrapper that unpacks required parameters for functions whose
    first two parameters are the x- and y-coordinates at which they should be
    evaluated.

    Args:
        func: Function that takes x- and y-coordinates as well as additional args.
        parameters: Parameters to unpack.

    Returns:
        Wrapper for func that unpacks keyword parameters, passes them to func, and
        returns the output.
    """

    def derivative_wrapper(x, y, kwargs, parameters):
        return func(x, y, *[kwargs[param] for param in parameters])

    return functools.partial(derivative_wrapper, parameters=parameters)


def downsample(image: jnp.ndarray, supersampling_factor: int) -> jnp.ndarray:
    """Downsamples image to correct for supersampling factor.

    Args:
        image: Image to downsample.
        supersampling_factor: Factor by which light rays were supersampled.

    Returns:
        Downsampled image.
    """
    n_x, n_y = image.shape
    image = jnp.reshape(image,
                        (n_x // supersampling_factor, supersampling_factor,
                         n_y // supersampling_factor, supersampling_factor))
    return jnp.sum(jnp.sum(image, axis=3), axis=1)


def magnitude_to_cps(magnitude: float, magnitude_zero_point: float) -> float:
    """Converts magnitude to counts per second.

    Args:
        magnitude: Input magnitude
        magnitude_zero_point: Zero point magnitude of the detector.

    Returns:
        Counts per second corresponding to input magnitude.
    """
    return 10**(-(magnitude - magnitude_zero_point) / 2.5)


def get_k_correction(z_light: float) -> float:
    """Return k-correction for a galaxy source at given redshift.

    Args:
        z_light: Redshift of galaxy light source

    Returns:
        K-correction factor.

    Notes:
		This code assumes the galaxy has a flat spectral wavelength density (and
		therefore 1/nu^2 spectral frequency density) and that the bandpass used
		for the absolute and apparent magntidue is the same
    """
    return 2.5 * jnp.log(1 + z_light)


def rotate_coordinates(
        grid_x: jnp.ndarray, grid_y: jnp.ndarray, angle: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate grid by rotation angle.

    Args:
        grid_x: X-coordinates to rotate.
        grid_y: Y-coordinates to rotate.
        angle: Angle of counterclockwise rotation.

    Returns:
        Rotated x- and y-coordinates.
    """
    complex_coords = jnp.exp(1j * angle) * ((grid_x) + (grid_y) * 1j)
    return complex_coords.real, complex_coords.imag


def random_permutation_iterator(
        array_to_cycle: jnp.ndarray, rng: Sequence[int]
    ) -> Any:
    """Yield a generator that cycles through random permutation of an array.

    Args:
        array_to_cycle: Array to cycle over random permutations of.
        rng: jax PRNG key.

    Returns:
        Generator function that cycles over random permutation of the array.
    """
    # Will loop forever.
    while True:
        rng_shuffle, rng = jax.random.split(rng)

        # Shuffle the array randomly and cycle through the list.
        shuffled_array = jax.random.permutation(rng_shuffle, array_to_cycle)
        for item in shuffled_array:
            yield item


def ellip_to_angle(
        ellip_x: jnp.ndarray, ellip_xy: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert from complex ellipticity moduli to angle and axis ratio.

    Args:
        ellip_x: Eccentricity in the x-direction.
        ellip_xy: Eccentricity in the xy-direction.

    Returns:
        Axis ratio and angle.
    """
    # Angle is encoded in the arctangent.
    angle = jnp.arctan2(ellip_xy, ellip_x) / 2

    # Axis ratio is a function of the magnitude.
    ellip_mag = jnp.sqrt(ellip_x ** 2 + ellip_xy ** 2)
    ellip_mag = jnp.minimum(ellip_mag, 0.9999)
    axis_ratio = (1 - ellip_mag) / (1 + ellip_mag)

    return axis_ratio, angle
