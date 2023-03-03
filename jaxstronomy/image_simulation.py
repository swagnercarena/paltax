# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strong lensing image simulation pipeline in jax.

This module includes functions to map from a strong lensing configuration to the
corresponding observed image using jax-optimized functions. Inspiration taken
from lenstronomy: https://github.com/lenstronomy/lenstronomy.
"""

import functools
from typing import Any, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from jaxstronomy import cosmology_utils
from jaxstronomy import utils


def generate_image(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    kwargs_lens_all: Mapping[str, Union[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    kwargs_source_slice: Mapping[str, jnp.ndarray],
    kwargs_lens_light_slice: Mapping[str, jnp.ndarray],
    kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_source: float,
    kwargs_detector: Mapping[str, Union[float, int]],
    all_models: Mapping[str, Sequence[Any]],
    apply_psf: bool = True,
) -> jnp.ndarray:
    """Generate an image given the source, lens light, and mass profiles.

    Args:
        grid_x: X-coordinates of grid on which image will be generated.
        grid_y: Y-coordinates of grid on which image will be generated.
        kwargs_lens_all: Keyword arguments and redshifts for each of the
            lensing components. This should include los_before, los_after,
            subhalos, and main_deflector.
        kwargs_source_slice: Keyword arguments to pass to brightness functions
            for each light model. Keys should include parameters for all source
            models (due to use of `jax.lax.switch`) and `model_index` which
            defines the model to pass the parameters to.
        kwargs_lens_light_slice: Keyword arguments to pass to brightness
            functions for each light model. Keys should include parameters for
            all source models (due to use of `jax.lax.switch`) and `model_index`
            which defines the model to pass the parameters to.
        kwargs_psf: Keyword arguments defining the point spread function. The
            psf is applied in the supersampled space, so the size of pixels
            should be defined with respect to the supersampled space.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_source: Redshift of the source.
        kwargs_detector: Keyword arguments defining the detector configuration.
        all_models: Tuple of model classes to consider for each component.
        apply_psf: Whether or not to convolve the final image with the point 
            spread function.

    Returns:
        Image after gravitational lensing at supersampling resolution. For
        consistency with lenstronomy, units are per pixel area of the detector,
        not the supersampling resolution.

    Notes:
        The parameters kwargs_detector and all_models must be made static if to
        jit compile this function.
    """
    image_array = source_surface_brightness(
        grid_x, grid_y, kwargs_lens_all, kwargs_source_slice, kwargs_detector,
        cosmology_params, z_source, all_models
    )
    image_array += lens_light_surface_brightness(
        grid_x, grid_y, kwargs_lens_light_slice, kwargs_detector,
        all_models["all_source_models"]
    )
    image = jnp.reshape(
        image_array,
        (
            kwargs_detector["n_x"] * kwargs_detector["supersampling_factor"],
            kwargs_detector["n_y"] * kwargs_detector["supersampling_factor"],
        ),
    )

    if apply_psf:
        image = psf_convolution(image, kwargs_psf, all_models["all_psf_models"])
    return image


def psf_convolution(
    image: jnp.ndarray,
    kwargs_psf: Mapping[str, Union[float, int, jnp.ndarray]],
    all_psf_models: Sequence[Any]
) -> jnp.ndarray:
    """Convolve image with the point spread function.

    Args:
        image: Image to convolve
        kwargs_psf: Keyword arguments defining the point spread function.
        all_psf_models: PSF models to use for model_index lookup.

    Returns:
        Convolved image.
    """
    # Psf not accounting for supersampling by default is prone to user error.
    # Consider changing.
    psf_functions = [model.convolve for model in all_psf_models]
    return jax.lax.switch(kwargs_psf["model_index"], psf_functions, image, kwargs_psf)


def noise_realization(
    image: jnp.ndarray,
    rng: Sequence[int],
    kwargs_detector: Mapping[str, Union[float, int]]
) -> jnp.ndarray:
    """Return noise realization for input image.

    Args:
        image: Image from which to draw poisson noise.
        rng: jax PRNG key used for noise realization.
        kwargs_detector: Keyword arguments defining the detector configuration.

    Returns:
        Noise realization for image.
    """
    rng_normal, rng_poisson = jax.random.split(rng)

    # Calculate expected background noise from detector kwargs.
    exposure_time_total = (
        kwargs_detector["exposure_time"] * kwargs_detector["num_exposures"]
    )
    read_noise = kwargs_detector["read_noise"] ** 2 * kwargs_detector["num_exposures"]
    sky_brightness_cps = utils.magnitude_to_cps(
        kwargs_detector["sky_brightness"], kwargs_detector["magnitude_zero_point"]
    )
    sky_brightness_tot = (
        exposure_time_total * sky_brightness_cps * kwargs_detector["pixel_width"] ** 2
    )
    background_noise = jnp.sqrt(read_noise + sky_brightness_tot) / exposure_time_total

    # By default all simulations are done in units of counts per second, but you
    # want to calculate poisson statistics in units of counts.
    flux_noise = jnp.sqrt(jax.nn.relu(image) / exposure_time_total)

    noise = jax.random.normal(rng_normal, image.shape) * background_noise
    noise += jax.random.normal(rng_poisson, image.shape) * flux_noise
    return noise


def lens_light_surface_brightness(
    theta_x: jnp.ndarray,
    theta_y: jnp.ndarray,
    kwargs_lens_light_slice: Mapping[str, jnp.ndarray],
    kwargs_detector: Mapping[str, Union[float, int]],
    all_source_models: Sequence[Any]
) -> jnp.ndarray:
    """Return the lens light surface brightness.

    Args:
        theta_x: X-coordinates in angular units.
        theta_y: Y-coordinates in angular units.
        kwargs_lens_light_slice: Keyword arguments to pass to brightness
            functions for each light model. Keys should include parameters for
            all source models (due to use of `jax.lax.switch`) and `model_index`
            which defines the model to pass the parameters to.
        kwargs_detector: Keyword arguments defining the detector configuration.
        all_source_models: Source models to use for model_index lookup.

    Returns:
        Surface brightness of lens light as 1D array.
    """
    lens_light_flux = _surface_brightness(
        theta_x, theta_y, kwargs_lens_light_slice, all_source_models
    )
    return lens_light_flux * kwargs_detector["pixel_width"] ** 2


def source_surface_brightness(
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    kwargs_lens_all: Mapping[str, Union[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    kwargs_source_slice: Mapping[str, jnp.ndarray],
    kwargs_detector: Mapping[str, jnp.ndarray],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_source: float,
    all_models: Mapping[str, Sequence[Any]],
) -> jnp.ndarray:
    """Return the lensed source surface brightness.

    Args:
        alpha_x: Initial x-component of deflection at each position.
        alpha_y: Initial y-component of deflection at each position.
        kwargs_lens_all: Keyword arguments and redshifts for each of the
            lensing components. This should include los_before, los_after,
            subhalos, and main_deflector.
        kwargs_detector: Keyword arguments defining the detector configuration. This
            includes potential supersampling in the lensing calculation.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_source: Redshift of the source.
        all_models: Tuple of model classes to consider for each component.

    Returns:
        Lensed source surface brightness as 1D array.
    """
    image_flux_array = _image_flux(
        alpha_x,
        alpha_y,
        kwargs_lens_all,
        kwargs_source_slice,
        cosmology_params,
        z_source,
        all_models
    )
    # Scale by pixel area to go from flux to surface brightness.
    return image_flux_array * kwargs_detector["pixel_width"] ** 2


def _image_flux(
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    kwargs_lens_all: Mapping[str, Union[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    kwargs_source_slice: Mapping[str, jnp.ndarray],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_source: float,
    all_models: Mapping[str, Sequence[Any]],
) -> jnp.ndarray:
    """Calculate image flux after ray tracing onto the source.

    Args:
        alpha_x: Initial x-component of deflection at each position.
        alpha_y: Initial y-component of deflectoin at each position.
        kwargs_lens_all: Keyword arguments and redshifts for each of the
            lensing components. This should include los_before, los_after,
            subhalos, and main_deflector.
        kwargs_source_slice: Keyword arguments to pass to brightness functions for
            each light model. Keys should include parameters for all source models
            (due to use of `jax.lax.switch`) and `model_index` which defines the model
            to pass the parameters to. Parameter values not relevant for the specified
            model are discarded. Values should be of type jnp.ndarray and have length
            equal to the total number of light models in the slice. For more detailed
            discussion see documentation of `jax.lax.scan`, which is used to iterate
            over the models.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_source: Redshift of the source.
        all_models: Tuple of model classes to consider for each component.

    Returns:
        Image flux.
    """
    x_source_comv, y_source_comv = _ray_shooting(
        alpha_x,
        alpha_y,
        kwargs_lens_all,
        cosmology_params,
        z_source,
        all_models,
    )
    x_source, y_source = cosmology_utils.comoving_to_angle(
        x_source_comv, y_source_comv, cosmology_params, z_source
    )
    return _surface_brightness(
        x_source, y_source, kwargs_source_slice, all_models["all_source_models"]
    )


def _ray_shooting(
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    kwargs_lens_all:  Mapping[str, Union[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_source: float,
    all_models: Mapping[str, Sequence[Any]]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ray shoot over all of the redshift slices between observer and source.

    Args:
        alpha_x: Initial x-component of deflection at each position.
        alpha_y: Initial y-component of deflectoin at each position.
        kwargs_lens_all: Keyword arguments and redshifts for each of the
            lensing components. This should include los_before, los_after,
            subhalos, and main_deflector.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_source: Redshift of the source.
        all_models: Tuple of model classes to consider for each component.

    Returns:
        Comoving x- and y-coordinate after ray shooting.
    """
    # Initially all our light rays are localized at the observer.
    comv_x = jnp.zeros_like(alpha_x)
    comv_y = jnp.zeros_like(alpha_y)

    z_lens_last = 0.0

    # The state to pass to scan.
    state = (comv_x, comv_y, alpha_x, alpha_y, z_lens_last)

    # Break the ray tracing into four parts: the first batch of line-of-sight
    # halos, the subhalos, the main deflector, and then the second batch
    # of line-of-sight halos. This circumvents issues with jax.lax.switch
    # evaluating the expensive lens models for each deflector.
    ray_shooting_step_los = functools.partial(
        _ray_shooting_step,
        cosmology_params=cosmology_params,
        z_source=z_source,
        all_lens_models=all_models["all_los_models"]
    )
    ray_shooting_step_main_deflector = functools.partial(
        _ray_shooting_step,
        cosmology_params=cosmology_params,
        z_source=z_source,
        all_lens_models=all_models["all_main_deflector_models"]
    )

    # Scan over all of the lens models in our system to calculate deflection and
    # ray shoot between lens models.
    state, _ = jax.lax.scan(
        ray_shooting_step_los,
        state,
        {
            "z_lens": kwargs_lens_all["z_array_los_before"],
            "kwargs_lens": kwargs_lens_all["kwargs_los_before"]
        },
    )
    # We can do all the subhalos at once, which is a lot faster for large
    # number of subhalos.
    # Use max instead of mean here, in case things are padded with zeros
    state, _ = _ray_shooting_group(
        state, kwargs_lens_all["kwargs_subhalos"], cosmology_params, z_source,
        jnp.max(kwargs_lens_all["z_array_subhalos"]),
        all_models["all_subhalo_models"])
    state, _ = _ray_shooting_group(
        state, kwargs_lens_all["kwargs_main_deflector"], cosmology_params,
        z_source, jnp.max(kwargs_lens_all["z_array_main_deflector"]),
        all_models["all_main_deflector_models"])
    state, _ = jax.lax.scan(
        ray_shooting_step_los,
        state,
        {
            "z_lens": kwargs_lens_all["z_array_los_after"],
            "kwargs_lens": kwargs_lens_all["kwargs_los_after"]
        },
    )

    comv_x, comv_y, alpha_x, alpha_y, z_lens_last = state

    # Continue the ray tracing until the source.
    delta_t = cosmology_utils.comoving_distance(cosmology_params, z_lens_last, z_source)
    comv_x, comv_y = _ray_step_add(comv_x, comv_y, alpha_x, alpha_y, delta_t)

    return comv_x, comv_y


def _ray_shooting_group(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
    kwargs_lens_slice: Mapping[str, jnp.ndarray],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_source: float,
    z_lens: float,
    all_lens_models: Sequence[Any]
) -> Tuple[Tuple, Tuple]:
    """Conduct ray shooting for a group of coplanar lens models.

    Args:
        state: The current comoving positions, deflections, and the previous lens
            model redshift.
        kwargs_lens_slice: Keyword arguments specifying the model (through
            `model_index` value) and the lens model parameters for each lens
            model in the slice. Due to the nature of `jax.lax.switch` this must
            have a key-value pair for all lens models included in
            `all_lens_models`, even if those lens models will not be used.
            `model_index` of -1 indicates that the model should be ignored.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_source: Redshift of the source.
        z_lens: Redshift of the coplanar group of lens models.
        all_lens_models: Lens models to use for model_index lookup.

    Returns:
        Two copies of the new state, which is a tuple of new comoving positions,
        new deflection, and the redshift of the current lens.
    """
    comv_x, comv_y, alpha_x, alpha_y, z_lens_last = state

    # The displacement from moving along the deflection direction.
    delta_t = cosmology_utils.comoving_distance(
        cosmology_params, z_lens_last, z_lens,
    )
    comv_x, comv_y = _ray_step_add(comv_x, comv_y, alpha_x, alpha_y, delta_t)
    alpha_x, alpha_y = _add_deflection_group(
        comv_x, comv_y, alpha_x, alpha_y, kwargs_lens_slice,
        cosmology_params, z_lens, z_source, all_lens_models)

    new_state = (comv_x, comv_y, alpha_x, alpha_y, z_lens)

    # Second return is required by scan, but will be ignored by the compiler.
    return new_state, new_state

def _ray_shooting_step(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
    kwargs_z_lens: Mapping[str, Union[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_source: float,
    all_lens_models: Sequence[Any]
) -> Tuple[Tuple, Tuple]:
    """Conduct ray shooting between two lens models.

    Args:
        state: The current comoving positions, deflections, and the previous lens
            model redshift.
        kwargs_z_lens: Dict with keys `z_lens`, the redshift of the next lens model,
            and 'kwargs_lens`, the keyword arguments specifying the next lens model
            (through `model_index` value) and the lens model parameters. Due to the
            requirements of `jax.lax.switch`, `kwargs_lens` must have a key-value pair
            for all lens models included in `all_lens_models`, even if those lens
            models will not be used.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_source: Redshift of the source.
        all_lens_models: Lens models to use for model_index lookup.

    Returns:
        Two copies of the new state, which is a tuple of new comoving positions,
        new deflection, and the redshift of the current lens.

    Notes:
        For use with `jax.lax.scan`.
    """
    comv_x, comv_y, alpha_x, alpha_y, z_lens_last = state

    # The displacement from moving along the deflection direction.
    delta_t = cosmology_utils.comoving_distance(
        cosmology_params, z_lens_last, kwargs_z_lens["z_lens"]
    )
    comv_x, comv_y = _ray_step_add(comv_x, comv_y, alpha_x, alpha_y, delta_t)
    alpha_x, alpha_y = _add_deflection(
        comv_x,
        comv_y,
        alpha_x,
        alpha_y,
        kwargs_z_lens["kwargs_lens"],
        cosmology_params,
        kwargs_z_lens["z_lens"],
        z_source,
        all_lens_models,
    )

    new_state = (comv_x, comv_y, alpha_x, alpha_y, kwargs_z_lens["z_lens"])

    # Second return is required by scan, but will be ignored by the compiler.
    return new_state, new_state


def _ray_step_add(
    comv_x: jnp.ndarray,
    comv_y: jnp.ndarray,
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    delta_t: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Conduct ray step by adding deflections to comoving positions.

    Args:
        comv_x: Comoving x-coordinate.
        comv_y: Comoving y-coordinate.
        alpha_x: Current physical x-component of deflection at each position.
        alpha_y: Current physical y-component of deflection at each position.
        delta_t: Comoving distance between current slice and next slice.

    Returns:
        Updated x- and y-coordinate in comoving Mpc.
    """
    return comv_x + alpha_x * delta_t, comv_y + alpha_y * delta_t


def _add_deflection_group(
    comv_x: jnp.ndarray,
    comv_y: jnp.ndarray,
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    kwargs_lens_slice: Mapping[str, jnp.ndarray],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_lens: float,
    z_source: float,
    all_lens_models: Sequence[Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the deflection angle for a group of co-planar lens models.

    Args:
        comv_x: Comoving x-coordinate.
        comv_y: Comoving y-coordinate.
        alpha_x: Current physical x-component of deflection at each position.
        alpha_y: Current physical y-component of deflection at each position.
        kwargs_lens_slice: Keyword arguments specifying the model (through
            `model_index` value) and the lens model parameters for each lens
            model in the slice. Due to the nature of `jax.lax.switch` this must
            have a key-value pair for all lens models included in
            `all_lens_models`, even if those lens models will not be used.
            `model_index` of -1 indicates that the model should be ignored.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_lens: Redshift of the slice (i.e. current redshift).
        z_source: Redshift of the source.
        all_lens_models: Lens models to use for model_index lookup.

    Returns:
        New x- and y-component of the deflection at each specified position.
    """
    theta_x, theta_y = cosmology_utils.comoving_to_angle(
        comv_x, comv_y, cosmology_params, z_lens
    )

    # All of our derivatives are defined in reduced coordinates. We want to
    # scan over all the calculation in reduced coordinates to improve
    # parallelization performance.
    calculate_derivatives = functools.partial(
        _calculate_derivatives, theta_x=theta_x, theta_y=theta_y,
        all_lens_models=all_lens_models)
    alpha_x_reduced_array, alpha_y_reduced_array = jax.vmap(
        calculate_derivatives, in_axes=0)(kwargs_lens_slice)
    alpha_x_reduced = jnp.sum(alpha_x_reduced_array, axis=0)
    alpha_y_reduced = jnp.sum(alpha_y_reduced_array, axis=0)

    alpha_x_update = cosmology_utils.reduced_to_physical(
        alpha_x_reduced, cosmology_params, z_lens, z_source
    )
    alpha_y_update = cosmology_utils.reduced_to_physical(
        alpha_y_reduced, cosmology_params, z_lens, z_source
    )

    return alpha_x - alpha_x_update, alpha_y - alpha_y_update


def _add_deflection(
    comv_x: jnp.ndarray,
    comv_y: jnp.ndarray,
    alpha_x: jnp.ndarray,
    alpha_y: jnp.ndarray,
    kwargs_lens: Mapping[str, Union[int, float, jnp.ndarray]],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_lens: float,
    z_source: float,
    all_lens_models: Sequence[Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the deflection for a specific lens model.

    Args:
        comv_x: Comoving x-coordinate.
        comv_y: Comoving y-coordinate.
        alpha_x: Current physical x-component of deflection at each position.
        alpha_y: Current physical y-component of deflection at each position.
        kwargs_lens: Keyword arguments specifying the model (through `model_index`
            value) and the lens model parameters. Due to the nature of
            `jax.lax.switch` this must have a key-value pair for all lens models
            included in `all_lens_models`, even if those lens models will not be used.
            `model_index` of -1 indicates that the previous total should be returned.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_lens: Redshift of the slice (i.e. current redshift).
        z_source: Redshift of the source.
        all_lens_models: Lens models to use for model_index lookup.

    Returns:
        New x- and y-component of the deflection at each specified position.
    """
    theta_x, theta_y = cosmology_utils.comoving_to_angle(
        comv_x, comv_y, cosmology_params, z_lens
    )

    # All of our derivatives are defined in reduced coordinates.
    alpha_x_reduced, alpha_y_reduced = _calculate_derivatives(
        kwargs_lens, theta_x, theta_y, all_lens_models
    )

    alpha_x_update = cosmology_utils.reduced_to_physical(
        alpha_x_reduced, cosmology_params, z_lens, z_source
    )
    alpha_y_update = cosmology_utils.reduced_to_physical(
        alpha_y_reduced, cosmology_params, z_lens, z_source
    )

    return alpha_x - alpha_x_update, alpha_y - alpha_y_update


def _calculate_derivatives(
    kwargs_lens: Mapping[str, Union[int, float, jnp.ndarray]],
    theta_x: jnp.ndarray,
    theta_y: jnp.ndarray,
    all_lens_models: Sequence[Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the derivatives for the specified lens model.

    Args:
        kwargs_lens: Keyword arguments specifying the model (through
            `model_index` value) and the lens model parameters. Due to the
            nature of `jax.lax.switch` this must have a key-value pair for all
            lens models included in `all_lens_models`, even if those lens models
            will not be used. `model_index` of -1 indicates that the previous
            total should be returned.
        theta_x: X-coordinate at which to evaluate the derivative.
        theta_y: Y-coordinate at which to evaluate the derivative.
        all_lens_models: Lens models to use for model_index lookup.

    Returns:
        Change in x- and y-component of derivative caused by lens model.
    """
    # The jax.lax.switch compilation requires that all of the functions take the
    # same inputs. We accomplish this using our wrapper and picking out only the
    # parameters required for that model.
    derivative_functions = [
        utils.unpack_parameters_xy(model.derivatives, model.parameters)
        for model in all_lens_models
    ]

    # Condition on model_index to allow for padding.
    def calculate_derivative():
        alpha_x, alpha_y = jax.lax.switch(
            kwargs_lens["model_index"],
            derivative_functions,
            theta_x,
            theta_y,
            kwargs_lens,
        )
        return (alpha_x, alpha_y)

    def identity():
        return (jnp.zeros_like(theta_x), jnp.zeros_like(theta_y))

    return jax.lax.cond(
        kwargs_lens["model_index"] == -1, identity, calculate_derivative
    )


def _surface_brightness(
    theta_x: jnp.ndarray,
    theta_y: jnp.ndarray,
    kwargs_source_slice: Mapping[str, jnp.ndarray],
    all_source_models: Sequence[Any]
) -> jnp.ndarray:
    """Return the surface brightness for a slice of light models.

    Args:
        theta_x: X-coordinate at which to evaluate the surface brightness.
        theta_y: Y-coordinate at which to evaluate the surface brightness.
        kwargs_source_slice: Keyword arguments to pass to brightness functions for
            each light model. Keys should include parameters for all source models
            included in `all_source_models`.
            (due to use of `jax.lax.switch`) and `model_index` which defines the model
            to pass the parameters to. Parameter values not relevant for the specified
            model are discarded. Values should be of type jnp.ndarray and have length
            equal to the total number of light models in the slice. For more detailed
            discussion see documentation of `jax.lax.scan`, which is used to iterate
            over the models.
        all_source_models: Source models to use for model_index lookup.

    Returns:
        Surface brightness summed over all sources.
    """
    add_surface_brightness = functools.partial(
        _add_surface_brightness,
        theta_x=theta_x,
        theta_y=theta_y,
        all_source_models=all_source_models,
    )
    brightness_total = jnp.zeros_like(theta_x)
    brightness_total, _ = jax.lax.scan(
        add_surface_brightness, brightness_total, kwargs_source_slice
    )
    return brightness_total


def _add_surface_brightness(
    prev_brightness: jnp.ndarray,
    kwargs_source: Mapping[str, Union[int, float, jnp.ndarray]],
    theta_x: jnp.ndarray,
    theta_y: jnp.ndarray,
    all_source_models: Sequence[Any]
) -> jnp.ndarray:
    """Return the surface brightness for a single light model.

    Args:
        prev_brightness: Previous brightness.
        kwargs_source: Kwargs to evaluate the source surface brightness.
        theta_x: X-coordinate at which to evaluate the surface brightness.
        theta_y: Y-coordinate at which to evaluate the surface brightness.
        all_source_models: Source models to use for model_index lookup.

    Returns:
        Surface brightness of the source at given coordinates.
    """
    source_functions = [
        utils.unpack_parameters_xy(model.function, model.parameters)
        for model in all_source_models
    ]
    brightness = jax.lax.switch(
        kwargs_source["model_index"], source_functions, theta_x, theta_y,
        kwargs_source
    )

    return prev_brightness + brightness, brightness
