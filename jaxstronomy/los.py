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
"""Draw masses and cocentrations for NFW line-of-sight halos.

This module contains the functions needed to turn the parameters of the
los halo distribution into masses, concentrations, and positions as defined in 
https://arxiv.org/pdf/1909.02573.pdf.
"""

from typing import Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from jaxstronomy import cosmology_utils
from jaxstronomy import nfw_functions
from jaxstronomy import power_law

# Parameters fit from simulations and fixed so that the integral for the nu 
# function over all nu gives 1.
NU_FUNC_A = 0.32218
NU_FUNC_Q = 0.3
NU_FUNC_LIT_A = 0.707


def nu_function(nu: jnp.ndarray):
    """Return nu f(nu) for the Sheth Tormen 2001 model.

    Args:
        nu: Values for nu.

    Returns:
        Value of nu times f(nu) for Sheth Tormen 2001 model. 
    """
    # Fundamental unit for the equation.
    nu_squared = NU_FUNC_LIT_A * jnp.square(nu)

    return (2 * NU_FUNC_A * (1 + nu_squared **(-NU_FUNC_Q)) * 
        jnp.sqrt(nu_squared / (2 * jnp.pi)) * jnp.exp(-nu_squared / 2))


def mass_function_exact(
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]], 
    masses: jnp.ndarray, z: float) -> jnp.ndarray:
    """Return the exact Sheth Tormen 2001 mass function.

    Args:
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        masses: Masses at which to evaluate mass function. In units of M_sun.
        z: Redshift at which to evaluate mass function.

    Returns:
        The value of the mass function for each provided mass. In units of the 
        physical number density 1/(M_sun * kpc^3).
    """
    # Convert to units of nu used by nu_function.
    h = cosmology_params['hubble_constant'] / 100
    delta_collapse = cosmology_utils.collapse_overdensity(cosmology_params, z)
    lagrangian_r = cosmology_utils.lagrangian_radius(cosmology_params, masses)
    sigma = cosmology_utils.sigma_tophat(cosmology_params, lagrangian_r, z)
    nu = delta_collapse / sigma

    # Calculate the mass function from its components
    nu_function_eval = nu_function(nu)
    derivative_sigma = cosmology_utils.derivative_log_sigma_log_r(
        cosmology_params, lagrangian_r, z)
    # Matter density is returned in units of M_sun * h ^ 2 / kpc ^ 3
    rho_matter = cosmology_utils.rho_matter(cosmology_params, z) * h ** 2

    return (-1 / 3 * nu_function_eval * rho_matter / masses ** 2 * 
        derivative_sigma)

def mass_function_power_law(
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]], z: float, 
    m_min: float, m_max: float) -> Tuple[float, float]:
    """Return the best fit power law parameters for the mass function.

    Args:
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z: Redhist at which to estimate mass function.
        m_min: Lower bound of masses to consider. In units of M_sun.
        m_max: Upper bound of masses to consider. In units of M_sun.

    Returns:
        Power law slope and norm that best approximates the exact mass function.
    """
    # TODO: For now the number of samples are hardcoded to avoid compilation
    # issues. May want to change this in the future.
    num_masses = 100
    masses = jnp.logspace(jnp.log10(m_min), jnp.log10(m_max), num_masses)
    log_masses = jnp.log(masses)
    log_mass_function = mass_function_exact(cosmology_params, masses, z)

    # Use the MLE estimate for the slope assuming Gaussian scatter on the log
    # quantity.
    slope_estimate = 1 / num_masses * jnp.sum(log_masses) * jnp.sum(log_masses)
    slope_estimate -= jnp.sum(log_mass_function * log_masses)
    slope_estimate /= (-jnp.sum(log_masses ** 2) + 
        1 / num_masses * jnp.sum(log_masses) ** 2)
    
    # Similar MLE estimate for the norm.
    norm_estimate = jnp.exp(1 / num_masses * 
        jnp.sum(log_mass_function - slope_estimate * log_masses))

    return slope_estimate, norm_estimate


def two_halo_boost(main_deflector_params: Mapping[str, float], 
    los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]], 
    z: float) -> float:
    """Calculate the boost from the two halo term caused by the host halo.

    Args:
        main_deflector_params: Parameters of the main deflector.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z: Redshift at which to calculate the boost.

    Returns:
        Boost at the given redshift.
    """
    # Extract the los and main deflector parameters we will need.
    dz = los_params['dz']
    host_mass = main_deflector_params['mass']
    z_lens = main_deflector_params['z_lens']
    r_min = los_params['r_min']
    r_max = los_params['r_max']
    h = cosmology_params['hubble_constant'] / 100

    # TODO: For now use a fixed range of redshift values to calculate the boost.
    z_range = jnp.linspace(z, z+dz, 100)

    # Only consider the two-point statistics within the radial limits.
    comoving_r = jnp.abs(jax.vmap(cosmology_utils.comoving_distance, 
        in_axes = [None, 0, None])(cosmology_params, z_range, z_lens))
    
    # The two halo term consists of the correlation function and the halo bias.
    two_halo = cosmology_utils.correlation_function(cosmology_params, 
        comoving_r, z_lens)
    two_halo *= cosmology_utils.halo_bias(cosmology_params, host_mass * h, 
        z_lens)

    return 1 + jnp.mean(two_halo)


def cone_angle_to_radius(main_deflector_params: Mapping[str, float], 
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]], 
    z: float) -> float:
    """Return the radius in kpc for the lightcone.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z: Redshift at which to calculate the radius.

    Returns:
        Radius of the lightcone. In units of kpc.
    """
    # Get the source, lens, and los parameters we will need.
    z_lens = main_deflector_params['z_lens']
    z_source = source_params['z_source']
    cone_angle = los_params['cone_angle']
    angle_buffer = los_params['angle_buffer']

    # If we are in front of the main deflector, the opening radius is a simple
    # conversion.
    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z)
    los_radius = kpa * cone_angle * 0.5

    # If we're behind the main deflector, then the light cone shrinks back to a
    # point at the source, with an additional buffer.
    scale_factor = angle_buffer
    scale_factor *= cosmology_utils.comoving_distance(cosmology_params, 0.0, 
        z_source)
    scale_factor /= cosmology_utils.comoving_distance(cosmology_params, z_lens,
        z_source)
    scale_factor *= cosmology_utils.comoving_distance(cosmology_params, z_lens, 
        z)
    scale_factor /= cosmology_utils.comoving_distance(cosmology_params, 0.0, z)

    # Define functions for use with jax.lax.cond.
    def before():
        return los_radius
    
    def after():
        return los_radius * (1-scale_factor)

    return jax.lax.cond(z < z_lens, before, after)


def volume_element(main_deflector_params: Mapping[str, float], 
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]], 
    z: float) -> float:
    """Return the volume element for the lightcone.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z: Redshift at which to calculate the volume element.

    Returns:
        Volume element of the lightcone. In units of kpc ** 3.
    """
    # Get the los parameters we will need.
    dz = los_params['dz']

    los_radius = cone_angle_to_radius(main_deflector_params, source_params, 
        los_params, cosmology_params, z + dz / 2)
    
    # Get the thickness of the slice in physical units
    dz_in_kpc = cosmology_utils.comoving_distance(cosmology_params, z, z + dz)
    # Comoving to physical and Mpc to kpc
    dz_in_kpc *= 1000 / (1 + z)

    return dz_in_kpc * jnp.pi * los_radius ** 2


def expected_num_halos(main_deflector_params: Mapping[str, float], 
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z) -> float:
    """Return the expected number of halos in a redshift slice.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z: Redhisft of the slice.

    Returns:
        Expected number of los halos in the slice.
    """
    # Extract the main deflector, source, and los parameters we need.
    m_min = los_params['m_min']
    m_max = los_params['m_max']

    slope_pl, norm_pl = mass_function_power_law(cosmology_params, z, m_min, 
        m_max)
    norm_pl *= volume_element(main_deflector_params, source_params,
        los_params, cosmology_params, z)
    norm_pl *= two_halo_boost(main_deflector_params, los_params,
        cosmology_params, z)

    return norm_pl * power_law.power_law_integrate(m_min , m_max, slope_pl)


def draw_redshifts(main_deflector_params: Mapping[str, float], 
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]], 
    z_min: float, z_max: float, rng: Sequence[int], 
    num_z_bins: int, pad_length: int) -> jnp.ndarray:
    """Draw the redshifts for the los halos within an interval.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_min: Minimum redshift of interval.
        z_max: Maximum redshift of interval.
        rng: jax PRNG key.
        num_z_bins: Number of redshift bins to consider.
        pad_length: Length to pad redshift draws to.

    Returns:
        Redshifts of los halos. A redshift of -1.0 indicates a padded halo.
    """
    rng_num, rng_cdf = jax.random.split(rng)
    # Extract the main deflector, source, and los parameters we will need.
    dz = los_params['dz']

    # Get the redshift bins which we'll use for our cdf interpolation.
    z_samples = jnp.linspace(z_min, z_max, num_z_bins)

    # Round the samples so that they are only in dz increments. This means many
    # samples will be rounded to the same value, so we will also need the
    # average number of samples being rounded into each bin.
    z_samples = jnp.round(z_samples / dz) * dz
    samples_per_bin = num_z_bins / (z_max - z_min) * dz

    # Calculate the number of expected halos in each bin to build our cdf.
    num_expected_samps = jax.vmap(expected_num_halos, 
        in_axes=[None, None, None, None, 0])(main_deflector_params, 
        source_params, los_params, cosmology_params, z_samples)
    num_expected_samps /= samples_per_bin

    # Get the number of los halos we will keep
    total_expected = jnp.sum(num_expected_samps)
    num_los = jax.random.poisson(rng_num, total_expected)

    # Use the expected samples to draw from the inverse cdf.
    cdf = jnp.cumsum(num_expected_samps) / total_expected
    cdf_draws = jax.random.uniform(rng_cdf, shape=(pad_length,))
    z_draws = jnp.interp(cdf_draws, cdf, z_samples)
    z_draws -= z_draws % dz

    # Calculate which z draws need to be padded out.
    indices = jnp.arange(pad_length)
    is_pad = indices >= num_los

    return z_draws * ~is_pad - is_pad


def draw_masses(los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_values: jnp.ndarray, rng: Sequence[int]) -> jnp.ndarray:
    """Draw the masses for the los halos within an interval.

    Args:
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_values: Redshifts of los halos to draw masses for.
        rng: jax PRNG key.

    Returns:
        Masses of los halos. In units of M_sun.
    """
    # Extract the los parameters we will need.
    m_min = los_params['m_min']
    m_max = los_params['m_max']

    # Get the position of each los halo in its respective cdf
    cdf_draws = jax.random.uniform(rng, shape=z_values.shape)

    # Function for mapping the indiviudal cdf draws to a mass given the slope.
    def draw_from_pl_cdf(m_min, m_max, slope_pl, cdf_draw):
        s_one = slope_pl + 1
        return (cdf_draw * (m_max ** s_one - m_min ** s_one) + 
            m_min ** s_one) ** (1 / s_one)

    slopes_pl, _ = jax.vmap(mass_function_power_law, 
        in_axes=[None, 0, None, None])(cosmology_params, z_values, m_min, m_max)
    return jax.vmap(draw_from_pl_cdf, in_axes=[None, None, 0, 0])(m_min, m_max, 
        slopes_pl, cdf_draws)


def draw_positions(main_deflector_params: Mapping[str, float], 
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    z_values: jnp.ndarray, rng: Sequence[int]) -> jnp.ndarray:
    """Return position draws for the los halos.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        z_values: Redshifts of the slices for which positions are being drawn.
        rng: jax PRNG key.

    Returns:
        x- and y-positions stacked over the last axis. In units of kpc.
    """
    rng_radius, rng_theta = jax.random.split(rng)

    # Draw the los positions uniformly within the slice.
    los_radius = jax.vmap(cone_angle_to_radius, 
        in_axes=[None, None, None, None, 0])(main_deflector_params, 
        source_params, los_params, cosmology_params, z_values)
    radius_draws = los_radius * jnp.sqrt(
        jax.random.uniform(rng_radius, shape=z_values.shape))
    theta_draws = 2 * jnp.pi * jax.random.uniform(rng_theta, 
        shape=z_values.shape)
    coords = [radius_draws * jnp.cos(theta_draws), 
        radius_draws * jnp.sin(theta_draws)]

    return jnp.stack(coords, axis=-1)


def convert_to_lensing(main_deflector_params: Mapping[str, float],
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    los_masses: jnp.ndarray, los_z: jnp.ndarray, los_cart_pos: jnp.ndarray,
    rng: Sequence[int]) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Convert los masses and positions into lensing quantities.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        los_mass: Masses of the los halos.
        los_z: Redshift of the los halos
        los_cart_pos: X- and y-coordinates of the los halos.
        rng: jax PRNG key.

    Returns:
        Redshifts and Dictionary with model index and lensing profile properties 
        for all of the los halos.
    """
    # Extract the redshifts we need
    z_lens = main_deflector_params['z_lens']
    z_source = source_params['z_source']

    # Calculate the concentration and radial position of the subhalos
    los_c = nfw_functions.mass_concentration(los_params, cosmology_params,
        los_masses, z_lens, rng)

    # Concert from masses and concentrations to nfw parameters
    los_r_two_hund = nfw_functions.r_two_hund_from_m(cosmology_params,
        los_masses, z_lens)
    los_r_scale = los_r_two_hund / los_c
    los_rho_nfw = nfw_functions.rho_nfw_from_c(cosmology_params, los_c, z_lens)

    # Convert to lensing units
    los_r_scale_ang, los_alpha_rs = nfw_functions.convert_to_lensing_nfw(
        cosmology_params, los_r_scale, los_z, los_rho_nfw, z_source)
    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
    subhalos_cart_pos_ang = los_cart_pos / kpa

    # There is only one model, the NFW. Subhalos with redshift -1.0 are treated
    # as padding models.
    subhalos_model_index = (jnp.full(los_masses.shape, -1) *
        jnp.int32(los_z == -1.0))

    subhalos_kwargs = {'model_index': subhalos_model_index,
        'scale_radius': los_r_scale_ang, 'alpha_rs': los_alpha_rs,
        'center_x': subhalos_cart_pos_ang[:,0],
        'center_y': subhalos_cart_pos_ang[:,1]}

    return los_z, subhalos_kwargs


def draw_los(main_deflector_params: Mapping[str, float],
    source_params: Mapping[str, float], los_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    rng: Sequence[int], num_z_bins: int, los_pad_length: int,
) -> Tuple[Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]], 
    Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]]:
    """Draw line-of-sight halos with redshift and lensing quantities.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters for the source.
        los_parms: Parameters of the los distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        rng: jax PRNG key.
        num_z_bins: Number of redshift bins to use in los draws. These bins 
            will be rounded to the nearest unit of dz and should be a large
            number.
        los_pad_length: Number of los halos to pad to.
    
    Returns:
        Redshifts and Dictionary with model index and lensing profile properties 
        for all of the los halos. First tuple is for the los halos before the
        main deflector and second tuple is for los halos after the main
        deflector.
    """
    rng_before, rng_after = jax.random.split(rng)
    rng_before_z, rng_before_mass, rng_before_pos, rng_before_convert = (
        jax.random.split(rng_before, 4))
    rng_after_z, rng_after_mass, rng_after_pos, rng_after_convert = (
        jax.random.split(rng_after, 4))

    # Draw the redshifts, masses and positions for our los halos up to the pad
    # before the main deflector.
    z_lens = main_deflector_params['z_lens']
    los_before_z = draw_redshifts(main_deflector_params, source_params, 
        los_params, cosmology_params, 0.0, z_lens, rng_before_z, num_z_bins, 
        los_pad_length)
    los_before_masses = draw_masses(los_params, cosmology_params, los_before_z, 
        rng_before_mass)
    los_before_cart_pos = draw_positions(main_deflector_params, source_params,
        los_params, cosmology_params, los_before_z, rng_before_pos)
    los_before_tuple = convert_to_lensing(main_deflector_params, source_params,
        los_params, cosmology_params, los_before_masses, los_before_z, 
        los_before_cart_pos, rng_before_convert)

    # Draw the redshifts, masses and positions for our los halos up to the pad
    # after the main deflector.
    z_source = source_params['z_source']
    los_before_z = draw_redshifts(main_deflector_params, source_params, 
        los_params, cosmology_params, z_lens, z_source, rng_after_z, num_z_bins, 
        los_pad_length)
    los_before_masses = draw_masses(los_params, cosmology_params, los_before_z, 
        rng_after_mass)
    los_before_cart_pos = draw_positions(main_deflector_params, source_params,
        los_params, cosmology_params, los_before_z, rng_after_pos)
    los_after_tuple = convert_to_lensing(main_deflector_params, source_params,
        los_params, cosmology_params, los_before_masses, los_before_z, 
        los_before_cart_pos, rng_after_convert)

    return los_before_tuple, los_after_tuple
