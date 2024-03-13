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
"""Draw subhalo masses and cocentrations for NFW subhalos.

This module containts the functions needed to draw subhalos for an underlying
distribution as defined in https://arxiv.org/pdf/1909.02573.pdf.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from paltax import cosmology_utils
from paltax import nfw_functions
from paltax import power_law


def host_scaling_function(host_mass: float, z_lens: float, k_one: float,
    k_two: float) -> float:
    """Return scaling for the subhalos mass function due to host mass.

    Args:
        host_mass: Mass of the host halo in units of M_sun.
        z_lens: The redshift of the host halo.
        k1: Amplitude of halo mass dependence.
        k2: Amplitude of the redshift scaling.

    Notes:
        Derived from galacticus in https://arxiv.org/pdf/1909.02573.pdf.
    """
    log_f = (k_one * jnp.log10(host_mass / 1e13) +
        k_two * jnp.log10(z_lens + 0.5))
    return 10 ** log_f


def draw_nfw_masses(main_deflector_params: Mapping[str, float],
    subhalo_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    rng: Sequence[int], pad_length: int) -> jnp.ndarray:
    """Draw from the subhalo mass function and return the masses.

    Args:
        main_deflector_params: Parameters of the main deflector.
        subhalo_parms: Parameters of the subhalo distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        rng: Jax PRNG key.
        pad_length: Length to pad mass draw to.

    Returns:
        Masses of the drawn halos in units of M_sun.
    """
    # Cap sigma_sub to 0 for draws.
    sigma_sub = jax.lax.max(0.0, subhalo_params['sigma_sub'])

    # Extract the remaining parameters from the dict for readability
    shmf_plaw_index = subhalo_params['shmf_plaw_index']
    m_pivot = subhalo_params['m_pivot']
    m_min = subhalo_params['m_min']
    m_max = subhalo_params['m_max']
    k_one = subhalo_params['k_one']
    k_two = subhalo_params['k_two']
    host_mass = main_deflector_params['mass']
    z_lens = main_deflector_params['z_lens']

    # Calculate the overall norm of the power law, including the host scaling
    f_host = host_scaling_function(host_mass, z_lens, k_one, k_two)
    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
    radius_e = kpa * main_deflector_params['theta_e']
    area_elem = jnp.pi * (3 * radius_e) ** 2

    # Fold the pivot mass into the norm directly
    norm = f_host * area_elem * sigma_sub * m_pivot ** (-shmf_plaw_index - 1)

    # Draw our masses
    return power_law.power_law_draw(m_min, m_max, shmf_plaw_index, norm, rng,
        pad_length)


def rejection_sampling(radial_coord: jnp.ndarray, r_two_hund: float,
    r_bound: float, rng: Sequence[int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return three-dimensional sampling of the halo positions with rejection.

    Args:
        radial_cord: Radial displacement from center for each halo.
        r_two_hund: Two-hundred times overdensity radius of the host halo,
            which will be used to set the bounds on the z-coordinate.
        r_bound: Bound for x- and y-coordinate in units of kpc.
        rng: jax PRNG key used for noise realization.

    Returns:
        Acceptance and three dimensional coordinates of each subhalo.
    """
    # Split the rng key for each draw.
    rng_theta, rng_phi = jax.random.split(rng)
    theta = jax.random.uniform(rng_theta, shape=radial_coord.shape)
    theta *= 2 * jnp.pi
    phi = jnp.arccos(1 - 2 * jax.random.uniform(rng_phi,
        shape=radial_coord.shape))

    # Get each of the cartesian coordinates
    cart_x = radial_coord * jnp.sin(phi) * jnp.cos(theta)
    cart_y = radial_coord * jnp.sin(phi) * jnp.sin(theta)
    cart_z = radial_coord * jnp.cos(phi)

    # Test if the coordinates fall within the boudns
    r_xy_inside = jnp.sqrt(cart_x ** 2 + cart_y ** 2) < r_bound
    z_inside = jnp.abs(cart_z) < r_two_hund
    is_inside = jnp.logical_and(r_xy_inside, z_inside)

    return (is_inside, jnp.stack([cart_x, cart_y, cart_z], axis=-1))


def sample_cored_nfw(main_deflector_params: Mapping[str, float],
    subhalo_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    rng: Sequence[int], n_subs: int, sampling_pad_length: int) -> jnp.ndarray:
    """ Sample positions for NFW subhalos.

    Args:
        main_deflector_params: Parameters of the main deflector.
        subhalo_params: Parameters of the subhalo population.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        rng: jax PRNG key used for noise realization.
        n_subs: Number of subhalos to draw positions for.
        sampling_pad_length: Number of positions to sample before conducting
            rejection sampling. For small numbers, the odds of all of the
            returned positions being within the host R200 and three times
            the Einstein radius goes down.

    Returns:
        x-y-z coordinates of the subhalos in units of kpc.

    Notes:
        Subhalo positions are drawn within the scale radius of the host in the
        z-direction and three times the Einstein radius in the xy-plane.
    """
    # Seperate rng keys for each of our random draws.
    rng_host, rng_radii, rng_pos = jax.random.split(rng, 3)

    # Extract the relevant host properties
    host_mass = main_deflector_params['mass']
    z_lens = main_deflector_params['z_lens']
    host_c = nfw_functions.mass_concentration(subhalo_params, cosmology_params,
        jnp.array([host_mass]), z_lens, rng_host)
    host_r_two_hund = nfw_functions.r_two_hund_from_m(cosmology_params,
        host_mass, z_lens)
    host_r_scale = host_r_two_hund / host_c
    host_rho_nfw = nfw_functions.rho_nfw_from_c(cosmology_params, host_c,
        z_lens)

    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
    radius_e_three = kpa * main_deflector_params['theta_e'] * 3

    # The maximum radius is the corner for the cylinder in which we're
    # rendering.
    r_max = jnp.sqrt((radius_e_three) ** 2 + host_r_two_hund ** 2)

    # Draw coordinates up to the sampling pad length.
    radial_coord = nfw_functions.cored_nfw_draws(host_r_two_hund / 2,
        host_rho_nfw, host_r_scale, r_max, rng_radii, sampling_pad_length)
    is_inside, cart_pos = rejection_sampling(radial_coord, host_r_two_hund,
        radius_e_three, rng_pos)

    # We'll sort the cartesian coordinates by whether or not they're inside
    # the bounds and only return n_sub coordinates. This does not guarantee
    # that all the samples returned will be within the boudns, but for
    # large enough sampling_pad_length, nearly all of them will be.
    return cart_pos[jnp.argsort(~is_inside)[:n_subs]]


def get_truncation_radius(subhalos_mass: jnp.ndarray,
    subhalos_radii: jnp.ndarray, m_pivot: Optional[float] = 1e7,
    r_pivot: Optional[float] = 50) -> jnp.ndarray:
    """Return truncation radius for the subhalos.

    Args:
        subhalo_mass: Mass of the subhalo in units of M_sun.
        subhalos_radii: Radius of the subhalos in units of kpc.
        m_pivot: Pivot mass for the scaling in units of M_sun.
        r_pivot: Pivot radius for the relation in units of kpc.

    Returns:
        Truncation radius in units of kpc.
    """
    # Set a minimum truncation radius of 1e-7 kpc to avoid poorly defined
    # models
    r_trunc_min = 1e-7
    return jax.lax.max((1.4 * (subhalos_mass / m_pivot) ** (1 / 3) *
        (subhalos_radii / r_pivot) ** (2 / 3)), r_trunc_min)


def convert_to_lensing(main_deflector_params: Mapping[str, float],
    source_params: Mapping[str, Union[int, float]],
    subhalo_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    subhalos_masses: jnp.ndarray, subhalos_cart_pos: jnp.ndarray,
    rng: Sequence[int]) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Convert subhalo masses and positions into lensing quantities.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters of the source.
        subhalo_params: Parameters of the subhalo population.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        subhalo_masses: Masses of the subhalos in units of M_sun.
        subhalos_cart_pos: x-y-z-position of the subhalos in units of kpc.
        rng: Jax PRNG key.

    Returns:
        Redshifts for each subhalo and the dictionary of lensing quantities.

    Notes:
        Model index set to -1 for subhalos with zero mass.
    """
    # Extract the redshifts we need
    z_lens = main_deflector_params['z_lens']
    z_source = source_params['z_source']

    # All the subhalos are at the main deflector redshift
    subhalos_z = jnp.full(subhalos_masses.shape, z_lens)

    # Calculate the concentration and radial position of the subhalos
    subhalos_c = nfw_functions.mass_concentration(subhalo_params, cosmology_params,
        subhalos_masses, z_lens, rng)
    subhalos_radii = jnp.sqrt(jnp.sum(subhalos_cart_pos ** 2, axis=-1))

    # Convert from masses and concentrations to nfw parameters
    subhalos_r_two_hund = nfw_functions.r_two_hund_from_m(cosmology_params,
        subhalos_masses, z_lens)
    subhalos_r_scale = subhalos_r_two_hund / subhalos_c
    subhalos_rho_nfw = nfw_functions.rho_nfw_from_c(cosmology_params,
        subhalos_c, z_lens)
    subhalos_r_trunc = get_truncation_radius(subhalos_masses, subhalos_radii)

    # Convert to lensing units
    lq_tuple = nfw_functions.convert_to_lensing_tnfw(cosmology_params,
        subhalos_r_scale, subhalos_z, subhalos_rho_nfw, subhalos_r_trunc,
        z_source)
    subhalos_r_scale_ang, subhlos_alpha_rs, subhalos_r_trunc_ang = lq_tuple
    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
    subhalos_cart_pos_ang = subhalos_cart_pos / kpa

    # There is only one model, the tNFW. Subhalos with mass 0 are treated
    # as padding models.
    subhalos_model_index = (jnp.full(subhalos_masses.shape, -1) *
        jnp.int32(subhalos_masses == 0))

    subhalos_kwargs = {'model_index': subhalos_model_index,
        'scale_radius': subhalos_r_scale_ang, 'alpha_rs': subhlos_alpha_rs,
        'trunc_radius': subhalos_r_trunc_ang,
        # Assume the first main deflector model sets the center.
        'center_x': (
            subhalos_cart_pos_ang[:,0] + main_deflector_params['center_x']),
        'center_y': (
            subhalos_cart_pos_ang[:,1] + main_deflector_params['center_y'])}

    return subhalos_z, subhalos_kwargs


def draw_subhalos(main_deflector_params: Mapping[str, float],
    source_params: Mapping[str, Union[int, float]],
    subhalo_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    rng: Sequence[int], subhalos_pad_length: int, sampling_pad_length: int,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Draw subhalos with redshift and lensing quantities.

    Args:
        main_deflector_params: Parameters of the main deflector.
        source_params: Parameters of the source.
        subhalo_params: Parameters of the subhalo population.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        rng: Jax PRNG key.
        subhalos_pad_length: Number of subhalos to draw, including padding. If
            this number is too small, you may undercount the number of subhalos
            in your system.
        sampling_pad_length: Number of positional samples to draw before
            rejection sampling the subhalo positions within the volume
            boundaries. Should be one to two order of magntidue larger than the
            subhalo pad length.

    Returns:
        Redshifts for each subhalo and the dictionary of lensing quantities.

    Notes:
        Model index set to -1 for subhalos with zero mass.
    """
    rng_masses, rng_pos, rng_convert = jax.random.split(rng, 3)

    # Draw the masses and positions for our subhalos up to the pad.
    subhalos_mass = draw_nfw_masses(main_deflector_params, subhalo_params,
        cosmology_params, rng_masses, subhalos_pad_length)
    subhalos_cart_pos = sample_cored_nfw(main_deflector_params, subhalo_params,
        cosmology_params, rng_pos, subhalos_pad_length, sampling_pad_length)

    # Return our lensing keyword dictionary and array of redshifts.
    return convert_to_lensing(main_deflector_params, source_params,
        subhalo_params, cosmology_params, subhalos_mass, subhalos_cart_pos,
        rng_convert)
