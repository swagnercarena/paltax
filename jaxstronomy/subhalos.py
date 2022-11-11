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

from typing import Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from jaxstronomy import power_law
from jaxstronomy import cosmology_utils
from jaxstronomy import nfw_functions


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
        TODO

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


def mass_concentration(subhalo_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    masses: Union[float, jnp.ndarray], z: float,
    rng: Sequence[int]) -> jnp.ndarray:
    """Return the concentration of halos at a given mass and redshift.

    Args:
        TODO

    Returns:
        Concentration for each halo.
    """
    # Pull out the mass-concentration parameters we need.
    c_zero = subhalo_params['c_zero']
    zeta = subhalo_params['conc_zeta']
    beta = subhalo_params['conc_beta']
    m_ref = subhalo_params['conc_m_ref']
    dex_scatter = subhalo_params['conc_m_scatter']

    # Use peak heights to calculate concentrations
    h = cosmology_params['hubble_constant'] / 100
    peak_heights = jax.vmap(cosmology_params.peak_height,
        in_axes=[None, 0, None])(cosmology_params, masses * h, z)
    peak_height_ref = cosmology_params.peak_height(cosmology_params, m_ref * h,
        0.0)
    concentrations = (c_zero * (1+z) ** zeta *
        (peak_heights / peak_height_ref) ** (-beta))

    # Add scatter and return concentrations
    conc_scatter = jax.random.normal(rng, shape=concentrations.shape)
    conc_scatter *= dex_scatter
    return 10**(jnp.log10(concentrations) + conc_scatter)


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
    phi = jnp.arccos(1 - 2 * jnp.random.uniform(rng_phi,
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

    TODO
    """

    rng_host, rng_pos = jax.random.split(rng)

    # Extract the relevant host properties
    host_mass = main_deflector_params['mass']
    z_lens = main_deflector_params['z_lens']
    host_c = mass_concentration(subhalo_params, cosmology_params,
        jnp.array([host_mass]), z_lens, rng_host)
    host_r_two_hund = nfw_functions.r_two_hund_from_m(cosmology_params,
        host_mass, z_lens)
    host_r_scale = host_r_two_hund / host_c
    host_rho_nfw = nfw_functions.rho_nfw_from_c(cosmology_params, host_mass,
        host_c)

    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z_lens)
    radius_e_three = kpa * main_deflector_params['theta_e'] * 3

    # The maximum radius is the corner for the cylinder in which we're
    # rendering.
    r_max = jnp.sqrt((radius_e_three) ** 2 + host_r_two_hund ** 2)

    # Draw coordinates up to the sampling pad length.
    radial_coord = nfw_functions.cored_nfw_draws(host_r_two_hund / 2,
        host_rho_nfw, host_r_scale, r_max, sampling_pad_length)
    is_inside, cart_pos = rejection_sampling(radial_coord, host_r_two_hund,
        radius_e_three, rng_pos)

    # We'll sort the cartesian coordinates by whether or not they're inside
    # the bounds and only return n_sub coordinates. This does not guarantee
    # that all the samples returned will be within the boudns, but for
    # large enough sampling_pad_length, nearly all of them will be.
    return cart_pos[jnp.flip(jnp.argsort(is_inside))[:n_subs]]

