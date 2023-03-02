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
"""Define functions useful for conversion between different NFW conventions.

This module contains the functions used to move between NFW conventions and
to transform NFW parameters into lensing inputs.
"""

from typing import Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from jaxstronomy import cosmology_utils


def r_two_hund_from_m(
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    masses: Union[jnp.ndarray, float], z: float) -> Union[float, jnp.ndarray]:
    """Calculate the two-hundred radial overdensity from the mass.

    Args:
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        masses: Mass values to calculate the overdensity radius for.
        z: Redshift at which to conduct the calculation.

    Returns:
        The overdensity radius corresponding to each provided mass.
    """
    # Calculate the radius using the critical density at the given redshift.
    h = cosmology_params['hubble_constant'] / 100.0
    rho_crit = cosmology_utils.rho_crit(cosmology_params, z) * h ** 2
    return (3 * masses / (4 * jnp.pi * rho_crit * 200)) ** (1.0 / 3.0)


def rho_nfw_from_c(
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    c: Union[jnp.ndarray, float], z: float):
    """Calculate the amplitude of the NFW profile.

    Args:
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        c: Concentration of the NFW.
        z: Redshift at which to define the amplitude

    Returns:
        Amplitude of the NFW
    """
    # Calculate the radius using the critical density at the given redshift.
    h = cosmology_params['hubble_constant'] / 100.0
    rho_crit = cosmology_utils.rho_crit(cosmology_params, z) * h ** 2
    return (c ** 3 * rho_crit * 200) / (3 * (jnp.log(1 + c) - c / (1+c)))


def _cored_nfw_integral(r_tidal: float, rho_nfw: float, r_scale: float,
    r_upper: jnp.ndarray) -> jnp.ndarray:
    """Integrate the cored NFW profile.

    Args:
        r_tidal: Tidal radius of the NFW profile. Units of kpc.
        rho_nfw: Normalization of the NFW profile.
        r_scale: Scale radius of NFW profile. Units of kpc.
        r_upper: Upper bounds to evaluate in units of kpc.

    Returns:
        Value of the integral for each r_upper.
    """
    # Convert to natural units for NFW
    x_tidal = r_tidal / r_scale
    x_upper = r_upper / r_scale

    # Get the NFW value in the core.
    linear_scaling = rho_nfw / (x_tidal * (1 + x_tidal) ** 2)

    # Calculate the cored component of the integral
    integral_values = (1.0 / 3.0 * linear_scaling *
        jnp.minimum(r_tidal,r_upper)**3)

    # Add the nfw component for x_upper > x_tidal
    lower_bound = 1 / (x_tidal + 1) + jnp.log(x_tidal + 1)
    upper_bound = 1 / (x_upper + 1) + jnp.log(x_upper + 1)
    nfw_integral = upper_bound - lower_bound
    add_nfw = r_upper > r_tidal
    integral_values += nfw_integral * rho_nfw * r_scale**3 * add_nfw

    return integral_values * 4 * jnp.pi


def cored_nfw_draws(r_tidal: float, rho_nfw: float, r_scale: float,
    r_max: float, rng: Sequence[int], n_draws: float) -> jnp.ndarray:
    """Return radial samples from a cored nfw profile.

    Args:
        r_tidal: Tidal radius of the NFW
        rho_nfw: Amplitude of the NFW
        r_scale: Scale radius of the NFW
        r_max: Maximum radius to draw halos within
        rng: Jax PRNG key.
        n_draws: Number of positions to draw.

    Returns:
        Radial position draws.
    """
    # Default to 1000 values for the inverse cdf interpolation.
    r_values = jnp.linspace(0, r_max, 1000)
    cdf_values = _cored_nfw_integral(r_tidal, rho_nfw, r_scale, r_values)

    # Normalize
    cdf_values /= jnp.max(cdf_values)

    # Interpolate the inverse cdf function and use that to draw radial
    # samples
    cdf_draws = jax.random.uniform(rng, shape=(n_draws,))
    return jnp.interp(cdf_draws, cdf_values, r_values)


def convert_to_lensing_nfw(
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    r_scale: float, z: float, rho_nfw: float, z_source: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert physical NFW parameters to lensing parameters.

    Args:
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        r_scale: Scale radius of the NFW.
        z: Redshift of the NFW.
        rho_nfw: Amplitude of the NFW.
        z_source: Redshift of the source.

    Returns:
        The lensing scale radius and the deflection strength at the scale
            radius.
    """
    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z)

    # Two of the parameters just need to be transformed to arcseconds.
    r_scale_ang = r_scale / kpa

    # Lensing alpha is defined with respect to sigma_crit.
    sigma_crit = cosmology_utils.calculate_sigma_crit(cosmology_params, z,
        z_source)
    alpha_rs = rho_nfw * (4 * r_scale ** 2 * (1 + jnp.log(0.5)))
    alpha_rs /= kpa * sigma_crit

    return r_scale_ang, alpha_rs


def convert_to_lensing_tnfw(
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    r_scale: float, z: float, rho_nfw: float, r_trunc: float, z_source: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert physical tNFW parameters to lensing parameters.

    Args:
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        r_scale: Scale radius of the tNFW.
        z: Redshift of the tNFW.
        rho_nfw: Amplitude of the tNFW.
        r_trunc: Truncation radius of the tNFW.
        z_source: Redshift of the source.

    Returns:
        The lensing scale radius, the deflection strength at the scale
            radius, and the truncation radius.
    """
    kpa = cosmology_utils.kpc_per_arcsecond(cosmology_params, z)

    # First two parameters can be transformed as for an NFW.
    r_scale_ang, alpha_rs = convert_to_lensing_nfw(cosmology_params, r_scale, z,
        rho_nfw, z_source)

    # Truncation radius just needs to be converted to radians.
    r_trunc_ang = r_trunc / kpa

    return r_scale_ang, alpha_rs, r_trunc_ang


def mass_concentration(substructure_params: Mapping[str, float],
    cosmology_params: Mapping[str, Union[float, int, jnp.ndarray]],
    masses: Union[float, jnp.ndarray], z: float,
    rng: Sequence[int]) -> jnp.ndarray:
    """Return the concentration of halos at a given mass and redshift.

    Args:
        substructure_params: Parameters of the substructure distribution.
        cosmology_params: Cosmological parameters that define the universe's
            expansion.
        masses: Masses to pull concentrations for. In units of M_sun.
        z: Redshift of the halos.
        rng: Jax PRNG key.

    Returns:
        Concentration for each halo.
    """
    # Pull out the mass-concentration parameters we need.
    c_zero = substructure_params['c_zero']
    zeta = substructure_params['conc_zeta']
    beta = substructure_params['conc_beta']
    m_ref = substructure_params['conc_m_ref']
    dex_scatter = substructure_params['conc_dex_scatter']

    # Use peak heights to calculate concentrations
    h = cosmology_params['hubble_constant'] / 100
    peak_heights = jax.vmap(cosmology_utils.peak_height,
        in_axes=[None, 0, None])(cosmology_params, masses * h, z)
    peak_height_ref = cosmology_utils.peak_height(cosmology_params, m_ref * h,
        0.0)
    concentrations = (c_zero * (1+z) ** zeta *
        (peak_heights / peak_height_ref) ** (-beta))

    # Add scatter and return concentrations
    conc_scatter = jax.random.normal(rng, shape=concentrations.shape)
    conc_scatter *= dex_scatter
    return 10**(jnp.log10(concentrations) + conc_scatter)
