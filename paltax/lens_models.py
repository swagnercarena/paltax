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
"""Implementations of mass profiles for lensing.

Implementation of mass profiles for lensing closely following implementations
in lenstronomy: https://github.com/lenstronomy/lenstronomy.
"""

from typing import Dict, Mapping, Tuple, Union

import jax
import jax.numpy as jnp

from paltax import utils

__all__ = ['EPL', 'EPLEllip', 'NFW', 'ShearCart', 'Shear', 'TNFW']


class _LensModelBase():
    """Base source model.

    Provides identity implementation of convert_to_angular for all lens
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
                expansion.

        Returns:
            Modified cosmology parameters.
        """
        return cosmology_params

    @staticmethod
    def convert_to_angular(
            all_kwargs:  Mapping[str, jnp.ndarray],
            cosmology_params: Dict[str, Union[float, int, jnp.ndarray]]
        ) -> Mapping[str, jnp.ndarray]:
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


class EPL(_LensModelBase):
    """Elliptical Power Law mass profile.

    Elliptical Power Law mass profile functions, with calculation following
    those described in Tessore & Metcalf (2015) and implementation closely
    following the EPL_numba class in Lenstronomy.
    """

    parameters = (
            'theta_e', 'slope', 'axis_ratio', 'angle', 'center_x', 'center_y'
    )

    @staticmethod
    def derivatives(
        x: jnp.ndarray, y: jnp.ndarray, theta_e: float, slope: float,
        axis_ratio: float, angle: float, center_x: float, center_y: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate the derivative of the potential for the EPL mass profile.

        Args:
            x: X-coordinates at which to evaluate the derivative.
            y: Y-coordinates at which to evaluate the derivative.
            theta_e: Einstein radius of the EPL profile.
            slope: Power-law slope of the EPL profile.
            axis_ratio: Axis ratio of the major and minor axis of ellipticity.
            angle: Clockwise angle of orientation of major axis.
            center_x: X-coordinate center of the EPL profile.
            center_y: Y-coordinate cetner of the EPL profile.

        Returns:
            X- and y-component of the derivatives.
        """
        # Transform parameters to Tessore & Metcalf (2015) definition
        z = jnp.exp(-1j * angle) * ((x - center_x) + (y - center_y) * 1j)
        scale_length = theta_e * jnp.sqrt(axis_ratio)
        complex_derivative = EPL._complex_derivative(
                z.real, z.imag, scale_length, axis_ratio, slope)
        complex_derivative *= jnp.exp(1j * angle)
        return complex_derivative.real, complex_derivative.imag

    @staticmethod
    def _complex_derivative(x: jnp.ndarray, y: jnp.ndarray, scale_length: float,
                            axis_ratio: float, slope: float) -> jnp.ndarray:
        """Calculate the complex derivative for the EPL mass profile.

        Args:
            x: X-coordinates at which to evaluate the derivative.
            y: Y-coordinates at which to evaluate the derivative.
            scale_length: Scale length of the EPL mass profile (related to
                Einstein Radius by axis ratio).
            axis_ratio: Axis ratio of the major and minor axis of ellipticity.
            slope: Power-law slope of the EPL profile.

        Returns:
            Complex derivative at each (x,y) coordinate pair.
        """
        ellip_vector = x * axis_ratio + 1j * y
        ellip_radius = jnp.abs(ellip_vector)
        ellip_angle = jnp.angle(ellip_vector)
        omega = EPL._hypergeometric_series(ellip_angle, slope, axis_ratio)
        return (2 * scale_length) / (1 + axis_ratio) * jnp.nan_to_num(
                (scale_length / ellip_radius)**(slope - 2), copy=False) * omega

    @staticmethod
    def _hypergeometric_series(ellip_angle: jnp.ndarray, slope: float,
                               axis_ratio: float) -> jnp.ndarray:
        """Calculate the hypergeometric series required for the derivative.

        Args:
            ellip_angle: The elliptical angles at which to evaluate the series.
            slope: Power-law slope of the EPL profile.
            axis_ratio: Axis ratio the major and minor axis of ellipticity.

        Returns:
            The hypergeometric series sum for each angle.
        """
        flattening = (1 - axis_ratio) / (1 + axis_ratio)
        omegas = jnp.zeros_like(ellip_angle)

        # Calculate the zeroeth term in the Fourier-type series expansion.
        four_n = 1 * jnp.exp(1j * ellip_angle)
        omegas += four_n
        four_factor = -flattening * jnp.exp(2j * ellip_angle)
        # Use the ratio between terms to calculate the rest of the series.
        # Consider making the number of terms an input to the calculation.
        for n in range(1, 200):
            four_n *= (2 * n - (3 - slope)) / (2 * n + (3 - slope))
            four_n *= four_factor
            omegas += four_n

        return omegas


class EPLEllip(_LensModelBase):
    """Elliptical Power Law mass profile with ellipticity inputs.

    Elliptical Power Law mass profile functions, with calculation following
    those described in Tessore & Metcalf (2015) and implementation closely
    following the EPL_numba class in Lenstronomy.
    """

    parameters = (
            'theta_e', 'slope', 'ellip_x', 'ellip_xy', 'center_x', 'center_y'
    )

    @staticmethod
    def derivatives(
        x: jnp.ndarray, y: jnp.ndarray, theta_e: float, slope: float,
        ellip_x: float, ellip_xy: float, center_x: float, center_y: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate the derivative of the potential for the EPL mass profile.

        Args:
            x: X-coordinates at which to evaluate the derivative.
            y: Y-coordinates at which to evaluate the derivative.
            theta_e: Einstein radius of the EPL profile.
            slope: Power-law slope of the EPL profile.
            ellip_x: X-componenet of the ellipticity.
            ellip_xy: XY-componenet of the ellipticity.
            center_x: X-coordinate center of the EPL profile.
            center_y: Y-coordinate cetner of the EPL profile.

        Returns:
            X- and y-component of the derivatives.
        """
        # Convert to angle and axis ratio and call the regular EPL class.
        axis_ratio, angle = utils.ellip_to_angle(ellip_x, ellip_xy)
        return EPL.derivatives(x, y, theta_e, slope, axis_ratio, angle,
                               center_x, center_y)


class NFW(_LensModelBase):
    """Navarro Frenk White (NFW) mass profile.

    NFW mass profile functions, with implementation closely following the NFW
    class in Lenstronomy.
    """

    parameters = ('scale_radius', 'alpha_rs', 'center_x', 'center_y')

    @staticmethod
    def derivatives(
        x: jnp.ndarray, y: jnp.ndarray, scale_radius: float, alpha_rs: float,
        center_x: float, center_y: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the NFW profile derivatives.

        Args:
            x: X-coordinates at which to evaluate the derivative
            y: Y-coordinates at which to evaluate the derivative
            scale_radius: Scale radius of NFW profile.
            alpha_rs: Derivative at the scale radius
            center_x: X-coordinate center of the NFW profile.
            center_y: Y-coordinate cetner of the NFW profile.

        Returns:
            X- and y-component of the derivative.
        """
        rho_input = NFW._alpha_to_rho(alpha_rs, scale_radius)
        x_centered = x - center_x
        y_centered = y - center_y
        radius = jnp.sqrt(x_centered ** 2 + y_centered ** 2)
        return NFW._nfw_derivatives(radius, scale_radius, rho_input, x_centered,
                                    y_centered)

    @staticmethod
    def _alpha_to_rho(alpha_rs: float, scale_radius: float) -> float:
        """Return the NFW profile normalization.

        Args:
            alpha_rs: Derivative at the scale radius
            scale_radius: Scale radius of NFW profile.

        Returns:
            NFW profile normalization.

        """
        return alpha_rs / (4. * scale_radius**2 * (1. + jnp.log(0.5)))

    @staticmethod
    def _nfw_derivatives(
        radius: jnp.ndarray, scale_radius: float, rho_input: float,
        x_centered: jnp.ndarray, y_centered: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the NFW profile derivatives.

        Args:
            radius: Radii at which to evaluate the derivatives.
            scale_radius: Scale radius of NFW profile.
            rho_input: Normalization of the NFW profile.
            x_centered: X-coordinate offset from NFW center.
            y_centered: Y-coordinate offset from NFW center.

        Returns:
            X- and y-component of the derivative.
        """
        reduced_radius = radius / scale_radius
        nfw_integral = NFW._nfw_integral(reduced_radius)
        derivative_norm = 4 * rho_input * scale_radius * nfw_integral
        derivative_norm /= reduced_radius**2

        return derivative_norm * x_centered, derivative_norm * y_centered

    @staticmethod
    def _nfw_integral(reduced_radius: jnp.ndarray) -> jnp.ndarray:
        """Return analytic solution to integral of NFW profile.

        Args:
            reduced_radius: Upper limits for integrals in reduced units.

        Returns:
            Solution to integral for each provided upper limit.
        """
        solution = jnp.zeros_like(reduced_radius)

        # There are three regimes for the analytic solution, which we will
        # calculate with masks to avoid indexing.
        is_below_one = reduced_radius < 1
        is_one = reduced_radius == 1
        is_above_one = reduced_radius > 1
        solution += jnp.nan_to_num(
                jnp.log(reduced_radius / 2.) + 1 /
                jnp.sqrt(1 - reduced_radius**2) *
                jnp.arccosh(1. / reduced_radius)) * is_below_one
        solution += (1 + jnp.log(1. / 2.)) * is_one
        solution += jnp.nan_to_num(
                jnp.log(reduced_radius / 2) + 1 /
                jnp.sqrt(reduced_radius**2 - 1) *
                jnp.arccos(1. / reduced_radius)) * is_above_one

        return solution


class ShearCart(_LensModelBase):
    """Shear mass profile in Cartesian representation.

    Shear mass profile functions, with implementation closely following the
    ShearGammaPsi class in Lenstronomy.
    """

    parameters = ('gamma_one', 'gamma_two', 'zero_x', 'zero_y')

    @staticmethod
    def derivatives(
        x: jnp.ndarray, y: jnp.ndarray, gamma_one: float, gamma_two: float,
        zero_x: float, zero_y: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the shear profile derivatives.

        Args:
            x: X-coordinates at which to evaluate the derivative
            y: Y-coordinates at which to evaluate the derivative
            gamma_one: Diagonal component of shear.
            gamma_two: Off-diagonal component of shear.
            zero_x: X-coordinate where shear is 0.
            zero_y: Y-coordinate where shear is 0.

        Returns:
            X- and y-component of the derivative.
        """
        x_centered = x - zero_x
        y_centered = y - zero_y
        return (gamma_one * x_centered + gamma_two * y_centered,
                        gamma_two * x_centered - gamma_one * y_centered)


class Shear(_LensModelBase):
    """Shear mass profile.

    Shear mass profile functions, with implementation closely following the
    ShearGammaPsi class in Lenstronomy.
    """

    parameters = ('gamma_ext', 'angle', 'zero_x', 'zero_y')

    @staticmethod
    def derivatives(
        x: jnp.ndarray, y: jnp.ndarray, gamma_ext: float, angle: float,
        zero_x: float, zero_y: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the shear profile derivatives.

        Args:
            x: X-coordinates at which to evaluate the derivative
            y: Y-coordinates at which to evaluate the derivative
            gamma_ext: Strength of the shear profile.
            angle: Clockwise angle of the shear vector with respect to the
                simulation grid.
            zero_x: X-coordinate where shear is 0.
            zero_y: Y-coordinate where shear is 0.

        Returns:
            X- and y-component of the derivative.
        """
        gamma_one, gamma_two = Shear._polar_to_cartesian(gamma_ext, angle)
        return ShearCart.derivatives(x, y, gamma_one, gamma_two, zero_x, zero_y)

    @staticmethod
    def _polar_to_cartesian(
        gamma_ext: float, angle: float
    ) -> Tuple[float, float]:
        """Return the two shear components used for the shear derivative.

        Args:
            gamma_ext: Strength of the shear profile.
            angle: Clockwise angle of the shear vector with respect to the
                simulation grid.

        Returns:
            Two shear components.
        """
        return gamma_ext * jnp.cos(2 * angle), gamma_ext * jnp.sin(2 * angle)


def _nfw_function_exact(reduced_radius: jnp.ndarray) -> jnp.ndarray:
    """Evaluate NFW function.

    Args:
        reduced_radius: Upper limits for integrals in reduced units.

    Returns:
        NFW function output.
    """
    nfw_function = jnp.zeros_like(reduced_radius)

    # There are three regimes for the NFW function, which we will calculate
    # with masks to avoid indexing.
    is_below_one = reduced_radius < 1
    is_one = reduced_radius == 1
    is_above_one = reduced_radius > 1
    nfw_function += jnp.nan_to_num(
        (1 - reduced_radius**2)**-.5 *
        jnp.arctanh((1 - reduced_radius**2)**.5)) * is_below_one
    nfw_function += is_one
    nfw_function += jnp.nan_to_num(
        (reduced_radius**2 - 1)**-.5 *
        jnp.arctan((reduced_radius**2 - 1)**.5)) * is_above_one

    return nfw_function


class TNFW(NFW):
    """Truncated Navarro Frenk White (TNFW) mass profile.

    TNFW mass profile functions, with implementation closely following the TNFW
    class in Lenstronomy.
    """

    # Static lookup table to accelerate computation.
    _TNFW_LOOKUP_RADII = jnp.logspace(-3, 4, 7001)
    _TNFW_DR = 0.001
    _TNFW_LOOKUP_VALUES = _nfw_function_exact(_TNFW_LOOKUP_RADII)
    parameters = (
        'scale_radius', 'alpha_rs', 'trunc_radius', 'center_x', 'center_y'
    )

    @staticmethod
    def derivatives(
        x: jnp.ndarray, y: jnp.ndarray, scale_radius: float, alpha_rs: float,
        trunc_radius: float, center_x: float, center_y: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the TNFW profile derivatives.

        Args:
            x: X-coordinates at which to evaluate the derivative
            y: Y-coordinates at which to evaluate the derivative
            scale_radius: Scale radius of TNFW profile.
            alpha_rs: Derivative at the scale radius
            trunc_radius: Truncation radius for TNFW profile.
            center_x: X-coordinate center of the TNFW profile.
            center_y: Y-coordinate cetner of the TNFW profile.

        Returns:
            X- and y-component of the derivative.
        """
        rho_input = TNFW._alpha_to_rho(alpha_rs, scale_radius)
        x_centered = x - center_x
        y_centered = y - center_y
        radius = jnp.sqrt(x_centered**2 + y_centered**2)
        return TNFW._tnfw_derivatives(radius, scale_radius, rho_input,
                                      trunc_radius, x_centered, y_centered)

    @staticmethod
    def _tnfw_derivatives(
        radius: jnp.ndarray, scale_radius: float, rho_input: float,
        trunc_radius: float, x_centered: jnp.ndarray, y_centered: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the TNFW profile derivatives.

        Args:
            radius: Radii at which to evaluate the derivatives.
            scale_radius: Scale radius of TNFW profile.
            rho_input: Normalization of the TNFW profile.
            trunc_radius: Truncation radius for TNFW profile.
            x_centered: X-coordinate offset from TNFW center.
            y_centered: Y-coordinate offset from TNFW center.

        Returns:
            X- and y-component of the derivative.
        """
        reduced_radius = radius / scale_radius
        reduced_trunc_radius = trunc_radius / scale_radius
        tnfw_integral = TNFW._tnfw_integral(reduced_radius,
                                            reduced_trunc_radius)
        derivative_norm = 4. * rho_input * scale_radius * tnfw_integral
        derivative_norm /= reduced_radius**2

        return derivative_norm * x_centered, derivative_norm * y_centered

    @staticmethod
    def _tnfw_integral(
        reduced_radius: jnp.ndarray, reduced_trunc_radius: float
    ) -> jnp.ndarray:
        """Return analytic solution to integral of TNFW profile.

        Args:
            reduced_radius: Upper limits for integrals in reduced units.
            reduced_trunc_radius: Reduced truncation radius for TNFW profile.

        Returns:
            Solution to integral for each provided upper limit.
        """
        solution = (reduced_trunc_radius**2 + 1 + 2 *
                    (reduced_radius**2 - 1))
        solution *= TNFW._nfw_function(reduced_radius)
        solution += reduced_trunc_radius * jnp.pi + (
                reduced_trunc_radius**2 - 1) * jnp.log(reduced_trunc_radius)
        solution += jnp.sqrt(reduced_trunc_radius**2 + reduced_radius**2) * (
                -jnp.pi + TNFW._tnfw_log(reduced_radius, reduced_trunc_radius) *
                (reduced_trunc_radius**2 - 1) * reduced_trunc_radius**-1)
        solution *= reduced_trunc_radius**2 * (reduced_trunc_radius**2 + 1)**-2
        return solution

    @staticmethod
    def _tnfw_log(
        reduced_radius: jnp.ndarray, reduced_trunc_radius: float
    ) -> jnp.ndarray:
        """Evaluate log expression that appears in the TNFW calculations.

        Args:
            reduced_radius: Upper limits for integrals in reduced units.
            reduced_trunc_radius: Reduced truncation radius for TNFW profile.

        Returns:
            Log calculation output.
        """
        return jnp.log(reduced_radius * (reduced_trunc_radius +
                                         jnp.sqrt(reduced_radius**2 +
                                                  reduced_trunc_radius**2))**-1)

    @staticmethod
    def _nfw_function(reduced_radius: jnp.ndarray) -> jnp.ndarray:
        """Evaluate NFW function.

        Args:
            reduced_radius: Upper limits for integrals in reduced units.

        Returns:
            NFW function output.
        """
        # Conduct a linear interpolation between the static lookup table
        # values.
        unrounded_i = (
            jnp.log10(reduced_radius) - jnp.log10(TNFW._TNFW_LOOKUP_RADII[0])
        ) / TNFW._TNFW_DR

        lookup_i_upper = jax.lax.min(
            jnp.ceil(unrounded_i).astype(int), len(TNFW._TNFW_LOOKUP_VALUES) - 1
        )
        lookup_i_lower = jax.lax.max(jnp.floor(unrounded_i).astype(int), 0)
        frac_i = unrounded_i % 1

        # Interpolation using the lookup table.
        interpolated = (1 - frac_i) * TNFW._TNFW_LOOKUP_VALUES[lookup_i_lower]
        interpolated += (frac_i) * TNFW._TNFW_LOOKUP_VALUES[lookup_i_upper]

        return interpolated
