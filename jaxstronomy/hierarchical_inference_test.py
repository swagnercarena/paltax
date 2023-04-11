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
"""Tests for hierarchical_inference.py.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numba
import numpy as np
from scipy import stats
from scipy import integrate
from jaxstronomy import hierarchical_inference


def _prepare_means_precisions(problem):
    # Return different combinations of precision matrices and means to test.
    mu_pred = np.ones(2)
    prec_pred = np.identity(2)
    mu_omega_i = np.ones(2)
    prec_omega_i = np.identity(2)
    mu_omega = np.ones(2)
    prec_omega = np.identity(2)

    if problem == 'cov':
        prec_omega_i *= 0.5

    if problem == 'mean_shift':
        prec_omega_i *= 0.5
        mu_omega_i *= 2
        mu_pred *= 0.5

    if problem == 'cov_complex':
        mu_omega_i *= 2
        mu_pred *= 0.5
        prec_pred = np.array([[1, 0.3], [0.3, 1]])
        prec_omega_i = np.array([[1, -0.3], [-0.3, 1]])
        prec_omega = np.array([[10, 0.05], [0.05, 10]])

    if problem == 'invalid':
        prec_pred = np.array([[1, 0.8], [0.8, 1]])
        prec_omega_i = np.array([[1, 0.0], [0.0, 1]])
        prec_omega = np.array([[0.5, 0.05], [0.05, 0.5]])

    return ([mu_pred, mu_omega, mu_omega_i],
            [prec_pred, prec_omega, prec_omega_i])


class HierarchicalInference(parameterized.TestCase):
    """Runs tests of various hierarchical inference functions."""

    def test_log_p_omega(self):
        # Test that the bounds of the evaluation funciton are returned.
        hyperparameters = np.array([1, 0.2])

        # Test that it works with numba
        @numba.njit()
        def eval_func_omega(hyperparameters):
            if hyperparameters[0] < 0:
                return np.nan
            else:
                return hyperparameters[0]*hyperparameters[1]

        self.assertEqual(hierarchical_inference.log_p_omega(
            hyperparameters, eval_func_omega), 0.2)

        hyperparameters = np.array([-1,0.2])
        self.assertEqual(hierarchical_inference.log_p_omega(
            hyperparameters, eval_func_omega), -np.inf)

    @parameterized.named_parameters(
            [(f'{problem}',problem) for problem in
             ['identity', 'cov', 'mean_shift', 'cov_complex', 'invalid']])
    def test_gaussian_product_analytical(self, problem):
        # Compare analytic results with numerical integration.
        means, precisions = _prepare_means_precisions(problem)
        mu_pred, mu_omega, mu_omega_i = means
        prec_pred, prec_omega, prec_omega_i = precisions

        # Build the scipy function we will call for integration
        def scipy_integrand(y, x, pred_pdf, omega_pdf, omega_i_pdf):
            array = np.array([x, y])
            integrand = pred_pdf.pdf(array) * omega_pdf.pdf(array)
            integrand /= omega_i_pdf.pdf(array)
            return integrand

        # Use scipy for the pdfs as well
        pred_pdf = stats.multivariate_normal(mean=mu_pred,
                                             cov=np.linalg.inv(prec_pred))
        omega_pdf = stats.multivariate_normal(mean=mu_omega,
                                              cov=np.linalg.inv(prec_omega))
        omega_i_pdf = stats.multivariate_normal(mean=mu_omega_i,
                                                cov=np.linalg.inv(prec_omega_i))

        if problem == 'invalid':
            self.assertAlmostEqual(
                -np.inf,
                hierarchical_inference.gaussian_product_analytical(
                    mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,
                    prec_omega)
            )
        else:
            numerical = integrate.dblquad(
                scipy_integrand, -20, 20, -20, 20,
                args=(pred_pdf, omega_pdf, omega_i_pdf))[0]
            self.assertAlmostEqual(
                np.log(numerical),
                hierarchical_inference.gaussian_product_analytical(
                    mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,
                    prec_omega)
            )


if __name__ == '__main__':
    absltest.main()
