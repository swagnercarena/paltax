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
"""Tests for hierarchical_inference.py.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numba
import numpy as np
from scipy import stats
from scipy import integrate
from paltax import hierarchical_inference


def _prepare_test_gaussian_product_analytical(problem):
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

['simple', 'invalid', 'covariance']
def _perpare_test_log_post_omega(problem):
    # Return hyperparameters to test.
    if problem == 'simple':
        return np.zeros(20)
    elif problem == 'invalid':
        return -np.ones(20)
    elif problem == 'covariance':
        return np.random.rand(20)


class HierarchicalInferenceTest(parameterized.TestCase):
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
        means, precisions = _prepare_test_gaussian_product_analytical(
            problem)
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


class ProbabilityClassAnalyticalTest(parameterized.TestCase):
    """Runs tests of various ProbabilityClassAnalytical functions."""

    def setUp(self):
        np.random.seed(2)
        return super().setUp()

    def test_set_predictions(self):
        # Test that setting the samples saves it globally.
        n_lenses = 1000
        mu_pred_array_input = np.random.randn(n_lenses, 10)
        prec_pred_array_input = np.tile(
            np.expand_dims(np.identity(10), axis=0), (n_lenses, 1, 1)
        )
        mu_omega_i = np.ones(10)
        cov_omega_i = np.identity(10)
        def eval_func_omega(_):
            return 0.0

        # Establish our ProbabilityClassAnalytical
        prob_class = hierarchical_inference.ProbabilityClassAnalytical(
            mu_omega_i, cov_omega_i, eval_func_omega)

        # Try setting the predictions
        prob_class.set_predictions(mu_pred_array_input, prec_pred_array_input)
        np.testing.assert_array_almost_equal(
            hierarchical_inference.MU_PRED_ARRAY, mu_pred_array_input)
        np.testing.assert_array_almost_equal(
            hierarchical_inference.PREC_PRED_ARRAY, prec_pred_array_input)

    def test_log_integral_product(self):
        # Test that the log integral product just sums the log of each integral.
        n_lenses = 1000
        mu_pred_array = np.random.randn(n_lenses, 10)
        prec_pred_array = np.tile(
            np.expand_dims(np.identity(10), axis=0),(n_lenses, 1, 1))
        mu_omega_i = np.ones(10)
        prec_omega_i = np.identity(10)
        mu_omega = np.ones(10)
        prec_omega = np.identity(10)

        # Calculate the value by hand.
        hand_integral = 0
        for mu_pred, prec_pred in zip(mu_pred_array, prec_pred_array):
            hand_integral += hierarchical_inference.gaussian_product_analytical(
                    mu_pred, prec_pred, mu_omega_i, prec_omega_i, mu_omega,
                    prec_omega)

        # Now use the class.
        prob_class = hierarchical_inference.ProbabilityClassAnalytical
        integral = prob_class.log_integral_product(
            mu_pred_array, prec_pred_array, mu_omega_i, prec_omega_i, mu_omega,
            prec_omega)

        self.assertAlmostEqual(integral, hand_integral)

    @parameterized.named_parameters(
            [(f'{problem}',problem) for problem in
             ['simple', 'invalid', 'covariance']])
    def test_log_post_omega(self, problem):
        # Test that the log_post_omega calculation includes both the integral
        # and the prior.
        n_lenses = 1000
        mu_pred_array_input = np.random.randn(n_lenses, 10)
        prec_pred_array_input = np.tile(np.expand_dims(np.identity(10), axis=0),
            (n_lenses, 1, 1))
        mu_omega_i = np.ones(10)
        cov_omega_i = np.identity(10)
        prec_omega_i = cov_omega_i

        @numba.njit()
        def eval_func_omega(hyperparameters):
            if np.any(hyperparameters[len(hyperparameters) // 2:] < 0):
                return -np.inf
            return 0

        # Establish our ProbabilityClassAnalytical
        prob_class = hierarchical_inference.ProbabilityClassAnalytical(
            mu_omega_i, cov_omega_i, eval_func_omega
        )
        prob_class.set_predictions(mu_pred_array_input, prec_pred_array_input)

        # Test a simple array of zeros
        hyperparameters = _perpare_test_log_post_omega(problem)
        mu_omega = hyperparameters[:10]
        prec_omega = np.linalg.inv(np.diag(np.exp(hyperparameters[10:]) ** 2))

        if problem == 'invalid':
            self.assertEqual(-np.inf,prob_class.log_post_omega(hyperparameters))
        else:
            hand_calc = prob_class.log_integral_product(
                mu_pred_array_input, prec_pred_array_input, mu_omega_i,
                prec_omega_i, mu_omega, prec_omega)
            self.assertAlmostEqual(
                hand_calc, prob_class.log_post_omega(hyperparameters))


if __name__ == '__main__':
    absltest.main()
