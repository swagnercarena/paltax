# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conduct hierarchical inference on a population of lenses.

This module contains the tools to conduct hierarchical inference on our
network posteriors. This code is modified from the inference code in the
paltas package.
"""
from typing import Any
import warnings

import numpy as np
import numba

# Global error filters for python warnings.
LINALGWARNING = True

# The predicted samples need to be set as a global variable for the pooling to
# be efficient when done by emcee. This will have shape (num_params,num_samps,
# batch_size).
PREDICT_SAMPS_HIER = None

# As with the predicted samples, the predicted mu and cov for the analytical
# calculations should also be set at the global level for optimal
# performance.
MU_PRED_ARRAY = None
PREC_PRED_ARRAY = None


def log_p_omega(hyperparameters: np.array, eval_func_omega: Any) -> float:
    """Calculate log p(omega) - the probability of the hyperparameters.

    Args:
        hyperparameters: Proposed hyperparameters for the population level lens
            parameter distribution.
        eval_func_omega: Mapping from (hyperparameters) to value of log
            p(omega).

    Returns:
        Value of log p(omega)
    """
    # Check for nans.
    logpdf = eval_func_omega(hyperparameters)

    if np.sum(np.isnan(logpdf))>0:
        logpdf = -np.inf

    return logpdf


@numba.njit
def gaussian_product_analytical(
    mu_pred: np.array, prec_pred: np.array, mu_omega_i: np.array,
    prec_omega_i: np.array, mu_omega,prec_omega: np.array) -> float:  # pragma: no cover
    """Calculate the log of the integral of importance sampling ratio.

    Calculate the log of the integral of p(xi_k|omega)*p(xi_k|d_k,omega_int)/
    p(xi_k|omega_int) when all three pdfs are Gaussian.

    Args:
        mu_pred: Mean output by the network
        prec_pred: Precision matrix output by the network
        mu_omega_i: Mean output of the interim prior
        prec_omega_i: Precision matrix of the interim prior
        mu_omega: Mean of the proposed hyperparameter posterior.
        prec_omega: Precision matrix of the proposed hyperparameter posterior.

    Returns:
        Log of the product of the three Gaussian integrated over all space.

    Notes:
        The equation used here breaks down when the combination of precision
        matrices does not yield a valid precision matrix. When this happen, the
        output will be -np.inf.
    """
    # Calculate the values of eta and the combined precision matrix
    prec_comb = prec_pred+prec_omega-prec_omega_i

    # The combined matrix is not guaranteed to be a valid precision matrix.
    # In those cases, return -np.inf. To check the matrix is positive definite,
    # we check that is symmetric and that its Cholesky decomposition exists
    # (see https://stackoverflow.com/questions/16266720).
    if not np.array_equal(prec_comb, prec_comb.T):
        return -np.inf
    try:
        np.linalg.cholesky(prec_comb)
    except Exception:  # LinAlgError, but numba can't match exceptions
        return -np.inf

    cov_comb = np.linalg.inv(prec_comb)
    eta_pred = np.dot(prec_pred,mu_pred)
    eta_omega_i = np.dot(prec_omega_i,mu_omega_i)
    eta_omega = np.dot(prec_omega,mu_omega)
    eta_comb = eta_pred + eta_omega - eta_omega_i

    # Now calculate each of the terms in our exponent
    exponent = 0
    exponent -= np.log(abs(np.linalg.det(prec_pred)))
    exponent -= np.log(abs(np.linalg.det(prec_omega)))
    exponent += np.log(abs(np.linalg.det(prec_omega_i)))
    exponent += np.log(abs(np.linalg.det(prec_comb)))
    exponent += np.dot(mu_pred.T,np.dot(prec_pred,mu_pred))
    exponent += np.dot(mu_omega.T,np.dot(prec_omega,mu_omega))
    exponent -= np.dot(mu_omega_i.T,np.dot(prec_omega_i,mu_omega_i))
    exponent -= np.dot(eta_comb.T,np.dot(cov_comb,eta_comb))

    return -0.5*exponent


class ProbabilityClassAnalytical:
    """Class for the hierarchical inference probability calculations.

    A class for the hierarchical inference probability calculations that
    works analytically for the case of Gaussian outputs, priors, and target
    distributions.

    Args:
        mu_omega_i: Mean of each parameters in the training distribution.
        cov_omega_i: Covariance matrix for the training distribution.
        eval_func_omega: Mapping from (hyperparameters) to value of log
            p(omega).
    """
    def __init__(self: Any, mu_omega_i: np.array, cov_omega_i: np.array,
                 eval_func_omega: np.array):
        # Save each parameter to the class
        self.mu_omega_i = mu_omega_i
        self.cov_omega_i = cov_omega_i
        # Store the precision matrix for later use.
        self.prec_omega_i = np.linalg.inv(cov_omega_i)
        self.eval_func_omega = eval_func_omega

        # A flag to make sure the prediction values are set
        self.predictions_init = False

    def set_predictions(self: Any, mu_pred_array: np.array,
                        prec_pred_array: np.array):
        """Set the global lens mean and covariance prediction values.

        Args:
            mu_pred_array Mean network prediction on each lens.
            prec_pred_array: Predicted precision matrix on each lens.
        """
        # Call up the globals and set them.
        global MU_PRED_ARRAY
        global PREC_PRED_ARRAY
        MU_PRED_ARRAY = mu_pred_array
        PREC_PRED_ARRAY = prec_pred_array

        # Set the flag for the predictions being initialized
        self.predictions_init = True

    @staticmethod
    @numba.njit
    def log_integral_product(
        mu_pred_array: np.array, prec_pred_array: np.array,
        mu_omega_i: np.array, prec_omega_i: np.array,
        mu_omega: np.array, prec_omega: np.array) -> float:  # pragma: no cover
        """Calculate the log importance sampling integral over all lenses.

        For the case of Gaussian distributions, calculate the log of the
        integral p(xi_k|omega)*p(xi_k|d_k,omega_int)/p(xi_k|omega_int) summed
        over all of the lenses in the sample.

        Args:
            mu_pred_array: Mean output by the network for each lens
            prec_pred_array: Precision matrix output by the network for each
                lens.
            mu_omega_i: Mean of the interim prior.
            prec_omega_i: Precision matrix of the interim prior.
            mu_omega: Mean of the proposed hyperparameter posterior.
            prec_omega: Precision matrix of the proposed hyperparameter
                posterior.

        Returns:
            Log importance sampling integral over all of the lenses.
        """
        # In log space, the product over lenses in the posterior becomes a sum
        integral = 0
        for mu_pred, prec_pred in zip(mu_pred_array, prec_pred_array):
            integral += gaussian_product_analytical(
                mu_pred, prec_pred, mu_omega_i,prec_omega_i,mu_omega,prec_omega)

        # Treat nan as probability 0.
        if np.isnan(integral):
            integral = -np.inf

        return integral

    def log_post_omega(self: Any, hyperparameters: np.array):
        """Calculate the log posterior of a specific distribution.

        Args:
            hyperparameters: Proposed hyperparameters describing the population
                level lens parameter distribution omega. Should be twice the
                number of parameters, the first half being the mean and the
                second half being the log of the standard deviation for each
                parameter.

        Returns:
            Log posterior of omega given the predicted samples.
        """

        if self.predictions_init is False:
            raise RuntimeError('Must set predictions or behaviour is '
                +'ill-defined.')

        global MU_PRED_ARRAY
        global PREC_PRED_ARRAY
        global LINALGWARNING

        # Start with the prior on omega
        lprior = log_p_omega(hyperparameters, self.eval_func_omega)

        # No need to evaluate the samples if the proposal is outside the prior.
        if lprior == -np.inf:
            return lprior

        # Extract mu_omega and prec_omega from the provided hyperparameters
        mu_omega = hyperparameters[:len(hyperparameters) // 2]
        cov_omega = np.diag(
            np.exp(hyperparameters[len(hyperparameters) // 2:] * 2)
        )
        try:
            prec_omega = np.linalg.inv(cov_omega)
        except np.linalg.LinAlgError:
            # Singular covariance matrix
            if LINALGWARNING:
                warnings.warn('Singular covariance matrix',
                    category=RuntimeWarning)
                LINALGWARNING = False
            return -np.inf

        try:
            like_ratio = self.log_integral_product(
                MU_PRED_ARRAY, PREC_PRED_ARRAY, self.mu_omega_i,
                self.prec_omega_i,mu_omega,prec_omega)
        except np.linalg.LinAlgError:
            # Something else was singular, too bad
            if LINALGWARNING:
                warnings.warn('Singular covariance matrix',
                    category=RuntimeWarning)
                LINALGWARNING = False
            return -np.inf

        # Return the likelihood and the prior combined
        return lprior + like_ratio
