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
"""Run hierarchical inference code."""

import os

from absl import app
from absl import flags
import emcee
import numba
import numpy as np

from paltax import hierarchical_inference


FLAGS = flags.FLAGS
flags.DEFINE_integer('n_lenses', 1, 'number of lenses.')
flags.DEFINE_string('backend_path', None,
                    'path to which to save chains backend.')
flags.DEFINE_string('npz_path', None,
                    'path to numpy file with mean and precision predictions.')
flags.DEFINE_integer('n_walkers', 40, 'number of walkers of emcee.')
flags.DEFINE_integer('n_emcee_samps', 10000, 'number of emcee samples to draw.')


def main(_):
    """Run hierarchical inference with configuration defined by flags."""
    n_lenses = FLAGS.n_lenses
    backend_path = FLAGS.backend_path
    n_walkers = FLAGS.n_walkers
    n_emcee_samps = FLAGS.n_emcee_samps

    npz_file = np.load(FLAGS.npz_path)
    mean_pred = npz_file['mean_pred'][:n_lenses]
    prec_pred = npz_file['prec_pred'][:n_lenses]

    # Hardcoded interim prior.
    train_mean = 2.0e-3
    train_std = 5.5e-3
    mu_omega_i = np.array([train_mean])
    cov_omega_i = np.diag(np.array([train_std]) ** 2)

    # A prior function that mainly just bounds the uncertainty estimation.
    @numba.njit()
    def eval_func_omega(hyperparameters):
        # Enforce that the SHMF normalization is not negative
        if hyperparameters[0] < -6e-3 or hyperparameters[0] > 6e-3:
            return -np.inf
        # Need to set bounds to avoid random singular matrix proposals
        if hyperparameters[1] < -12 or hyperparameters[1] > np.log(1e-3):
            return -np.inf
        return 0

    # Initialize our class and then give it the network predictions.
    # These are set to global variables in case you want to use
    # pooling.
    prob_class = hierarchical_inference.ProbabilityClassAnalytical(
        mu_omega_i, cov_omega_i, eval_func_omega
    )
    prob_class.set_predictions(mu_pred_array=mean_pred,
                               prec_pred_array=prec_pred)

    n_dim = 2
    initial_std = np.array([1e-3, 1])
    cur_state = (np.random.rand(n_walkers, n_dim) * 2 - 1) * initial_std
    cur_state += np.concatenate(
        [mu_omega_i, np.log(np.diag(np.sqrt(cov_omega_i)))]
    )

    # If backend is already initialized, do not override the current state.
    if os.path.exists(backend_path):
        cur_state = None
    backend = emcee.backends.HDFBackend(backend_path)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                    prob_class.log_post_omega,backend=backend)

    sampler.run_mcmc(cur_state, n_emcee_samps, progress=True)


if __name__ == '__main__':
    app.run(main)
