# coding=utf-8

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
"""Configuration file for warm dark matter.
"""

from paltax.InputConfigs import input_config_br
from paltax.input_pipeline import encode_constant, encode_normal

def get_config():
    """Get the hyperparameter configuration"""
    config = input_config_br.get_config()

    mass_conc_params = {
        'c_zero': encode_normal(mean=16.0, std=2.0),
        'conc_zeta': encode_normal(mean=-0.3, std=0.1),
        'conc_beta': encode_normal(mean=0.55, std=0.3),
        'conc_m_ref': encode_constant(1e8),
        'conc_dex_scatter': encode_normal(mean=0.1, std=0.06),
    }

    config['lensing_config']['subhalo_params'] = {
        'sigma_sub': encode_normal(mean=2.0e-3, std=1.1e-3),
        'log_m_hm': encode_normal(mean=8.0, std=1.0),
        'shmf_plaw_index': encode_normal(mean=-1.9, std=0.05),
        'm_pivot': encode_constant(1e10),
        'm_min': encode_constant(1e7),
        'm_max': encode_constant(1e10),
        'k_one': encode_constant(0.0),
        'k_two': encode_constant(0.0),
    }
    config['lensing_config']['subhalo_params'].update(mass_conc_params)

    config['kwargs_simulation'] = {
        'num_z_bins': 1000,
        'los_pad_length': 10,
        'subhalos_pad_length': 8192,
        'subhalos_n_chunks': 32,
        'sampling_pad_length': 2000000,
    }

    # Control how often cosmology parameters are updated by source models
    # Set to 1 to update every step, higher values update less frequently
    config['cosmology_update_frequency'] = 32

    config['truth_parameters'] = (
        [
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'source_params', 'source_params', 'subhalo_params',
            'subhalo_params', 'subhalo_params'
        ],
        [
            'theta_e', 'slope', 'center_x', 'center_y', 'ellip_x', 'ellip_xy',
            'gamma_one', 'gamma_two', 'center_x', 'center_y', 'sigma_sub',
            'shmf_plaw_index', 'log_m_hm'
        ],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    )

    return config
