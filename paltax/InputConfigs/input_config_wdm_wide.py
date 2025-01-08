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
"""Configuration file for warm dark matter with wide prior.
"""

from paltax.InputConfigs import input_config_wdm
from paltax.input_pipeline import encode_constant, encode_normal

def get_config():
    """Get the hyperparameter configuration"""
    config = input_config_wdm.get_config()

    config['lensing_config']['subhalo_params']['sigma_sub'] = (
        encode_normal(mean=2.0e-3, std=2.2e-3)
    )
    config['lensing_config']['subhalo_params']['log_m_hm'] = (
        encode_normal(mean=8.0, std=2.0)
    )
    config['lensing_config']['subhalo_params']['shmf_plaw_index'] = (
        encode_constant(constant=-1.9)
    )

    config['truth_parameters'] = (
        [
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'source_params', 'source_params', 'subhalo_params',
            'subhalo_params'
        ],
        [
            'theta_e', 'slope', 'center_x', 'center_y', 'ellip_x', 'ellip_xy',
            'gamma_one', 'gamma_two', 'center_x', 'center_y', 'sigma_sub',
            'log_m_hm'
        ],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    )

    return config
