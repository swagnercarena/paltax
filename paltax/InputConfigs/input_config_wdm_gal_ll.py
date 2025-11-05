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
"""Configuration file for warm dark matter with galaxy-like lens light.
"""
import pathlib

import jax.numpy as jnp

from paltax import source_models
from paltax.InputConfigs import input_config_wdm
from paltax.input_pipeline import encode_constant, encode_normal, encode_uniform

def get_config():
    """Get the hyperparameter configuration"""
    config = input_config_wdm.get_config()

    # Change the lens light model to pull from the COSMOS catalog.
    config['lensing_config']['lens_light_params'] = {
        'galaxy_index': encode_uniform(minimum=0.0, maximum=1.0),
        'output_ab_zeropoint': encode_constant(25.127),
        'catalog_ab_zeropoint': encode_constant(25.127),
        'z_source': encode_constant(0.5),
        'amp': encode_uniform(minimum=6.0, maximum=16.0),
        'angle': encode_uniform(minimum=0.0, maximum=2 * jnp.pi),
        'center_x': encode_normal(mean=0.0, std=0.16),
        'center_y': encode_normal(mean=0.0, std=0.16)
    }
    cosmos_path = str(pathlib.Path(__file__).parent.parent.parent)
    cosmos_path += '/datasets/cosmos/ddprism_v1.h5'
    config['all_models']['all_lens_light_models'] = (
        source_models.CosmosCatalog(cosmos_path),
    )

    config['truth_parameters'] = (
        [
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'main_deflector_params', 'main_deflector_params',
            'source_params', 'source_params', 'source_params',
            'lens_light_params', 'lens_light_params', 'lens_light_params',
            'subhalo_params', 'subhalo_params', 'subhalo_params'
        ],
        [
            'theta_e', 'slope', 'center_x', 'center_y', 'ellip_x', 'ellip_xy',
            'gamma_one', 'gamma_two', 'amp', 'center_x', 'center_y', 'amp',
            'center_x', 'center_y', 'sigma_sub', 'shmf_plaw_index', 'log_m_hm'
        ],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    return config
