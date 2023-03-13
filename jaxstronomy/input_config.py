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
"""Configuration file for generating jaxstronomy image outputs.
"""

import jax
import jax.numpy as jnp

from jaxstronomy.input_pipeline import encode_normal, encode_uniform
from jaxstronomy.input_pipeline import encode_constant
from jaxstronomy import lens_models
from jaxstronomy import psf_models
from jaxstronomy import source_models


def get_config():
    """Return the config for input generation.
    """
    config = {}

    # Specify the configuration for each one of the parameters.
    config['lensing_config'] = {
        'los_params':{
            'delta_los': encode_constant(0.0),
            'r_min': encode_constant(0.5),
            'r_max': encode_constant(10.0),
            'm_min': encode_constant(1e7),
            'm_max': encode_constant(1e10),
            'dz': encode_constant(0.01),
            'cone_angle': encode_constant(8.0),
            'angle_buffer': encode_constant(0.8),
            'c_zero': encode_constant(18),
            'conc_zeta': encode_constant(-0.2),
            'conc_beta': encode_constant(0.8),
            'conc_m_ref': encode_constant(1e8),
            'conc_dex_scatter': encode_constant(0.1)
        },
        'main_deflector_params': {
            'mass': encode_constant(1e13),
            'z_lens': encode_constant(0.5),
            'theta_e': encode_normal(mean=1.1, std=0.15),
            'slope': encode_normal(mean=2.0, std=0.1),
            'center_x': encode_normal(mean=0.0, std=0.16),
            'center_y': encode_normal(mean=0.0, std=0.16),
            'axis_ratio': encode_normal(mean=1.0, std=0.1),
            'angle': encode_uniform(minimum=0.0, maximum=2 * jnp.pi),
            'gamma_ext': encode_normal(mean=0.0, std=0.1)
        },
        'subhalo_params':{
            'sigma_sub': encode_normal(mean=2.0e-3, std=1.1e-3),
            'shmf_plaw_index': encode_uniform(minimum=-2.02, maximum=-1.92),
            'm_pivot': encode_constant(1e10),
            'm_min': encode_constant(7e7),
            'm_max': encode_constant(1e10),
            'k_one': encode_constant(0.0),
            'k_two': encode_constant(0.0),
            'c_zero': encode_constant(18),
            'conc_zeta': encode_constant(-0.2),
            'conc_beta': encode_constant(0.8),
            'conc_m_ref': encode_constant(1e8),
            'conc_dex_scatter': encode_constant(0.1)
        },
        'source_params':{
            'galaxy_index': encode_uniform(minimum=0.0, maximum=1.0),
            'output_ab_zeropoint': encode_constant(25.0),
            'catalog_ab_zeropoint': encode_constant(25.0),
            'z_source': encode_constant(1.5),
            'amp': encode_uniform(minimum=1.0, maximum=10.0),
            'sersic_radius': encode_uniform(minimum=1.0, maximum=3.0),
            'n_sersic': encode_uniform(minimum=1.0, maximum=1.5),
            'axis_ratio': encode_normal(mean=1.0, std=0.05),
            'angle': encode_uniform(minimum=0.0, maximum=2 * jnp.pi),
            'center_x': encode_normal(mean=0.0, std=0.16),
            'center_y': encode_normal(mean=0.0, std=0.16)
        },
        'lens_light_params':{
            'amp': encode_constant(0.0),
            'sersic_radius': encode_uniform(minimum=1.0, maximum=3.0),
            'n_sersic': encode_uniform(minimum=1.0, maximum=4.0),
            'axis_ratio': encode_normal(mean=1.0, std=0.05),
            'angle': encode_uniform(minimum=0.0, maximum=2 * jnp.pi),
            'center_x': encode_normal(mean=0.0, std=0.16),
            'center_y': encode_normal(mean=0.0, std=0.16)
        }
    }

    # The remaining parameters should not be drawn from random distributions.
    config['kwargs_detector'] = {
        'n_x': 128, 'n_y': 128, 'pixel_width': 0.04, 'supersampling_factor': 2,
        'exposure_time': 1024, 'num_exposures': 2.0, 'sky_brightness': 22,
        'magnitude_zero_point': 25, 'read_noise': 3.0
    }
    cosmos_path = '/scratch/users/swagnerc/datasets/cosmos/cosmos_galaxies_train.npz'
    config['all_models'] = {
        'all_los_models': (lens_models.NFW(),),
        'all_subhalo_models': (lens_models.TNFW(),),
        'all_main_deflector_models': (lens_models.EPL(), lens_models.Shear()),
        'all_source_models': (source_models.CosmosCatalog(cosmos_path),),
        'all_lens_light_models': (source_models.SersicElliptic(),),
        'all_psf_models': (psf_models.Gaussian(),)
    }
    config['cosmology_params'] = {
        'omega_m_zero': encode_constant(0.3089),
        'omega_b_zero': encode_constant(0.0486),
        'omega_de_zero': encode_constant(0.6910088292453472),
        'omega_rad_zero': encode_constant(9.117075466e-5),
        'temp_cmb_zero': encode_constant(2.7255),
        'hubble_constant': encode_constant(67.74),
        'n_s': encode_constant(0.9667),
        'sigma_eight': encode_constant(0.815)
    }
    config['kwargs_simulation'] = {
        'num_z_bins': 1000,
        'los_pad_length': 10,
        'subhalos_pad_length': 750,
        'sampling_pad_length': 200000,
    }

    config['rng'] = jax.random.PRNGKey(0)

    config['kwargs_psf'] = {'model_index': 0, 'fwhm': 0.04, 'pixel_width': 0.02}

    config['principal_md_index'] = 0
    config['principal_source_index'] = 0
    config['truth_parameters'] = (
        ['main_deflector_params', 'main_deflector_params',
         'main_deflector_params', 'main_deflector_params', 'subhalo_params'],
        ['theta_e', 'slope', 'center_x', 'center_y', 'sigma_sub'])

    return config
