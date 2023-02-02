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
"""Code for drawing batches of example images used to train a neural
network.
"""

import functools

import jax.numpy as jnp
import jax
from jaxstronomy import cosmology_utils
from jaxstronomy import los
from jaxstronomy import subhalos
from jaxstronomy import image_simulation
from jaxstronomy import utils
from jaxstronomy import lens_models
from jaxstronomy import source_models
from jaxstronomy import psf_models

# Most important parameters
rng = jax.random.PRNGKey(0)

# Set the simulation parameters we will keep static.
los_params = {'delta_los': 1.1, 'r_min':0.5, 'r_max':10.0, 'm_min': 1e8,
    'm_max': 1e10, 'dz':0.1, 'cone_angle': 8.0, 'angle_buffer': 0.8,
    'c_zero': 18, 'conc_zeta': -0.2, 'conc_beta': 0.8, 'conc_m_ref': 1e8,
    'conc_dex_scatter': 0.0}
cosmology_params_init = {'omega_m_zero': 0.3089, 'omega_b_zero': 0.0486,
    'omega_de_zero': 0.6910088292453472, 'omega_rad_zero': 9.117075466e-5,
    'temp_cmb_zero': 2.7255, 'hubble_constant': 67.74, 'n_s': 0.9667,
    'sigma_eight': 0.8159}
kwargs_detector = {'n_x': 124, 'n_y': 124, 'pixel_width': 0.04,
    'supersampling_factor': 2, 'exposure_time': 1024, 'num_exposures': 2.0,
    'sky_brightness': 22, 'magnitude_zero_point': 25,
    'read_noise': 3.0}
# TODO the bounds on the cosmology lookup table are being set manually here.
cosmology_params = cosmology_utils.add_lookup_tables_to_cosmology_params(
    cosmology_params_init, 1.5, los_params['dz'] / 2, 1e-4, 1e3, 2)
grid_x, grid_y = utils.coordinates_evaluate(kwargs_detector['n_x'],
    kwargs_detector['n_y'], kwargs_detector['pixel_width'],
    kwargs_detector['supersampling_factor'])
# TODO using a very simple PSF
kwargs_psf = {'model_index': 0, 'fwhm': 0.04, 'pixel_width': 0.02}
r_min = cosmology_utils.lagrangian_radius(cosmology_params, los_params['m_min'] / 10)
r_max = cosmology_utils.lagrangian_radius(cosmology_params, los_params['m_max'] * 10)
cosmology_params = cosmology_utils.add_lookup_tables_to_cosmology_params(
    cosmology_params_init, 1.5, los_params['dz'] / 2, r_min, r_max, 1000)
cosmology_params = los.add_los_lookup_tables_to_cosmology_params(los_params,
    cosmology_params, 1.5)

all_models = {'all_los_models': (lens_models.NFW,),
    'all_subhalo_models': (lens_models.TNFW,),
    'all_main_deflector_models': (lens_models.Shear, lens_models.EPL),
    'all_source_models': (source_models.SersicElliptic,),
    'all_psf_models': (psf_models.Gaussian,)}

# TODO hard code a sampler here. Will want to break this out later.
def draw_sample(rng):

    rng_te, rng_cx, rng_cy, rng = jax.random.split(rng, 4)
    rng_sl, rng_ar, rng_ge, rng_ang, rng = jax.random.split(rng, 5)
    main_deflector_params = {'model_index': jnp.array([0,1]),
                             'z_lens': jnp.full((2,), 0.5),
                             'theta_e': jax.random.normal(rng_te, shape=(2,)) * 0.15 + 1.1,
                             'slope': jax.random.normal(rng_sl, shape=(2,)) * 0.1 + 2.0,
                             'center_x': jax.random.normal(rng_cx, shape=(2,)) * 0.16,
                             'center_y': jax.random.normal(rng_cy, shape=(2,)) * 0.16,
                             'axis_ratio': jax.random.normal(rng_ar, shape=(2,)) * 0.05 + 1.0,
                             'angle': jax.random.uniform(rng_ang, shape=(2,)) * 2 * jnp.pi,
                             'gamma_ext': jax.random.normal(rng_ge, shape=(2,)) * 0.05}
    main_deflector_params_substructure = {'mass': 1e13, 'z_lens': 0.5,
                                          'theta_e': main_deflector_params['theta_e'][1],
                                          'center_x': main_deflector_params['center_x'][1],
                                          'center_y': main_deflector_params['center_y'][1]}
    rng_ss, rng_pi, rng = jax.random.split(rng, 3)
    subhalo_params = {'sigma_sub': jax.random.normal(rng_ss) * 1.1e-3 + 2.0e-3,
                      'shmf_plaw_index': jax.random.uniform(rng_pi) * 0.1 - 2.02,
                      'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'k_one': 0.0, 'k_two': 0.0,
                      'c_zero': 18, 'conc_zeta': -0.2, 'conc_beta': 0.8, 'conc_m_ref': 1e8,
                      'conc_dex_scatter': 0.0}
    rng_amp, rng_sr, rng_ns, rng_ar, rng_ang, rng_cx, rng_cy, rng = jax.random.split(rng, 8)
    source_params = {'model_index': jnp.full((1,), 0), 'z_source': 1.5,
                     'amp': jax.random.uniform(rng_amp, shape=(1,)) * 9.0 + 1.0,
                     'sersic_radius': jax.random.uniform(rng_sr, shape=(1,)) * 2.0 + 1.0,
                     'n_sersic': jax.random.uniform(rng_ns, shape=(1,)) * 3.0 + 1.0,
                     'axis_ratio': jax.random.normal(rng_ar, shape=(1,)) * 0.05 + 1.0,
                     'angle': jax.random.uniform(rng_ang, shape=(1,)) * 2 * jnp.pi,
                     'center_x': jax.random.normal(rng_cx, shape=(1,)) * 0.16,
                     'center_y': jax.random.normal(rng_cy, shape=(1,)) * 0.16}
    return main_deflector_params, source_params, subhalo_params, main_deflector_params_substructure

# TODO these values are also hard coded in the script and should be read from
# a configuration file in the future.
num_z_bins = 1000
los_pad_length = 200
subhalos_pad_length = 200
sampling_pad_length = 10000

draw_los_jit = jax.jit(functools.partial(los.draw_los, num_z_bins=num_z_bins,
    los_pad_length=los_pad_length))
draw_subhalos_jit = jax.jit(functools.partial(subhalos.draw_subhalos,
    subhalos_pad_length=subhalos_pad_length,
    sampling_pad_length=sampling_pad_length))
image_simulation_jit = jax.jit(functools.partial(
    image_simulation.generate_image, kwargs_detector=kwargs_detector,
    all_models=all_models))

draw_sample_vmap = jax.jit(jax.vmap(draw_sample))
draw_los_vmap = jax.jit(jax.vmap(draw_los_jit, in_axes=[0, 0, None, None, 0]))
draw_subhalos_vmap = jax.jit(jax.vmap(draw_subhalos_jit,
    in_axes=[0, 0, 0, None, 0]))
image_simulation_vmap = jax.jit(jax.vmap(image_simulation_jit,
    in_axes=[None, None, 0, 0, 0, None, None, 0]))
downsample_vmap = jax.jit(jax.vmap(functools.partial(utils.downsample,
    supersampling_factor=2)))


def draw_images(rng, batch_size):

    rng_array = jax.random.split(rng, batch_size)

    (main_deflector_params, source_params, subhalo_params,
        main_deflector_params_sub) = draw_sample_vmap(rng_array)
    los_before_tuple, los_after_tuple = draw_los_vmap(main_deflector_params_sub,
        source_params, los_params, cosmology_params, rng_array)
    subhalos_z, subhalos_kwargs = draw_subhalos_vmap(main_deflector_params_sub,
        source_params, subhalo_params, cosmology_params, rng_array)

    kwargs_lens_all = {'z_array_los_before': los_before_tuple[0],
        'kwargs_los_before': los_before_tuple[1],
        'z_array_los_after': los_after_tuple[0],
        'kwargs_los_after': los_after_tuple[1],
        'kwargs_main_deflector': main_deflector_params,
        'z_array_main_deflector': main_deflector_params['z_lens'],
        'z_array_subhalos': subhalos_z, 'kwargs_subhalos': subhalos_kwargs}
    z_source = source_params.pop('z_source')

    # TODO For now the truths are just brute-force normalized sigma_sub
    truth = (subhalo_params['sigma_sub'] - 2e-3) / 1.1e-3
    image = downsample_vmap(
        image_simulation_vmap(grid_x, grid_y, kwargs_lens_all, source_params,
        source_params, kwargs_psf, cosmology_params, z_source))
    image /= jnp.expand_dims(jnp.expand_dims(
        jnp.std(image.reshape(batch_size, -1), axis=-1), axis=-1), axis=-1)

    return image, truth
