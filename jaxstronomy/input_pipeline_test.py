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
"""Tests for input_pipeline.py."""

import functools

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxstronomy import cosmology_utils
from jaxstronomy import input_pipeline
from jaxstronomy import lens_models
from jaxstronomy import psf_models
from jaxstronomy import source_models


def _prepare_lensing_config():
    encode_normal = input_pipeline.encode_normal
    encode_uniform = input_pipeline.encode_uniform
    encode_constant = input_pipeline.encode_constant
    lensing_config = {
            'los_params':{
            'delta_los': encode_constant(10.0),
            'r_min': encode_constant(0.5),
            'r_max': encode_constant(10.0),
            'm_min': encode_constant(1e8),
            'm_max': encode_constant(1e10),
            'dz': encode_constant(0.1),
            'cone_angle': encode_constant(8.0),
            'angle_buffer': encode_constant(0.8),
            'c_zero': encode_constant(18),
            'conc_zeta': encode_constant(-0.2),
            'conc_beta': encode_constant(0.8),
            'conc_m_ref': encode_constant(1e8),
            'conc_dex_scatter': encode_constant(0.0)
        },
        'main_deflector_params': {
            'mass': encode_constant(1e13),
            'z_lens': encode_constant(0.5),
            'theta_e': encode_constant(1.1),
            'slope': encode_normal(mean=2.0, std=0.1),
            'center_x': encode_normal(mean=0.0, std=0.16),
            'center_y': encode_normal(mean=0.0, std=0.16),
            'axis_ratio': encode_normal(mean=1.0, std=0.05),
            'angle': encode_uniform(minimum=0.0, maximum=2 * jnp.pi),
            'gamma_ext': encode_normal(mean=0.0, std=0.05)
        },
        'subhalo_params':{
            'sigma_sub': encode_normal(mean=2.0e-3, std=1.1e-3),
            'shmf_plaw_index': encode_uniform(minimum=-2.02, maximum=-1.92),
            'm_pivot': encode_constant(1e10),
            'm_min': encode_constant(1e6),
            'm_max': encode_constant(1e9),
            'k_one': encode_constant(0.0),
            'k_two': encode_constant(0.0),
            'c_zero': encode_constant(18),
            'conc_zeta': encode_constant(-0.2),
            'conc_beta': encode_constant(0.8),
            'conc_m_ref': encode_constant(1e8),
            'conc_dex_scatter': encode_constant(0.0)
        },
        'source_params':{
            'z_source': encode_constant(1.5),
            'amp': encode_uniform(minimum=1.0, maximum=10.0),
            'sersic_radius': encode_uniform(minimum=1.0, maximum=4.0),
            'n_sersic': encode_uniform(minimum=1.0, maximum=4.0),
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
    return lensing_config

def _perpare_cosmology_params():
    encode_constant = input_pipeline.encode_constant
    return {
        'omega_m_zero': encode_constant(0.3089),
        'omega_b_zero': encode_constant(0.0486),
        'omega_de_zero': encode_constant(0.6910088292453472),
        'omega_rad_zero': encode_constant(9.117075466e-5),
        'temp_cmb_zero': encode_constant(2.7255),
        'hubble_constant': encode_constant(67.74),
        'n_s': encode_constant(0.9667),
        'sigma_eight': encode_constant(0.8159),
    }

class InputPipelineTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of input pipeline functions."""

    @chex.all_variants
    def test_encode_normal(self):
        # Test that the encoding behaves as expected
        encode_normal = self.variant(input_pipeline.encode_normal)
        mean = 1.2
        std = 0.4
        encoded = encode_normal(mean=mean, std=std)
        expected = np.array([0.0, 0.0, 0.0, 1.0, mean, std, 0.0])
        np.testing.assert_array_almost_equal(encoded, expected)

    @chex.all_variants
    def test_encode_uniform(self):
        # Test that the encoding behaves as expected
        encode_uniform = self.variant(input_pipeline.encode_uniform)
        minimum = 0.2
        maximum = 2.01
        encoded = encode_uniform(minimum=minimum, maximum=maximum)
        expected = np.array([1.0, minimum, maximum, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(encoded, expected)

    @chex.all_variants
    def test_encode_constant(self):
        # That that the encoding behaves as expected.
        encode_constant = self.variant(input_pipeline.encode_constant)
        constant = 1.43
        encoded = encode_constant(constant=constant)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, constant])
        np.testing.assert_array_almost_equal(encoded, expected)

    @chex.all_variants
    def test_decode_maximum(self):
        # Test that for the constant, uniform, and gaussian case it gives
        # the constant, the maximum, and the mean plus five sigma.
        mean = 1.8
        std = 1.0
        minimum = 0.8
        maximum = 12.0
        constant = 13.4
        constant_encoding = input_pipeline.encode_constant(constant)
        uniform_encoding = input_pipeline.encode_uniform(minimum=minimum,
                                                         maximum=maximum)
        normal_encoding = input_pipeline.encode_normal(mean=mean, std=std)
        decode_maximum = self.variant(input_pipeline.decode_maximum)

        self.assertAlmostEqual(decode_maximum(constant_encoding), constant,
                               places=6)
        self.assertAlmostEqual(decode_maximum(uniform_encoding), maximum,
                               places=6)
        self.assertAlmostEqual(decode_maximum(normal_encoding), mean + 5 * std,
                               places=6)

    @chex.all_variants
    def test_decode_minimum(self):
        # Test that for the constant, uniform, and gaussian case it gives
        # the constant, the maximum, and the mean plus five sigma.
        mean = 1.8
        std = 1.0
        minimum = 0.8
        maximum = 12.0
        constant = 13.4
        constant_encoding = input_pipeline.encode_constant(constant)
        uniform_encoding = input_pipeline.encode_uniform(minimum=minimum,
                                                         maximum=maximum)
        normal_encoding = input_pipeline.encode_normal(mean=mean, std=std)
        decode_minimum = self.variant(input_pipeline.decode_minimum)

        self.assertAlmostEqual(decode_minimum(constant_encoding), constant,
                               places=6)
        self.assertAlmostEqual(decode_minimum(uniform_encoding), minimum,
                               places=6)
        self.assertAlmostEqual(decode_minimum(normal_encoding), mean - 5 * std,
                               places=6)

    @chex.all_variants
    def test_draw_from_encoding(self):
        # Test that the encoding gives reasonable draws from the distribution
        # for the normal, uniform, and constant encoding.
        draw_from_encoding = self.variant(input_pipeline.draw_from_encoding)
        draw_from_encoding_vmap = jax.vmap(draw_from_encoding,
                                           in_axes=[None, 0])
        n_draws = 100000
        rng_list = jax.random.split(jax.random.PRNGKey(0), n_draws)

        constant = 12.0
        constant_encoding = input_pipeline.encode_constant(constant)
        constant_draws = draw_from_encoding_vmap(constant_encoding, rng_list)
        np.testing.assert_array_almost_equal(constant_draws,
                                             np.ones(n_draws) * constant)

        mean = 1.8
        std = 1.0
        normal_encoding = input_pipeline.encode_normal(mean=mean, std=std)
        normal_draws = draw_from_encoding_vmap(normal_encoding, rng_list)
        self.assertAlmostEqual(jnp.mean(normal_draws), mean, places=2)
        self.assertAlmostEqual(jnp.std(normal_draws), std, places=2)

        minimum = 0.0
        maximum = 4.0
        uniform_encoding = input_pipeline.encode_uniform(minimum=minimum,
                                                         maximum=maximum)
        uniform_draws = draw_from_encoding_vmap(uniform_encoding, rng_list)
        self.assertAlmostEqual(jnp.mean(uniform_draws < 1.0), 0.25, places=2)
        self.assertAlmostEqual(jnp.mean(uniform_draws < 2.0), 0.5, places=2)
        self.assertAlmostEqual(jnp.mean(uniform_draws < 3.0), 0.75, places=2)
        self.assertAlmostEqual(jnp.mean(uniform_draws < 4.0), 1.0, places=2)

    @chex.all_variants
    def test_normalize_params(self):
        # Test that the draws are normalized as expected.
        normalize_param = self.variant(input_pipeline.normalize_param)
        normalize_param_vmap = jax.vmap(normalize_param, in_axes=[0, None])
        draw_from_encoding_vmap = jax.jit(jax.vmap(
            input_pipeline.draw_from_encoding, in_axes=[None, 0]))

        n_draws = 100000
        rng_list = jax.random.split(jax.random.PRNGKey(0), n_draws)

        constant = 12.0
        constant_encoding = input_pipeline.encode_constant(constant)
        constant_draws = draw_from_encoding_vmap(constant_encoding, rng_list)
        normalized_draws = normalize_param_vmap(constant_draws,
                                                constant_encoding)
        np.testing.assert_array_almost_equal(normalized_draws, np.zeros(n_draws))

        mean = 1.8
        std = 1.0
        normal_encoding = input_pipeline.encode_normal(mean=mean, std=std)
        normal_draws = draw_from_encoding_vmap(normal_encoding, rng_list)
        normalized_draws = normalize_param_vmap(normal_draws, normal_encoding)
        self.assertAlmostEqual(jnp.mean(normalized_draws), 0.0, places=2)
        self.assertAlmostEqual(jnp.std(normalized_draws), 1.0, places=2)

        minimum = 0.0
        maximum = 4.0
        uniform_encoding = input_pipeline.encode_uniform(minimum=minimum,
                                                         maximum=maximum)
        uniform_draws = draw_from_encoding_vmap(uniform_encoding, rng_list)
        normalized_draws = normalize_param_vmap(uniform_draws, uniform_encoding)
        self.assertAlmostEqual(jnp.mean(normalized_draws < 0.25), 0.25,
                               places=2)
        self.assertAlmostEqual(jnp.mean(normalized_draws < 0.50), 0.5, places=2)
        self.assertAlmostEqual(jnp.mean(normalized_draws < 0.75), 0.75,
                               places=2)
        self.assertAlmostEqual(jnp.mean(normalized_draws < 1.00), 1.0, places=2)

    def test_generate_grids(self):
        # Test that the grid has the expected dimensions.
        kd = {
            'n_x': 128, 'n_y': 128, 'pixel_width': 0.04,
            'supersampling_factor': 2.0
        }
        config = {'kwargs_detector': kd}

        grid_x, grid_y = input_pipeline.generate_grids(config)

        size = kd['n_x'] * kd['n_y'] * kd['supersampling_factor'] ** 2
        self.assertTupleEqual(grid_x.shape, (size,))
        self.assertTupleEqual(grid_y.shape, (size,))
        angular_size = ((kd['n_x'] * kd['supersampling_factor'] - 1) *
                        kd['pixel_width'] / kd['supersampling_factor'])
        self.assertAlmostEqual(jnp.max(grid_x) - jnp.min(grid_x), angular_size)

    def test_initialize_cosmology_params(self):
        # Test that the cosmology params generated can handle the range of
        # values defined by the config.
        encode_constant = input_pipeline.encode_constant
        dz = 0.01
        config = {}
        config['cosmology_params'] = _perpare_cosmology_params()
        config['lensing_config'] = {
            'source_params': {'z_source': encode_constant(2.0)},
            'subhalo_params': {
                'm_max': encode_constant(1e10),
                'm_min': encode_constant(1e8)
                },
            'los_params': {
                'dz': encode_constant(dz),
                'm_max': encode_constant(1e9),
                'm_min': encode_constant(1e7)
            }
        }
        rng = jax.random.PRNGKey(0)

        cosmology_params = input_pipeline.intialize_cosmology_params(
            config, rng
        )

        # Test that the cosmology params do not cause issues if the masses
        # are within the specified range.
        r_min = cosmology_utils.lagrangian_radius(cosmology_params, 1e6)
        r_max = cosmology_utils.lagrangian_radius(cosmology_params, 1e11)
        # Cosmological values are calculated in incriments of dz / 2.
        self.assertAlmostEqual(cosmology_params['dz'], dz / 2)
        self.assertAlmostEqual(cosmology_params['r_min'], r_min, places=6)
        self.assertAlmostEqual(cosmology_params['r_max'], r_max, places=6)

    @chex.all_variants
    def test_draw_sample(self):
        # Test that the samples drawn match the expected distribution.
        mean = 1.0
        std = 0.5
        normal_encoding = input_pipeline.encode_normal(mean=mean, std=std)
        constant = 12.0
        constant_encoding = input_pipeline.encode_constant(constant=constant)
        encoded_configuration = {
            'node_a': {'node_aa': normal_encoding, 'node_ab': normal_encoding},
            'node_b': constant_encoding
        }
        n_draws = 100000
        rng_list = jax.random.split(jax.random.PRNGKey(0), n_draws)

        draw_sample = self.variant(input_pipeline.draw_sample)
        draw_sample_vmap = jax.vmap(draw_sample, in_axes=[None, 0])

        sampled_configuration = draw_sample_vmap(encoded_configuration,
                                                 rng_list)
        np.testing.assert_array_almost_equal(
            sampled_configuration['node_b'], np.ones(n_draws) * constant
        )
        self.assertAlmostEqual(
            jnp.mean(sampled_configuration['node_a']['node_aa']), mean,
            places=2)
        self.assertAlmostEqual(
            jnp.mean(sampled_configuration['node_a']['node_ab']), mean,
            places=2)
        self.assertAlmostEqual(
            jnp.std(sampled_configuration['node_a']['node_aa']), std,
            places=2)
        self.assertAlmostEqual(
            jnp.std(sampled_configuration['node_a']['node_ab']), std,
            places=2)

    @chex.all_variants
    def test_extract_multiple_models(self):
        # Test that the code scales well to multiple models and a single model.
        constant = 12.0
        mean = 0.9
        std = 0.3
        encoded_configuration = {
            'param_1': input_pipeline.encode_constant(constant),
            'param_2': input_pipeline.encode_normal(mean=mean, std=std),
        }
        n_models = 100000
        rng = jax.random.PRNGKey(0)

        extract_multiple_models = self.variant(functools.partial(
            input_pipeline.extract_multiple_models, n_models=n_models
        ))
        sampled_configuration = extract_multiple_models(encoded_configuration,
                                                        rng)

        self.assertTupleEqual(sampled_configuration['param_1'].shape, (100000,))
        np.testing.assert_array_almost_equal(sampled_configuration['param_1'],
                                             np.ones(n_models) * constant)
        self.assertAlmostEqual(jnp.mean(sampled_configuration['param_2']),
                               mean, places=2)
        self.assertAlmostEqual(jnp.std(sampled_configuration['param_2']),
                               std, places=2)
        np.testing.assert_array_almost_equal(
            sampled_configuration['model_index'], np.arange(n_models))

        n_models = 1
        extract_multiple_models = self.variant(functools.partial(
            input_pipeline.extract_multiple_models, n_models=n_models
        ))
        sampled_configuration = extract_multiple_models(encoded_configuration,
                                                        rng)
        self.assertTupleEqual(sampled_configuration['param_1'].shape, (1,))
        np.testing.assert_array_almost_equal(
            sampled_configuration['model_index'], np.arange(n_models))


    @chex.all_variants(without_device=False)
    def test_extract_multiple_models_angular(self):
        # Test that the code draws the parameters for multiple models and
        # converts the required parameters to angular units.
        encode_normal = input_pipeline.encode_normal
        encode_uniform = input_pipeline.encode_uniform
        encode_constant = input_pipeline.encode_constant
        encoded_configuration = {
            'galaxy_index': encode_constant(0.7),
            'output_ab_zeropoint': encode_constant(23.5),
            'catalog_ab_zeropoint': encode_constant(25.6),
            'z_source': encode_constant(1.5),
            'amp': encode_uniform(minimum=1.0, maximum=10.0),
            'sersic_radius': encode_uniform(minimum=1.0, maximum=4.0),
            'n_sersic': encode_uniform(minimum=1.0, maximum=4.0),
            'axis_ratio': encode_normal(mean=1.0, std=0.05),
            'angle': encode_uniform(minimum=0.0, maximum=2 * jnp.pi),
            'center_x': encode_normal(mean=0.0, std=0.16),
            'center_y': encode_normal(mean=0.0, std=0.16)
        }
        rng = jax.random.PRNGKey(0)
        config = {}
        config['cosmology_params'] = _perpare_cosmology_params()
        config['lensing_config'] = _prepare_lensing_config()
        cosmology_params = input_pipeline.intialize_cosmology_params(config,
                                                                     rng)
        all_models = (
            source_models.SersicElliptic(),
            source_models.CosmosCatalog(
                'test_files/cosmos_galaxies_testing.npz'
            )
        )

        extract_multiple_models_angular = self.variant(functools.partial(
            input_pipeline.extract_multiple_models_angular,
            all_models=all_models
        ))

        sampled_configuration = extract_multiple_models_angular(
            encoded_configuration, rng, cosmology_params
        )
        for image in sampled_configuration['image']:
            np.testing.assert_array_almost_equal(
                image,
                all_models[1].images[1] / all_models[1].pixel_sizes[1] ** 2
            )


    @chex.all_variants(without_device=False)
    def test_extract_truth_values(self):
        # Test that the extracted parameters match the values fed into the
        # dictionary.
        all_params = {
            'a': {'a': 0.0, 'b': 0.5,
                  'c': 12.2},
            'b': {'a': 1.0, 'b': 0.7,
                  'c': 10.2}}
        constant = 1.0
        mean = 2.0
        std = 1.0
        minimum = -1.0
        maximum = 12.0
        lensing_config = {
            'a': {'a': input_pipeline.encode_constant(constant),
                  'b': input_pipeline.encode_normal(mean, std),
                  'c': input_pipeline.encode_uniform(minimum, maximum)},
            'b': {'a': input_pipeline.encode_constant(constant),
                  'b': input_pipeline.encode_normal(mean, std),
                  'c': input_pipeline.encode_uniform(minimum, maximum)}}
        extract_objects = ['a','a','b']
        extract_keys = ['a','b','c']
        truth_parameters = (extract_objects, extract_keys)
        extract_truth_values = self.variant(functools.partial(
            input_pipeline.extract_truth_values,
            truth_parameters=truth_parameters))

        parameter_array = extract_truth_values(all_params, lensing_config)
        self.assertTupleEqual(parameter_array.shape, (3,))
        np.testing.assert_array_almost_equal(
            parameter_array, np.array([0.0, (0.5 - mean) / std,
                                       (10.2 - minimum) / (maximum - minimum)])
        )


    @chex.all_variants(without_device=False)
    def test_draw_image_and_truth(self):
        # Test that the images have reasonable shape and that the truth values
        # are drawn correctly.
        config = {}
        config['cosmology_params'] = _perpare_cosmology_params()
        config['lensing_config'] = _prepare_lensing_config()
        rng = jax.random.PRNGKey(0)

        cosmology_params = input_pipeline.intialize_cosmology_params(config,
                                                                     rng)
        n_x = 16
        n_y = 16
        config['kwargs_detector'] = {
            'n_x': n_x, 'n_y': n_y, 'pixel_width': 0.4,
            'supersampling_factor': 2, 'exposure_time': 1024,
            'num_exposures': 1.0, 'sky_brightness': 22,
            'magnitude_zero_point': 25, 'read_noise': 3.0
        }
        grid_x, grid_y = input_pipeline.generate_grids(config)
        all_models = {
            'all_los_models': (lens_models.NFW(),),
            'all_subhalo_models': (lens_models.TNFW(),),
            'all_main_deflector_models': (lens_models.Shear(),
                                          lens_models.EPL()),
            'all_source_models': (source_models.SersicElliptic(),),
            'all_psf_models': (psf_models.Gaussian(),)
        }
        principal_md_index = 0
        principal_source_index = 0
        kwargs_simulation = {
            'num_z_bins': 1000,
            'los_pad_length': 10,
            'subhalos_pad_length': 10,
            'sampling_pad_length': 1000,
        }
        kwargs_psf = {'model_index': 0, 'fwhm': 0.04, 'pixel_width': 0.02}
        truth_parameters = (
            ['main_deflector_params', 'subhalo_params'],
            ['theta_e', 'sigma_sub']
        )

        draw_image_and_truth = self.variant(functools.partial(
            input_pipeline.draw_image_and_truth, all_models=all_models,
            principal_md_index=principal_md_index,
            principal_source_index=principal_source_index,
            kwargs_simulation=kwargs_simulation,
            kwargs_detector=config['kwargs_detector'],
            kwargs_psf=kwargs_psf, truth_parameters=truth_parameters))
        image, truth = draw_image_and_truth(config['lensing_config'],
                                            cosmology_params, grid_x, grid_y,
                                            rng)

        # Just test that the data and truths vary and that they are normalized
        # as expected.
        self.assertTupleEqual(image.shape, (n_x, n_y))
        self.assertAlmostEqual(1.0, jnp.std(image), places=6)
        self.assertTupleEqual(truth.shape, (2,))
        self.assertEqual(truth[0], 0.0)
