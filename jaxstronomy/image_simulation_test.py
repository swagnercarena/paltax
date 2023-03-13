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
"""Tests for image_simulation.py."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from jaxstronomy import cosmology_utils
from jaxstronomy import image_simulation
from jaxstronomy import lens_models
from jaxstronomy import psf_models
from jaxstronomy import source_models
from jaxstronomy import utils

COSMOLOGY_PARAMS_LENSTRONOMY = immutabledict({
    'omega_m_zero': 0.30966,
    'omega_b_zero': 0.0486,
    'omega_de_zero': 0.688846306,
    'omega_rad_zero': 0.0014937,
    'temp_cmb_zero': 2.7255,
    'hubble_constant': 67.66,
    'n_s': 0.9667,
    'sigma_eight': 0.8159,
})


def _prepare_cosmology_params(cosmology_params_init, z_lookup_max, dz):
    # Only generate a lookup table for values we need.
    z_lookup_max = max(z_lookup_max, 1e-7)
    dz = max(dz, 1e-7)
    # We won't be using the linear density field lookup tables, so provide
    # reasonable radial bounds and only 2 bins.
    return cosmology_utils.add_lookup_tables_to_cosmology_params(
        dict(cosmology_params_init), z_lookup_max, dz, 1e-3, 2e-3, 2)


def _prepare_kwargs_detector():
    return {
        'n_x': 2,
        'n_y': 2,
        'pixel_width': 0.04,
        'supersampling_factor': 2,
        'exposure_time': 1024,
        'num_exposures': 2.0,
        'sky_brightness': 22,
        'magnitude_zero_point': 25,
        'read_noise': 3.0
    }


def _prepare_kwargs_psf():
    kwargs_detector = _prepare_kwargs_detector()
    all_psf_models = _prepare_all_psf_models()
    params = tuple(sorted(set(itertools.chain(
        *[model.parameters for model in all_psf_models]))))
    kwargs_psf = {param: 0.05 for param in params}
    kwargs_psf['pixel_width'] = (
        kwargs_detector['pixel_width'] / kwargs_detector['supersampling_factor']
    )
    x = jnp.arange(-0.5, 0.5, 0.04) / 0.02
    kernel = jnp.outer(jnp.exp(-x**2), jnp.exp(-x**2))
    kwargs_psf['kernel_point_source'] = kernel
    kwargs_psf['model_index'] = psf_models.__all__.index('Pixel')
    return kwargs_psf


def _prepare_x_y():
    rng = jax.random.PRNGKey(3)
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.normal(rng_x, shape=(3,)) * 1e3
    y = jax.random.normal(rng_y, shape=(3,)) * 1e3
    return x, y


def _prepare_x_y_angular():
    x, y = _prepare_x_y()
    return x / 1e3, y / 1e3


def _prepare_alpha_x_alpha_y():
    rng = jax.random.PRNGKey(2)
    rng_x, rng_y = jax.random.split(rng)
    alpha_x = jax.random.normal(rng_x, shape=(3,)) * 2
    alpha_y = jax.random.normal(rng_y, shape=(3,)) * 2
    return alpha_x, alpha_y


def _prepare_image():
    return jax.random.uniform(jax.random.PRNGKey(3), shape=(4, 4))


def _prepare_all_psf_models():
    models = psf_models.__all__
    return tuple([psf_models.__getattribute__(model)() for model in models])

def _prepare_all_lens_models(model_group):
    if model_group == 'los':
        models = ['NFW']
    elif model_group == 'subhalos':
        models = ['TNFW']
    elif model_group == 'main_deflector':
        models = ['Shear', 'EPL']
    elif model_group == 'all':
        models = ['NFW', 'TNFW', 'Shear', 'EPL']
    else:
        raise ValueError(f'Unsupported lens_models specification {model_group}')

    return tuple([lens_models.__getattribute__(model) for model in models])


def _prepare_all_source_models():
    all_source_models = []
    for model in source_models.__all__:
        # CosmosCatalog model required initialization parameters.
        if model != 'CosmosCatalog':
            all_source_models.append(
                source_models.__getattribute__(model)()
            )
        else:
            all_source_models.append(
                source_models.__getattribute__(model)(
                    'test_files/cosmos_galaxies_testing.npz'
                )
            )
    return tuple(all_source_models)


def _prepare_all_models():
    model_group = 'all'
    all_lens_models = _prepare_all_lens_models(model_group)
    all_models = {}
    all_models['all_los_models'] = all_lens_models
    all_models['all_subhalo_models'] = all_lens_models
    all_models['all_main_deflector_models'] = all_lens_models
    all_models['all_source_models'] = _prepare_all_source_models()
    all_models['all_lens_light_models'] = _prepare_all_source_models()
    all_models['all_psf_models'] = _prepare_all_psf_models()

    return all_models


def _prepare_kwargs_lens(model_group):
    all_lens_models = _prepare_all_lens_models(model_group)
    params = tuple(sorted(set(itertools.chain(
        *[model.parameters for model in all_lens_models]))))
    return {param: 0.5 for param in params}


def _prepare_kwargs_lens_all():
    model_group = 'all'

    kwargs_lens_all = {}
    kwargs_lens_all['kwargs_los_before'] = _prepare_kwargs_lens(model_group)
    kwargs_lens_all['kwargs_subhalos'] = _prepare_kwargs_lens(model_group)
    kwargs_lens_all['kwargs_main_deflector'] = _prepare_kwargs_lens(model_group)
    kwargs_lens_all['kwargs_los_after'] = _prepare_kwargs_lens(model_group)
    # Set all values to jnp.ndarrays.
    for _, kwargs_lens in kwargs_lens_all.items():
        for kwarg in kwargs_lens:
            kwargs_lens[kwarg] = jnp.ones((4,)) * 0.5
        # Should be NFW, Shear, TNFW, and then padded indices.
        # Make sure that a padded model does not change the output.
        kwargs_lens['model_index'] = jnp.array([0, 2, 1, -1])

    kwargs_lens_all['z_array_los_before'] = jnp.array([0.1] * 4)
    kwargs_lens_all['z_array_subhalos'] = jnp.array([0.2] * 4)
    kwargs_lens_all['z_array_main_deflector'] = jnp.array([0.2] * 4)
    kwargs_lens_all['z_array_los_after'] = jnp.array([0.3] * 4)

    # Turn off the subhalo models.
    kwargs_lens_all['kwargs_subhalos']['model_index'] = jnp.array([-1] * 4)

    return kwargs_lens_all


def _prepare_kwargs_source():
    all_source_models = _prepare_all_source_models()
    all_source_model_parameters = tuple(sorted(set(itertools.chain(
        *[model.parameters for model in all_source_models]))))
    kwargs_source = dict([
        (param, 0.5) for param in all_source_model_parameters
    ])
    kwargs_source['image'] = jax.random.normal(
        jax.random.PRNGKey(3), shape=(16, 16))
    kwargs_source['model_index'] = source_models.__all__.index('Interpol')
    return kwargs_source


def _prepare_kwargs_source_slice():
    kwargs_source_slice = _prepare_kwargs_source()
    # Modify kwargs_source_slice to deal with interpolation and give sersic
    # profile meaningful brightness.
    for kwarg in kwargs_source_slice:
        kwargs_source_slice[kwarg] = jnp.ones((2,)) * 0.5
    kwargs_source_slice['image'] = jax.random.uniform(
        jax.random.PRNGKey(3), shape=(2, 16, 16))
    kwargs_source_slice['n_sersic'] = jnp.ones((2,)) * 3.0
    kwargs_source_slice['model_index'] = jnp.array([
        source_models.__all__.index('Interpol'),
        source_models.__all__.index('SersicElliptic')
    ])
    return kwargs_source_slice


def _prepare__ray_shooting_step_expected(z_lens, z_source):
    if z_lens == 0.1 and z_source == 0.5:
        return (
            [-1455.0688, 261.96478, -263.31418],
            [311.67578, -1881.2992, 1816.3245],
            [-0.42915863, 3.7673984, -2.9809818],
            [-1.632507, -0.58538866, 0.6765887],
            0.1,
        )
    elif z_lens == 0.5 and z_source == 1.0:
        return (
            [-2219.5464, 5967.2534, -4809.161],
            [-2153.1443, -2859.7515, 2950.956],
            [-0.34272414, 3.638215, -2.8426921],
            [-1.4692892, -0.5461657, 0.694726],
            0.5,
        )
    else:
        raise ValueError(
            f'Unsupported z_lens and z_source = ({z_lens},{z_source})')


def _prepare__add_deflection_expected(z_lens, z_source):
    if z_lens == 0.1 and z_source == 0.5:
        return jnp.array([[-0.42978406, 3.7907193, -3.0125906],
                        [-1.6494985, -0.5895851, 0.66275066]])
    elif z_lens == 0.5 and z_source == 1.0:
        return jnp.array([[-0.11232474, 3.921643, -2.5539021],
                        [-1.5750934, -0.37239552, 0.2341882]])
    else:
        raise ValueError(
            f'Unsupported z_lens and z_source = ({z_lens},{z_source})')


class ImageSimulationTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants(without_device=False)
    def test_generate_image(self):
        all_models = _prepare_all_models()
        kwargs_lens_all = _prepare_kwargs_lens_all()
        kwargs_source_slice = _prepare_kwargs_source_slice()
        kwargs_lens_light_slice = _prepare_kwargs_source_slice()
        kwargs_detector = _prepare_kwargs_detector()
        kwargs_psf = _prepare_kwargs_psf()
        z_source = 0.5
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, 0.1
        )
        # Need to evaluate on coordinate grid to match lenstronomy.
        grid_x, grid_y = utils.coordinates_evaluate(
            kwargs_detector['n_x'], kwargs_detector['n_y'],
            kwargs_detector['pixel_width'],
            kwargs_detector['supersampling_factor'])
        expected = jnp.array(
            [[0.00221893, 0.0030974], [0.00328375, 0.00458501]]
        )

        g_image = self.variant(
            functools.partial(
                image_simulation.generate_image,
                kwargs_detector=kwargs_detector,
                all_models=all_models))

        result = utils.downsample(
        g_image(grid_x, grid_y, kwargs_lens_all, kwargs_source_slice,
                kwargs_lens_light_slice, kwargs_psf, cosmology_params,
                z_source),
        kwargs_detector['supersampling_factor'])
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    @chex.all_variants
    def test_psf_convolution(self):
        all_psf_models = _prepare_all_psf_models()
        kwargs_psf = _prepare_kwargs_psf()
        image = _prepare_image()

        convolve = self.variant(functools.partial(
            image_simulation.psf_convolution, all_psf_models=all_psf_models))

        # Default model is Pixel
        expected = psf_models.Pixel.convolve(image, kwargs_psf)
        np.testing.assert_allclose(convolve(image, kwargs_psf), expected,
                                   rtol=1e-5)

        # Try Gaussian model
        kwargs_psf['model_index'] = psf_models.__all__.index('Gaussian')
        expected = psf_models.Gaussian.convolve(image, kwargs_psf)
        np.testing.assert_allclose(convolve(image, kwargs_psf), expected,
                                   rtol=1e-5)

    @chex.all_variants
    def test_noise_realization(self):
        kwargs_detector = _prepare_kwargs_detector()
        image = _prepare_image()

        # We can't just pull the expectation from lenstronomy, since the noise is a
        # random realization and lenstronomy uses numpy for it's random functions.
        # Instead we pull the magnitude of the normal draw from lenstronomy.
        background_noise = 0.0040833212572525535
        flux_noise = jnp.array(
            [[0.02159131, 0.02026811, 0.00825452, 0.01600841],
             [0.02006242, 0.01973204, 0.01489258, 0.01015701],
             [0.01921518, 0.01649803, 0.00997489, 0.01330792],
             [0.01392349, 0.00903829, 0.01201941, 0.01346962]])

        rng_noise = jax.random.PRNGKey(0)
        rng_normal, rng_poisson = jax.random.split(rng_noise)
        expected = (
            background_noise * jax.random.normal(rng_normal, shape=image.shape)
            + flux_noise * jax.random.normal(rng_poisson, shape=image.shape))

        noise = self.variant(image_simulation.noise_realization)

        # Noise matches very closely, but numbers are small so use atol.
        np.testing.assert_allclose(
            noise(image, rng_noise, kwargs_detector), expected, rtol=0,
            atol=1e-7)

    @chex.all_variants
    def test_lens_light_surface_brightness(self):
        all_source_models = _prepare_all_source_models()
        kwargs_lens_light_slice = _prepare_kwargs_source_slice()
        kwargs_detector = _prepare_kwargs_detector()
        # Need to evaluate on coordinate grid to match lenstronomy.
        theta_x, theta_y = utils.coordinates_evaluate(
            kwargs_detector['n_x'], kwargs_detector['n_y'],
            kwargs_detector['pixel_width'],
            kwargs_detector['supersampling_factor'])
        expected = jnp.array(
            [[0.00074752, 0.00076053], [0.00083884, 0.00086197]]
        )

        ll_surface_brightness = self.variant(functools.partial(
            image_simulation.lens_light_surface_brightness,
            all_source_models=all_source_models))

        image = utils.downsample(
            jnp.reshape(
                ll_surface_brightness(theta_x, theta_y, kwargs_lens_light_slice,
                                    kwargs_detector),
                (
                    kwargs_detector['n_x'] *
                    kwargs_detector['supersampling_factor'],
                    kwargs_detector['n_y'] *
                    kwargs_detector['supersampling_factor'],
                ),
            ),
            kwargs_detector['supersampling_factor'],
        )

        np.testing.assert_allclose(image, expected, rtol=1e-5)

    @chex.all_variants(without_device=False)
    def test_source_surface_brightness(self):
        all_models = _prepare_all_models()
        kwargs_lens_all = _prepare_kwargs_lens_all()
        kwargs_source_slice = _prepare_kwargs_source_slice()
        z_source = 0.5
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, 0.1
        )
        kwargs_detector = _prepare_kwargs_detector()
        # Need to evaluate on coordinate grid to match lenstronomy.
        alpha_x, alpha_y = utils.coordinates_evaluate(
            kwargs_detector['n_x'], kwargs_detector['n_y'],
            kwargs_detector['pixel_width'],
            kwargs_detector['supersampling_factor'])
        expected = jnp.array(
            [[0.00325632, 0.00348312], [0.00369543, 0.00387607]]
        )

        s_surface_brightness = self.variant(
            functools.partial(image_simulation.source_surface_brightness,
                all_models=all_models))

        # Lenstronomy comparison is after downsampling.
        image = utils.downsample(
            jnp.reshape(
                s_surface_brightness(alpha_x, alpha_y, kwargs_lens_all,
                                    kwargs_source_slice, kwargs_detector,
                                    cosmology_params, z_source),
                (
                    kwargs_detector['n_x'] *
                    kwargs_detector['supersampling_factor'],
                    kwargs_detector['n_y'] *
                    kwargs_detector['supersampling_factor'],
                ),
            ),
            kwargs_detector['supersampling_factor'],
        )
        np.testing.assert_allclose(image, expected, rtol=1e-3)

    @chex.all_variants(without_device=False)
    def test__image_flux(self):
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        all_models = _prepare_all_models()
        kwargs_lens_all = _prepare_kwargs_lens_all()
        kwargs_source_slice = _prepare_kwargs_source_slice()
        z_source = 0.5
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, 0.1
        )
        expected = jnp.array([2.08355751e-01, 4.74442133e-05, 2.49857234e-04])

        image_flux = self.variant(functools.partial(
            image_simulation._image_flux, all_models=all_models))

        np.testing.assert_allclose(
            image_flux(alpha_x, alpha_y, kwargs_lens_all, kwargs_source_slice,
                    cosmology_params, z_source),
            expected,
            rtol=1e-2)

    @chex.all_variants(without_device=False)
    def test_ray_shooting(self):
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        all_models = _prepare_all_models()
        kwargs_lens_all = _prepare_kwargs_lens_all()
        z_source = 0.5
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY,  z_source, 0.1
        )
        expected = jnp.array([[4268.8287053, 6375.01729267, -2325.55000691],
                            [-2971.59041917, -10392.18963907, 9207.33107629]])

        ray_shooting = self.variant(functools.partial(
            image_simulation._ray_shooting, all_models=all_models))

        np.testing.assert_allclose(
            jnp.array(
                ray_shooting(alpha_x, alpha_y, kwargs_lens_all,
                cosmology_params, z_source)),
            expected,
            rtol=1e-2)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_z_lens_{z_lens}_z_source_{z_source}', z_lens, z_source)
        for z_lens, z_source in zip([0.1, 0.5], [0.5, 1.0])
    ])
    def test__ray_shooting_group(self, z_lens, z_source):
        x, y, = _prepare_x_y()
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        z_lens_last = 0.05
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, z_lens_last
        )
        state = (x, y, alpha_x, alpha_y, z_lens_last)
        expected_state = state
        model_group = 'subhalos'
        all_lens_models = _prepare_all_lens_models(model_group)

        # Make it four deflectors and make sure they add up correctly.
        kwargs_lens = _prepare_kwargs_lens(model_group)
        kwargs_lens_slice = {}
        for kwarg in kwargs_lens:
            kwargs_lens_slice[kwarg] = jnp.ones(4) * kwargs_lens[kwarg]
        kwargs_lens['model_index'] = 0
        kwargs_lens_slice['model_index'] = jnp.array([0] * 4)

        for _ in range(4):
            expected_state, _ = image_simulation._ray_shooting_step(
                expected_state, {'z_lens': z_lens, 'kwargs_lens': kwargs_lens},
                cosmology_params, z_source, all_lens_models)

        ray_shooting_group = self.variant(functools.partial(
            image_simulation._ray_shooting_group,
            all_lens_models=all_lens_models))

        new_state, new_state_copy = ray_shooting_group(
            state, kwargs_lens_slice, cosmology_params, z_source, z_lens)

        for si in range(len(new_state)):
            np.testing.assert_allclose(new_state[si], expected_state[si],
                                       rtol=1e-5)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_z_lens_{z_lens}_z_source_{z_source}', z_lens, z_source)
        for z_lens, z_source in zip([0.1, 0.5], [0.5, 1.0])
    ])
    def test__ray_shooting_step(self, z_lens, z_source):
        x, y, = _prepare_x_y()
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        z_lens_last = 0.05
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, z_lens_last
        )
        state = (x, y, alpha_x, alpha_y, z_lens_last)
        model_group = 'subhalos'
        all_lens_models = _prepare_all_lens_models(model_group)
        kwargs_lens = _prepare_kwargs_lens(model_group)
        kwargs_lens['model_index'] = 0

        expected = _prepare__ray_shooting_step_expected(z_lens, z_source)

        ray_shooting_step = self.variant(functools.partial(
            image_simulation._ray_shooting_step,
            all_lens_models=all_lens_models))

        new_state, new_state_copy = ray_shooting_step(
            state,
            {
                'z_lens': z_lens,
                'kwargs_lens': kwargs_lens
            },
            cosmology_params,
            z_source,
        )

        for si in range(len(new_state)):
            np.testing.assert_allclose(new_state[si], expected[si],
                                       rtol=1e-2)
            np.testing.assert_allclose(new_state_copy[si], expected[si],
                                       rtol=1e-2)

    @chex.all_variants
    def test__ray_step_add(self):
        x, y = _prepare_x_y()
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        delta_t = 1e3
        expected = jnp.array([[-1852.1759, 3225.5708, -2624.6494],
                            [-968.6724, -2389.5552, 2405.7075]])

        ray_step_add = self.variant(image_simulation._ray_step_add)

        np.testing.assert_allclose(
            jnp.array(ray_step_add(x, y, alpha_x, alpha_y, delta_t)),
            expected,
            rtol=1e-6)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_z_lens_{z_lens}_z_source_{z_source}', z_lens, z_source)
        for z_lens, z_source in zip([0.1, 0.5], [0.5, 1.0])
    ])
    def test__add_deflection_group(self, z_lens, z_source):
        x, y = _prepare_x_y()
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        model_group = 'subhalos'
        all_lens_models = _prepare_all_lens_models(model_group)

        # Make it four deflectors and make sure they add up correctly.
        kwargs_lens = _prepare_kwargs_lens(model_group)
        kwargs_lens_slice = {}
        for kwarg in kwargs_lens:
            kwargs_lens_slice[kwarg] = jnp.ones(4) * kwargs_lens[kwarg]
        kwargs_lens['model_index'] = 0
        kwargs_lens_slice['model_index'] = jnp.array([0] * 4)

        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, z_lens
        )

        alpha_x_expected = jnp.copy(alpha_x)
        alpha_y_expected = jnp.copy(alpha_y)
        for _ in range(4):
            alpha_x_expected, alpha_y_expected = (
                image_simulation._add_deflection(
                    x, y, alpha_x_expected, alpha_y_expected, kwargs_lens,
                    cosmology_params, z_lens, z_source, all_lens_models))

        add_deflection_group = self.variant(
            functools.partial(image_simulation._add_deflection_group,
            all_lens_models=all_lens_models))

        np.testing.assert_allclose(
            jnp.array(
                add_deflection_group(x, y, alpha_x, alpha_y, kwargs_lens_slice,
                            cosmology_params, z_lens, z_source)),
            jnp.array([alpha_x_expected, alpha_y_expected]), rtol=1e-5)

    @chex.all_variants
    @parameterized.named_parameters([
        (f'_z_lens_{z_lens}_z_source_{z_source}', z_lens, z_source)
        for z_lens, z_source in zip([0.1, 0.5], [0.5, 1.0])
    ])
    def test__add_deflection(self, z_lens, z_source):
        x, y = _prepare_x_y()
        alpha_x, alpha_y = _prepare_alpha_x_alpha_y()
        model_group = 'subhalos'
        all_lens_models = _prepare_all_lens_models(model_group)
        kwargs_lens = _prepare_kwargs_lens(model_group)
        kwargs_lens['model_index'] = 0
        cosmology_params = _prepare_cosmology_params(
            COSMOLOGY_PARAMS_LENSTRONOMY, z_source, z_lens
        )

        expected = _prepare__add_deflection_expected(z_lens, z_source)

        add_deflection = self.variant(
            functools.partial(image_simulation._add_deflection,
            all_lens_models=all_lens_models))

        np.testing.assert_allclose(
            jnp.array(
                add_deflection(x, y, alpha_x, alpha_y, kwargs_lens,
                            cosmology_params, z_lens, z_source)),
            expected,
            rtol=1e-2)

    @chex.all_variants
    def test__calculate_derivatives(self):
        x, y = _prepare_x_y_angular()
        model_group = 'los'
        all_lens_models = _prepare_all_lens_models(model_group)
        kwargs_lens = _prepare_kwargs_lens(model_group)
        kwargs_lens['model_index'] = 0
        expected = jnp.array([[-0.42939782, -0.16496612, -0.05045471],
                            [0.03707895, -0.3547466, 0.47889906]])

        calculate_derivatives = self.variant(functools.partial(
            image_simulation._calculate_derivatives,
            all_lens_models=all_lens_models))
        a_stack = jnp.array(calculate_derivatives(kwargs_lens, x, y))

        np.testing.assert_allclose(a_stack, expected, rtol=1e-5)

        # Also check that model_index of -1 returns the identity
        kwargs_lens['model_index'] = -1
        a_stack = jnp.array(calculate_derivatives(kwargs_lens, x, y))
        np.testing.assert_allclose(a_stack, jnp.zeros_like(expected), rtol=1e-5)

    @chex.all_variants
    def test__surface_brightness(self):
        x, y = _prepare_x_y_angular()
        all_source_models = _prepare_all_source_models()
        kwargs_source_slice = _prepare_kwargs_source_slice()
        expected = jnp.array([0.28131152, 0.26588708, 0.32980748])

        surface_brightness = self.variant(functools.partial(
            image_simulation._surface_brightness,
            all_source_models=all_source_models))

        np.testing.assert_allclose(
            surface_brightness(x, y, kwargs_source_slice), expected, rtol=1e-5)

    @chex.all_variants
    def test__add_surface_brightness(self):
        x, y = _prepare_x_y_angular()
        all_source_models = _prepare_all_source_models()
        kwargs_source = _prepare_kwargs_source()
        expected = jnp.array([-0.0836301, 0.06599196, 0.23668436])

        add_surface_brightness = self.variant(
            functools.partial(image_simulation._add_surface_brightness,
            all_source_models=all_source_models))

        # Use x as previous surface brightness.
        tb, b = add_surface_brightness(x, kwargs_source, x, y)

        np.testing.assert_allclose(tb, expected + x, rtol=1e-5)
        np.testing.assert_allclose(b, expected, rtol=1e-5)


if __name__ == '__main__':
    absltest.main()
