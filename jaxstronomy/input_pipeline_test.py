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

from jaxstronomy import input_pipeline
from jaxstronomy import lens_models

class InputPipelineTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of image simulation functions."""

    @chex.all_variants
    def test_draw_sample(self):
        # Test that the function returns the dictionaries that are expected
        # and that those dictionaries contain the correct parameter values.
        draw_sample = self.variant(input_pipeline.draw_sample)
        rng = jax.random.PRNGKey(0)

        md_params, source_params, sub_params, mds_params = draw_sample(rng)

        np.testing.assert_array_equal(md_params['model_index'],
            np.array([0, 1]))
        np.testing.assert_array_equal(md_params['z_lens'], np.array([0.5, 0.5]))
        self.assertGreater(jnp.min(md_params['theta_e']), 0.0)
        self.assertLess(jnp.min(md_params['theta_e']), 3.0)

        np.testing.assert_array_equal(source_params['model_index'],
            np.array([0]))

        self.assertNotAlmostEqual(sub_params['sigma_sub'], 1.1e-3)

        # Make sure theta_e and the center is being drawn from the right lens
        # model.
        epl_index = md_params['model_index'] == input_pipeline.all_models[
            'all_main_deflector_models'].index(lens_models.EPL)
        self.assertAlmostEqual(mds_params['theta_e'],
            md_params['theta_e'][epl_index])
        self.assertAlmostEqual(mds_params['center_x'],
            md_params['center_x'][epl_index])
        self.assertAlmostEqual(mds_params['center_y'],
            md_params['center_y'][epl_index])

    @chex.all_variants
    def test_draw_images(self):
        # Test that the distribution of truths is reasonable and that the
        # images looks good.
        # TODO this test shouldn't take this long, but too many things are
        # hard coded in input_pipeline.py Once I've fixed that I should
        # circle back to this test.
        batch_size = 4
        draw_images = self.variant(functools.partial(input_pipeline.draw_images,
            batch_size=batch_size))
        rng = jax.random.PRNGKey(0)

        image, truth = draw_images(rng)

        # Just test that the data and truths vary and that they are normalized
        # as expected.
        np.testing.assert_array_less(jnp.zeros(image.shape[1:]),
            jnp.std(image, axis=0))
        np.testing.assert_array_almost_equal(
            jnp.std(image.reshape(batch_size, -1)), jnp.ones(batch_size),
            decimal=2)
        self.assertLess(jnp.abs(jnp.mean(truth)), 0.3)
        self.assertGreater(jnp.std(truth), 0.6)
