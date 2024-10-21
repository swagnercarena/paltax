# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for train.py."""

import functools
import pathlib
from tempfile import TemporaryDirectory

from absl.testing import parameterized
import chex
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from paltax import input_pipeline
from paltax import maf_flow
from paltax import models
from paltax import train
from paltax import train_nf
from paltax import train_test


def _create_train_state():
    """Create generic train state for testing."""
    rng = jax.random.PRNGKey(0)
    config = ml_collections.ConfigDict()
    embedding_module = models.ResNet18VerySmall(
        num_outputs=2, dtype=jnp.float32
    )
    maf_module = maf_flow.MAF(
        n_dim=2, n_maf_layers=2, hidden_dims=[4,4], activation='gelu'
    )
    model = maf_flow.EmbeddedFlow(embedding_module, maf_module)
    image_size = 4
    parameter_dim = 2
    learning_rate_schedule = train_test._get_learning_rate()

    train_state = train_nf.create_train_state_nf(
        rng, config, model, image_size, parameter_dim,
        learning_rate_schedule
    )

    return train_state


def _create_test_config():
    """Return a test configuration."""
    config = ml_collections.ConfigDict()
    config.rng_key = 0
    config.train_type = 'NF'

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/InputConfigs/input_config_test.py'

    # As defined in the `models` module.
    config.embedding_model = 'ResNet18VerySmall'
    config.embedding_dim = 3
    config.n_maf_layer = 2
    config.hidden_dims = [8, 8]
    config.activation = 'gelu'
    config.n_atoms = 16

    config.momentum = 0.9
    config.batch_size = 32

    config.cache = False
    config.half_precision = False

    # One step of training and one step of refinement.
    config.steps_per_epoch = ml_collections.config_dict.FieldReference(1)
    config.num_initial_train_steps = config.get_ref('steps_per_epoch') * 1
    config.num_steps_per_refinement = config.get_ref('steps_per_epoch') * 1
    config.num_train_steps = config.get_ref('steps_per_epoch') * 2
    config.num_refinements = ((
        config.get_ref('num_train_steps') -
         config.get_ref('num_initial_train_steps')) //
        config.get_ref('num_steps_per_refinement'))

    # Decide how often to save the model in checkpoints.
    config.keep_every_n_steps = config.get_ref('steps_per_epoch')

    # Parameters of the learning rate schedule
    config.learning_rate = 0.01
    config.warmup_steps = 1 * config.get_ref('steps_per_epoch')
    config.refinement_base_value_multiplier = 0.5
    config.flow_weight_schedule_type = 'linear'

    config.mu_prior = jnp.zeros(5)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape)) / 25

    return config


class TrainNFTests(chex.TestCase, parameterized.TestCase):
    """Runs tests of training functions."""

    def test_initialized(self):
        # Test the the model includes initialized weights and biases and
        # can be applied.
        rng = jax.random.PRNGKey(0)
        embedding_module = models.ResNet18VerySmall(
            num_outputs=2, dtype=jnp.float32
        )
        maf_module = maf_flow.MAF(
            n_dim=2, n_maf_layers=2, hidden_dims=[4,4], activation='gelu'
        )
        model = maf_flow.EmbeddedFlow(embedding_module, maf_module)
        image_size = 128
        params, batch_stats = train_nf.initialized(rng, image_size, 2, model)

        self.assertTupleEqual(
            params['embedding_module']['conv_init']['kernel'].shape,
            (7, 7, 1, 8)
        )
        self.assertTupleEqual(
            batch_stats['embedding_module']['bn_init']['mean'].shape,
            (8,)
        )

        output, _ = model.apply(
            {'params':params, 'batch_stats':batch_stats},
            jax.random.normal(rng, shape=(3, 2)),
            jax.random.normal(rng, shape=(3, image_size, image_size, 1)),
            mutable=['batch_stats']
        )
        self.assertTupleEqual(output.shape, (3,))

    def test_get_optimizer(self):
        # Test that an optimizer is returned that forks on the parameter type.
        optimizer = 'adam'
        learning_rate_schedule = 1e-3
        params = {
            'train_param': jnp.ones((5,5)),
            'freeze_param': jnp.zeros(2, dtype=int)
        }

        optimizer, opt_mask = train_nf.get_optimizer(
            optimizer, learning_rate_schedule, params
        )
        self.assertTrue(opt_mask['freeze_param'])
        self.assertFalse(opt_mask['train_param'])

    def test_create_state_nf(self):
        # Test that the train state is created without issue.
        train_state = _create_train_state()
        self.assertEqual(train_state.step, 0)
        opt_mask = train_state.opt_mask
        self.assertFalse(
            opt_mask['embedding_module']['Dense_0']['bias']
        )
        self.assertTrue(
            opt_mask['flow_module']['permute_layers_0']['permutation']
        )

    @chex.all_variants
    def test_draw_sample(self):
        # Test that the samples are drawn from the lensing config and flow
        # with the correct ratio.
        rng = jax.random.PRNGKey(0)
        config = _create_test_config()
        input_config = train._get_config(config.input_config_path)
        input_config['lensing_config']['main_deflector_params']['theta_e'] = (
            input_pipeline.encode_constant(1.0)
        )
        cosmology_params = input_pipeline.initialize_cosmology_params(
            input_config, rng
        )
        n_dim = len(input_config['truth_parameters'][0])
        n_cont = 2
        maf_module = maf_flow.MAF(
            n_dim=n_dim, n_maf_layers=2, hidden_dims=[4,4], activation='gelu'
        )
        maf_params = maf_module.init(
            rng, jnp.ones((1, n_dim)), jnp.ones((1, n_cont))
        )

        draw_sample = self.variant(
            functools.partial(
                train_nf.draw_sample, batch_size=config.batch_size,
                input_config=input_config, cosmology_params=cosmology_params,
                normalize_config=input_config['lensing_config'],
                flow_module=maf_module
            ),
        )

        # Test that when no weight is on the flow, all the draws come from the
        # lensing config.
        flow_weight = 0.0
        context = jnp.zeros(n_cont)
        truth, nan_fraction = draw_sample(
            rng, context, maf_params, flow_weight
        )
        self.assertAlmostEqual(nan_fraction, 0.0)
        self.assertAlmostEqual(jnp.std(truth[:, 0]), 0.0)

        # Test that with 50 percent weight on the flow half the draws come from
        # the flow.
        flow_weight = 0.5
        truth_flow, nan_fraction = draw_sample(
            rng, context, maf_params, flow_weight
        )
        self.assertAlmostEqual(nan_fraction, 0.0)
        self.assertAlmostEqual(jnp.std(truth_flow[16:, 0]), 0.0)
        self.assertGreater(jnp.std(truth_flow[:16, 0]), 0.0)

        # Test that the context impacts the output.
        context = jnp.ones(n_cont)
        truth_flow_nc, nan_fraction = draw_sample(
            rng, context, maf_params, flow_weight
        )
        self.assertAlmostEqual(nan_fraction, 0.0)
        np.testing.assert_array_almost_equal(
            truth_flow[16:], truth_flow_nc[16:]
        )
        np.testing.assert_array_less(
            jnp.zeros(truth_flow[:16].shape),
            jnp.abs(truth_flow[:16] - truth_flow_nc[:16])
        )

    @chex.all_variants(without_device=False)
    def test_extract_flow_context(self):
        # Test that the context and the flow parameters are extracted
        # correctly from the call.
        # Manually define the model and parameters in the same way
        # that train state will define them.
        rng = jax.random.PRNGKey(0)
        config = _create_test_config()
        parameter_dim = 2
        image_size = 16
        embedding_module = getattr(models, config.embedding_model)(
            num_outputs=config.embedding_dim
        )
        flow_module = maf_flow.MAF(
            parameter_dim, config.n_maf_layer, config.hidden_dims,
            config.activation
        )
        model = maf_flow.EmbeddedFlow(embedding_module, flow_module)
        params, batch_stats = train_nf.initialized(
            rng, image_size, parameter_dim, model
        )

        # Extract context.
        context = jax.random.normal(rng, (10, image_size, image_size, 1))
        embed_context, _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            context, method='embed_context', mutable=('batch_stats',),
            train=False
        )

        # Create the train state and extract the parameters and context.
        learning_rate_schedule = train_test._get_learning_rate()
        train_state = train_nf.create_train_state_nf(
            rng, config, model, image_size, parameter_dim,
            learning_rate_schedule
        )

        extract_flow_context = self.variant(train_nf.extract_flow_context)
        flow_params, embed_context_test = extract_flow_context(
            train_state, context[0]
        )

        # Check that the resulting contexts are equivalent.
        np.testing.assert_array_almost_equal(
            embed_context[0], embed_context_test
        )
        np.testing.assert_array_almost_equal(
            flow_params['params']['made_layers_0']['MaskedDense_0']['kernel'],
            params['flow_module']['made_layers_0']['MaskedDense_0']['kernel']
        )

    def test_get_flow_weight_schedule(self):
        # Check that the flow weight schedule is zero at first and then
        # linearly increases as the training continues.
        config = _create_test_config()
        config.get_ref('steps_per_epoch').set(4)
        flow_weight_schedule = train_nf.get_flow_weight_schedule(config)
        self.assertEqual(flow_weight_schedule(0), 0.0)
        self.assertEqual(flow_weight_schedule(3), 0.0)
        self.assertEqual(flow_weight_schedule(5), 0.25)
        self.assertEqual(flow_weight_schedule(7), 0.75)

    @chex.all_variants
    def test_gaussian_log_prob(self):
        # Test that the gaussian log probability is correctly calculated.
        mean = jnp.array([0.0, 1.0])
        prec = jnp.eye(2) * 0.5
        truth = jnp.array([
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]]
        ])
        gaussian_log_prob = self.variant(train_nf.gaussian_log_prob)

        result = gaussian_log_prob(mean, prec, truth)
        expected = jnp.array([[0.0, -0.25], [-0.5, -0.25]])
        np.testing.assert_array_almost_equal(result, expected)

    @chex.all_variants
    def test_apt_loss(self):
        # Test that the apt_loss converges to the correct value on certain
        # limits.
        rng = jax.random.PRNGKey(0)
        n_samples = 100
        n_atoms = 1_000

        apt_loss = self.variant(train_nf.apt_loss)

        # Test for the case where posterior and prior are the same.
        samples = jax.random.normal(rng, (n_samples, n_atoms))
        log_posterior = -0.5 * jnp.square(samples)
        log_prior = -0.5 * jnp.square(samples)
        # Should just be -log(1/n_atoms).
        loss = apt_loss(log_posterior, log_prior)
        np.testing.assert_array_almost_equal(
            loss,
            jnp.sum(np.ones_like(loss) * jnp.log(n_atoms))
        )

    @chex.all_variants
    def test_apt_get_atoms(self):
        # Test that the correct shape is returned, and that no batch gets
        # assigned contrastive examples with the same index.
        batch_size = 10
        truth = jnp.arange(batch_size)
        rng = jax.random.PRNGKey(0)
        n_atoms = 4

        apt_get_atoms = self.variant(train_nf.apt_get_atoms, static_argnums=[2])

        # Check for a subsample of atoms
        truth_atoms = apt_get_atoms(rng, truth, n_atoms)
        self.assertTupleEqual(truth_atoms.shape, (batch_size, n_atoms))
        np.testing.assert_array_almost_equal(
            truth_atoms[:, 0], truth
        )
        self.assertAlmostEqual(
            0.0, jnp.sum(truth_atoms[:, 1:] == truth[:, None])
        )

        # Check for a larger number of atoms
        n_atoms = batch_size
        truth_atoms = apt_get_atoms(rng, truth, n_atoms)
        for i in range(batch_size):
            np.testing.assert_array_almost_equal(
                jnp.sort(truth_atoms[i]), truth
            )

    def test_train_step(self):
        # Test that a single iteration of the train step succesfully executes.
        rng = jax.random.PRNGKey(0)
        state = _create_train_state()
        batch_size = 8

        # Set up our objects for training.
        image = jnp.ones((batch_size, 4, 4, 1))
        truth = jnp.ones((batch_size, 2))
        batch = {'truth': truth, 'image': image}
        mu_prior = jnp.zeros(2)
        prec_prior = jnp.eye(2)
        learning_rate_schedule = train_test._get_learning_rate()
        n_atoms = batch_size

        rng_train = jax.random.split(rng, num=jax.device_count())
        p_train_step = jax.pmap(
            functools.partial(
                train_nf.train_step,
                learning_rate_schedule=learning_rate_schedule,
                n_atoms=n_atoms, opt_mask=state.opt_mask
            ),
            in_axes=(0, 0, 0, None, None),
            axis_name='batch'
        )
        new_state, metrics = p_train_step(
            rng_train, jax_utils.replicate(state), jax_utils.replicate(batch),
            mu_prior, prec_prior
        )
        self.assertEqual(new_state.step, 1)
        self.assertTrue('loss' in metrics)

    def test_train_and_evaluate_nf(self):
        # Test that we can initialize a model from a test configuration and
        # train it.
        config = _create_test_config()
        input_config = train._get_config(config.input_config_path)
        rng = jax.random.PRNGKey(2)
        target_image = jax.random.normal(rng, shape=(4, 4))

        log_prob_batches = {
            'target_train':
            {
                'truth': jnp.ones((32,5)),
                'image': jnp.ones((32, 4, 4))
            }
        }

        config.wandb_mode = 'disabled'

        with TemporaryDirectory() as tmp_dir_name:
            state = train_nf.train_and_evaluate_nf(
                config, input_config, tmp_dir_name, rng, target_image,
                log_prob_batches=log_prob_batches
            )
            self.assertEqual(state.step, 2)
