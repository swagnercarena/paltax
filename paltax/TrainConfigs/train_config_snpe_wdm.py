import pathlib

import ml_collections
from ml_collections.config_dict import FieldReference
import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.rng_key = 0

    # Determine the training scheme for this config.
    config.train_type = 'SNPE'

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/../InputConfigs/input_config_wdm.py'

    # As defined in the `models` module.
    config.model = 'ResNet50'

    config.momentum = 0.9
    config.batch_size = 8

    config.cache = False
    config.half_precision = False

    # Need to set the boundaries of how long the model will train generically
    # and when the sequential training will turn on.
    config.steps_per_epoch = FieldReference(15_600) # Assuming 4 GPUs
    config.num_initial_train_steps = config.get_ref('steps_per_epoch') * 10
    config.num_steps_per_refinement = config.get_ref('steps_per_epoch') * 10
    config.num_train_steps = config.get_ref('steps_per_epoch') * 500
    config.num_refinements = ((
        config.get_ref('num_train_steps') -
         config.get_ref('num_initial_train_steps')) //
        config.get_ref('num_steps_per_refinement'))

    # Decide how often to save the model in checkpoints.
    config.keep_every_n_steps = config.get_ref('steps_per_epoch')

    # Parameters of the learning rate schedule
    config.learning_rate = 0.01
    config.warmup_steps = 10 * config.get_ref('steps_per_epoch')
    config.refinement_base_value_multiplier = 1e-1

    # Sequential prior and initial proposal
    config.mu_prior = jnp.zeros(11)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape)) / 25
    config.mu_prop_init = jnp.zeros(11)
    config.prec_prop_init = jnp.diag(jnp.ones(config.mu_prop_init.shape))

    # Decay factor that controls how the sequential proposals are built.
    config.prop_decay_factor = 0.0

    # The std deviation and mean normalization imposed by the input config.
    # This is currently hard coded into the trianing config, but should
    # be dealt with dynamically in the future.
    config.std_norm = jnp.array(
        [0.15, 0.1, 0.16, 0.16, 0.1, 0.1, 0.05, 0.05, 0.16, 0.16, 1.0]
    )
    config.mean_norm = jnp.array(
        [1.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]
    )

    return config
