import ml_collections
import pathlib
import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.rng_key = 0

    # Determine the training scheme for this config.
    config.train_type = 'SNPE'

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = pathlib.Path(__file__).parent.resolve()
    config.input_config_path += '../InputConfigs/input_config_br.py'

    # As defined in the `models` module.
    config.model = 'ResNet50'

    config.momentum = 0.9
    config.batch_size = 32

    config.cache = False
    config.half_precision = False

    # Need to set the boundaries of how long the model will train generically
    # and when the sequential training will turn on.
    config.steps_per_epoch = 3900 # Assuming 4 GPUs
    config.num_initial_train_steps = config.steps_per_epoch * 10
    config.num_steps_per_refinement = config.steps_per_epoch * 10
    config.num_train_steps = config.steps_per_epoch * 500
    config.num_refinements = int((
        config.num_train_steps - config.num_initial_train_steps) /
        config.num_steps_per_refinement)

    # Decide how often to save the model in checkpoints.
    config.keep_every_n_steps = config.steps_per_epoch

    # Parameters of the learning rate schedule
    config.learning_rate = 0.01
    config.warmup_steps = 10 * config.steps_per_epoch
    config.refinement_base_value_multiplier = 1e-1

    config.mu_prior = jnp.zeros(11)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape)) / 25
    config.mu_prop_init = jnp.zeros(11)
    config.prec_prop_init = jnp.diag(jnp.ones(config.mu_prop_init.shape))
    config.prop_decay_factor = 0.0

    return config
