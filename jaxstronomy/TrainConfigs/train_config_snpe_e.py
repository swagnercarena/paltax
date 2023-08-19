import ml_collections
import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = 'ResNet50'

    config.momentum = 0.9
    config.batch_size = 32

    config.cache = False
    config.half_precision = False

    # Need to set the boundaries of how long the model will train generically
    # and when the sequential training will turn on.
    config.steps_per_epoch = 15600
    config.num_initial_train_steps = config.steps_per_epoch * 100
    config.num_steps_per_refinement = config.steps_per_epoch * 10
    config.num_train_steps = config.steps_per_epoch * 500
    config.num_refinements = int((
        config.num_train_steps - config.num_initial_train_steps) /
        config.num_steps_per_refinement)

    # Decide how often to save the model in checkpoints.
    config.keep_every_n_steps = config.steps_per_epoch

    # Parameters of the learning rate schedule
    config.warmup_steps = 10 * config.steps_per_epoch
    config.refinement_base_value_multiplier = 1e-2

    config.mu_prior = jnp.zeros(11)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape)) / 25
    config.mu_prop_init = jnp.zeros(11)
    config.prec_prop_init = jnp.diag(jnp.ones(config.mu_prop_init.shape))

    return config
