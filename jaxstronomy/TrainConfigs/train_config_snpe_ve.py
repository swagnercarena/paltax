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

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.steps_per_epoch = 15600
    config.num_initial_train_steps = 70200 
    config.num_steps_per_refinement = config.steps_per_epoch * 495
    config.num_refinements = 1

    config.num_train_steps = config.num_initial_train_steps
    config.num_train_steps += config.num_steps_per_refinement * config.num_refinements

    config.keep_every_n_steps = 500

    # Parameters of the learning rate schedule
    config.schedule_function_type = 'cosine'
    config.warmup_steps = 10 * config.steps_per_epoch

    config.mu_prior = jnp.zeros(5)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape)) / 25
    config.mu_prop_init = jnp.zeros(5)
    config.prec_prop_init = jnp.diag(jnp.ones(config.mu_prop_init.shape))

    return config
