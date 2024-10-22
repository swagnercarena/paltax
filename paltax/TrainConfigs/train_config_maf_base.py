import pathlib

import ml_collections
from ml_collections.config_dict import FieldReference
import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.rng_key = 0

    # Determine the training scheme for this config.
    config.train_type = 'NF'

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/../InputConfigs/input_config_wdm.py'

    # As defined in the `models` module.
    config.embedding_model = 'ResNet50'
    config.embedding_dim = 32
    config.n_maf_layer = 5
    config.hidden_dims = [64, 64]
    config.activation = 'gelu'
    config.n_atoms = 64

    config.momentum = 0.9
    config.batch_size = 128

    config.cache = False
    config.half_precision = False

    # Need to set the boundaries of how long the model will train generically
    # and when the sequential training will turn on.
    config.steps_per_epoch = FieldReference(978) # Assuming 4 GPUs
    config.num_train_steps = config.get_ref('steps_per_epoch') * 50

    # Parameters for training the flow
    config.flow_weight_schedule_type = 'power'
    config.flow_weight_schedule_power = 1.0
    config.num_steps_per_refinement = config.get_ref('steps_per_epoch') * 10
    config.num_initial_train_steps = config.get_ref('steps_per_epoch') * 10

    # Decide how often to save the model in checkpoints.
    config.keep_every_n_steps = config.get_ref('steps_per_epoch')

    # Parameters of the learning rate schedule
    config.learning_rate = 0.001
    config.schedule_function_type = 'cosine'
    config.warmup_steps = 10 * config.get_ref('steps_per_epoch')

    # Sequential prior and initial proposal
    config.mu_prior = jnp.zeros(13)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape))

    # Wandb options.
    config.wandb_mode = 'online'
    config.wandb_project = 'sl-wdm-maf'
    config.wandb_run_name = None

    return config
