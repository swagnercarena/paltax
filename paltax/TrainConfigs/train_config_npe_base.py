import pathlib

import ml_collections
from ml_collections.config_dict import FieldReference


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.rng_key = 0

    # Determine the training scheme for this config.
    config.train_type = 'NPE'

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/../InputConfigs/input_config_br.py'

    # As defined in the `models` module.
    config.model = 'ResNet50'

    config.momentum = 0.9
    config.batch_size = 32

    config.cache = False
    config.half_precision = False

    config.steps_per_epoch = FieldReference(3900)
    config.num_train_steps = 500 * config.get_ref('steps_per_epoch')
    config.keep_every_n_steps = config.get_ref('steps_per_epoch')

    # Parameters of the learning rate schedule
    config.learning_rate = 0.01
    config.schedule_function_type = 'cosine'
    config.warmup_steps = 10 * config.get_ref('steps_per_epoch')

    return config
