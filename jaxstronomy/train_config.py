import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = 'ResNet50'

    config.momentum = 0.9
    config.batch_size = 256

    config.cache = False
    config.half_precision = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.steps_per_epoch = 1950
    config.warmup_steps = 10 * config.steps_per_epoch
    config.num_train_steps = 500 * config.steps_per_epoch
    config.keep_every_n_steps = 500
    config.num_outputs = 2
    return config
