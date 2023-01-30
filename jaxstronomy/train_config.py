import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = 'ResNet50'
    # `name` argument of tensorflow_datasets.builder()
    config.dataset = 'imagenet2012:5.*.*'

    config.momentum = 0.9
    config.batch_size = 4

    config.log_every_steps = 100

    config.cache = False
    config.half_precision = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = 4
    config.steps_per_epoch = 2
    config.num_outputs = 2
    return config
