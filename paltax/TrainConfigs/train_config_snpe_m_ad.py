from paltax.TrainConfigs import train_config_snpe_base


def get_config():
    """Get the default hyperparameter configuration."""
    config = train_config_snpe_base.get_config()

    # Need to set the boundaries of how long the model will train generically
    # and when the sequential training will turn on.
    config.steps_per_epoch = 3900
    config.num_initial_train_steps = config.steps_per_epoch * 200
    config.num_steps_per_refinement = config.steps_per_epoch * 10
    config.num_train_steps = config.steps_per_epoch * 500
    config.num_refinements = int((
        config.num_train_steps - config.num_initial_train_steps) /
        config.num_steps_per_refinement)

    return config
