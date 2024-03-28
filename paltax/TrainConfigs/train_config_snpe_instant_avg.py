from paltax.TrainConfigs import train_config_snpe_base


def get_config():
    """Get the default hyperparameter configuration."""
    config = train_config_snpe_base.get_config()

    # Change to averaging the posteriors.
    config.prop_decay_factor = -1.0

    return config
