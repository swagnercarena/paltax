from  paltax.TrainConfigs import train_config_npe_base


def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_npe_base.get_config()

    # As defined in the `models` module.
    config.model = 'ResNet18Small'

    return config
