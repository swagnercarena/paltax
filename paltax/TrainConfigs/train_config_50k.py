from  paltax.TrainConfigs import train_config_npe_base


def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_npe_base.get_config()

    # Limit the number of unique batches.
    config.num_unique_batches = 391

    return config
