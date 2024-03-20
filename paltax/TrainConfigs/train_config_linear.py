from  paltax.TrainConfigs import train_config_npe_base


def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_npe_base.get_config()

    # Parameters of the learning rate schedule
    config.schedule_function_type = 'linear'
    config.end_value_multiplier = 0.01

    return config
