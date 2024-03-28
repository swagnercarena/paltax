from  paltax.TrainConfigs import train_config_npe_base


def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_npe_base.get_config()

    # Parameters of the learning rate schedule
    config.schedule_function_type = 'exp_decay'
    config.decay_rate = 0.98

    return config
