from  paltax.TrainConfigs import train_config_linear


def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_linear.get_config()

    # Parameters of the learning rate schedule
    config.learning_rate = 0.001

    return config
