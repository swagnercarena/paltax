from  paltax.TrainConfigs import train_config_maf_base

def get_config():
    """Get the hyperparameter configuration."""
    config = train_config_maf_base.get_config()

    config.hidden_dims = [128, 128]

    return config
