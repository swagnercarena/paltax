from  paltax.TrainConfigs import train_config_maf_base

def get_config():
    """Get the hyperparameter configuration."""
    config = train_config_maf_base.get_config()

    config.embedding_dim = 1024
    config.hidden_dims = [1024, 1024]

    return config
