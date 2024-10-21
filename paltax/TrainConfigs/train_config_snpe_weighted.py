import pathlib

from paltax.TrainConfigs import train_config_snpe_base


def get_config():
    """Get the default hyperparameter configuration."""
    config = train_config_snpe_base.get_config()

    # Need to set the boundaries of how long the model will train generically
    # and when the sequential training will turn on.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/../InputConfigs/input_weighted_config_br.py'

    return config
