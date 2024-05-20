import pathlib

from paltax.TrainConfigs import train_config_snpe_base


def get_config():
    """Get the default hyperparameter configuration."""
    config = train_config_snpe_base.get_config()

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/../InputConfigs/input_config_sr_50p.py'

    return config
