import pathlib

import jax.numpy as jnp

from  paltax.TrainConfigs import train_config_maf_base


def get_config():
    """Get the default hyperparameter configuration."""
    config = train_config_maf_base.get_config()

    # Search for the input configuration relative to this config file to ease
    # use accross filesystems.
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/../InputConfigs/input_config_wdm_wide.py'

    # Sequential prior and initial proposal
    config.mu_prior = jnp.zeros(12)
    config.prec_prior = jnp.diag(jnp.ones(config.mu_prior.shape))

    config.wandb_project = 'sl-wdm-wide'

    return config
