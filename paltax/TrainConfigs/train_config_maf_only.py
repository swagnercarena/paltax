from  paltax.TrainConfigs import train_config_maf_base

def get_config():
    """Get the hyperparameter configuration."""
    config = train_config_maf_base.get_config()

    # Don't update the embedding and allow for a uniform prior
    config.update_embedding = False
    config.prec_prior *= 0

    # No refinements by default to improve stability of the MAF.
    config.num_steps_per_refinement = config.get_ref('num_train_steps')

    # Go to large weights instantly.
    config.num_initial_train_steps = 0
    config.flow_weight_schedule_type = 'constant'
    config.flow_weight_constant = 1.0

    return config
