"""Push-T low-dimensional contact-Langevin bridge with fixed damping."""

from action_bridge.configs.base import pusht_lowdim_config


def get_config():
    config = pusht_lowdim_config("continuous")
    config.reference.type = "contact_langevin"
    config.reference.coordinate_mode = "absolute_action"
    config.reference.potential_type = "none"
    config.reference.gamma_mode = "constant"
    config.reference.gamma_const = 0.2
    config.reference.sigma = 7.0
    config.reference.beta_kl = 0.001
    config.model.control_scale = 1.0
    config.loss.beta_z_start = 0.001
    config.loss.beta_z_end = 0.01
    config.loss.beta_z_warmup_steps = 5000
    return config
