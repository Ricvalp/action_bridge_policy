"""Push-T low-dimensional contact-Langevin bridge with learned damping."""

from action_bridge.configs.pusht_lowdim_contact_fixed_damping import get_config as fixed_config


def get_config():
    config = fixed_config()
    config.reference.gamma_mode = "learned_scalar"
    config.reference.gamma_min = 0.0
    config.reference.gamma_max = 0.95
    return config
