"""Push-T low-dimensional contact-Langevin bridge with quadratic potential."""

from action_bridge.configs.pusht_lowdim_contact_learned_gamma import get_config as learned_gamma_config


def get_config():
    config = learned_gamma_config()
    config.reference.potential_type = "quadratic"
    config.reference.stiffness_mode = "learned_diag"
    config.reference.attractor_mode = "learned"
    config.reference.k_min = 0.0
    config.reference.k_max = 2.0
    return config
