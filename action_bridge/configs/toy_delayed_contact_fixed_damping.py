"""Delayed toy contact-Langevin bridge with fixed damping in raw delta space."""

from action_bridge.configs.base import toy_delayed_config


def get_config():
    config = toy_delayed_config("categorical")
    config.reference.type = "contact_langevin"
    config.reference.coordinate_mode = "raw_action"
    config.reference.potential_type = "none"
    config.reference.gamma_mode = "constant"
    config.reference.gamma_const = 0.2
    config.reference.beta_kl = 0.01
    config.reference.sigma = 0.05
    config.model.control_scale = 1.0
    return config
