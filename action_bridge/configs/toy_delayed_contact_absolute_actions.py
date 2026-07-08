"""Delayed toy contact-Langevin bridge trained on absolute target actions."""

from action_bridge.configs.toy_delayed_contact_fixed_damping import get_config as fixed_config


def get_config():
    config = fixed_config()
    config.data.train_absolute_actions = True
    config.data.env_accepts_absolute_actions = False
    config.reference.coordinate_mode = "absolute_action"
    return config
