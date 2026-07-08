"""Delayed toy contact-Langevin bridge over absolute paths decoded to deltas."""

from action_bridge.configs.toy_delayed_contact_fixed_damping import get_config as fixed_config


def get_config():
    config = fixed_config()
    config.data.train_absolute_actions = False
    config.reference.coordinate_mode = "absolute_from_delta"
    return config
