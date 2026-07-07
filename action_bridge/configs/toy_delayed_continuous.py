"""Delayed-branch toy config with a continuous chunk latent."""

from action_bridge.configs.base import toy_delayed_config


def get_config():
    return toy_delayed_config("continuous")
