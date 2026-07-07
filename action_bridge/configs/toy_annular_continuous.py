"""Annular toy config with a continuous chunk latent."""

from action_bridge.configs.base import toy_annular_config


def get_config():
    return toy_annular_config("continuous")
