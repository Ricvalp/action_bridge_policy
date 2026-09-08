"""No-latent Action Bridge starting configuration for Robomimic Square."""

from action_bridge.configs.base import mujoco_config


def get_config():
    return mujoco_config("robomimic_square", "none")
