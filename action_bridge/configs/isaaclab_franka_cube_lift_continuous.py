"""Continuous-latent Action Bridge baseline for PHI Franka cube lift."""

from action_bridge.configs.base import isaaclab_franka_cube_lift_config


def get_config():
    return isaaclab_franka_cube_lift_config("continuous")
