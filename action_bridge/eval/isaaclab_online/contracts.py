"""Simulator-free constants for the first PHI Isaac Lab policy contract."""

from __future__ import annotations

BENCHMARK_NAME = "isaaclab_franka_cube_lift"
TASK_ID = "franka_cube_lift"
VARIATION_ID = 0
OBSERVATION_PROFILE = "phi.isaaclab.franka_cube_lift.state.v1"
ACTION_PROFILE = "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v1"
OBSERVATION_DIM = 35
ACTION_DIM = 8
CONTROL_TIMESTEP_S = 0.02

COLLECTION_SCHEMA_NAME = "phi.isaaclab.episode_hdf5"
COLLECTION_SCHEMA_VERSION = 1
ONLINE_SCHEMA_NAME = "action_bridge.isaaclab_online"
ONLINE_SCHEMA_VERSION = 1

POSITION_LOWER_M = (0.20, -0.50, 0.02)
POSITION_UPPER_M = (0.80, 0.50, 0.80)
POSITION_PROJECTION = "clamp"
QUATERNION_ORDER = "xyzw"
QUATERNION_PROJECTION = "normalize_nonnegative_w"
QUATERNION_EPSILON = 1e-8
TCP_POSE_SLICE = (18, 25)
GRIPPER_THRESHOLD = 0.0
GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0

SUPPORTED_POLICY_TYPES = {
    "action_bridge",
    "ar_bc",
    "autoregressive_bc",
    "bc_smooth",
    "direct_bc",
}
SUPPORTED_LATENT_COMMITMENTS = {"chunk", "episode"}

__all__ = [
    "ACTION_DIM",
    "ACTION_PROFILE",
    "BENCHMARK_NAME",
    "COLLECTION_SCHEMA_NAME",
    "COLLECTION_SCHEMA_VERSION",
    "CONTROL_TIMESTEP_S",
    "GRIPPER_CLOSE_ACTION",
    "GRIPPER_OPEN_ACTION",
    "GRIPPER_THRESHOLD",
    "OBSERVATION_DIM",
    "OBSERVATION_PROFILE",
    "ONLINE_SCHEMA_NAME",
    "ONLINE_SCHEMA_VERSION",
    "POSITION_LOWER_M",
    "POSITION_PROJECTION",
    "POSITION_UPPER_M",
    "QUATERNION_EPSILON",
    "QUATERNION_ORDER",
    "QUATERNION_PROJECTION",
    "SUPPORTED_LATENT_COMMITMENTS",
    "SUPPORTED_POLICY_TYPES",
    "TASK_ID",
    "TCP_POSE_SLICE",
    "VARIATION_ID",
]
