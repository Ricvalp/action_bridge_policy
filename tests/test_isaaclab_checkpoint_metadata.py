from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

import pytest

from action_bridge.config import load_config
from action_bridge.eval.isaaclab_online.metadata import (
    OnlineEvaluationMetadata,
    OnlineMetadataError,
    validate_checkpoint_config,
)
from action_bridge.training.isaaclab_online_metadata import (
    configure_isaaclab_online_metadata,
)


def _metadata_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_name": "action_bridge.isaaclab_online",
        "schema_version": 2,
        "task_name": "franka_cube_lift",
        "variation_id": 0,
        "observation_profile": "phi.isaaclab.franka_cube_lift.state.v2",
        "action_profile": "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v2",
        "observation_dim": 35,
        "action_dim": 8,
        "observation_history": 2,
        "action_history": 2,
        "action_horizon": 4,
        "actions_per_plan": 1,
        "control_timestep_s": 0.02,
        "normalization": {
            "type": "standard",
            "eps": 1e-6,
            "obs_mean": [0.0] * 35,
            "obs_std": [1.0] * 35,
            "action_mean": [0.0] * 8,
            "action_std": [1.0] * 8,
        },
        "collection_identity": {
            "schema_name": "phi.isaaclab.episode_hdf5",
            "schema_version": 2,
            "manifest_sha256": "a" * 64,
        },
        "action_projection": {
            "position_lower_m": [0.2, -0.5, 0.02],
            "position_upper_m": [0.8, 0.5, 0.8],
            "position_projection": "clamp",
            "quaternion_order": "xyzw",
            "quaternion_projection": "normalize_positive_first_largest_absolute_xyzw_component",
            "quaternion_epsilon": 1e-8,
            "gripper_threshold": 0.0,
            "gripper_open_action": 1.0,
            "gripper_close_action": -1.0,
        },
        "policy_type": "direct_bc",
        "latent_commitment": "chunk",
        "deterministic_latent": True,
    }
    value.update(overrides)
    return value


def _checkpoint_config(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": "isaaclab_franka_cube_lift",
        "obs_dim": metadata["observation_dim"],
        "action_dim": metadata["action_dim"],
        "obs_history": metadata["observation_history"],
        "action_history": metadata["action_history"],
        "chunk_horizon": metadata["action_horizon"],
        "model": {"policy_type": metadata["policy_type"]},
        "inference": {
            "deterministic": metadata["deterministic_latent"],
            "latent_commitment": metadata["latent_commitment"],
        },
        "data": {
            "observation_profile": metadata["observation_profile"],
            "action_profile": metadata["action_profile"],
            "normalize": True,
            "normalization_stats": deepcopy(metadata["normalization"]),
            "collection_identity": deepcopy(metadata["collection_identity"]),
        },
    }


class _Dataset:
    obs_dim = 35
    action_dim = 8
    observation_profile = "phi.isaaclab.franka_cube_lift.state.v2"
    action_profile = "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v2"
    normalization_stats: ClassVar[dict[str, object]] = deepcopy(
        _metadata_dict()["normalization"]
    )
    collection_identity: ClassVar[dict[str, object]] = deepcopy(
        _metadata_dict()["collection_identity"]
    )


def test_isaaclab_configs_bind_the_exact_lowdim_contract() -> None:
    continuous = load_config("isaaclab_franka_cube_lift_continuous")
    no_latent = load_config("isaaclab_franka_cube_lift_no_latent")
    direct = load_config("isaaclab_franka_cube_lift_direct_chunk_bc")
    for config in (continuous, no_latent, direct):
        assert config.benchmark == "isaaclab_franka_cube_lift"
        assert (config.obs_dim, config.action_dim) == (35, 8)
        assert (config.obs_history, config.action_history, config.chunk_horizon) == (2, 2, 4)
        assert config.eval.actions_per_plan == 1
        assert config.data.observation_profile == "phi.isaaclab.franka_cube_lift.state.v2"
        assert (
            config.data.action_profile
            == "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v2"
        )
    assert continuous.model.latent_type == "continuous"
    assert no_latent.model.latent_type == "none"
    assert direct.model.policy_type == "direct_bc"


def test_metadata_round_trip_and_checkpoint_validation() -> None:
    value = _metadata_dict()
    metadata = OnlineEvaluationMetadata.from_mapping(value)
    assert metadata.to_json_dict() == value
    validate_checkpoint_config(_checkpoint_config(value), metadata)


@pytest.mark.parametrize("profile_key", ["observation_profile", "action_profile"])
def test_checkpoint_config_rejects_profile_drift(profile_key: str) -> None:
    value = _metadata_dict()
    metadata = OnlineEvaluationMetadata.from_mapping(value)
    config = _checkpoint_config(value)
    config["data"][profile_key] = "phi.isaaclab.incompatible.v999"

    with pytest.raises(OnlineMetadataError, match=f"data.{profile_key}"):
        validate_checkpoint_config(config, metadata)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("task_name",), "another_task"),
        (("actions_per_plan",), 2),
        (("deterministic_latent",), False),
        (("action_projection", "position_lower_m"), [0.0, -0.5, 0.02]),
        (("action_projection", "quaternion_order"), "wxyz"),
        (("collection_identity", "schema_name"), "phi.isaaclab.unknown"),
    ],
)
def test_metadata_rejects_semantic_drift(path: tuple[str, ...], replacement: object) -> None:
    value = _metadata_dict()
    target = value
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = replacement
    with pytest.raises(OnlineMetadataError):
        OnlineEvaluationMetadata.from_mapping(value)


def test_training_metadata_binds_dataset_and_projection_contract() -> None:
    config = load_config("isaaclab_franka_cube_lift_direct_chunk_bc")
    metadata = configure_isaaclab_online_metadata(
        config, _Dataset(), _Dataset(), _Dataset()
    )
    assert metadata == _metadata_dict()
    assert config.online_evaluation.to_dict() == metadata
    assert config.data.collection_identity.to_dict() == _Dataset.collection_identity


def test_training_metadata_rejects_split_drift_and_multi_action_execution() -> None:
    config = load_config("isaaclab_franka_cube_lift_continuous")
    validation = _Dataset()
    validation.normalization_stats = deepcopy(_Dataset.normalization_stats)
    validation.normalization_stats["action_mean"] = [0.1] + [0.0] * 7
    with pytest.raises(ValueError, match="validation dataset normalization_stats"):
        configure_isaaclab_online_metadata(
            config, _Dataset(), validation, _Dataset()
        )

    config = load_config("isaaclab_franka_cube_lift_continuous")
    config.eval.actions_per_plan = 2
    with pytest.raises(ValueError, match="actions_per_plan=1"):
        configure_isaaclab_online_metadata(
            config, _Dataset(), _Dataset(), _Dataset()
        )


@pytest.mark.parametrize("profile_key", ["observation_profile", "action_profile"])
def test_training_metadata_rejects_dataset_and_config_profile_drift(
    profile_key: str,
) -> None:
    config = load_config("isaaclab_franka_cube_lift_direct_chunk_bc")
    validation = _Dataset()
    setattr(validation, profile_key, "phi.isaaclab.incompatible.v999")
    with pytest.raises(ValueError, match=f"validation dataset {profile_key}"):
        configure_isaaclab_online_metadata(
            config, _Dataset(), validation, _Dataset()
        )

    config = load_config("isaaclab_franka_cube_lift_direct_chunk_bc")
    setattr(config.data, profile_key, "phi.isaaclab.incompatible.v999")
    with pytest.raises(ValueError, match=f"config data.{profile_key}"):
        configure_isaaclab_online_metadata(
            config, _Dataset(), _Dataset(), _Dataset()
        )


def test_training_metadata_rejects_stale_resume_contract() -> None:
    config = load_config("isaaclab_franka_cube_lift_continuous")
    config.online_evaluation = {"schema_name": "stale"}
    with pytest.raises(ValueError, match="disagrees"):
        configure_isaaclab_online_metadata(
            config, _Dataset(), _Dataset(), _Dataset()
        )
