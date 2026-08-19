"""Build checkpoint-safe MuJoCo online metadata from instantiated datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from phi_mujoco.dataset import SCHEMA_NAME as COLLECTION_SCHEMA_NAME
from phi_mujoco.dataset import SCHEMA_VERSION as COLLECTION_SCHEMA_VERSION
from phi_mujoco.tasks.planar_reach import (
    ACTION_PROFILE,
    DEFAULT_ACTION_REPEAT,
    MODEL_TIMESTEP_S,
    OBSERVATION_PROFILE,
    TASK_ID,
    VARIATION_ID,
)

from action_bridge.config import to_plain_dict


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _dataset_value(dataset: object, name: str) -> object:
    if not hasattr(dataset, name):
        raise AttributeError(f"MuJoCo dataset is missing required attribute {name!r}")
    return getattr(dataset, name)


def _same_dataset_contract(train_dataset: object, other: object, *, split: str) -> None:
    for name in ("obs_dim", "action_dim", "normalization_stats", "collection_identity"):
        train_value = to_plain_dict(_dataset_value(train_dataset, name))
        other_value = to_plain_dict(_dataset_value(other, name))
        if train_value != other_value:
            raise ValueError(f"MuJoCo {split} dataset {name} disagrees with the train dataset")


def configure_mujoco_online_metadata(
    config: Any,
    train_dataset: object,
    validation_dataset: object,
    test_dataset: object,
) -> dict[str, object]:
    """Validate dataset semantics and attach exact online checkpoint metadata."""

    _same_dataset_contract(train_dataset, validation_dataset, split="validation")
    _same_dataset_contract(train_dataset, test_dataset, split="test")
    if int(_dataset_value(train_dataset, "obs_dim")) != OBSERVATION_PROFILE.shape[0]:
        raise ValueError("MuJoCo dataset observation width disagrees with its named profile")
    if int(_dataset_value(train_dataset, "action_dim")) != ACTION_PROFILE.shape[0]:
        raise ValueError("MuJoCo dataset action width disagrees with its named profile")

    observation_history = int(config.obs_history)
    action_history = int(config.action_history)
    action_horizon = int(config.chunk_horizon)
    if observation_history < 1:
        raise ValueError("obs_history must be positive")
    if action_history < 2:
        raise ValueError("action_history must be at least 2 for Action Bridge generation")
    if action_horizon < 1:
        raise ValueError("chunk_horizon must be positive")
    actions_per_plan = int(config.eval.actions_per_plan)
    if actions_per_plan != 1:
        raise ValueError(
            "the first MuJoCo online contract requires eval.actions_per_plan=1 so live "
            "observation history remains aligned with training"
        )
    latent_commitment = str(config.inference.latent_commitment)
    if latent_commitment not in {"chunk", "episode"}:
        raise ValueError("inference.latent_commitment must be 'chunk' or 'episode' for MuJoCo")
    deterministic_latent = config.inference.deterministic
    clip_actions = config.eval.clip_actions
    if not isinstance(deterministic_latent, bool):
        raise TypeError("inference.deterministic must be a boolean")
    if not isinstance(clip_actions, bool):
        raise TypeError("eval.clip_actions must be a boolean")

    stats = _mapping(
        _dataset_value(train_dataset, "normalization_stats"),
        name="dataset.normalization_stats",
    )
    if stats.get("type") != "standard":
        raise ValueError("MuJoCo online evaluation requires standard normalization")
    identity = _mapping(
        _dataset_value(train_dataset, "collection_identity"),
        name="dataset.collection_identity",
    )
    expected_identity = {
        "schema_name": COLLECTION_SCHEMA_NAME,
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "manifest_sha256": identity.get("manifest_sha256"),
    }
    if identity != expected_identity:
        raise ValueError("MuJoCo collection identity is incomplete or contains unknown fields")

    data_config = config.data
    data_config.normalization_stats = to_plain_dict(stats)
    data_config.collection_identity = to_plain_dict(identity)
    config.obs_dim = OBSERVATION_PROFILE.shape[0]
    config.action_dim = ACTION_PROFILE.shape[0]
    metadata: dict[str, object] = {
        "schema_name": "action_bridge.mujoco_online",
        "schema_version": 1,
        "task_name": TASK_ID,
        "variation_id": VARIATION_ID,
        "observation_profile": OBSERVATION_PROFILE.name,
        "action_profile": ACTION_PROFILE.name,
        "observation_dim": OBSERVATION_PROFILE.shape[0],
        "action_dim": ACTION_PROFILE.shape[0],
        "observation_history": observation_history,
        "action_history": action_history,
        "action_horizon": action_horizon,
        "actions_per_plan": actions_per_plan,
        "action_lower": list(ACTION_PROFILE.lower),
        "action_upper": list(ACTION_PROFILE.upper),
        "control_timestep_s": MODEL_TIMESTEP_S * DEFAULT_ACTION_REPEAT,
        "normalization": to_plain_dict(stats),
        "collection_identity": to_plain_dict(identity),
        "policy_type": str(config.model.policy_type),
        "deterministic_latent": deterministic_latent,
        "latent_commitment": latent_commitment,
        "clip_actions": clip_actions,
    }
    existing = config.get("online_evaluation")
    if existing is not None and to_plain_dict(existing) != metadata:
        raise ValueError(
            "checkpoint/config MuJoCo online metadata disagrees with the instantiated dataset"
        )
    config.online_evaluation = metadata
    return metadata


__all__ = ["configure_mujoco_online_metadata"]
