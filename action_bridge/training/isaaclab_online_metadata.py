"""Build checkpoint-safe Isaac Lab metadata from instantiated datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from action_bridge.config import to_plain_dict
from action_bridge.eval.isaaclab_online.contracts import (
    ACTION_DIM,
    ACTION_PROFILE,
    COLLECTION_SCHEMA_NAME,
    COLLECTION_SCHEMA_VERSION,
    CONTROL_TIMESTEP_S,
    OBSERVATION_DIM,
    OBSERVATION_PROFILE,
    ONLINE_SCHEMA_NAME,
    ONLINE_SCHEMA_VERSION,
    TASK_ID,
    VARIATION_ID,
)
from action_bridge.eval.isaaclab_online.metadata import ActionProjection


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _dataset_value(dataset: object, name: str) -> object:
    if not hasattr(dataset, name):
        raise AttributeError(f"Isaac Lab dataset is missing required attribute {name!r}")
    return getattr(dataset, name)


def _same_dataset_contract(train_dataset: object, other: object, *, split: str) -> None:
    for name in ("obs_dim", "action_dim", "normalization_stats", "collection_identity"):
        train_value = to_plain_dict(_dataset_value(train_dataset, name))
        other_value = to_plain_dict(_dataset_value(other, name))
        if train_value != other_value:
            raise ValueError(f"Isaac Lab {split} dataset {name} disagrees with the train dataset")


def configure_isaaclab_online_metadata(
    config: Any,
    train_dataset: object,
    validation_dataset: object,
    test_dataset: object,
) -> dict[str, object]:
    """Validate dataset semantics and attach the exact batched online contract."""

    _same_dataset_contract(train_dataset, validation_dataset, split="validation")
    _same_dataset_contract(train_dataset, test_dataset, split="test")
    if int(_dataset_value(train_dataset, "obs_dim")) != OBSERVATION_DIM:
        raise ValueError("Isaac Lab dataset observation width disagrees with its named profile")
    if int(_dataset_value(train_dataset, "action_dim")) != ACTION_DIM:
        raise ValueError("Isaac Lab dataset action width disagrees with its named profile")

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
            "Isaac Lab v1 requires eval.actions_per_plan=1 so vectorized live "
            "observation history remains aligned with training"
        )
    latent_commitment = str(config.inference.latent_commitment)
    if latent_commitment not in {"chunk", "episode"}:
        raise ValueError("inference.latent_commitment must be 'chunk' or 'episode' for Isaac Lab")
    deterministic_latent = config.inference.deterministic
    if deterministic_latent is not True:
        raise ValueError("Isaac Lab v1 requires inference.deterministic=true")

    stats = _mapping(
        _dataset_value(train_dataset, "normalization_stats"),
        name="dataset.normalization_stats",
    )
    if stats.get("type") != "standard":
        raise ValueError("Isaac Lab online evaluation requires standard normalization")
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
        raise ValueError("Isaac Lab collection identity is incomplete or contains unknown fields")

    config.data.normalization_stats = to_plain_dict(stats)
    config.data.collection_identity = to_plain_dict(identity)
    config.obs_dim = OBSERVATION_DIM
    config.action_dim = ACTION_DIM
    metadata: dict[str, object] = {
        "schema_name": ONLINE_SCHEMA_NAME,
        "schema_version": ONLINE_SCHEMA_VERSION,
        "task_name": TASK_ID,
        "variation_id": VARIATION_ID,
        "observation_profile": OBSERVATION_PROFILE,
        "action_profile": ACTION_PROFILE,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "observation_history": observation_history,
        "action_history": action_history,
        "action_horizon": action_horizon,
        "actions_per_plan": actions_per_plan,
        "control_timestep_s": CONTROL_TIMESTEP_S,
        "normalization": to_plain_dict(stats),
        "collection_identity": to_plain_dict(identity),
        "action_projection": ActionProjection().to_json_dict(),
        "policy_type": str(config.model.policy_type),
        "latent_commitment": latent_commitment,
        "deterministic_latent": deterministic_latent,
    }
    existing = config.get("online_evaluation")
    if existing is not None and to_plain_dict(existing) != metadata:
        raise ValueError(
            "checkpoint/config Isaac Lab online metadata disagrees with the instantiated dataset"
        )
    config.online_evaluation = metadata
    return metadata


__all__ = ["configure_isaaclab_online_metadata"]
