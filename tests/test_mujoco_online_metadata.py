from __future__ import annotations

import json
from copy import deepcopy

import pytest

from action_bridge.eval.mujoco_online.metadata import (
    OnlineEvaluationMetadata,
    OnlineMetadataError,
    load_online_metadata,
    resolve_online_metadata,
    validate_checkpoint_config,
)


def _metadata_dict(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_name": "action_bridge.mujoco_online",
        "schema_version": 1,
        "task_name": "planar_reach",
        "variation_id": 0,
        "observation_profile": "phi.mujoco.planar_reach.state.v1",
        "action_profile": "phi.mujoco.planar_reach.joint_torque.v1",
        "observation_dim": 8,
        "action_dim": 2,
        "observation_history": 2,
        "action_history": 2,
        "action_horizon": 3,
        "actions_per_plan": 1,
        "action_lower": [-2.0, -2.0],
        "action_upper": [2.0, 2.0],
        "control_timestep_s": 0.02,
        "normalization": {
            "type": "standard",
            "eps": 1e-6,
            "obs_mean": [0.0] * 8,
            "obs_std": [1.0] * 8,
            "action_mean": [0.0, 0.0],
            "action_std": [1.0, 1.0],
        },
        "collection_identity": {
            "schema_name": "phi.mujoco.episode_npz",
            "schema_version": 1,
            "manifest_sha256": "a" * 64,
        },
        "policy_type": "direct_bc",
        "latent_commitment": "episode",
        "deterministic_latent": True,
        "clip_actions": False,
    }
    value.update(overrides)
    return value


def _checkpoint_config(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": "mujoco_planar_reach",
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
            "normalize": True,
            "normalization_stats": deepcopy(metadata["normalization"]),
            "collection_identity": deepcopy(metadata["collection_identity"]),
        },
    }


def test_metadata_round_trip_uses_exact_phi_mujoco_contract() -> None:
    value = _metadata_dict()
    metadata = OnlineEvaluationMetadata.from_mapping(value)

    assert metadata.to_json_dict() == value
    validate_checkpoint_config(_checkpoint_config(value), metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_name", "another_task"),
        ("observation_profile", "unknown"),
        ("action_history", 1),
        ("actions_per_plan", 2),
        ("action_lower", [-1.0, -2.0]),
        ("control_timestep_s", 0.01),
        ("latent_commitment", "sticky"),
        ("clip_actions", 1),
    ],
)
def test_metadata_rejects_runtime_semantic_drift(field: str, value: object) -> None:
    with pytest.raises(OnlineMetadataError):
        OnlineEvaluationMetadata.from_mapping(_metadata_dict(**{field: value}))


def test_metadata_rejects_unknown_keys_and_invalid_statistics() -> None:
    with pytest.raises(OnlineMetadataError, match="extra"):
        OnlineEvaluationMetadata.from_mapping(_metadata_dict(unknown=True))

    value = _metadata_dict()
    normalization = deepcopy(value["normalization"])
    assert isinstance(normalization, dict)
    normalization["obs_std"] = [1.0] * 7 + [0.0]
    with pytest.raises(OnlineMetadataError, match="positive"):
        OnlineEvaluationMetadata.from_mapping(_metadata_dict(normalization=normalization))


def test_strict_json_loader_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "online.json"
    path.write_text('{"schema_name":"one","schema_name":"two"}', encoding="utf-8")

    with pytest.raises(OnlineMetadataError, match="duplicate JSON key"):
        load_online_metadata(path)


def test_metadata_resolution_requires_explicit_agreement(tmp_path) -> None:
    embedded = _metadata_dict()
    checkpoint = {"online_evaluation": embedded}
    assert resolve_online_metadata(checkpoint).to_json_dict() == embedded

    explicit = deepcopy(embedded)
    explicit["clip_actions"] = True
    path = tmp_path / "online.json"
    path.write_text(json.dumps(explicit), encoding="utf-8")
    with pytest.raises(OnlineMetadataError, match="disagrees"):
        resolve_online_metadata(checkpoint, explicit_path=path)


def test_checkpoint_config_must_repeat_data_contract() -> None:
    value = _metadata_dict()
    metadata = OnlineEvaluationMetadata.from_mapping(value)
    config = _checkpoint_config(value)
    data = config["data"]
    assert isinstance(data, dict)
    identity = data["collection_identity"]
    assert isinstance(identity, dict)
    identity["manifest_sha256"] = "b" * 64

    with pytest.raises(OnlineMetadataError, match="collection_identity disagrees"):
        validate_checkpoint_config(config, metadata)
