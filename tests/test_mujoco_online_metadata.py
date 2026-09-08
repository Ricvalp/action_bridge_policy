from __future__ import annotations

from copy import deepcopy

import pytest
from phi_mujoco.integrations import get_integration
from phi_mujoco.offline import StandardNormalization

from action_bridge.eval.mujoco_online.metadata import (
    OnlineEvaluationMetadata,
    OnlineMetadataError,
    resolve_online_metadata,
    validate_checkpoint_config,
)


def make_metadata(
    name="robomimic_square", *, policy_type="direct_bc", actions_per_plan=2
):
    spec = get_integration(name).spec
    obs_dim, action_dim = spec.observations["state"].shape[0], spec.action.shape[0]
    return OnlineEvaluationMetadata(
        integration=spec,
        normalization=StandardNormalization(
            "state",
            (0, 1),
            1e-6,
            (0.0,) * obs_dim,
            (1.0,) * obs_dim,
            (0.0,) * action_dim,
            (1.0,) * action_dim,
        ),
        collection_identity={"manifest_sha256": "a" * 64, "data_sha256": "b" * 64},
        splits={"train": [0, 1], "val": [2], "test": []},
        observation_history=2,
        action_history=2,
        action_horizon=3,
        actions_per_plan=actions_per_plan,
        policy_type=policy_type,
        latent_commitment="episode",
        deterministic_latent=True,
        clip_actions=True,
    )


def make_config(metadata=None):
    metadata = metadata or make_metadata()
    return {
        "benchmark": "mujoco",
        "obs_dim": metadata.observation_dim,
        "action_dim": metadata.action_dim,
        "obs_history": 2,
        "action_history": 2,
        "chunk_horizon": 3,
        "model": {
            "policy_type": metadata.policy_type,
            "hidden_dim": 8,
            "h_emb_dim": 8,
            "depth": 2,
        },
        "inference": {"deterministic": True, "latent_commitment": "episode"},
        "eval": {"actions_per_plan": metadata.actions_per_plan, "clip_actions": True},
        "data": {
            "integration": metadata.integration.name,
            "observation_profile": metadata.integration.observation_profile,
            "action_profile": metadata.integration.action_profile,
            "normalization": metadata.normalization.to_dict(),
            "collection_identity": metadata.collection_identity,
        },
        "online_evaluation": metadata.to_json_dict(),
    }


@pytest.mark.parametrize(
    "name", ["planar_reach", "robomimic_square", "robomimic_tool_hang"]
)
def test_metadata_roundtrip_and_config_agreement(name):
    metadata = make_metadata(name)
    assert OnlineEvaluationMetadata.from_mapping(metadata.to_json_dict()) == metadata
    validate_checkpoint_config(make_config(metadata), metadata)


@pytest.mark.parametrize(
    "field", ["observation_profile", "action_profile", "upstream_task_id"]
)
def test_stale_integration_profile_is_rejected(field):
    data = make_metadata().to_json_dict()
    data["integration"][field] = "stale"
    with pytest.raises(OnlineMetadataError, match="integration/profile"):
        OnlineEvaluationMetadata.from_mapping(data)


def test_old_or_missing_metadata_is_rejected():
    with pytest.raises(OnlineMetadataError, match="embedded"):
        resolve_online_metadata({})
    data = make_metadata().to_json_dict()
    data["schema_version"] = 1
    with pytest.raises(OnlineMetadataError, match="retrain"):
        OnlineEvaluationMetadata.from_mapping(data)


def test_normalization_must_use_exact_train_partition():
    data = make_metadata().to_json_dict()
    data["normalization"]["source_episode_indices"] = [0, 2]
    with pytest.raises(OnlineMetadataError, match="train split"):
        OnlineEvaluationMetadata.from_mapping(data)


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "obs_dim", 8),
        ("data", "integration", "robomimic_tool_hang"),
        ("eval", "actions_per_plan", 3),
        ("inference", "deterministic", False),
        ("data", "observation_profile", "stale"),
        ("data", "action_profile", "stale"),
    ],
)
def test_checkpoint_config_drift_is_rejected(section, field, value):
    metadata = make_metadata()
    config = deepcopy(make_config(metadata))
    (config if section is None else config[section])[field] = value
    with pytest.raises(OnlineMetadataError, match="disagrees"):
        validate_checkpoint_config(config, metadata)
