from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

import pytest

from action_bridge.config import load_config
from action_bridge.training.mujoco_online_metadata import (
    configure_mujoco_online_metadata,
)


class _Dataset:
    obs_dim = 8
    action_dim = 2
    normalization_stats: ClassVar[dict[str, object]] = {
        "type": "standard",
        "eps": 1e-6,
        "obs_mean": [0.0] * 8,
        "obs_std": [1.0] * 8,
        "action_mean": [0.0] * 2,
        "action_std": [1.0] * 2,
    }
    collection_identity: ClassVar[dict[str, object]] = {
        "schema_name": "phi.mujoco.episode_npz",
        "schema_version": 1,
        "manifest_sha256": "a" * 64,
    }


def test_training_metadata_binds_dataset_profiles_geometry_and_normalization() -> None:
    config = load_config("mujoco_planar_reach_continuous")
    metadata = configure_mujoco_online_metadata(
        config,
        _Dataset(),
        _Dataset(),
        _Dataset(),
    )
    assert metadata["schema_name"] == "action_bridge.mujoco_online"
    assert metadata["task_name"] == "planar_reach"
    assert metadata["observation_dim"] == 8
    assert metadata["action_dim"] == 2
    assert metadata["action_lower"] == [-2.0, -2.0]
    assert metadata["control_timestep_s"] == pytest.approx(0.02)
    assert metadata["normalization"] == _Dataset.normalization_stats
    assert metadata["collection_identity"] == _Dataset.collection_identity
    assert metadata["latent_commitment"] == "chunk"
    assert metadata["clip_actions"] is False
    assert config.online_evaluation.to_dict() == metadata
    assert config.data.collection_identity.to_dict() == _Dataset.collection_identity


def test_training_metadata_rejects_split_contract_drift() -> None:
    config = load_config("mujoco_planar_reach_continuous")
    validation = _Dataset()
    validation.normalization_stats = deepcopy(_Dataset.normalization_stats)
    validation.normalization_stats["action_mean"] = [0.1, 0.0]
    with pytest.raises(ValueError, match="validation dataset normalization_stats"):
        configure_mujoco_online_metadata(
            config,
            _Dataset(),
            validation,
            _Dataset(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("action_history", 1, "at least 2"),
        ("actions_per_plan", 2, "requires eval.actions_per_plan=1"),
        ("latent_commitment", "trajectory", "must be 'chunk' or 'episode'"),
    ],
)
def test_training_metadata_rejects_unsupported_online_geometry(
    field: str,
    value: int | str,
    message: str,
) -> None:
    config = load_config("mujoco_planar_reach_continuous")
    if field == "actions_per_plan":
        config.eval.actions_per_plan = value
    elif field == "latent_commitment":
        config.inference.latent_commitment = value
    else:
        setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        configure_mujoco_online_metadata(config, _Dataset(), _Dataset(), _Dataset())


def test_training_metadata_rejects_resume_metadata_drift() -> None:
    config = load_config("mujoco_planar_reach_continuous")
    config.online_evaluation = {"schema_name": "stale"}
    with pytest.raises(ValueError, match="disagrees"):
        configure_mujoco_online_metadata(config, _Dataset(), _Dataset(), _Dataset())
