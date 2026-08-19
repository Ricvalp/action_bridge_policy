from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ml_collections import ConfigDict

from action_bridge.config import load_config
from action_bridge.training.train_toy import save_checkpoint


def _objects():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


def test_mujoco_configs_bind_the_expected_lowdim_contract() -> None:
    bridge = load_config("mujoco_planar_reach_continuous")
    bridge_no_latent = load_config("mujoco_planar_reach_no_latent")
    baseline = load_config("mujoco_planar_reach_direct_chunk_bc")
    for config in (bridge, bridge_no_latent, baseline):
        assert config.benchmark == "mujoco_planar_reach"
        assert (config.obs_dim, config.action_dim) == (8, 2)
        assert (config.obs_history, config.action_history, config.chunk_horizon) == (
            2,
            2,
            4,
        )
        assert config.eval.actions_per_plan == 1
        assert config.data.observation_profile == "phi.mujoco.planar_reach.state.v1"
        assert config.data.action_profile == "phi.mujoco.planar_reach.joint_torque.v1"
    assert bridge.model.policy_type == "action_bridge"
    assert bridge_no_latent.model.policy_type == "action_bridge"
    assert bridge_no_latent.model.latent_type == "none"
    assert baseline.model.policy_type == "direct_bc"


def test_torch_checkpoint_copies_online_evaluation_metadata_to_top_level(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    metadata = {
        "schema_name": "action_bridge.mujoco_online",
        "schema_version": 1,
    }
    config = ConfigDict({"online_evaluation": metadata})
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, step=7, best_metric=0.5)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["online_evaluation"] == metadata
    assert payload["config"]["online_evaluation"] == metadata


def test_checkpoint_without_online_metadata_preserves_legacy_shape(
    tmp_path: Path,
) -> None:
    model, optimizer = _objects()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, ConfigDict(), step=1, best_metric=1.0)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "online_evaluation" not in payload


def test_checkpoint_rejects_non_mapping_online_metadata(tmp_path: Path) -> None:
    model, optimizer = _objects()
    path = tmp_path / "nested" / "checkpoint.pt"
    with pytest.raises(TypeError, match="online_evaluation"):
        save_checkpoint(
            path,
            model,
            optimizer,
            ConfigDict({"online_evaluation": ["invalid"]}),
            step=1,
            best_metric=1.0,
        )
    assert not path.exists()
    assert not path.parent.exists()
