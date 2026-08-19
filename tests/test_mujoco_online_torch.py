from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from phi_mujoco.evaluation import PolicyInput

from action_bridge.config import load_config, to_plain_dict
from action_bridge.eval.mujoco_online.metadata import OnlineEvaluationMetadata
from action_bridge.eval.mujoco_online.torch_backend import (
    TorchInferenceBackend,
    load_torch_policy_adapter,
)
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.training.common import build_model


def _metadata_dict(*, policy_type: str = "direct_bc") -> dict[str, object]:
    return {
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
        "policy_type": policy_type,
        "latent_commitment": "episode",
        "deterministic_latent": True,
        "clip_actions": False,
    }


def _direct_config() -> dict[str, object]:
    metadata = _metadata_dict()
    return {
        "benchmark": "mujoco_planar_reach",
        "obs_dim": 8,
        "action_dim": 2,
        "obs_history": 2,
        "action_history": 2,
        "chunk_horizon": 3,
        "model": {
            "policy_type": "direct_bc",
            "hidden_dim": 8,
            "h_emb_dim": 8,
            "depth": 2,
        },
        "inference": {
            "deterministic": True,
            "latent_commitment": "episode",
        },
        "data": {
            "normalize": True,
            "normalization_stats": metadata["normalization"],
            "collection_identity": metadata["collection_identity"],
        },
        "online_evaluation": metadata,
    }


def _policy_input() -> PolicyInput:
    return PolicyInput(
        observation=np.arange(8, dtype=np.float32) / 10.0,
        task_name="planar_reach",
        variation_id=0,
        episode_step=0,
    )


def test_real_torch_checkpoint_inference_matches_reconstructed_model(
    tmp_path: Path,
) -> None:
    config = _direct_config()
    torch.manual_seed(3)
    model = build_model(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    expected = torch.tensor([0.10, -0.10, 0.20, -0.20, 0.30, -0.30], dtype=torch.float32)
    with torch.no_grad():
        model.head[-1].bias.copy_(expected)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "config": config,
            "online_evaluation": config["online_evaluation"],
            "model_state": model.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="trusted_checkpoint"):
        load_torch_policy_adapter(checkpoint)
    adapter = load_torch_policy_adapter(
        checkpoint,
        trusted_checkpoint=True,
        device="cpu",
    )
    adapter.reset(task_name="planar_reach", variation_id=0, seed=11)
    output = adapter.predict(_policy_input())

    obs_hist = torch.from_numpy(np.repeat(_policy_input().observation[None, None], 2, axis=1))
    act_hist = torch.zeros((1, 2, 2), dtype=torch.float32)
    with torch.inference_mode():
        direct_output = model(obs_hist, act_hist).numpy()[0]
    np.testing.assert_array_equal(output.actions, direct_output)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert adapter.checkpoint_identifier == f"sha256:{digest}"


def test_continuous_action_bridge_deterministic_latent_is_prior_mean() -> None:
    config = load_config("mujoco_planar_reach_continuous")
    config.chunk_horizon = 3
    config.model.hidden_dim = 8
    config.model.h_emb_dim = 8
    config.model.z_embed_dim = 4
    config.model.z_dim = 2
    config.reference.hidden_dim = 8
    config.reference.time_emb_dim = 4
    model = build_model(config)
    assert isinstance(model, ActionBridgePolicy)
    metadata = OnlineEvaluationMetadata.from_mapping(_metadata_dict(policy_type="action_bridge"))
    backend = TorchInferenceBackend(
        model=model,
        metadata=metadata,
        device=torch.device("cpu"),
    )
    history_embedding = torch.randn(1, 8)

    latent, latent_embedding, _ = backend._latent(model, history_embedding)
    expected_mean, _ = model.latent.prior_params(history_embedding)

    assert latent is not None
    torch.testing.assert_close(latent, expected_mean)
    torch.testing.assert_close(latent_embedding, model.latent.embed(expected_mean))


def test_action_bridge_episode_latent_is_reused_until_reset(monkeypatch) -> None:
    config = load_config("mujoco_planar_reach_continuous")
    config.chunk_horizon = 3
    config.model.hidden_dim = 8
    config.model.h_emb_dim = 8
    config.model.z_embed_dim = 4
    config.model.z_dim = 2
    config.reference.hidden_dim = 8
    config.reference.time_emb_dim = 4
    model = build_model(config)
    assert isinstance(model, ActionBridgePolicy)
    metadata = OnlineEvaluationMetadata.from_mapping(_metadata_dict(policy_type="action_bridge"))
    backend = TorchInferenceBackend(
        model=model,
        metadata=metadata,
        device=torch.device("cpu"),
    )
    calls = 0
    original = backend._latent

    def counted_latent(model, history_embedding):
        nonlocal calls
        calls += 1
        return original(model, history_embedding)

    monkeypatch.setattr(backend, "_latent", counted_latent)
    batch = {
        "obs_hist": np.zeros((1, 2, 8), dtype=np.float32),
        "act_hist": np.zeros((1, 2, 2), dtype=np.float32),
    }
    backend.reset(seed=7)
    _, first_diagnostics = backend.predict(batch)
    _, second_diagnostics = backend.predict(batch)
    assert calls == 1
    assert first_diagnostics["episode_latent_reused"] is False
    assert second_diagnostics["episode_latent_reused"] is True

    backend.reset(seed=8)
    backend.predict(batch)
    assert calls == 2


def test_loader_rejects_checkpoint_config_drift_before_model_reconstruction(
    tmp_path: Path,
) -> None:
    config = _direct_config()
    config["obs_dim"] = 7
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "config": config,
            "online_evaluation": config["online_evaluation"],
            "model_state": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="obs_dim=7 disagrees"):
        load_torch_policy_adapter(checkpoint, trusted_checkpoint=True)


def test_checkpoint_config_is_plain_json_compatible() -> None:
    # Guard the test fixture against accidentally relying on ConfigDict semantics.
    assert to_plain_dict(_direct_config()) == _direct_config()
