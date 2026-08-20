from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from action_bridge.eval.isaaclab_online.torch_backend import load_torch_policy_adapter
from action_bridge.training.common import build_model


def _metadata() -> dict[str, object]:
    return {
        "schema_name": "action_bridge.isaaclab_online",
        "schema_version": 1,
        "task_name": "franka_cube_lift",
        "variation_id": 0,
        "observation_profile": "phi.isaaclab.franka_cube_lift.state.v1",
        "action_profile": "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v1",
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
            "schema_version": 1,
            "manifest_sha256": "a" * 64,
        },
        "action_projection": {
            "position_lower_m": [0.2, -0.5, 0.02],
            "position_upper_m": [0.8, 0.5, 0.8],
            "position_projection": "clamp",
            "quaternion_order": "xyzw",
            "quaternion_projection": "normalize_nonnegative_w",
            "quaternion_epsilon": 1e-8,
            "gripper_threshold": 0.0,
            "gripper_open_action": 1.0,
            "gripper_close_action": -1.0,
        },
        "policy_type": "direct_bc",
        "latent_commitment": "chunk",
        "deterministic_latent": True,
    }


def _config() -> dict[str, object]:
    metadata = _metadata()
    return {
        "benchmark": "isaaclab_franka_cube_lift",
        "obs_dim": 35,
        "action_dim": 8,
        "obs_history": 2,
        "action_history": 2,
        "chunk_horizon": 4,
        "model": {
            "policy_type": "direct_bc",
            "hidden_dim": 8,
            "h_emb_dim": 8,
            "depth": 1,
        },
        "inference": {"deterministic": True, "latent_commitment": "chunk"},
        "data": {
            "normalize": True,
            "normalization_stats": metadata["normalization"],
            "collection_identity": metadata["collection_identity"],
        },
        "online_evaluation": metadata,
    }


@dataclass
class _Input:
    observation: torch.Tensor
    episode_ids: torch.Tensor
    step_indices: torch.Tensor
    task_id: str = "franka_cube_lift"
    variation_id: int = 0


def test_checkpoint_loader_reconstructs_batched_tensor_policy(tmp_path: Path) -> None:
    config = _config()
    model = build_model(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    action = torch.tensor([0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, -1.0])
    with torch.no_grad():
        model.head[-1].bias.copy_(action.repeat(4))
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
        load_torch_policy_adapter(checkpoint, device="cpu")
    adapter = load_torch_policy_adapter(
        checkpoint, trusted_checkpoint=True, device="cpu"
    )
    adapter.reset()
    observations = torch.zeros((3, 35), dtype=torch.float32)
    observations[:, 18:25] = action[:7]
    output = adapter.predict(
        _Input(
            observations,
            torch.zeros(3, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
        )
    )

    assert output.shape == (3, 8)
    assert output.device.type == "cpu"
    torch.testing.assert_close(output, action.repeat(3, 1))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert adapter.checkpoint_identifier == f"sha256:{digest}"


def test_loader_rejects_checkpoint_geometry_before_model_reconstruction(
    tmp_path: Path,
) -> None:
    config = _config()
    config["obs_dim"] = 34
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "config": config,
            "online_evaluation": config["online_evaluation"],
            "model_state": {},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="obs_dim=34 disagrees"):
        load_torch_policy_adapter(
            checkpoint, trusted_checkpoint=True, device="cpu"
        )


def test_loader_refuses_silent_cuda_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    model = build_model(config)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "config": config,
            "online_evaluation": config["online_evaluation"],
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing silent fallback"):
        load_torch_policy_adapter(
            checkpoint,
            trusted_checkpoint=True,
            device="cuda:0",
            strict_device=True,
        )


def test_loader_requires_explicit_cuda_index_in_strict_mode(tmp_path: Path) -> None:
    config = _config()
    model = build_model(config)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "config": config,
            "online_evaluation": config["online_evaluation"],
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="explicit device index"):
        load_torch_policy_adapter(
            checkpoint,
            trusted_checkpoint=True,
            device="cuda",
            strict_device=True,
        )
