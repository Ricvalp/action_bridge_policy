from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from test_mujoco_online_adapter import policy_input
from test_mujoco_online_metadata import make_config, make_metadata

from action_bridge.config import load_config
from action_bridge.eval.mujoco_online.torch_backend import (
    TorchInferenceBackend,
    load_torch_policy_adapter,
)
from action_bridge.training.common import build_model
from action_bridge.training.train_toy import save_checkpoint


@pytest.mark.parametrize("name", ["robomimic_square", "robomimic_tool_hang"])
def test_training_checkpoint_reload_matches_model(tmp_path, name):
    metadata = make_metadata(name)
    config = make_config(metadata)
    model = build_model(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.linspace(-0.3, 0.3, 21))
    checkpoint = tmp_path / "checkpoint.pt"
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(checkpoint, model, optimizer, config, 1, 0.1)
    with pytest.raises(ValueError, match="trusted_checkpoint"):
        load_torch_policy_adapter(checkpoint)
    adapter = load_torch_policy_adapter(checkpoint, trusted_checkpoint=True)
    adapter.reset(integration=metadata.integration, seed=123)
    with torch.inference_mode():
        expected = model(
            torch.zeros(1, 2, metadata.observation_dim), torch.zeros(1, 2, 7)
        )[0]
    np.testing.assert_allclose(
        adapter.predict(policy_input(metadata, 0)), expected[0].numpy()
    )
    np.testing.assert_allclose(
        adapter.predict(policy_input(metadata, 1)), expected[1].numpy()
    )
    assert (
        adapter.checkpoint_identifier
        == f"sha256:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"
    )
    assert (
        load_torch_policy_adapter(
            checkpoint, trusted_checkpoint=True, actions_per_plan=1
        ).actions_per_plan
        == 1
    )


def test_config_drift_fails_before_model_load(tmp_path):
    config = make_config()
    config["obs_dim"] = 8
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "config": config,
            "online_evaluation": config["online_evaluation"],
            "model_state": {},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="obs_dim disagrees"):
        load_torch_policy_adapter(checkpoint, trusted_checkpoint=True)


@pytest.mark.parametrize("latent_type", ["continuous", "categorical", "none"])
def test_action_bridge_training_checkpoint_can_be_reloaded(tmp_path, latent_type):
    config = load_config("mujoco_robomimic_square")
    metadata = make_metadata(policy_type="action_bridge")
    config.chunk_horizon = 3
    config.model.latent_type = latent_type
    config.model.hidden_dim = config.model.h_emb_dim = 8
    config.model.z_embed_dim, config.model.z_dim = 4, 2
    config.reference.hidden_dim, config.reference.time_emb_dim = 8, 4
    config.inference.latent_commitment = metadata.latent_commitment
    config.eval.actions_per_plan = metadata.actions_per_plan
    config.data.normalization = metadata.normalization.to_dict()
    config.data.normalization_stats = {
        "type": "standard",
        **metadata.normalization.to_dict(),
    }
    config.data.collection_identity = metadata.collection_identity
    config.online_evaluation = metadata.to_json_dict()
    model = build_model(config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint, model, torch.optim.Adam(model.parameters()), config, 1, 0.1
    )
    adapter = load_torch_policy_adapter(checkpoint, trusted_checkpoint=True)
    adapter.reset(integration=metadata.integration, seed=123)
    action = adapter.predict(policy_input(metadata, 0))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    adapter.integration.validate_action(action)


@pytest.mark.parametrize("latent_type", ["continuous", "categorical", "none"])
def test_action_bridge_latents_are_repeatable_and_episode_commitment_is_preserved(
    latent_type,
):
    config = load_config("mujoco_planar_reach_continuous")
    config.obs_dim, config.action_dim, config.chunk_horizon = 23, 7, 3
    config.model.latent_type = latent_type
    config.model.hidden_dim = config.model.h_emb_dim = 8
    config.model.z_embed_dim, config.model.z_dim = 4, 2
    config.reference.hidden_dim, config.reference.time_emb_dim = 8, 4
    metadata = make_metadata(policy_type="action_bridge")
    model = build_model(config)
    backend = TorchInferenceBackend(
        model=model, metadata=metadata, device=torch.device("cpu")
    )
    batch = {
        "obs_hist": np.zeros((1, 2, 23), dtype=np.float32),
        "act_hist": np.zeros((1, 2, 7), dtype=np.float32),
    }
    backend.reset(seed=7)
    first, first_info = backend.predict(batch)
    second, second_info = backend.predict(batch)
    np.testing.assert_array_equal(first, second)
    assert first_info["episode_latent_reused"] is False
    assert second_info["episode_latent_reused"] is True
    backend.reset(seed=7)
    repeated, info = backend.predict(batch)
    np.testing.assert_array_equal(first, repeated)
    assert info["episode_latent_reused"] is False
