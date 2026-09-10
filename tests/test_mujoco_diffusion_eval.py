from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from test_mujoco_online_adapter import policy_input
from test_mujoco_online_metadata import make_metadata

from action_bridge.config import load_config
from action_bridge.eval.eval_mujoco import evaluate_mujoco_offline
from action_bridge.eval.mujoco_online.torch_backend import (
    TorchInferenceBackend,
    load_torch_policy_adapter,
)
from action_bridge.eval.rollout import predict_actions
from action_bridge.training.common import build_model
from action_bridge.training.train_toy import save_checkpoint


@pytest.fixture
def tiny_diffusion():
    metadata = make_metadata(policy_type="diffusion")
    stats = replace(
        metadata.normalization,
        obs_mean=(0.25,) * metadata.observation_dim,
        obs_std=(2.0,) * metadata.observation_dim,
        action_mean=tuple(np.linspace(-0.2, 0.3, 7)),
        action_std=tuple(np.linspace(0.1, 1.3, 7)),
    )
    metadata = replace(
        metadata, action_horizon=8, normalization=stats, latent_commitment="chunk"
    )
    config = load_config("mujoco_robomimic_square_diffusion")
    config.model.unet_channels = [8, 16, 32]
    config.model.hidden_dim = config.model.h_emb_dim = 16
    config.model.time_emb_dim = 16
    config.model.num_inference_steps = 5
    config.chunk_horizon = metadata.action_horizon
    config.eval.actions_per_plan = metadata.actions_per_plan
    config.eval.batch_size = 2
    config.eval.sampling_seed = 123
    config.data.normalization = stats.to_dict()
    config.data.normalization_stats = stats.to_dict()
    config.data.collection_identity = metadata.collection_identity
    config.online_evaluation = metadata.to_json_dict()
    return build_model(config).eval(), config, metadata


class _ValidationDataset:
    def __init__(self, metadata, *, target_offset=0.0):
        from phi_mujoco.integrations import get_integration

        self.integration = get_integration(metadata.integration.name)
        self.spec = self.integration.spec
        self.stats = metadata.normalization
        self.horizon = metadata.action_horizon
        self.target_offset = target_offset

    def __len__(self):
        return 3

    def __getitem__(self, index):
        actions = np.full((self.horizon, 7), index * 0.1 + self.target_offset)
        return {
            "obs_hist": self.stats.normalize_observations(np.full((2, 23), index * 0.1)),
            "act_hist": self.stats.normalize_actions(np.full((2, 7), index * 0.05)),
            "future_actions": self.stats.normalize_actions(actions),
        }


def test_diffusion_validation_is_seeded_target_free_and_preserves_training_rng(
    tiny_diffusion, tmp_path, monkeypatch
):
    model, config, metadata = tiny_diffusion
    dataset = _ValidationDataset(metadata)
    generated = []
    generate = model.generate

    def record_prediction(obs_hist, act_hist, *, generator=None):
        # This signature also rejects any attempt to pass future target actions.
        assert generator is not None
        prediction = generate(obs_hist, act_hist, generator=generator)
        generated.append(prediction.clone())
        return prediction

    def reject_training_loss(*args, **kwargs):
        raise AssertionError("validation must sample actions, not call diffusion_loss")

    monkeypatch.setattr(model, "generate", record_prediction)
    monkeypatch.setattr(model, "diffusion_loss", reject_training_loss)
    torch.manual_seed(777)
    training_rng = torch.get_rng_state().clone()
    first = evaluate_mujoco_offline(
        model, dataset, config, torch.device("cpu"), output_dir=tmp_path
    )
    repeated = evaluate_mujoco_offline(model, dataset, config, torch.device("cpu"))
    assert torch.equal(training_rng, torch.get_rng_state())
    assert first == repeated
    assert torch.equal(generated[0], generated[2])
    assert torch.equal(generated[1], generated[3])
    assert "normalized_path_kl_energy" not in first
    assert first["evaluated_chunks"] == 3
    assert first["evaluated_batches"] == 2

    stats = metadata.normalization
    predicted = stats.denormalize_actions(torch.cat(generated[:2]).numpy())
    targets = stats.denormalize_actions(
        np.stack([dataset[index]["future_actions"] for index in range(len(dataset))])
    )
    assert first["action_mse"] == pytest.approx(
        float(np.mean((predicted.astype(np.float64) - targets) ** 2)), rel=1e-6
    )
    offline_metadata = json.loads(
        (tmp_path / "metrics" / "mujoco_offline_metadata.json").read_text()
    )
    assert offline_metadata["prediction_protocol"] == "conditional_diffusion_chunk"
    assert offline_metadata["sampling_seed"] == 123
    assert "path_kl_coordinates" not in offline_metadata

    shifted = evaluate_mujoco_offline(
        model,
        _ValidationDataset(metadata, target_offset=0.7),
        config,
        torch.device("cpu"),
    )
    assert torch.equal(generated[0], generated[4])
    assert torch.equal(generated[1], generated[5])
    assert shifted["action_mse"] != first["action_mse"]
    config.eval.sampling_seed = 124
    evaluate_mujoco_offline(model, dataset, config, torch.device("cpu"))
    assert not torch.equal(generated[0], generated[6])
    assert torch.equal(training_rng, torch.get_rng_state())


def test_non_diffusion_validation_loader_does_not_consume_training_rng(tiny_diffusion):
    _, config, metadata = tiny_diffusion

    class ConstantPolicy(torch.nn.Module):
        def forward(self, obs_hist, act_hist):
            return obs_hist.new_zeros(obs_hist.shape[0], 8, 7)

    model = ConstantPolicy()
    training_rng = torch.get_rng_state().clone()
    evaluate_mujoco_offline(
        model, _ValidationDataset(metadata), config, torch.device("cpu")
    )
    assert torch.equal(training_rng, torch.get_rng_state())


def test_diffusion_online_rng_matches_shared_prediction_and_resets(tiny_diffusion):
    model, _, metadata = tiny_diffusion
    backend = TorchInferenceBackend(
        model=model, metadata=metadata, device=torch.device("cpu")
    )
    batch = {
        "obs_hist": np.zeros((1, 2, 23), dtype=np.float32),
        "act_hist": np.zeros((1, 2, 7), dtype=np.float32),
    }
    tensor_batch = {key: torch.from_numpy(value) for key, value in batch.items()}
    expected_rng = torch.Generator().manual_seed(123)
    training_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="reset"):
        backend.predict(batch)
    backend.reset(seed=123)
    first, diagnostics = backend.predict(batch)
    second, _ = backend.predict(batch)
    for actual in (first, second):
        expected = predict_actions(model, tensor_batch, generator=expected_rng)
        assert set(expected) == {"actions"}
        np.testing.assert_array_equal(actual, expected["actions"].numpy())
    assert not np.array_equal(first, second)
    assert diagnostics["sampler"] == "ddim"
    assert diagnostics["sampling_seed"] == 123
    assert "latent_commitment" not in diagnostics
    assert "normalized_path_kl_energy" not in diagnostics
    backend.reset(seed=123)
    np.testing.assert_array_equal(first, backend.predict(batch)[0])
    backend.reset(seed=124)
    assert not np.array_equal(first, backend.predict(batch)[0])
    assert torch.equal(training_rng, torch.get_rng_state())


def test_diffusion_checkpoint_rebuild_normalization_clipping_and_chunk_history(
    tiny_diffusion, tmp_path, monkeypatch
):
    model, config, metadata = tiny_diffusion
    # Zero noise predictions deliberately leave large denoised samples, so the
    # existing raw-controller clipping path is exercised as well as normalization.
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    path = tmp_path / "diffusion.pt"
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), config, 7, 0.5)
    with pytest.raises(ValueError, match="trusted_checkpoint"):
        load_torch_policy_adapter(path)
    adapter = load_torch_policy_adapter(path, trusted_checkpoint=True)
    assert adapter.metadata == metadata
    assert adapter.actions_per_plan == 2
    assert load_torch_policy_adapter(
        path, trusted_checkpoint=True, actions_per_plan=8
    ).actions_per_plan == 8
    observed_histories = []
    generate = adapter.backend.model.generate

    def record_histories(obs_hist, act_hist, *, generator=None):
        observed_histories.append((obs_hist.numpy().copy(), act_hist.numpy().copy()))
        return generate(obs_hist, act_hist, generator=generator)

    monkeypatch.setattr(adapter.backend.model, "generate", record_histories)
    stats = metadata.normalization
    normalized_observations = stats.normalize_observations(np.zeros((2, 23)))
    normalized_actions = stats.normalize_actions(np.zeros((2, 7)))
    expected_rng = torch.Generator().manual_seed(123)
    initial = model.generate(
        torch.from_numpy(normalized_observations)[None],
        torch.from_numpy(normalized_actions)[None],
        generator=expected_rng,
    ).numpy()[0]
    raw = stats.denormalize_actions(initial)
    assert np.any(np.abs(raw) > 1.0)
    expected = np.clip(raw, -1.0, 1.0)
    adapter.reset(integration=metadata.integration, seed=123)
    first = adapter.predict(policy_input(metadata, 0))
    second = adapter.predict(policy_input(metadata, 1))
    np.testing.assert_array_equal(first, expected[0])
    np.testing.assert_array_equal(second, expected[1])
    assert len(observed_histories) == 1
    np.testing.assert_array_equal(observed_histories[0][0], normalized_observations[None])
    np.testing.assert_array_equal(observed_histories[0][1], normalized_actions[None])
    adapter.predict(policy_input(metadata, 2))
    assert len(observed_histories) == 2
    np.testing.assert_array_equal(
        observed_histories[1][0],
        stats.normalize_observations(np.stack([np.ones(23), np.full(23, 2)]))[None],
    )
    np.testing.assert_array_equal(
        observed_histories[1][1], stats.normalize_actions(np.stack([first, second]))[None]
    )
    adapter.reset(integration=metadata.integration, seed=123)
    np.testing.assert_array_equal(first, adapter.predict(policy_input(metadata, 0)))
