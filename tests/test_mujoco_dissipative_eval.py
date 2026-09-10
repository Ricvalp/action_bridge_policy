from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
import torch
from phi_mujoco.integrations import get_integration

from test_mujoco_online_adapter import policy_input
from test_mujoco_online_metadata import make_metadata
from test_mujoco_training import _cache

from action_bridge.config import apply_overrides, load_config
from action_bridge.eval.eval_mujoco import evaluate_mujoco_offline
from action_bridge.eval.mujoco_online.torch_backend import load_torch_policy_adapter
from action_bridge.eval.rollout import generate_chunk
from action_bridge.training.common import build_model, load_config_from_checkpoint
from action_bridge.training.train_mujoco import train
from action_bridge.training.train_toy import save_checkpoint


CONFIGS = [
    "mujoco_robomimic_square_dissipative",
    "mujoco_robomimic_square_dissipative_stopgrad",
]


def _tiny_config(name, latent_type="none"):
    return apply_overrides(load_config(name), [
        "device=cpu",
        f"model.latent_type={latent_type}",
        "model.hidden_dim=16",
        "model.h_emb_dim=16",
        "model.time_emb_dim=16",
        "model.encoder_depth=2",
        "model.control_depth=2",
        "model.z_dim=2",
        "model.z_embed_dim=8",
        "reference.hidden_dim=8",
        "reference.time_emb_dim=16",
    ])


def _objects(name, latent_type="none"):
    config = _tiny_config(name, latent_type)
    metadata = make_metadata(policy_type="action_bridge")
    stats = replace(
        metadata.normalization,
        obs_mean=(0.25,) * metadata.observation_dim,
        obs_std=(2.0,) * metadata.observation_dim,
        action_mean=tuple(np.linspace(-0.2, 0.3, 7)),
        action_std=tuple(np.linspace(0.2, 1.4, 7)),
    )
    metadata = replace(
        metadata, normalization=stats, action_horizon=8,
        actions_per_plan=4, latent_commitment="chunk",
    )
    config.data.normalization = stats.to_dict()
    config.data.normalization_stats = stats.to_dict()
    config.data.collection_identity = metadata.collection_identity
    config.online_evaluation = metadata.to_json_dict()
    config.eval.batch_size = 2
    model = build_model(config).eval()
    # Deliberately separate the live and EMA drifts without changing the fixed
    # diffusion scale. Wrong-reference inference must not pass by coincidence.
    references = [(model.reference_process, 3.0)]
    if hasattr(model, "reference_process_ema"):
        references.append((model.reference_process_ema, -3.0))
    with torch.no_grad():
        for reference, attractor in references:
            for key, parameter in reference.named_parameters():
                if key != "log_sigma":
                    parameter.zero_()
            reference.m_net[-1].bias.fill_(attractor)
    return model, config, metadata


@torch.no_grad()
def _manual_chunk(model, batch, reference):
    history = model.encode_history(batch["obs_hist"], batch["act_hist"])
    _, latent = model.sample_prior_z(
        history, mode="mode", deterministic_continuous=True
    )
    q, p = model.coordinate_adapter.init_qp_from_history(batch)
    positions = [q]
    for step in range(model.chunk_horizon):
        q, p, _, _ = model.contact_step(
            q, p, history, step, latent,
            obs_state=batch["obs_hist"][:, -1], reference=reference,
        )
        positions.append(q)
    return model.coordinate_adapter.decode_raw_actions(torch.stack(positions, dim=1))


def _checkpoint(tmp_path, model, config):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), config, 7, 0.5)
    return load_torch_policy_adapter(path, trusted_checkpoint=True, device="cpu")


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("latent_type", ["none", "continuous"])
def test_checkpoint_and_online_inference_use_the_trained_reference(
    tmp_path, name, latent_type
):
    model, config, metadata = _objects(name, latent_type)
    batch = {"obs_hist": torch.zeros(2, 2, 23), "act_hist": torch.zeros(2, 2, 7)}
    generated = generate_chunk(model, **batch, deterministic=True)
    selected = model.inference_reference()
    expected = _manual_chunk(model, batch, selected)
    torch.testing.assert_close(generated["actions"], expected)
    assert torch.isfinite(expected).all()
    assert generated["path_kl_steps"].shape == (2, 8)
    if name.endswith("stopgrad"):
        assert selected is model.reference_process_ema
        assert not torch.allclose(
            expected, _manual_chunk(model, batch, model.reference_process)
        )
    else:
        assert selected is model.reference_process

    adapter = _checkpoint(tmp_path, model, config)
    assert adapter.metadata == metadata
    assert adapter.actions_per_plan == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(adapter.backend.model.state_dict()[key], value)
    if name.endswith("stopgrad"):
        assert adapter.backend.model.inference_reference() is adapter.backend.model.reference_process_ema
        assert all(not parameter.requires_grad for parameter in adapter.backend.model.reference_process_ema.parameters())
    adapter.backend.reset(seed=123)
    predicted, diagnostics = adapter.backend.predict({key: value.numpy() for key, value in batch.items()})
    np.testing.assert_allclose(predicted, expected.numpy(), rtol=1e-6, atol=1e-6)
    assert diagnostics["normalized_path_kl_energy"] == pytest.approx(
        float(generated["path_kl_energy"].mean()), rel=1e-6
    )


class _ValidationDataset:
    def __init__(self, metadata):
        self.integration = get_integration(metadata.integration.name)
        self.spec = self.integration.spec
        self.stats = metadata.normalization

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "obs_hist": self.stats.normalize_observations(np.full((2, 23), index * 0.1)),
            "act_hist": self.stats.normalize_actions(np.full((2, 7), index * 0.05)),
            "future_actions": self.stats.normalize_actions(np.full((8, 7), index * 0.03)),
        }


@pytest.mark.parametrize("name", CONFIGS)
def test_offline_validation_matches_restored_online_chunks_and_denormalizes_once(tmp_path, name):
    model, config, metadata = _objects(name)
    dataset = _ValidationDataset(metadata)
    adapter = _checkpoint(tmp_path, model, config)
    adapter.backend.reset(seed=123)
    batch = {
        key: np.stack([dataset[index][key] for index in range(len(dataset))])
        for key in ("obs_hist", "act_hist", "future_actions")
    }
    predictions, _ = adapter.backend.predict({key: batch[key] for key in ("obs_hist", "act_hist")})
    raw = metadata.normalization.denormalize_actions(predictions).astype(np.float64)
    target = metadata.normalization.denormalize_actions(batch["future_actions"]).astype(np.float64)
    assert np.any(np.abs(raw) > 1)  # Offline errors must use unclipped predictions.
    expected = float(np.mean((raw - target) ** 2))
    metrics = evaluate_mujoco_offline(model, dataset, config, torch.device("cpu"))
    assert metrics["action_mse"] == pytest.approx(expected, rel=1e-6)
    assert metrics["evaluated_chunks"] == 3
    assert metrics["evaluated_batches"] == 2
    assert metrics["predicted_bound_violation_rate"] > 0
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("name", CONFIGS)
def test_execution_four_buffers_projected_commands_and_updates_every_observation(tmp_path, name, monkeypatch):
    model, config, metadata = _objects(name)
    adapter = _checkpoint(tmp_path, model, config)
    histories = []
    predict = adapter.backend.predict

    def record(batch):
        output = predict(batch)
        histories.append(({key: value.copy() for key, value in batch.items()}, output[0].copy()))
        return output

    monkeypatch.setattr(adapter.backend, "predict", record)
    adapter.reset(integration=metadata.integration, seed=123)
    executed = [adapter.predict(policy_input(metadata, step)) for step in range(4)]
    assert len(histories) == 1
    stats = metadata.normalization
    raw = stats.denormalize_actions(histories[0][1])[0, :4]
    assert np.any(np.abs(raw) > 1)
    np.testing.assert_allclose(executed, adapter.integration.project_action(raw), atol=1e-6)
    np.testing.assert_array_equal(histories[0][0]["act_hist"], stats.normalize_actions(np.zeros((2, 7)))[None])

    adapter.predict(policy_input(metadata, 4))
    assert len(histories) == 2
    np.testing.assert_array_equal(
        histories[1][0]["obs_hist"],
        stats.normalize_observations(np.stack([np.full(23, 3), np.full(23, 4)]))[None],
    )
    np.testing.assert_array_equal(
        histories[1][0]["act_hist"], stats.normalize_actions(np.stack(executed[-2:]))[None]
    )


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("latent_type", ["none", "continuous"])
def test_shared_training_and_resume_preserve_dissipative_reference(tmp_path, name, latent_type):
    bundle = _cache(tmp_path / "cache", "robomimic_square")
    config = apply_overrides(_tiny_config(name, latent_type), [
        f"data.cache_root={bundle.root}",
        f"output_dir={tmp_path / 'runs'}",
        "run_id=dissipative",
        "optim.batch_size=2",
        "optim.max_steps=2",
        "logging.progress=false",
        "logging.log_every_steps=1",
        "logging.eval_every_steps=1",
        "logging.validation_max_batches=1",
        "eval.batch_size=2",
        "eval.offline_max_batches=1",
    ])
    run = train(config)
    path = run / "checkpoints" / "latest.pt"
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["config"]["reference"]["type"] == "contact_langevin"
    assert checkpoint["online_evaluation"]["action_horizon"] == 8
    assert checkpoint["online_evaluation"]["actions_per_plan"] == 4
    assert checkpoint["config"]["checkpoint_metric"] == "val_action_mse"
    if name.endswith("stopgrad"):
        ema_keys = [key for key in checkpoint["model_state"] if key.startswith("reference_process_ema.")]
        assert ema_keys
        assert any(not torch.equal(
            checkpoint["model_state"][key],
            checkpoint["model_state"][key.replace("reference_process_ema.", "reference_process.")],
        ) for key in ema_keys)
    with (run / "metrics" / "val_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert checkpoint["best_metric"] == min(float(row["action_mse"]) for row in rows)
    restored = load_torch_policy_adapter(path, trusted_checkpoint=True)
    assert restored.backend.model.uses_contact_langevin
    resumed = load_config_from_checkpoint(path)
    resumed.optim.max_steps = 3
    assert train(resumed) == run
    assert torch.load(path, weights_only=False)["step"] == 3
