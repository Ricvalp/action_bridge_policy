from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict
from phi_mujoco.integrations import get_integration

from action_bridge.config import apply_overrides, load_config
from action_bridge.eval.eval_mujoco import evaluate_mujoco_offline
from action_bridge.training.common import build_model


class _Dataset:
    def __init__(self, integration_name, targets, histories, mean, std):
        self.integration = get_integration(integration_name)
        self.spec = self.integration.spec
        self.targets = targets
        self.histories = histories
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "obs_hist": torch.zeros(2, self.spec.observations["state"].shape[0]),
            "act_hist": (self.histories[index] - self.mean) / self.std,
            "future_actions": (self.targets[index] - self.mean) / self.std,
        }


class _ConstantChunkPolicy(torch.nn.Module):
    def __init__(self, chunk):
        super().__init__()
        self.register_buffer("chunk", chunk)

    def forward(self, obs_hist, act_hist):
        assert not self.training
        return self.chunk.expand(obs_hist.shape[0], -1, -1)


@pytest.mark.parametrize(
    ("integration_name", "action_dim", "bound"),
    [("planar_reach", 2, 2.0), ("robomimic_square", 7, 1.0)],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_offline_metrics_use_unclipped_action_profile_values(
    tmp_path, integration_name, action_dim, bound, normalize
):
    # Fit from training actions and reuse these statistics for held-out targets.
    center = torch.linspace(-0.125, 0.125, action_dim)
    scale = torch.linspace(0.125, 0.5, action_dim)
    training_actions = torch.stack([center - scale, center + scale])
    mean = training_actions.mean(dim=0) if normalize else torch.zeros(action_dim)
    std = (
        training_actions.std(dim=0, correction=0)
        if normalize
        else torch.ones(action_dim)
    )
    target_chunk = torch.linspace(-0.5, 0.5, 3 * action_dim).reshape(3, action_dim)
    targets = torch.stack([target_chunk + index * 0.05 for index in range(3)])
    histories = torch.stack(
        [torch.full((2, action_dim), -0.25 + index * 0.125) for index in range(3)]
    )
    predicted_chunk = target_chunk + 0.25
    predicted_chunk[0, 0] = bound + 0.5
    predicted_chunk[1, -1] = -bound - 0.5
    model = _ConstantChunkPolicy((predicted_chunk - mean) / std)
    stats = {"action_mean": mean.tolist(), "action_std": std.tolist()}
    config = ConfigDict(
        {
            "data": {
                "integration": integration_name,
                "normalize": normalize,
                "normalization_stats": stats,
            },
            "eval": {"batch_size": 2},
            "inference": {"deterministic": True},
        }
    )
    dataset = _Dataset(integration_name, targets, histories, mean, std)
    metrics = evaluate_mujoco_offline(
        model, dataset, config, torch.device("cpu"), output_dir=tmp_path
    )

    predicted = predicted_chunk.expand_as(targets)
    difference = predicted - targets
    assert metrics["action_mse"] == pytest.approx(float(difference.square().mean()))
    assert metrics["action_l1"] == pytest.approx(float(difference.abs().mean()))
    assert metrics["first_action_mse"] == pytest.approx(
        float(difference[:, 0].square().mean())
    )
    assert metrics["chunk_boundary_mse"] == pytest.approx(
        float((predicted[:, 0] - histories[:, -1]).square().mean())
    )
    assert metrics["predicted_action_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(predicted, dim=-1).mean())
    )
    assert metrics["target_action_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(targets, dim=-1).mean())
    )
    assert metrics["predicted_bound_violation_rate"] == pytest.approx(
        2 / (3 * action_dim)
    )
    assert metrics["normalized_path_kl_energy"] == 0.0
    assert metrics["evaluated_chunks"] == 3.0
    assert metrics["evaluated_batches"] == 2.0
    # Clipping must never make reported prediction errors look artificially small.
    clipped = torch.from_numpy(dataset.integration.project_action(predicted.numpy()))
    assert metrics["action_mse"] > float((clipped - targets).square().mean())
    assert not model.training
    metrics_directory = tmp_path / "metrics"
    assert (
        json.loads((metrics_directory / "mujoco_offline_metrics.json").read_text())
        == metrics
    )
    metadata = json.loads(
        (metrics_directory / "mujoco_offline_metadata.json").read_text()
    )
    assert metadata["integration_spec"] == dataset.spec.to_dict()
    assert metadata["normalization_stats"] == (stats if normalize else None)
    assert metadata["prediction_protocol"] == "direct_chunk"


def test_offline_metrics_reject_nonfinite_predictions():
    target = torch.zeros(1, 3, 7)
    history = torch.zeros(1, 2, 7)
    dataset = _Dataset(
        "robomimic_square", target, history, torch.zeros(7), torch.ones(7)
    )
    model = _ConstantChunkPolicy(torch.full((3, 7), np.nan))
    with pytest.raises(ValueError, match="non-finite actions"):
        evaluate_mujoco_offline(model, dataset, {}, torch.device("cpu"))


def test_continuous_latent_mse_uses_prior_and_autoregressive_actions(monkeypatch):
    torch.manual_seed(0)
    config = apply_overrides(
        load_config("mujoco_robomimic_square"),
        [
            "model.latent_type=continuous",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.encoder_depth=1",
            "model.control_depth=1",
            "chunk_horizon=4",
            "eval.batch_size=2",
            "inference.deterministic=true",
        ],
    )
    mean = torch.linspace(-0.2, 0.2, 7)
    std = torch.linspace(0.1, 0.7, 7)
    config.data.normalization_stats = {
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
    }
    targets = torch.linspace(-0.8, 0.8, 3 * 4 * 7).reshape(3, 4, 7)
    targets[-1] += 2.0  # Different error in the last, partial batch.
    histories = torch.linspace(-0.3, 0.3, 3 * 2 * 7).reshape(3, 2, 7)
    dataset = _Dataset("robomimic_square", targets, histories, mean, std)
    model = build_model(config).eval()

    def reject_future_actions(*args, **kwargs):
        raise AssertionError("Validation must not use the future-action posterior")

    monkeypatch.setattr(model.latent, "posterior_params", reject_future_actions)
    monkeypatch.setattr(model.latent.future_encoder, "forward", reject_future_actions)
    batch = {
        key: torch.stack([dataset[index][key] for index in range(len(dataset))])
        for key in ("obs_hist", "act_hist", "future_actions")
    }
    with torch.no_grad():
        history_embedding = model.encode_history(batch["obs_hist"], batch["act_hist"])
        prior_mean, _ = model.latent.prior_params(history_embedding)
        latent_embedding = model.latent.embed(prior_mean)

        def chunk(*, teacher_forcing):
            previous_previous = batch["act_hist"][:, -2]
            previous = batch["act_hist"][:, -1]
            predictions = []
            for step in range(config.chunk_horizon):
                reference, _ = model.reference_process(
                    previous, previous_previous, history_embedding, step
                )
                predicted = reference + model.control(
                    previous,
                    previous_previous,
                    history_embedding,
                    step,
                    latent_embedding,
                )
                predictions.append(predicted)
                previous_previous, previous = (
                    previous,
                    (
                        batch["future_actions"][:, step]
                        if teacher_forcing
                        else predicted
                    ),
                )
            return torch.stack(predictions, dim=1)

        autoregressive = chunk(teacher_forcing=False)
        teacher_forced = chunk(teacher_forcing=True)

    assert not torch.allclose(autoregressive[:, 1:], teacher_forced[:, 1:])
    raw_prediction = autoregressive * std + mean
    errors = (raw_prediction.double() - targets.double()).square()
    metrics = evaluate_mujoco_offline(model, dataset, config, torch.device("cpu"))
    assert metrics["action_mse"] == pytest.approx(float(errors.mean()), rel=1e-6)
    assert metrics["action_mse"] != pytest.approx(
        float((errors[:2].mean() + errors[2:].mean()) / 2)
    )
    assert metrics["evaluated_chunks"] == 3.0
    assert metrics["evaluated_batches"] == 2.0
