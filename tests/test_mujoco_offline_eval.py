from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ml_collections import ConfigDict
from phi_mujoco.integrations import get_integration

from action_bridge.eval.eval_mujoco import evaluate_mujoco_offline


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


def test_offline_metrics_reject_nonfinite_predictions():
    target = torch.zeros(1, 3, 7)
    history = torch.zeros(1, 2, 7)
    dataset = _Dataset(
        "robomimic_square", target, history, torch.zeros(7), torch.ones(7)
    )
    model = _ConstantChunkPolicy(torch.full((3, 7), np.nan))
    with pytest.raises(ValueError, match="non-finite actions"):
        evaluate_mujoco_offline(model, dataset, {}, torch.device("cpu"))
