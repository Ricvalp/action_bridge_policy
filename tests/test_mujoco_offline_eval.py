from __future__ import annotations

import math

import torch
from ml_collections import ConfigDict

from action_bridge.eval.eval_mujoco import evaluate_mujoco_offline
from action_bridge.models.baselines import DirectChunkBCPolicy


class _Dataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        del index
        return {
            "obs_hist": torch.zeros(2, 8),
            "act_hist": torch.zeros(2, 2),
            "future_actions": torch.zeros(3, 2),
        }


def test_offline_mujoco_metrics_are_reported_in_physical_torque_units(tmp_path) -> None:
    model = DirectChunkBCPolicy(
        obs_dim=8,
        action_dim=2,
        obs_history=2,
        action_history=2,
        chunk_horizon=3,
        model_config={"hidden_dim": 8, "h_emb_dim": 8, "depth": 1},
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    config = ConfigDict(
        {
            "data": {
                "normalize": True,
                "normalization_stats": {
                    "type": "standard",
                    "eps": 1e-6,
                    "obs_mean": [0.0] * 8,
                    "obs_std": [1.0] * 8,
                    "action_mean": [0.5, -0.5],
                    "action_std": [0.25, 0.5],
                },
            },
            "eval": {"batch_size": 2},
            "inference": {"deterministic": True},
        }
    )
    metrics = evaluate_mujoco_offline(
        model,
        _Dataset(),
        config,
        torch.device("cpu"),
        output_dir=tmp_path,
    )
    assert metrics["action_mse_nm2"] == 0.0
    assert metrics["action_l1_nm"] == 0.0
    assert metrics["predicted_torque_norm_nm"] == math.sqrt(0.5)
    assert metrics["predicted_bound_violation_rate"] == 0.0
    assert metrics["evaluated_chunks"] == 3.0
    assert (tmp_path / "metrics" / "mujoco_offline_metrics.json").is_file()
