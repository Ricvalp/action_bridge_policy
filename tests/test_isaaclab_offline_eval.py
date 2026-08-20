from __future__ import annotations

import torch
from ml_collections import ConfigDict

from action_bridge.eval.eval_isaaclab import evaluate_isaaclab_offline
from action_bridge.models.baselines import DirectChunkBCPolicy

_MEAN_ACTION = [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 1.0]


class _Dataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        del index
        return {
            "obs_hist": torch.zeros(2, 35),
            "act_hist": torch.zeros(2, 8),
            "future_actions": torch.zeros(4, 8),
        }


def test_offline_metrics_use_physical_pose_and_gripper_semantics(tmp_path) -> None:
    model = DirectChunkBCPolicy(
        obs_dim=35,
        action_dim=8,
        obs_history=2,
        action_history=2,
        chunk_horizon=4,
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
                    "obs_mean": [0.0] * 35,
                    "obs_std": [1.0] * 35,
                    "action_mean": _MEAN_ACTION,
                    "action_std": [1.0] * 8,
                },
            },
            "eval": {"batch_size": 2},
            "inference": {"deterministic": True},
        }
    )
    metrics = evaluate_isaaclab_offline(
        model,
        _Dataset(),
        config,
        torch.device("cpu"),
        output_dir=tmp_path,
    )
    assert metrics["action_mse"] == 0.0
    assert metrics["tcp_position_mse_m2"] == 0.0
    assert metrics["tcp_quaternion_geodesic_rad"] == 0.0
    assert metrics["predicted_quaternion_norm_abs_error"] == 0.0
    assert metrics["gripper_accuracy"] == 1.0
    assert metrics["evaluated_chunks"] == 3.0
    assert (tmp_path / "metrics" / "isaaclab_offline_metrics.json").is_file()
