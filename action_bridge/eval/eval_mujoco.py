"""Task-neutral offline metrics for PHI MuJoCo action chunks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_bridge.config import to_plain_dict
from action_bridge.eval.rollout import predict_actions
from action_bridge.training.common import (
    move_to_device,
    save_json,
)


def _normalization(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    plain_config = to_plain_dict(config)
    if not isinstance(plain_config, Mapping):
        raise TypeError("MuJoCo evaluation config must be a mapping")
    data = plain_config.get("data", {})
    if not isinstance(data, Mapping) or not bool(data.get("normalize", False)):
        return None
    stats = data.get("normalization_stats")
    if not isinstance(stats, Mapping):
        raise TypeError(
            "normalized MuJoCo evaluation requires data.normalization_stats"
        )
    return stats


def _denormalize_actions(
    actions: torch.Tensor,
    stats: Mapping[str, Any] | None,
) -> torch.Tensor:
    if stats is None:
        return actions
    mean = torch.as_tensor(
        stats["action_mean"], dtype=actions.dtype, device=actions.device
    )
    std = torch.as_tensor(
        stats["action_std"], dtype=actions.dtype, device=actions.device
    )
    return actions * std + mean


@torch.no_grad()
def evaluate_mujoco_offline(
    model: torch.nn.Module,
    dataset: object,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    output_dir: Path | None = None,
    max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate action chunks in the integration's original action coordinates.

    Undo training normalization before computing errors. These coordinates are
    torques for planar reach and normalized controller commands for Robomimic.
    Predictions remain unclipped for errors; the integration's public projection
    reports how many action values would change before execution. Predictions
    use the history-conditioned prior and their own previous generated actions,
    never the future-conditioned posterior or teacher-forced expert actions.
    """

    eval_config = config.get("eval", {})
    batch_size = int(eval_config.get("batch_size", 256))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    stats = _normalization(config)
    integration = dataset.integration
    spec = dataset.spec
    action_dim = spec.action.shape[0]

    squared_error = 0.0
    absolute_error = 0.0
    first_squared_error = 0.0
    boundary_error = 0.0
    predicted_norm = 0.0
    target_norm = 0.0
    path_kl = 0.0
    bound_violations = 0
    action_values = 0
    actions = 0
    first_action_values = 0
    chunks = 0
    batches = 0

    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        prediction = predict_actions(
            model,
            {"obs_hist": batch["obs_hist"], "act_hist": batch["act_hist"]},
            deterministic=bool(config.get("inference", {}).get("deterministic", True)),
            mode="mode",
        )
        predicted = _denormalize_actions(prediction["actions"], stats).to(torch.float64)
        target = _denormalize_actions(batch["future_actions"], stats).to(torch.float64)
        history = _denormalize_actions(batch["act_hist"], stats).to(torch.float64)
        if (
            predicted.shape != target.shape
            or predicted.ndim != 3
            or predicted.shape[-1] != action_dim
        ):
            raise ValueError(
                f"MuJoCo prediction and target must have matching [B,H,{action_dim}] shapes; "
                f"got {tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        if not torch.isfinite(predicted).all():
            raise ValueError("MuJoCo offline prediction contains non-finite actions")

        difference = predicted - target
        squared_error += float(difference.square().sum().cpu())
        absolute_error += float(difference.abs().sum().cpu())
        first_squared_error += float(difference[:, 0].square().sum().cpu())
        boundary_error += float((predicted[:, 0] - history[:, -1]).square().sum().cpu())
        predicted_norm += float(torch.linalg.vector_norm(predicted, dim=-1).sum().cpu())
        target_norm += float(torch.linalg.vector_norm(target, dim=-1).sum().cpu())
        path_kl += float(prediction["path_kl_energy"].to(torch.float64).sum().cpu())
        raw_actions = predicted.cpu().numpy().astype(spec.action.dtype)
        projected_actions = integration.project_action(raw_actions)
        bound_violations += int(np.count_nonzero(raw_actions != projected_actions))
        action_values += int(target.numel())
        actions += int(target.shape[0] * target.shape[1])
        first_action_values += int(target[:, 0].numel())
        chunks += int(target.shape[0])
        batches += 1

    if batches == 0 or chunks == 0 or action_values == 0:
        raise ValueError("MuJoCo offline evaluation received no batches")
    metrics = {
        "action_mse": squared_error / action_values,
        "action_l1": absolute_error / action_values,
        "first_action_mse": first_squared_error / first_action_values,
        "chunk_boundary_mse": boundary_error / first_action_values,
        "predicted_action_norm": predicted_norm / actions,
        "target_action_norm": target_norm / actions,
        "predicted_bound_violation_rate": bound_violations / action_values,
        "normalized_path_kl_energy": path_kl / chunks,
        "evaluated_chunks": float(chunks),
        "evaluated_batches": float(batches),
    }
    if output_dir is not None:
        save_json(output_dir / "metrics" / "mujoco_offline_metrics.json", metrics)
        save_json(
            output_dir / "metrics" / "mujoco_offline_metadata.json",
            {
                "integration_spec": spec.to_dict(),
                "prediction_protocol": "prior_autoregressive_chunk",
                "action_metric_coordinates": "original action profile, before projection",
                "action_metric_units": (
                    "action profile units; squared for MSE metrics. "
                    "Robomimic uses normalized controller commands; planar reach uses Nm."
                ),
                "normalization_stats": stats,
                "path_kl_coordinates": (
                    "training-normalized actions"
                    if stats is not None
                    else "original action profile"
                ),
            },
        )
    return metrics


__all__ = ["evaluate_mujoco_offline"]
