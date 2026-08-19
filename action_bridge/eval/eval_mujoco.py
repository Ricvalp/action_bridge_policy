"""Task-neutral offline metrics for PHI MuJoCo action chunks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from action_bridge.config import to_plain_dict
from action_bridge.eval.rollout import predict_actions
from action_bridge.training.common import (
    move_to_device,
    save_json,
    writable_numpy_collate,
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
        raise TypeError("normalized MuJoCo evaluation requires data.normalization_stats")
    return stats


def _denormalize_actions(
    actions: torch.Tensor,
    stats: Mapping[str, Any] | None,
) -> torch.Tensor:
    if stats is None:
        return actions
    mean = torch.as_tensor(stats["action_mean"], dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(stats["action_std"], dtype=actions.dtype, device=actions.device)
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
    """Evaluate predicted torque chunks in physical units.

    This evaluator deliberately contains no planar-arm plotting or simulator
    rollout logic.  Closed-loop metrics are produced by phi-mujoco's native
    ``EvaluationRunner`` through the downstream online adapter.
    """

    eval_config = config.get("eval", {})
    batch_size = int(eval_config.get("batch_size", 256))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=writable_numpy_collate,
    )
    stats = _normalization(config)
    lower = torch.tensor([-2.0, -2.0], dtype=torch.float64, device=device)
    upper = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)

    squared_error = 0.0
    absolute_error = 0.0
    first_squared_error = 0.0
    boundary_error = 0.0
    predicted_norm = 0.0
    target_norm = 0.0
    path_kl = 0.0
    bound_violations = 0
    target_saturated = 0
    action_values = 0
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
            batch,
            deterministic=bool(config.get("inference", {}).get("deterministic", True)),
            mode="mode",
        )
        predicted = _denormalize_actions(prediction["actions"], stats).to(torch.float64)
        target = _denormalize_actions(batch["future_actions"], stats).to(torch.float64)
        history = _denormalize_actions(batch["act_hist"], stats).to(torch.float64)
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 2:
            raise ValueError(
                "MuJoCo prediction and target must have matching [B,H,2] shapes; "
                f"got {tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        if not torch.isfinite(predicted).all():
            raise ValueError("MuJoCo offline prediction contains non-finite torques")

        difference = predicted - target
        squared_error += float(difference.square().sum().cpu())
        absolute_error += float(difference.abs().sum().cpu())
        first_squared_error += float(difference[:, 0].square().sum().cpu())
        boundary_error += float((predicted[:, 0] - history[:, -1]).square().sum().cpu())
        predicted_norm += float(torch.linalg.vector_norm(predicted, dim=-1).sum().cpu())
        target_norm += float(torch.linalg.vector_norm(target, dim=-1).sum().cpu())
        path_kl += float(prediction["path_kl_energy"].to(torch.float64).sum().cpu())
        bound_violations += int(((predicted < lower) | (predicted > upper)).sum().cpu())
        target_saturated += int(
            ((target - lower).abs() <= 1e-6).logical_or((target - upper).abs() <= 1e-6).sum().cpu()
        )
        action_values += int(target.numel())
        first_action_values += int(target[:, 0].numel())
        chunks += int(target.shape[0])
        batches += 1

    if batches == 0 or chunks == 0 or action_values == 0:
        raise ValueError("MuJoCo offline evaluation received no batches")
    horizon_actions = action_values // 2
    metrics = {
        "action_mse_nm2": squared_error / action_values,
        "action_l1_nm": absolute_error / action_values,
        "first_action_mse_nm2": first_squared_error / first_action_values,
        "chunk_boundary_mse_nm2": boundary_error / first_action_values,
        "predicted_torque_norm_nm": predicted_norm / horizon_actions,
        "target_torque_norm_nm": target_norm / horizon_actions,
        "predicted_bound_violation_rate": bound_violations / action_values,
        "target_saturation_rate": target_saturated / action_values,
        "path_kl_energy": path_kl / chunks,
        "evaluated_chunks": float(chunks),
        "evaluated_batches": float(batches),
    }
    if output_dir is not None:
        save_json(output_dir / "metrics" / "mujoco_offline_metrics.json", metrics)
    return metrics


__all__ = ["evaluate_mujoco_offline"]
