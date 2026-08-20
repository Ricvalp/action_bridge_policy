"""Simulator-free metrics for PHI Isaac Lab Franka action chunks."""

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
        raise TypeError("Isaac Lab evaluation config must be a mapping")
    data = plain_config.get("data", {})
    if not isinstance(data, Mapping) or not bool(data.get("normalize", False)):
        return None
    stats = data.get("normalization_stats")
    if not isinstance(stats, Mapping):
        raise TypeError("normalized Isaac Lab evaluation requires data.normalization_stats")
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


def _quaternion_error_xyzw(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    invalid = (predicted_norm < 1e-8) | (target_norm < 1e-8)
    predicted_unit = predicted / predicted_norm.clamp_min(1e-8).unsqueeze(-1)
    target_unit = target / target_norm.clamp_min(1e-8).unsqueeze(-1)
    # q and -q encode the same orientation.
    dot = (predicted_unit * target_unit).sum(dim=-1).abs().clamp(max=1.0)
    angle = 2.0 * torch.acos(dot)
    return angle, invalid, (predicted_norm - 1.0).abs()


@torch.no_grad()
def evaluate_isaaclab_offline(
    model: torch.nn.Module,
    dataset: object,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    output_dir: Path | None = None,
    max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate absolute TCP-pose/gripper chunks in physical units.

    Closed-loop task success is intentionally not approximated here; it is
    measured by phi-isaaclab's native batched evaluator.
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

    action_squared_error = 0.0
    position_squared_error = 0.0
    position_absolute_error = 0.0
    quaternion_error = 0.0
    quaternion_norm_error = 0.0
    invalid_quaternions = 0
    correct_gripper = 0
    first_position_squared_error = 0.0
    first_quaternion_error = 0.0
    first_correct_gripper = 0
    boundary_position_squared_error = 0.0
    boundary_quaternion_error = 0.0
    path_kl = 0.0
    chunks = 0
    actions = 0
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
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 8:
            raise ValueError(
                "Isaac Lab prediction and target must have matching [B,H,8] shapes; "
                f"got {tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        if not torch.isfinite(predicted).all():
            raise ValueError("Isaac Lab offline prediction contains non-finite actions")

        difference = predicted - target
        position_difference = difference[..., :3]
        angular, invalid, norm_error = _quaternion_error_xyzw(
            predicted[..., 3:7], target[..., 3:7]
        )
        boundary_angular, _, _ = _quaternion_error_xyzw(
            predicted[:, 0, 3:7], history[:, -1, 3:7]
        )
        predicted_gripper = predicted[..., 7] >= 0.0
        target_gripper = target[..., 7] >= 0.0

        action_squared_error += float(difference.square().sum().cpu())
        position_squared_error += float(position_difference.square().sum().cpu())
        position_absolute_error += float(position_difference.abs().sum().cpu())
        quaternion_error += float(angular.sum().cpu())
        quaternion_norm_error += float(norm_error.sum().cpu())
        invalid_quaternions += int(invalid.sum().cpu())
        correct_gripper += int((predicted_gripper == target_gripper).sum().cpu())
        first_position_squared_error += float(position_difference[:, 0].square().sum().cpu())
        first_quaternion_error += float(angular[:, 0].sum().cpu())
        first_correct_gripper += int(
            (predicted_gripper[:, 0] == target_gripper[:, 0]).sum().cpu()
        )
        boundary_position_squared_error += float(
            (predicted[:, 0, :3] - history[:, -1, :3]).square().sum().cpu()
        )
        boundary_quaternion_error += float(boundary_angular.sum().cpu())
        path_kl += float(prediction["path_kl_energy"].to(torch.float64).sum().cpu())
        chunks += int(target.shape[0])
        actions += int(target.shape[0] * target.shape[1])
        batches += 1

    if batches == 0 or chunks == 0 or actions == 0:
        raise ValueError("Isaac Lab offline evaluation received no batches")
    metrics = {
        "action_mse": action_squared_error / (actions * 8),
        "tcp_position_mse_m2": position_squared_error / (actions * 3),
        "tcp_position_l1_m": position_absolute_error / (actions * 3),
        "tcp_quaternion_geodesic_rad": quaternion_error / actions,
        "predicted_quaternion_norm_abs_error": quaternion_norm_error / actions,
        "predicted_invalid_quaternion_rate": invalid_quaternions / actions,
        "gripper_accuracy": correct_gripper / actions,
        "first_tcp_position_mse_m2": first_position_squared_error / (chunks * 3),
        "first_tcp_quaternion_geodesic_rad": first_quaternion_error / chunks,
        "first_gripper_accuracy": first_correct_gripper / chunks,
        "chunk_boundary_tcp_position_mse_m2": boundary_position_squared_error / (chunks * 3),
        "chunk_boundary_tcp_quaternion_geodesic_rad": boundary_quaternion_error / chunks,
        "path_kl_energy": path_kl / chunks,
        "evaluated_chunks": float(chunks),
        "evaluated_batches": float(batches),
    }
    if output_dir is not None:
        save_json(output_dir / "metrics" / "isaaclab_offline_metrics.json", metrics)
    return metrics


__all__ = ["evaluate_isaaclab_offline"]
