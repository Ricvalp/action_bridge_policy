"""Losses and metrics for action bridge policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_phi(horizon: int, final_value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if horizon <= 1:
        return torch.ones(horizon, device=device, dtype=dtype)
    s = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    return 1.0 + (final_value - 1.0) * s


def bridge_path_energy(init_action: torch.Tensor, actions: torch.Tensor, phi_final: float = 1.4) -> torch.Tensor:
    path = torch.cat([init_action[:, None], actions], dim=1)
    diffs = path[:, 1:] - path[:, :-1]
    sq = diffs.pow(2).sum(dim=-1)
    phi = linear_phi(actions.shape[1], phi_final, actions.device, actions.dtype)
    return (sq * phi[None, :]).mean()


def action_jerk_loss(actions: torch.Tensor, context_actions: torch.Tensor | None = None) -> torch.Tensor:
    if context_actions is not None and context_actions.shape[1] >= 2:
        path = torch.cat([context_actions[:, -2:], actions], dim=1)
    else:
        path = actions
    if path.shape[1] < 3:
        return torch.zeros((), device=actions.device, dtype=actions.dtype)
    jerk = path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]
    return jerk.pow(2).sum(dim=-1).mean()


def action_path_length(actions: torch.Tensor, init_action: torch.Tensor) -> torch.Tensor:
    path = torch.cat([init_action[:, None], actions], dim=1)
    return torch.linalg.norm(path[:, 1:] - path[:, :-1], dim=-1).sum(dim=1)


def action_jerk_metric(actions: torch.Tensor, context_actions: torch.Tensor) -> torch.Tensor:
    if context_actions.shape[1] >= 2:
        path = torch.cat([context_actions[:, -2:], actions], dim=1)
    else:
        path = actions
    if path.shape[1] < 3:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    jerk = path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]
    return torch.linalg.norm(jerk, dim=-1).mean(dim=1)


def chunk_discontinuity(actions: torch.Tensor, context_actions: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(actions[:, 0] - context_actions[:, -1], dim=-1)


def action_batch_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    init_action: torch.Tensor,
    context_actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mse = F.mse_loss(pred_actions, target_actions)
    endpoint_mse = F.mse_loss(pred_actions[:, -1], target_actions[:, -1])
    first_mse = F.mse_loss(pred_actions[:, 0], target_actions[:, 0])
    return {
        "action_mse": mse,
        "action_endpoint_mse": endpoint_mse,
        "action_first_mse": first_mse,
        "action_path_length": action_path_length(pred_actions, init_action).mean(),
        "action_jerk": action_jerk_metric(pred_actions, context_actions).mean(),
        "action_chunk_discontinuity": chunk_discontinuity(pred_actions, context_actions).mean(),
    }
