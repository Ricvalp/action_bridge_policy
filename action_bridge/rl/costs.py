"""Regularization costs for contact bridge RL."""

from __future__ import annotations

import torch


def compute_ref_cost(info: dict, upto: int) -> torch.Tensor:
    """Path-control energy over the first ``upto`` chunk steps."""

    u = info["u_seq"][:, : int(upto)]
    return 0.5 * u.pow(2).sum(dim=-1).sum(dim=-1)


def compute_ref_cost_mean(info: dict, upto: int) -> torch.Tensor:
    """Mean per-step path-control energy, easier to tune than the sum."""

    u = info["u_seq"][:, : int(upto)]
    return 0.5 * u.pow(2).sum(dim=-1).mean(dim=-1)


def compute_bc_cost(actions: torch.Tensor, bc_actions: torch.Tensor) -> torch.Tensor:
    return (actions - bc_actions).pow(2).mean(dim=(1, 2))


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if int(duration) <= 0:
        return float(end)
    frac = min(max(float(step) / float(duration), 0.0), 1.0)
    return float(start) + frac * (float(end) - float(start))
