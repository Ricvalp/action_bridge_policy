"""Passive demonstration targets for dissipative contact-reference losses."""

from __future__ import annotations

from typing import Any, Dict

import torch

from action_bridge.data.action_coordinates import ActionCoordinateAdapter


def damped_continuation_projection(
    q_seq: torch.Tensor,
    p_seq: torch.Tensor,
    dt: float = 1.0,
    alpha_max: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Project demonstrated next velocity onto ``{alpha p_k}``.

    Args:
        q_seq: Teacher-forced phase positions with shape ``[B, H + 1, A]``.
        p_seq: Teacher-forced phase velocities with shape ``[B, H + 1, A]``.
        dt: Integrator step used by the contact reference.
        alpha_max: Maximum continuation coefficient.
        eps: Numerical floor for nearly stationary histories.

    Returns:
        A dictionary with passive next ``q``/``p`` targets for each chunk step.
    """

    q = q_seq[:, :-1]
    p = p_seq[:, :-1]
    p_star_next = p_seq[:, 1:]
    numerator = (p_star_next * p).sum(dim=-1, keepdim=True)
    denominator = p.pow(2).sum(dim=-1, keepdim=True).clamp_min(float(eps))
    alpha = (numerator / denominator).clamp(0.0, float(alpha_max))
    p_bar_next = alpha * p
    q_bar_next = q + float(dt) * p_bar_next
    return {
        "q_bar_next": q_bar_next,
        "p_bar_next": p_bar_next,
        "alpha": alpha.squeeze(-1),
        "speed": torch.linalg.norm(p, dim=-1),
        "demo_next_speed": torch.linalg.norm(p_star_next, dim=-1),
        "residual_next_velocity": p_star_next - p_bar_next,
    }


def passive_target_from_batch(
    adapter: ActionCoordinateAdapter,
    batch: Dict[str, Any],
    target_type: str = "damped_continuation",
    alpha_max: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Build a passive target from a training/evaluation batch."""

    q_seq = adapter.build_q_sequence(batch)
    p_seq = adapter.build_p_sequence(q_seq, batch)
    if target_type != "damped_continuation":
        raise ValueError(f"Unsupported passive target type {target_type!r}.")
    out = damped_continuation_projection(q_seq, p_seq, dt=adapter.dt, alpha_max=alpha_max, eps=eps)
    out["q_seq"] = q_seq
    out["p_seq"] = p_seq
    raw_steps = []
    for k in range(q_seq.shape[1] - 1):
        raw_steps.append(adapter.decode_step(q_seq[:, k], out["q_bar_next"][:, k]))
    out["projected_raw_actions"] = torch.stack(raw_steps, dim=1)
    return out
