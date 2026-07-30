"""Passive demonstration targets for dissipative contact-reference losses."""

from __future__ import annotations

from typing import Any, Dict

import torch

from action_bridge.data.action_coordinates import ActionCoordinateAdapter
from action_bridge.models.geometric_pusht import default_t_polygon_local, sample_polygon_boundary_local


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
        "projection_coefficient": alpha.squeeze(-1),
        "speed": torch.linalg.norm(p, dim=-1),
        "demo_next_speed": torch.linalg.norm(p_star_next, dim=-1),
        "residual_next_velocity": p_star_next - p_bar_next,
    }


def ema_smoothed_velocity_projection(
    q_seq: torch.Tensor,
    p_seq: torch.Tensor,
    dt: float = 1.0,
    ema_decay: float = 0.85,
) -> Dict[str, torch.Tensor]:
    """Use a causal EMA of demonstrated velocity as the passive target."""

    decay = float(ema_decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}.")
    q = q_seq[:, :-1]
    p_star_next = p_seq[:, 1:]
    smoothed = p_seq[:, 0]
    smoothed_steps = []
    for k in range(p_star_next.shape[1]):
        smoothed = decay * smoothed + (1.0 - decay) * p_star_next[:, k]
        smoothed_steps.append(smoothed)
    p_bar_next = torch.stack(smoothed_steps, dim=1)
    q_bar_next = q + float(dt) * p_bar_next
    coefficient = torch.full_like(p_star_next[..., 0], decay)
    return {
        "q_bar_next": q_bar_next,
        "p_bar_next": p_bar_next,
        "alpha": coefficient,
        "projection_coefficient": coefficient,
        "speed": torch.linalg.norm(p_seq[:, :-1], dim=-1),
        "demo_next_speed": torch.linalg.norm(p_star_next, dim=-1),
        "residual_next_velocity": p_star_next - p_bar_next,
    }


def _pusher_boundary_distance(
    future_obs_raw: torch.Tensor,
    boundary_samples_per_edge: int,
) -> torch.Tensor:
    if future_obs_raw.shape[-1] < 5:
        raise ValueError(
            "contact_direction requires Push-T observations "
            "[pusher_x, pusher_y, block_x, block_y, block_theta]."
        )
    pusher = future_obs_raw[..., :2]
    block = future_obs_raw[..., 2:4]
    theta = future_obs_raw[..., 4]
    delta = pusher - block
    c = torch.cos(theta)
    s = torch.sin(theta)
    pusher_local = torch.stack(
        [
            c * delta[..., 0] + s * delta[..., 1],
            -s * delta[..., 0] + c * delta[..., 1],
        ],
        dim=-1,
    )
    polygon = default_t_polygon_local().to(device=future_obs_raw.device, dtype=future_obs_raw.dtype)
    boundary, _ = sample_polygon_boundary_local(polygon, max(1, int(boundary_samples_per_edge)))
    return torch.linalg.norm(pusher_local[..., None, :] - boundary, dim=-1).amin(dim=-1)


def contact_direction_projection(
    q_seq: torch.Tensor,
    p_seq: torch.Tensor,
    future_obs_raw: torch.Tensor,
    dt: float = 1.0,
    lambda_parallel: float = 0.8,
    lambda_perp: float = 0.2,
    contact_distance: float = 18.0,
    contact_temperature: float = 4.0,
    boundary_samples_per_edge: int = 8,
    goal_xy: tuple[float, float] = (256.0, 256.0),
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Keep goal-aligned demonstrated velocity near contact.

    The contact score is a smooth function of the pusher-center distance to
    the transformed Push-T boundary. Far from contact the passive target is
    the full demonstrated velocity; near contact its lateral component is
    damped more strongly than its goal-aligned component.
    """

    if not 0.0 <= float(lambda_perp) <= float(lambda_parallel) <= 1.0:
        raise ValueError(
            "contact-direction coefficients must satisfy "
            f"0 <= lambda_perp <= lambda_parallel <= 1, got "
            f"{lambda_perp} and {lambda_parallel}."
        )
    if float(contact_temperature) <= 0.0:
        raise ValueError(f"contact_temperature must be positive, got {contact_temperature}.")
    horizon = p_seq.shape[1] - 1
    if future_obs_raw.ndim != 3 or future_obs_raw.shape[1] < horizon:
        raise ValueError(
            f"contact_direction requires future_obs_raw [B,H,D] with H >= {horizon}, "
            f"got {tuple(future_obs_raw.shape)}."
        )

    obs = future_obs_raw[:, :horizon]
    q = q_seq[:, :-1]
    p_star_next = p_seq[:, 1:]
    block = obs[..., 2:4]
    goal = torch.as_tensor(goal_xy, device=block.device, dtype=block.dtype)
    direction = goal - block
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(float(eps))

    parallel = (p_star_next * direction).sum(dim=-1, keepdim=True) * direction
    perpendicular = p_star_next - parallel
    boundary_distance = _pusher_boundary_distance(obs, boundary_samples_per_edge)
    contact_score = torch.sigmoid(
        (float(contact_distance) - boundary_distance) / float(contact_temperature)
    )
    projected_at_contact = float(lambda_parallel) * parallel + float(lambda_perp) * perpendicular
    p_bar_next = (
        (1.0 - contact_score[..., None]) * p_star_next
        + contact_score[..., None] * projected_at_contact
    )
    q_bar_next = q + float(dt) * p_bar_next
    return {
        "q_bar_next": q_bar_next,
        "p_bar_next": p_bar_next,
        "alpha": contact_score,
        "projection_coefficient": contact_score,
        "speed": torch.linalg.norm(p_seq[:, :-1], dim=-1),
        "demo_next_speed": torch.linalg.norm(p_star_next, dim=-1),
        "residual_next_velocity": p_star_next - p_bar_next,
        "contact_score": contact_score,
        "contact_boundary_distance": boundary_distance,
        "contact_direction": direction,
    }


def passive_target_from_batch(
    adapter: ActionCoordinateAdapter,
    batch: Dict[str, Any],
    target_type: str = "damped_continuation",
    alpha_max: float = 1.0,
    ema_decay: float = 0.85,
    contact_lambda_parallel: float = 0.8,
    contact_lambda_perp: float = 0.2,
    contact_distance: float = 18.0,
    contact_temperature: float = 4.0,
    contact_boundary_samples_per_edge: int = 8,
    contact_goal_xy: tuple[float, float] = (256.0, 256.0),
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Build a passive target from a training/evaluation batch."""

    q_seq = adapter.build_q_sequence(batch)
    p_seq = adapter.build_p_sequence(q_seq, batch)
    if target_type == "damped_continuation":
        out = damped_continuation_projection(
            q_seq,
            p_seq,
            dt=adapter.dt,
            alpha_max=alpha_max,
            eps=eps,
        )
    elif target_type == "ema_smoothed_velocity":
        out = ema_smoothed_velocity_projection(
            q_seq,
            p_seq,
            dt=adapter.dt,
            ema_decay=ema_decay,
        )
    elif target_type == "contact_direction":
        if "future_obs_raw" not in batch:
            raise KeyError(
                "contact_direction requires batch['future_obs_raw']; "
                "use PushTLowDimDataset so contact geometry is evaluated in pixels."
            )
        out = contact_direction_projection(
            q_seq,
            p_seq,
            batch["future_obs_raw"],
            dt=adapter.dt,
            lambda_parallel=contact_lambda_parallel,
            lambda_perp=contact_lambda_perp,
            contact_distance=contact_distance,
            contact_temperature=contact_temperature,
            boundary_samples_per_edge=contact_boundary_samples_per_edge,
            goal_xy=contact_goal_xy,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported passive target type {target_type!r}.")
    out["q_seq"] = q_seq
    out["p_seq"] = p_seq
    raw_steps = []
    for k in range(q_seq.shape[1] - 1):
        raw_steps.append(adapter.decode_step(q_seq[:, k], out["q_bar_next"][:, k]))
    out["projected_raw_actions"] = torch.stack(raw_steps, dim=1)
    return out
