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


def _entropic_ot_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 40,
) -> torch.Tensor:
    """Uniform entropic OT cost via log-domain Sinkhorn scaling.

    This is a small local replacement for the geomloss `SamplesLoss("sinkhorn")`
    used by the CWG codebase. It uses squared Euclidean cost and uniform masses.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sinkhorn inputs must be 2D point clouds.")
    eps = max(float(epsilon), 1e-6)
    cost = torch.cdist(x, y, p=2).pow(2)
    log_kernel = -cost / eps
    log_a = x.new_full((x.shape[0],), -torch.log(x.new_tensor(float(x.shape[0]))))
    log_b = y.new_full((y.shape[0],), -torch.log(y.new_tensor(float(y.shape[0]))))
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    log_plan = u[:, None] + log_kernel + v[None, :]
    plan = torch.exp(log_plan)
    return (plan * cost).sum()


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 40,
    debiased: bool = True,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence between two empirical point clouds."""

    xy = _entropic_ot_cost(x, y, epsilon=epsilon, iterations=iterations)
    if not debiased:
        return xy
    xx = _entropic_ot_cost(x, x, epsilon=epsilon, iterations=iterations)
    yy = _entropic_ot_cost(y, y, epsilon=epsilon, iterations=iterations)
    return xy - 0.5 * xx - 0.5 * yy


def joint_context_action_points(
    actions: torch.Tensor,
    history: torch.Tensor,
    context_weight: float = 0.05,
) -> torch.Tensor:
    """Build joint `(context, action)` points for context-aware OT matching."""

    if actions.ndim not in (2, 3):
        raise ValueError("actions must have shape (B, A) or (B, P, A).")
    if context_weight <= 0:
        return actions.reshape(-1, actions.shape[-1])

    scaled_history = F.normalize(history.detach(), dim=-1) * (float(context_weight) ** 0.5)
    if actions.ndim == 3:
        hist = scaled_history[:, None, :].expand(actions.shape[0], actions.shape[1], -1)
    else:
        hist = scaled_history
    return torch.cat([hist.reshape(actions.reshape(-1, actions.shape[-1]).shape[0], -1), actions.reshape(-1, actions.shape[-1])], dim=-1)


def sinkhorn_marginal_matching(
    particles: torch.Tensor,
    target_actions: torch.Tensor,
    history: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 40,
    context_weight: float = 0.05,
    intermediate_weight: float = 1.0,
    endpoint_weight: float = 1.0,
) -> torch.Tensor:
    """Match predicted action marginals to expert action marginals."""

    horizon = particles.shape[2]
    total = particles.new_zeros(())
    normalizer = 0.0
    for k in range(horizon):
        weight = endpoint_weight if k == horizon - 1 else intermediate_weight
        if weight == 0:
            continue
        pred_points = joint_context_action_points(particles[:, :, k], history, context_weight)
        target_points = joint_context_action_points(target_actions[:, k], history, context_weight)
        total = total + weight * sinkhorn_divergence(
            pred_points,
            target_points,
            epsilon=epsilon,
            iterations=iterations,
        )
        normalizer += float(weight)
    return total / max(1.0, normalizer)


def joint_context_path_points(
    paths: torch.Tensor,
    context: torch.Tensor,
    context_weight: float = 1.0,
) -> torch.Tensor:
    """Build joint `(context, full_action_path)` points for path-level OT.

    `paths` can be either `(B, H, A)` expert chunks or `(B, P, H, A)`
    generated particle chunks. The raw context coordinates are used directly
    because the synthetic state lives in a stable `[0, 1]` geometry.
    """

    if paths.ndim not in (3, 4):
        raise ValueError("paths must have shape (B, H, A) or (B, P, H, A).")
    if context.ndim != 2:
        raise ValueError("context must have shape (B, C).")
    flat_paths = paths.reshape(-1, paths.shape[-2] * paths.shape[-1])
    if context_weight <= 0:
        return flat_paths

    scaled_context = context.detach() * (float(context_weight) ** 0.5)
    if paths.ndim == 4:
        scaled_context = scaled_context[:, None, :].expand(paths.shape[0], paths.shape[1], -1)
    return torch.cat([scaled_context.reshape(flat_paths.shape[0], -1), flat_paths], dim=-1)


def sinkhorn_path_matching(
    particles: torch.Tensor,
    target_actions: torch.Tensor,
    context: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 40,
    context_weight: float = 1.0,
) -> torch.Tensor:
    """Match generated full action paths to expert full action paths."""

    pred_points = joint_context_path_points(particles, context, context_weight)
    target_points = joint_context_path_points(target_actions, context, context_weight)
    return sinkhorn_divergence(pred_points, target_points, epsilon=epsilon, iterations=iterations)


def sinkhorn_bridge_energy(
    init_particles: torch.Tensor,
    particles: torch.Tensor,
    history: torch.Tensor,
    phi_final: float = 1.4,
    epsilon: float = 0.05,
    iterations: int = 40,
    context_weight: float = 0.05,
) -> torch.Tensor:
    """CWG-style energy between consecutive particle marginals."""

    horizon = particles.shape[2]
    phi = linear_phi(horizon, phi_final, particles.device, particles.dtype)
    prev = init_particles
    total = particles.new_zeros(())
    for k in range(horizon):
        prev_points = joint_context_action_points(prev, history, context_weight)
        curr_points = joint_context_action_points(particles[:, :, k], history, context_weight)
        total = total + phi[k] * sinkhorn_divergence(
            curr_points,
            prev_points,
            epsilon=epsilon,
            iterations=iterations,
        )
        prev = particles[:, :, k]
    return total / max(1, horizon)


def particle_diversity(particles: torch.Tensor) -> torch.Tensor:
    """Mean per-step standard deviation across generated particles."""

    if particles.shape[1] <= 1:
        return particles.new_zeros(())
    return particles.std(dim=1).norm(dim=-1).mean()


def particle_path_diversity(particles: torch.Tensor) -> torch.Tensor:
    """Mean standard deviation of whole action paths across particles."""

    if particles.shape[1] <= 1:
        return particles.new_zeros(())
    flat = particles.reshape(particles.shape[0], particles.shape[1], -1)
    return flat.std(dim=1).norm(dim=-1).mean()
