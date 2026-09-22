"""Small observation-local entropic OT blocks, not global conditional transport.

The adapter supplies observed-context distances and compatibility. Output indices
select real endpoints; expert chunks are never barycentrically averaged.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def sinkhorn_coupling(cost, compatible, *, entropy=.1, iterations=500, tolerance=1e-7):
    """Uniform masses for one [N,N] or many [B,N,N] independent local blocks.

    One batched log-domain update serves every block; there is no Python loop
    over records or GPU synchronization per neighbourhood.
    """
    if cost.ndim not in (2, 3) or cost.shape[-2] != cost.shape[-1] or cost.shape[-1] == 0:
        raise ValueError("local OT expects nonempty square [N,N] or [B,N,N] costs")
    if compatible.shape != cost.shape or compatible.dtype != torch.bool:
        raise ValueError("compatible must be a boolean matrix matching cost")
    if not bool(compatible.diagonal(dim1=-2, dim2=-1).all()):
        raise ValueError("each natural diagonal pair must remain feasible")
    if entropy <= 0 or iterations < 1:
        raise ValueError("entropy and iteration count must be positive")
    if not bool(torch.isfinite(cost[compatible]).all()):
        raise ValueError("allowed transport costs must be finite")
    original_dtype = cost.dtype
    log_kernel = (-cost.double() / entropy).masked_fill(~compatible, -torch.inf)
    n = cost.shape[-1]
    log_mass = -math.log(n)
    log_u = torch.zeros(cost.shape[:-1], device=cost.device, dtype=torch.float64)
    log_v = torch.zeros_like(log_u)
    for iteration in range(iterations):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_mass - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        if iteration % 10 == 0 or iteration == iterations - 1:
            coupling = (log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).exp()
            error = torch.max(torch.abs(coupling.sum(-1) - 1 / n))
            if error <= tolerance:
                break
    # Rounding down leaves nonnegative deficits. A diagonal correction alone is
    # not generally enough, so expose convergence instead of changing hard masks.
    if float(error) > 1e-3:
        raise RuntimeError(
            "local Sinkhorn did not preserve uniform marginals within 1e-3; "
            "increase ot_iterations or entropy (do not relax context compatibility)"
        )
    return coupling.to(original_dtype)


@torch.no_grad()
def batched_local_ot_pair(source, target, context_distance, compatible, *, completion_ids=None,
                          radius=None, entropy=.1, context_weight=1., iterations=500,
                          target_indices=None, generator=None):
    """Pair many local [B,N,H,d] blocks in one vectorized Sinkhorn call.

    Supply ``target_indices=[0]`` for a uniformly chosen anchor placed first in a
    neighbourhood block. Neighbours must not silently replace uniform target
    sampling. With no selection, every target in an already uniformly drawn block
    is used once. A finite discrete source draw need not be a permutation. Returned
    diagnostics are per-block tensors, so callers need only synchronize for logs.
    """
    if source.ndim != 4 or source.shape != target.shape:
        raise ValueError("source and target must share shape [B,N,H,action_dim]")
    batch, n = source.shape[:2]
    if context_distance.shape != (batch, n, n) or compatible.shape != (batch, n, n):
        raise ValueError("the adapter must provide [B,N,N] context distances and compatibility")
    if bool((context_distance < 0).any()) or not bool(torch.isfinite(context_distance).all()):
        raise ValueError("context distances must be finite and nonnegative")
    allowed = compatible.clone().bool()
    if completion_ids is not None:
        if completion_ids.shape != (batch, n):
            raise ValueError("completion_ids must have shape [B,N]")
        allowed &= completion_ids[:, :, None] == completion_ids[:, None, :]
    if radius is not None:
        if radius < 0:
            raise ValueError("context radius must be nonnegative")
        allowed &= context_distance <= radius
    # Every original record is always a valid exact-context natural pairing.
    allowed.diagonal(dim1=-2, dim2=-1).fill_(True)
    action_cost = torch.cdist(source.flatten(2), target.flatten(2)).square() / (2 * source[0, 0].numel())
    cost = action_cost + context_weight * context_distance.square()
    coupling = sinkhorn_coupling(cost, allowed, entropy=entropy, iterations=iterations)
    targets = (torch.arange(n, device=source.device) if target_indices is None
               else torch.as_tensor(target_indices, device=source.device, dtype=torch.long))
    if targets.ndim != 1 or len(targets) == 0 or bool(((targets < 0) | (targets >= n)).any()):
        raise ValueError("target_indices must be a nonempty vector of valid block indices")
    columns = coupling.transpose(-2, -1).index_select(1, targets)
    source_indices = torch.multinomial(columns.reshape(-1, n), 1, generator=generator).reshape(batch, -1)
    rows, columns = coupling.sum(-1), coupling.sum(-2)
    targets = targets[None].expand(batch, -1)
    batch_ids = torch.arange(batch, device=source.device)[:, None]
    displacement = context_distance[batch_ids, source_indices, targets]
    off_diagonal = ~torch.eye(n, device=source.device, dtype=torch.bool)
    metrics = {
        "context_displacement": displacement.mean(-1),
        "max_context_displacement": displacement.max(-1).values,
        "off_diagonal_mass": (coupling * off_diagonal).sum((-1, -2)),
        "action_cost_before": action_cost.diagonal(dim1=-2, dim2=-1).mean(-1),
        "action_cost_after": (coupling * action_cost).sum((-1, -2)),
        "row_marginal_error": (rows - 1 / n).abs().max(-1).values,
        "column_marginal_error": (columns - 1 / n).abs().max(-1).values,
        "inactive": ~(allowed & off_diagonal).any(dim=(-1, -2)),
    }
    return source_indices, targets, metrics


@torch.no_grad()
def local_ot_pair(source, target, context_distance, compatible, *, completion_ids=None,
                  radius=None, entropy=.1, context_weight=1., iterations=500,
                  target_indices=None, generator=None):
    """Convenient single-block interface; training uses its batched counterpart."""
    if source.ndim != 3 or source.shape != target.shape:
        raise ValueError("source and target must share shape [N,H,action_dim]")
    source_ids, target_ids, metrics = batched_local_ot_pair(
        source[None], target[None], context_distance[None], compatible[None],
        completion_ids=None if completion_ids is None else completion_ids[None],
        radius=radius, entropy=entropy, context_weight=context_weight,
        iterations=iterations, target_indices=target_indices, generator=generator,
    )
    return source_ids[0], target_ids[0], {
        key: bool(value[0]) if key == "inactive" else float(value[0])
        for key, value in metrics.items()
    }


__all__ = ["sinkhorn_coupling", "local_ot_pair", "batched_local_ot_pair"]
