"""Toy rollout metrics for path coherence and geometry."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from action_bridge.eval.rollout import actions_to_positions


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def collision_stats(positions: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> Dict[str, torch.Tensor]:
    dist = torch.linalg.norm(positions - center[:, None, :], dim=-1)
    clearance = dist - radius[:, None]
    collision = clearance.min(dim=1).values < 0.0
    return {
        "collision": collision,
        "collision_rate": collision.float().mean(),
        "min_clearance": clearance.min(dim=1).values.mean(),
    }


def action_smoothness(actions: torch.Tensor, act_hist: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if act_hist is not None and act_hist.shape[1] >= 2:
        path = torch.cat([act_hist[:, -2:], actions], dim=1)
    elif act_hist is not None and act_hist.shape[1] >= 1:
        path = torch.cat([act_hist[:, -1:], actions], dim=1)
    else:
        path = actions
    if path.shape[1] >= 2:
        accel = path[:, 1:] - path[:, :-1]
        accel_energy = accel.pow(2).sum(dim=-1).sum(dim=1)
    else:
        accel_energy = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    if path.shape[1] >= 4:
        jerk = path[:, 3:] - 3.0 * path[:, 2:-1] + 3.0 * path[:, 1:-2] - path[:, :-3]
        jerk_energy = jerk.pow(2).sum(dim=-1).sum(dim=1)
    else:
        jerk_energy = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    return {"acceleration_energy": accel_energy.mean(), "jerk_energy": jerk_energy.mean()}


def toy_actions_to_positions(initial_pos: torch.Tensor, actions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    is_absolute = batch.get("action_is_absolute")
    if is_absolute is not None and bool(is_absolute.reshape(-1)[0].detach().cpu().item()):
        return torch.cat([initial_pos[:, None], actions], dim=1)
    return actions_to_positions(initial_pos, actions)


def delayed_mode_metrics(positions: torch.Tensor, center: torch.Tensor, radius: torch.Tensor, collision: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch = positions.shape[0]
    modes = []
    switches = []
    hybrids = []
    for i in range(batch):
        x = positions[i, :, 0]
        y = positions[i, :, 1]
        cx = center[i, 0]
        cy = center[i, 1]
        band = radius[i] + 0.08
        mask = (x > cx - band) & (x < cx + band)
        if not bool(mask.any()):
            nearest = torch.argmin((x - cx).abs())
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[nearest] = True
        signed = y[mask] - cy
        mode = torch.where(signed.mean() >= 0.0, torch.ones((), device=positions.device), -torch.ones((), device=positions.device))
        above = bool((signed > 0.025).any())
        below = bool((signed < -0.025).any())
        switch = above and below
        hybrid = switch or bool(collision[i])
        modes.append(mode)
        switches.append(torch.tensor(float(switch), device=positions.device))
        hybrids.append(torch.tensor(float(hybrid), device=positions.device))
    mode_tensor = torch.stack(modes)
    switch_tensor = torch.stack(switches)
    hybrid_tensor = torch.stack(hybrids)
    valid = (~collision) & (hybrid_tensor < 0.5)
    return {
        "mode_switch_rate": switch_tensor.mean(),
        "hybrid_rate": hybrid_tensor.mean(),
        "valid_top_rate": ((mode_tensor > 0) & valid).float().mean(),
        "valid_bottom_rate": ((mode_tensor < 0) & valid).float().mean(),
        "top_rate": (mode_tensor > 0).float().mean(),
        "bottom_rate": (mode_tensor < 0).float().mean(),
    }


def annular_mode_metrics(positions: torch.Tensor, center: torch.Tensor, collision: torch.Tensor) -> Dict[str, torch.Tensor]:
    modes = []
    switch_counts = []
    hybrids = []
    pos_np = positions.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy()
    collision_np = collision.detach().cpu().numpy()
    for i in range(pos_np.shape[0]):
        rel = pos_np[i] - center_np[i][None]
        theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        delta = theta[-1] - theta[0]
        modes.append(1.0 if delta >= 0 else -1.0)
        dtheta = np.diff(theta)
        signs = np.sign(dtheta[np.abs(dtheta) > 1e-3])
        sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        switch_counts.append(float(sign_changes))
        hybrids.append(float(sign_changes > 1 or collision_np[i]))
    mode = torch.tensor(modes, device=positions.device, dtype=positions.dtype)
    switches = torch.tensor(switch_counts, device=positions.device, dtype=positions.dtype)
    hybrid = torch.tensor(hybrids, device=positions.device, dtype=positions.dtype)
    return {
        "cw_rate": (mode < 0).float().mean(),
        "ccw_rate": (mode > 0).float().mean(),
        "mode_switch_count": switches.mean(),
        "hybrid_rate": hybrid.mean(),
    }


def feature_mmd(x: torch.Tensor, y: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_zeros(())
    all_points = torch.cat([x, y], dim=0)
    dists = torch.cdist(all_points, all_points).pow(2)
    if gamma is None:
        positive = dists[dists > 0]
        median = positive.median() if positive.numel() else x.new_tensor(1.0)
        gamma = float(1.0 / (2.0 * median.clamp_min(1e-6).item()))
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return kxx + kyy - 2.0 * kxy


def trajectory_features(positions: torch.Tensor, actions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    context = batch["context"]
    goal = context["goal"]
    center = context["obstacle_center"]
    radius = context["obstacle_radius"]
    final_error = torch.linalg.norm(positions[:, -1] - goal, dim=-1)
    distances = torch.linalg.norm(positions - center[:, None], dim=-1)
    min_clearance = distances.min(dim=1).values - radius
    path_length = torch.linalg.norm(positions[:, 1:] - positions[:, :-1], dim=-1).sum(dim=1)
    smooth = action_smoothness(actions, batch.get("act_hist"))
    mean_y = positions[:, :, 1].mean(dim=1)
    return torch.stack(
        [
            final_error,
            min_clearance,
            path_length,
            mean_y,
            smooth["acceleration_energy"].expand_as(final_error),
            smooth["jerk_energy"].expand_as(final_error),
        ],
        dim=-1,
    )


def compute_toy_metrics(
    pred_actions: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    path_kl_energy: Optional[torch.Tensor] = None,
    benchmark: str = "toy_delayed",
) -> Dict[str, float]:
    target_actions = batch["future_actions"]
    true_positions = batch["future_positions"]
    pred_positions = toy_actions_to_positions(true_positions[:, 0], pred_actions, batch)
    context = batch["context"]
    center = context["obstacle_center"]
    radius = context["obstacle_radius"]
    goal = context["goal"]

    collisions = collision_stats(pred_positions, center, radius)
    smooth = action_smoothness(pred_actions, batch["act_hist"])
    final_goal_error = torch.linalg.norm(pred_positions[:, -1] - goal, dim=-1).mean()
    path_length = torch.linalg.norm(pred_positions[:, 1:] - pred_positions[:, :-1], dim=-1).sum(dim=1).mean()
    action_mse = F.mse_loss(pred_actions, target_actions)
    feature_distance = feature_mmd(trajectory_features(pred_positions, pred_actions, batch), trajectory_features(true_positions, target_actions, batch))
    metrics = {
        "action_mse": _as_float(action_mse),
        "goal_error": _as_float(final_goal_error),
        "path_length": _as_float(path_length),
        "collision_rate": _as_float(collisions["collision_rate"]),
        "min_clearance": _as_float(collisions["min_clearance"]),
        "acceleration_energy": _as_float(smooth["acceleration_energy"]),
        "jerk_energy": _as_float(smooth["jerk_energy"]),
        "trajectory_feature_mmd": _as_float(feature_distance),
        "path_KL_energy": _as_float(path_kl_energy.mean()) if path_kl_energy is not None else 0.0,
    }
    if benchmark == "toy_annular":
        mode = annular_mode_metrics(pred_positions, center, collisions["collision"])
    else:
        mode = delayed_mode_metrics(pred_positions, center, radius, collisions["collision"])
    metrics.update({key: _as_float(value) for key, value in mode.items()})
    return metrics


def average_metric_dicts(items: list) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*(item.keys() for item in items)))
    out = {}
    for key in keys:
        values = [item[key] for item in items if key in item and np.isfinite(item[key])]
        if values:
            out[key] = float(np.mean(values))
    return out
