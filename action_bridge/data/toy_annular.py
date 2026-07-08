"""Annular clockwise/counterclockwise obstacle avoidance dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from action_bridge.data.chunking import (
    build_chunk_indices,
    split_trajectory_ids,
    trajectory_batch_item,
)


@dataclass
class AnnularConfig:
    num_contexts: int = 512
    paired_fraction: float = 0.3
    trajectory_len: int = 64
    chunk_horizon: int = 16
    obs_history: int = 2
    action_history: int = 2
    obstacle_center: tuple = (0.5, 0.5)
    obstacle_radius: float = 0.15
    margin: float = 0.08
    r_min: float = 0.28
    r_max: float = 0.48
    min_start_goal_distance: float = 0.35
    require_interaction: bool = True
    interaction_distance_threshold: float = 0.18
    p_min: float = 0.08
    temperature: float = 0.08
    speed_noise_std: float = 0.002
    seed: int = 0
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    train_absolute_actions: bool = False
    env_accepts_absolute_actions: bool = False


def _angle_delta(theta_s: float, theta_g: float, ccw: bool) -> float:
    if ccw:
        return (theta_g - theta_s) % (2.0 * np.pi)
    return (theta_s - theta_g) % (2.0 * np.pi)


def _line_distance_to_center(start: np.ndarray, goal: np.ndarray, center: np.ndarray) -> float:
    seg = goal - start
    denom = float(np.dot(seg, seg))
    if denom < 1e-10:
        return float(np.linalg.norm(start - center))
    alpha = float(np.dot(center - start, seg) / denom)
    alpha = min(1.0, max(0.0, alpha))
    closest = start + alpha * seg
    return float(np.linalg.norm(closest - center))


def _sample_point(rng: np.random.Generator, center: np.ndarray, cfg: AnnularConfig) -> np.ndarray:
    radius = rng.uniform(cfg.r_min, cfg.r_max)
    theta = rng.uniform(-np.pi, np.pi)
    return (center + radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)).astype(np.float32)


def sample_start_goal(rng: np.random.Generator, cfg: AnnularConfig) -> tuple:
    center = np.asarray(cfg.obstacle_center, dtype=np.float32)
    for _ in range(10000):
        start = _sample_point(rng, center, cfg)
        goal = _sample_point(rng, center, cfg)
        if np.linalg.norm(start - goal) < cfg.min_start_goal_distance:
            continue
        if cfg.require_interaction:
            distance = _line_distance_to_center(start, goal, center)
            if distance - cfg.obstacle_radius > cfg.interaction_distance_threshold:
                continue
        return start, goal
    raise RuntimeError("Could not sample an annular start/goal pair with the configured constraints.")


def mode_lengths_and_probs(start: np.ndarray, goal: np.ndarray, cfg: AnnularConfig) -> tuple:
    center = np.asarray(cfg.obstacle_center, dtype=np.float32)
    r_clear = cfg.obstacle_radius + cfg.margin
    s = start - center
    g = goal - center
    r_s = float(np.linalg.norm(s))
    r_g = float(np.linalg.norm(g))
    theta_s = float(np.arctan2(s[1], s[0]))
    theta_g = float(np.arctan2(g[1], g[0]))
    delta_ccw = _angle_delta(theta_s, theta_g, True)
    delta_cw = _angle_delta(theta_s, theta_g, False)
    length_ccw = abs(r_s - r_clear) + r_clear * delta_ccw + abs(r_g - r_clear)
    length_cw = abs(r_s - r_clear) + r_clear * delta_cw + abs(r_g - r_clear)
    logits = np.array([-length_cw / cfg.temperature, -length_ccw / cfg.temperature], dtype=np.float64)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    p_ccw = cfg.p_min + (1.0 - 2.0 * cfg.p_min) * float(probs[1])
    p_cw = 1.0 - p_ccw
    return length_cw, length_ccw, p_cw, p_ccw


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    diffs = points[1:] - points[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total < 1e-8:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, count)
    out = np.zeros((count, 2), dtype=np.float32)
    seg_idx = 0
    for i, target in enumerate(targets):
        while seg_idx < len(seg_lengths) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        denom = max(float(seg_lengths[seg_idx]), 1e-8)
        alpha = (target - cumulative[seg_idx]) / denom
        out[i] = (1.0 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1]
    return out


def generate_annular_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    mode: int,
    rng: np.random.Generator,
    cfg: AnnularConfig,
) -> tuple:
    center = np.asarray(cfg.obstacle_center, dtype=np.float32)
    r_clear = cfg.obstacle_radius + cfg.margin
    theta_s = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
    theta_g = float(np.arctan2(goal[1] - center[1], goal[0] - center[0]))
    start_clear = center + r_clear * np.array([np.cos(theta_s), np.sin(theta_s)], dtype=np.float32)
    goal_clear = center + r_clear * np.array([np.cos(theta_g), np.sin(theta_g)], dtype=np.float32)
    if mode > 0:
        delta = _angle_delta(theta_s, theta_g, True)
        arc_thetas = theta_s + np.linspace(0.0, delta, 48)
    else:
        delta = _angle_delta(theta_s, theta_g, False)
        arc_thetas = theta_s - np.linspace(0.0, delta, 48)
    arc = center[None] + r_clear * np.stack([np.cos(arc_thetas), np.sin(arc_thetas)], axis=1)
    raw = np.concatenate([start[None], start_clear[None], arc.astype(np.float32), goal_clear[None], goal[None]], axis=0)
    positions = _resample_polyline(raw, cfg.trajectory_len + 1)
    actions = positions[1:] - positions[:-1]
    if cfg.speed_noise_std > 0:
        actions = actions + rng.normal(0.0, cfg.speed_noise_std, size=actions.shape).astype(np.float32)
        positions = np.concatenate([positions[:1], positions[:1] + np.cumsum(actions, axis=0)], axis=0)
        positions[-1] = goal
        actions = positions[1:] - positions[:-1]
    return positions.astype(np.float32), actions.astype(np.float32)


def generate_annular_arrays(config: Optional[Dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
    data = dict(config or {})
    data.update(kwargs)
    cfg = AnnularConfig(**{k: v for k, v in data.items() if k in AnnularConfig.__dataclass_fields__})
    rng = np.random.default_rng(cfg.seed)
    paired_contexts = int(round(cfg.num_contexts * cfg.paired_fraction))
    single_contexts = max(0, cfg.num_contexts - paired_contexts)
    num_traj = paired_contexts * 2 + single_contexts

    positions = np.zeros((num_traj, cfg.trajectory_len + 1, 2), dtype=np.float32)
    actions = np.zeros((num_traj, cfg.trajectory_len, 2), dtype=np.float32)
    starts = np.zeros((num_traj, 2), dtype=np.float32)
    goals = np.zeros((num_traj, 2), dtype=np.float32)
    modes = np.zeros((num_traj,), dtype=np.int64)
    centers = np.repeat(np.asarray(cfg.obstacle_center, dtype=np.float32)[None], num_traj, axis=0)
    radii = np.full((num_traj,), cfg.obstacle_radius, dtype=np.float32)
    p_cw = np.zeros((num_traj,), dtype=np.float32)
    p_ccw = np.zeros((num_traj,), dtype=np.float32)
    length_cw = np.zeros((num_traj,), dtype=np.float32)
    length_ccw = np.zeros((num_traj,), dtype=np.float32)

    idx = 0
    for context_id in range(cfg.num_contexts):
        start, goal = sample_start_goal(rng, cfg)
        lcw, lccw, pcw, pccw = mode_lengths_and_probs(start, goal, cfg)
        if context_id < paired_contexts:
            mode_order = [1, -1] if rng.random() < 0.5 else [-1, 1]
        else:
            mode_order = [1 if rng.random() < pccw else -1]
        for mode in mode_order:
            positions[idx], actions[idx] = generate_annular_trajectory(start, goal, mode, rng, cfg)
            starts[idx] = start
            goals[idx] = goal
            modes[idx] = mode
            p_cw[idx] = pcw
            p_ccw[idx] = pccw
            length_cw[idx] = lcw
            length_ccw[idx] = lccw
            idx += 1

    return {
        "positions": torch.from_numpy(positions),
        "actions": torch.from_numpy(actions),
        "starts": torch.from_numpy(starts),
        "goals": torch.from_numpy(goals),
        "modes": torch.from_numpy(modes),
        "obstacle_centers": torch.from_numpy(centers),
        "obstacle_radii": torch.from_numpy(radii),
        "extra_context": {
            "p_cw_true": torch.from_numpy(p_cw),
            "p_ccw_true": torch.from_numpy(p_ccw),
            "length_cw": torch.from_numpy(length_cw),
            "length_ccw": torch.from_numpy(length_ccw),
        },
        "config": asdict(cfg),
    }


class AnnularObstacleDataset(Dataset):
    """Sliding-window annular chunks with known true mode probabilities."""

    def __init__(self, config: Optional[Dict] = None, split: str = "train", **kwargs):
        data = dict(config or {})
        data.update(kwargs)
        self.cfg = AnnularConfig(**{k: v for k, v in data.items() if k in AnnularConfig.__dataclass_fields__})
        arrays = generate_annular_arrays(asdict(self.cfg))
        self.positions = arrays["positions"].float()
        if self.cfg.train_absolute_actions:
            self.actions = arrays["positions"][:, 1:].float()
        else:
            self.actions = arrays["actions"].float()
        self.starts = arrays["starts"].float()
        self.goals = arrays["goals"].float()
        self.modes = arrays["modes"].long()
        self.obstacle_centers = arrays["obstacle_centers"].float()
        self.obstacle_radii = arrays["obstacle_radii"].float()
        self.extra_context = arrays.get("extra_context", {})
        traj_ids = split_trajectory_ids(
            self.positions.shape[0],
            split,
            train_fraction=self.cfg.train_fraction,
            val_fraction=self.cfg.val_fraction,
        )
        self.indices = build_chunk_indices(
            self.positions.shape[0],
            self.cfg.trajectory_len,
            self.cfg.obs_history,
            self.cfg.action_history,
            self.cfg.chunk_horizon,
            traj_ids,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        traj_id, t = self.indices[idx]
        return trajectory_batch_item(
            self.positions,
            self.actions,
            self.modes,
            self.starts,
            self.goals,
            self.obstacle_centers,
            self.obstacle_radii,
            self.extra_context,
            traj_id,
            t,
            self.cfg.obs_history,
            self.cfg.action_history,
            self.cfg.chunk_horizon,
            action_representation="absolute" if self.cfg.train_absolute_actions else "delta",
        )
