"""Delayed-branch top/bottom obstacle avoidance dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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
class DelayedBranchConfig:
    num_contexts: int = 512
    paired_fraction: float = 0.5
    trajectory_len: int = 64
    chunk_horizon: int = 16
    obs_history: int = 2
    action_history: int = 2
    obstacle_center: tuple = (0.5, 0.5)
    obstacle_radius: float = 0.13
    lane_margin: float = 0.08
    start_mean: tuple = (0.12, 0.5)
    start_jitter: tuple = (0.03, 0.08)
    goal_mean: tuple = (0.88, 0.5)
    goal_jitter: tuple = (0.03, 0.08)
    shared_prefix_steps: int = 8
    shared_prefix_target_x: float = 0.30
    action_noise_std: float = 0.005
    speed: float = 0.035
    seed: int = 0
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    train_absolute_actions: bool = False
    env_accepts_absolute_actions: bool = False


def _as_pair(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _clip_point(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.02, 0.98).astype(np.float32)


def _step_toward(pos: np.ndarray, target: np.ndarray, speed: float) -> np.ndarray:
    delta = target.astype(np.float32) - pos.astype(np.float32)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-8:
        return np.zeros(2, dtype=np.float32)
    step = min(float(speed), dist)
    return (delta / dist * step).astype(np.float32)


def _lane_waypoints(goal: np.ndarray, mode: int, cfg: DelayedBranchConfig) -> list:
    center = _as_pair(cfg.obstacle_center)
    lane_y = center[1] + float(mode) * (cfg.obstacle_radius + cfg.lane_margin)
    lane_y = float(np.clip(lane_y, 0.08, 0.92))
    return [
        np.array([cfg.shared_prefix_target_x, lane_y], dtype=np.float32),
        np.array([0.55, lane_y], dtype=np.float32),
        np.array([0.72, lane_y], dtype=np.float32),
        goal.astype(np.float32),
    ]


def simulate_delayed_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    mode: int,
    rng: np.random.Generator,
    cfg: DelayedBranchConfig,
) -> tuple:
    positions = np.zeros((cfg.trajectory_len + 1, 2), dtype=np.float32)
    actions = np.zeros((cfg.trajectory_len, 2), dtype=np.float32)
    positions[0] = start.astype(np.float32)
    waypoints = _lane_waypoints(goal, mode, cfg)
    waypoint_idx = 0
    shared_target = np.array([cfg.shared_prefix_target_x, start[1]], dtype=np.float32)

    for t in range(cfg.trajectory_len):
        pos = positions[t]
        if t < cfg.shared_prefix_steps:
            action = _step_toward(pos, shared_target, cfg.speed)
        else:
            while waypoint_idx < len(waypoints) - 1 and np.linalg.norm(pos - waypoints[waypoint_idx]) < cfg.speed * 1.5:
                waypoint_idx += 1
            action = _step_toward(pos, waypoints[waypoint_idx], cfg.speed)
            if cfg.action_noise_std > 0:
                action = action + rng.normal(0.0, cfg.action_noise_std, size=2).astype(np.float32)
                norm = float(np.linalg.norm(action))
                if norm > cfg.speed * 1.4:
                    action = action / (norm + 1e-8) * cfg.speed * 1.4
        actions[t] = action.astype(np.float32)
        positions[t + 1] = _clip_point(pos + actions[t])
    return positions, actions


def generate_delayed_branch_arrays(config: Optional[Dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
    data = dict(config or {})
    data.update(kwargs)
    cfg = DelayedBranchConfig(**{k: v for k, v in data.items() if k in DelayedBranchConfig.__dataclass_fields__})
    rng = np.random.default_rng(cfg.seed)

    paired_contexts = int(round(cfg.num_contexts * cfg.paired_fraction))
    single_contexts = max(0, cfg.num_contexts - paired_contexts)
    num_traj = paired_contexts * 2 + single_contexts

    positions = np.zeros((num_traj, cfg.trajectory_len + 1, 2), dtype=np.float32)
    actions = np.zeros((num_traj, cfg.trajectory_len, 2), dtype=np.float32)
    starts = np.zeros((num_traj, 2), dtype=np.float32)
    goals = np.zeros((num_traj, 2), dtype=np.float32)
    modes = np.zeros((num_traj,), dtype=np.int64)
    centers = np.repeat(_as_pair(cfg.obstacle_center)[None], num_traj, axis=0).astype(np.float32)
    radii = np.full((num_traj,), cfg.obstacle_radius, dtype=np.float32)

    idx = 0
    for context_id in range(cfg.num_contexts):
        start = _clip_point(_as_pair(cfg.start_mean) + rng.uniform(-1.0, 1.0, size=2).astype(np.float32) * _as_pair(cfg.start_jitter))
        goal = _clip_point(_as_pair(cfg.goal_mean) + rng.uniform(-1.0, 1.0, size=2).astype(np.float32) * _as_pair(cfg.goal_jitter))
        if context_id < paired_contexts:
            mode_order = [1, -1] if rng.random() < 0.5 else [-1, 1]
        else:
            mode_order = [1 if rng.random() < 0.5 else -1]
        for mode in mode_order:
            positions[idx], actions[idx] = simulate_delayed_trajectory(start, goal, mode, rng, cfg)
            starts[idx] = start
            goals[idx] = goal
            modes[idx] = mode
            idx += 1

    p_ccw = (modes > 0).astype(np.float32)
    extra = {
        "p_ccw_true": torch.from_numpy(p_ccw),
        "p_cw_true": torch.from_numpy(1.0 - p_ccw),
    }
    return {
        "positions": torch.from_numpy(positions),
        "actions": torch.from_numpy(actions),
        "starts": torch.from_numpy(starts),
        "goals": torch.from_numpy(goals),
        "modes": torch.from_numpy(modes),
        "obstacle_centers": torch.from_numpy(centers),
        "obstacle_radii": torch.from_numpy(radii),
        "extra_context": extra,
        "config": asdict(cfg),
    }


def save_delayed_branch(path: Path, config: Optional[Dict] = None, **kwargs) -> Path:
    arrays = generate_delayed_branch_arrays(config, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(arrays, path)
    return path


class DelayedBranchObstacleDataset(Dataset):
    """Sliding-window delayed-branch chunks with the common pilot batch schema."""

    def __init__(self, config: Optional[Dict] = None, split: str = "train", data_path: Optional[Path] = None, **kwargs):
        data = dict(config or {})
        data.update(kwargs)
        if data_path is not None:
            arrays = torch.load(Path(data_path), map_location="cpu")
            cfg_dict = dict(arrays.get("config", {}))
            cfg_dict.update(data)
            self.cfg = DelayedBranchConfig(**{k: v for k, v in cfg_dict.items() if k in DelayedBranchConfig.__dataclass_fields__})
        else:
            self.cfg = DelayedBranchConfig(**{k: v for k, v in data.items() if k in DelayedBranchConfig.__dataclass_fields__})
            arrays = generate_delayed_branch_arrays(asdict(self.cfg))
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
