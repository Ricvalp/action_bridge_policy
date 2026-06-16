"""Synthetic state-action data for execution-time action bridge policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PointObstacleConfig:
    """Small Push-T-like 2D control surrogate.

    A point agent starts on the left, must reach a goal on the right, and has
    paired top/bottom expert modes around a circular obstacle.
    """

    box_min: float = 0.0
    box_max: float = 1.0
    obstacle_center_x: float = 0.5
    obstacle_center_y: float = 0.5
    obstacle_radius: float = 0.16
    agent_radius: float = 0.015
    dt: float = 0.055
    max_action: float = 1.0
    expert_speed: float = 0.85
    expert_noise: float = 0.035
    waypoint_tol: float = 0.055
    lane_margin: float = 0.14
    goal_x: float = 0.88
    goal_y: float = 0.5
    goal_jitter: float = 0.055
    start_x: float = 0.12
    start_y: float = 0.5
    start_jitter_x: float = 0.035
    start_jitter_y: float = 0.08
    success_radius: float = 0.075

    @property
    def obstacle_center(self) -> np.ndarray:
        return np.array([self.obstacle_center_x, self.obstacle_center_y], dtype=np.float32)


def state_from_pos_goal(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.array([pos[0], pos[1], goal[0], goal[1]], dtype=np.float32)


def clip_action(action: np.ndarray, cfg: PointObstacleConfig) -> np.ndarray:
    norm = np.linalg.norm(action)
    if norm > cfg.max_action:
        action = action * (cfg.max_action / (norm + 1e-8))
    return action.astype(np.float32)


def segment_crosses_obstacle(
    p0: np.ndarray,
    p1: np.ndarray,
    cfg: PointObstacleConfig,
    margin: float = 0.0,
) -> bool:
    center = cfg.obstacle_center
    seg = p1 - p0
    denom = float(np.dot(seg, seg))
    if denom < 1e-10:
        dist = np.linalg.norm(p0 - center)
    else:
        alpha = float(np.dot(center - p0, seg) / denom)
        alpha = min(1.0, max(0.0, alpha))
        closest = p0 + alpha * seg
        dist = np.linalg.norm(closest - center)
    return bool(dist < cfg.obstacle_radius + cfg.agent_radius + margin)


def step_state(
    state: np.ndarray,
    action: np.ndarray,
    cfg: PointObstacleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one step and return contact flags [wall, obstacle]."""

    pos = state[:2].copy()
    goal = state[2:].copy()
    action = clip_action(action, cfg)
    next_pos = pos + cfg.dt * action
    contacts = np.zeros(2, dtype=np.float32)

    low = cfg.box_min + cfg.agent_radius
    high = cfg.box_max - cfg.agent_radius
    clipped = np.clip(next_pos, low, high)
    if not np.allclose(clipped, next_pos):
        contacts[0] = 1.0
    next_pos = clipped

    center = cfg.obstacle_center
    delta = next_pos - center
    dist = np.linalg.norm(delta)
    min_dist = cfg.obstacle_radius + cfg.agent_radius
    if dist < min_dist:
        normal = delta / (dist + 1e-8)
        next_pos = center + normal * min_dist
        contacts[1] = 1.0
    elif segment_crosses_obstacle(pos, next_pos, cfg):
        contacts[1] = 1.0

    return state_from_pos_goal(next_pos.astype(np.float32), goal.astype(np.float32)), contacts


def sample_start_goal(rng: np.random.Generator, cfg: PointObstacleConfig) -> tuple[np.ndarray, np.ndarray]:
    start = np.array(
        [
            cfg.start_x + rng.uniform(-cfg.start_jitter_x, cfg.start_jitter_x),
            cfg.start_y + rng.uniform(-cfg.start_jitter_y, cfg.start_jitter_y),
        ],
        dtype=np.float32,
    )
    goal = np.array(
        [
            cfg.goal_x,
            cfg.goal_y + rng.uniform(-cfg.goal_jitter, cfg.goal_jitter),
        ],
        dtype=np.float32,
    )
    return start, goal


def expert_waypoints(goal: np.ndarray, mode: int, cfg: PointObstacleConfig) -> list[np.ndarray]:
    lane_y = cfg.obstacle_center_y + mode * (cfg.obstacle_radius + cfg.lane_margin)
    lane_y = float(np.clip(lane_y, cfg.box_min + 0.08, cfg.box_max - 0.08))
    return [
        np.array([0.30, lane_y], dtype=np.float32),
        np.array([0.55, lane_y], dtype=np.float32),
        np.array([0.72, lane_y], dtype=np.float32),
        goal.astype(np.float32),
    ]


def expert_action(
    state: np.ndarray,
    waypoints: list[np.ndarray],
    waypoint_idx: int,
    rng: np.random.Generator,
    cfg: PointObstacleConfig,
) -> tuple[np.ndarray, int]:
    pos = state[:2]
    while waypoint_idx < len(waypoints) - 1 and np.linalg.norm(pos - waypoints[waypoint_idx]) < cfg.waypoint_tol:
        waypoint_idx += 1
    target = waypoints[waypoint_idx]
    direction = target - pos
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    action = cfg.expert_speed * direction
    if cfg.expert_noise > 0.0:
        action = action + rng.normal(0.0, cfg.expert_noise, size=2).astype(np.float32)
    return clip_action(action, cfg), waypoint_idx


def simulate_expert_trajectory(
    length: int,
    start: np.ndarray,
    goal: np.ndarray,
    mode: int,
    rng: np.random.Generator,
    cfg: PointObstacleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.zeros((length + 1, 4), dtype=np.float32)
    actions = np.zeros((length, 2), dtype=np.float32)
    contacts = np.zeros((length, 2), dtype=np.float32)
    states[0] = state_from_pos_goal(start, goal)
    waypoints = expert_waypoints(goal, mode, cfg)
    waypoint_idx = 0
    for t in range(length):
        actions[t], waypoint_idx = expert_action(states[t], waypoints, waypoint_idx, rng, cfg)
        states[t + 1], contacts[t] = step_state(states[t], actions[t], cfg)
    return states, actions, contacts


def generate_dataset(
    path: Path,
    num_trajectories: int,
    trajectory_length: int,
    seed: int,
    cfg: PointObstacleConfig,
    paired_modes: bool = True,
) -> Path:
    """Generate paired top/bottom expert demonstrations."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    states = np.zeros((num_trajectories, trajectory_length + 1, 4), dtype=np.float32)
    actions = np.zeros((num_trajectories, trajectory_length, 2), dtype=np.float32)
    contacts = np.zeros((num_trajectories, trajectory_length, 2), dtype=np.float32)
    modes = np.zeros((num_trajectories,), dtype=np.int64)

    idx = 0
    while idx < num_trajectories:
        start, goal = sample_start_goal(rng, cfg)
        if paired_modes and idx + 1 < num_trajectories:
            mode_order = [1, -1] if rng.random() < 0.5 else [-1, 1]
        else:
            mode_order = [1 if rng.random() < 0.5 else -1]

        for mode in mode_order:
            if idx >= num_trajectories:
                break
            states[idx], actions[idx], contacts[idx] = simulate_expert_trajectory(
                trajectory_length, start, goal, mode, rng, cfg
            )
            modes[idx] = mode
            idx += 1

    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        contacts=contacts,
        modes=modes,
        config=np.array(str(asdict(cfg))),
    )
    return path


class ActionChunkDataset(Dataset):
    """Sliding-window state/action chunk dataset."""

    def __init__(
        self,
        npz_path: Path,
        context: int,
        horizon: int,
        split: str,
        train_fraction: float = 0.8,
    ):
        data = np.load(npz_path)
        states = data["states"]
        actions = data["actions"]
        contacts = data["contacts"]
        modes = data["modes"]
        n_train = int(train_fraction * len(states))
        if split == "train":
            traj_ids = range(0, n_train)
        elif split == "test":
            traj_ids = range(n_train, len(states))
        else:
            raise ValueError(f"Unknown split: {split}")

        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        self.contacts = torch.from_numpy(contacts)
        self.modes = torch.from_numpy(modes)
        self.context = context
        self.horizon = horizon
        self.indices: list[tuple[int, int]] = []
        trajectory_length = actions.shape[1]
        for traj_id in traj_ids:
            for t in range(context, trajectory_length - horizon + 1):
                self.indices.append((traj_id, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_id, t = self.indices[idx]
        c = self.context
        h = self.horizon
        return {
            "context_states": self.states[traj_id, t - c + 1 : t + 1],
            "context_actions": self.actions[traj_id, t - c : t],
            "future_actions": self.actions[traj_id, t : t + h],
            "future_states": self.states[traj_id, t + 1 : t + h + 1],
            "future_contacts": self.contacts[traj_id, t : t + h],
            "mode": self.modes[traj_id],
            "traj_id": torch.tensor(traj_id, dtype=torch.long),
            "time_index": torch.tensor(t, dtype=torch.long),
        }


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}
