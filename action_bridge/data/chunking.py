"""Chunking helpers shared by toy and Push-T-style datasets."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def build_chunk_indices(
    num_trajectories: int,
    trajectory_len: int,
    obs_history: int,
    action_history: int,
    chunk_horizon: int,
    traj_ids: Sequence[int],
) -> List[Tuple[int, int]]:
    start_t = max(obs_history - 1, action_history)
    end_t = trajectory_len - chunk_horizon
    indices: List[Tuple[int, int]] = []
    for traj_id in traj_ids:
        if traj_id < 0 or traj_id >= num_trajectories:
            raise IndexError(f"Trajectory id {traj_id} out of range 0..{num_trajectories - 1}.")
        for t in range(start_t, end_t + 1):
            indices.append((int(traj_id), int(t)))
    return indices


def split_trajectory_ids(
    num_trajectories: int,
    split: str,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> range:
    n_train = int(train_fraction * num_trajectories)
    n_val = int(val_fraction * num_trajectories)
    if split == "train":
        return range(0, n_train)
    if split == "val":
        return range(n_train, min(num_trajectories, n_train + n_val))
    if split == "test":
        return range(min(num_trajectories, n_train + n_val), num_trajectories)
    if split == "all":
        return range(0, num_trajectories)
    raise ValueError(f"Unknown split {split!r}; expected train, val, test, or all.")


def trajectory_batch_item(
    positions: torch.Tensor,
    actions: torch.Tensor,
    modes: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    obstacle_centers: torch.Tensor,
    obstacle_radii: torch.Tensor,
    extra_context: dict,
    traj_id: int,
    t: int,
    obs_history: int,
    action_history: int,
    chunk_horizon: int,
) -> dict:
    pos = positions[traj_id]
    act = actions[traj_id]
    start = t - obs_history + 1
    obs_pos = pos[start : t + 1]
    goal = goals[traj_id].expand(obs_history, -1)
    obs_hist = torch.cat([obs_pos, goal], dim=-1)

    future_positions = pos[t : t + chunk_horizon + 1]
    future_actions = act[t : t + chunk_horizon]
    act_hist = act[t - action_history : t]

    context = {
        "start": starts[traj_id],
        "goal": goals[traj_id],
        "obstacle_center": obstacle_centers[traj_id],
        "obstacle_radius": obstacle_radii[traj_id],
        "traj_id": torch.tensor(traj_id, dtype=torch.long),
        "time_index": torch.tensor(t, dtype=torch.long),
    }
    for key, value in extra_context.items():
        context[key] = value[traj_id]

    mode_sign = modes[traj_id]
    label = (mode_sign > 0).long()
    true_probs = torch.stack([1.0 - context.get("p_ccw_true", label.float()), context.get("p_ccw_true", label.float())])
    if "p_cw_true" in context and "p_ccw_true" in context:
        true_probs = torch.stack([context["p_cw_true"], context["p_ccw_true"]])

    return {
        "obs_hist": obs_hist.float(),
        "act_hist": act_hist.float(),
        "future_actions": future_actions.float(),
        "future_positions": future_positions.float(),
        "mode_label": label,
        "mode_sign": mode_sign.long(),
        "true_mode_probs": true_probs.float(),
        "context": context,
    }
