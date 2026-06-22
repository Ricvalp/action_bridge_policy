"""Visualize expert marginal distributions in the delayed-mode dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import PointObstacleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/point_modes_prefix6_n1000_t72_s7.npz"),
        help="Path to a generated delayed-prefix .npz dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/dataset_viz"),
        help="Directory for visualization outputs.",
    )
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--shared_prefix_steps", type=int, default=6)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--split", choices=["all", "train", "test"], default="test")
    parser.add_argument("--max_contexts", type=int, default=96)
    parser.add_argument("--time_slices", type=int, default=7)
    return parser.parse_args()


def split_ids(num_items: int, split: str, train_fraction: float) -> np.ndarray:
    n_train = int(train_fraction * num_items)
    if split == "train":
        return np.arange(0, n_train)
    if split == "test":
        return np.arange(n_train, num_items)
    return np.arange(num_items)


def paired_ids(states: np.ndarray, modes: np.ndarray, ids: np.ndarray, t: int) -> list[tuple[int, int]]:
    id_set = set(int(i) for i in ids)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for traj_id in ids:
        traj_id = int(traj_id)
        if traj_id in used:
            continue
        candidates = []
        adjacent = traj_id + 1 if traj_id % 2 == 0 else traj_id - 1
        if adjacent in id_set:
            candidates.append(adjacent)
        same_context = np.max(np.abs(states[:, t] - states[traj_id, t]), axis=1) < 1e-5
        candidates.extend(int(i) for i in np.flatnonzero(same_context) if int(i) in id_set)
        partner = None
        for candidate in candidates:
            if candidate == traj_id or candidate in used:
                continue
            if int(modes[candidate]) != -int(modes[traj_id]):
                continue
            if np.max(np.abs(states[candidate, t] - states[traj_id, t])) < 1e-5:
                partner = candidate
                break
        if partner is None:
            continue
        top_id = traj_id if int(modes[traj_id]) > 0 else partner
        bottom_id = partner if top_id == traj_id else traj_id
        pairs.append((top_id, bottom_id))
        used.add(traj_id)
        used.add(partner)
    return pairs


def setup_world_axis(ax, cfg: PointObstacleConfig) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ax.plot(
        cfg.obstacle_center_x + cfg.obstacle_radius * np.cos(theta),
        cfg.obstacle_center_y + cfg.obstacle_radius * np.sin(theta),
        color="black",
        linewidth=0.9,
    )
    ax.set_xlim(cfg.box_min, cfg.box_max)
    ax.set_ylim(cfg.box_min, cfg.box_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_position_marginals(
    states: np.ndarray,
    pairs: list[tuple[int, int]],
    t: int,
    time_indices: np.ndarray,
    cfg: PointObstacleConfig,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(time_indices), figsize=(2.6 * len(time_indices), 2.8), squeeze=False)
    top_ids = np.array([pair[0] for pair in pairs], dtype=np.int64)
    bottom_ids = np.array([pair[1] for pair in pairs], dtype=np.int64)
    for ax, step in zip(axes[0], time_indices, strict=True):
        setup_world_axis(ax, cfg)
        top_pos = states[top_ids, t + step, :2]
        bottom_pos = states[bottom_ids, t + step, :2]
        ax.scatter(top_pos[:, 0], top_pos[:, 1], color="tab:blue", alpha=0.45, s=13, label="top")
        ax.scatter(bottom_pos[:, 0], bottom_pos[:, 1], color="tab:orange", alpha=0.45, s=13, label="bottom")
        ax.set_title(f"position k={int(step)}", fontsize=8)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_action_marginals(
    actions: np.ndarray,
    pairs: list[tuple[int, int]],
    t: int,
    time_indices: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(time_indices), figsize=(2.6 * len(time_indices), 2.8), squeeze=False)
    top_ids = np.array([pair[0] for pair in pairs], dtype=np.int64)
    bottom_ids = np.array([pair[1] for pair in pairs], dtype=np.int64)
    for ax, step in zip(axes[0], time_indices, strict=True):
        action_step = min(int(step), actions.shape[1] - t - 1)
        top_act = actions[top_ids, t + action_step]
        bottom_act = actions[bottom_ids, t + action_step]
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)
        ax.scatter(top_act[:, 0], top_act[:, 1], color="tab:blue", alpha=0.45, s=13, label="top")
        ax.scatter(bottom_act[:, 0], bottom_act[:, 1], color="tab:orange", alpha=0.45, s=13, label="bottom")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"action k={action_step}", fontsize=8)
        ax.set_xlabel("vx", fontsize=7)
        ax.set_ylabel("vy", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset)
    states = data["states"]
    actions = data["actions"]
    modes = data["modes"]
    ids = split_ids(len(states), args.split, args.train_fraction)
    t = max(args.context, args.shared_prefix_steps)
    if t + args.horizon >= states.shape[1]:
        raise ValueError("context + horizon exceeds trajectory length")
    pairs = paired_ids(states, modes, ids, t)[: args.max_contexts]
    if not pairs:
        raise ValueError("No paired top/bottom contexts found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    time_indices = np.unique(np.linspace(0, args.horizon, min(args.horizon + 1, args.time_slices), dtype=np.int64))
    stem = args.dataset.stem
    pos_path = args.out_dir / f"{stem}_{args.split}_gt_position_marginals.png"
    act_path = args.out_dir / f"{stem}_{args.split}_gt_action_marginals.png"
    plot_position_marginals(states, pairs, t, time_indices, PointObstacleConfig(), pos_path)
    plot_action_marginals(actions, pairs, t, time_indices, act_path)
    print(f"paired contexts: {len(pairs)}")
    print(pos_path)
    print(act_path)


if __name__ == "__main__":
    main()
