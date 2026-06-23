"""Visualize synthetic expert trajectories used by the action bridge sandbox."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data import PointObstacleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/point_modes_n256_t48_s7.npz"),
        help="Path to a generated .npz dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/dataset_viz"),
        help="Directory for visualization outputs.",
    )
    parser.add_argument("--max_trajectories", type=int, default=96)
    parser.add_argument("--grid_examples", type=int, default=12)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_indices(num_items: int, args: argparse.Namespace) -> np.ndarray:
    n_train = int(args.train_fraction * num_items)
    if args.split == "train":
        candidates = np.arange(0, n_train)
    elif args.split == "test":
        candidates = np.arange(n_train, num_items)
    else:
        candidates = np.arange(num_items)
    rng = np.random.default_rng(args.seed)
    if len(candidates) > args.max_trajectories:
        candidates = rng.choice(candidates, size=args.max_trajectories, replace=False)
    return np.sort(candidates)


def setup_axis(ax, cfg: PointObstacleConfig) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ox = cfg.obstacle_center_x + cfg.obstacle_radius * np.cos(theta)
    oy = cfg.obstacle_center_y + cfg.obstacle_radius * np.sin(theta)
    ax.plot(ox, oy, color="black", linewidth=1.2)
    ax.set_xlim(cfg.box_min, cfg.box_max)
    ax.set_ylim(cfg.box_min, cfg.box_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_overlay(states: np.ndarray, modes: np.ndarray, indices: np.ndarray, cfg: PointObstacleConfig, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axis(ax, cfg)
    colors = {1: "tab:blue", -1: "tab:orange"}
    labels = {1: "top mode", -1: "bottom mode"}
    used_labels = set()
    for idx in indices:
        pos = states[idx, :, :2]
        mode = int(modes[idx])
        label = labels[mode] if mode not in used_labels else None
        used_labels.add(mode)
        ax.plot(pos[:, 0], pos[:, 1], color=colors[mode], alpha=0.35, linewidth=1.0, label=label)
        ax.scatter(pos[0, 0], pos[0, 1], color=colors[mode], s=8, alpha=0.7)
        ax.scatter(states[idx, 0, 2], states[idx, 0, 3], color="tab:green", marker="*", s=18, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_title(f"Expert trajectories ({len(indices)} shown)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_grid(states: np.ndarray, modes: np.ndarray, indices: np.ndarray, cfg: PointObstacleConfig, path: Path) -> None:
    n = min(len(indices), 12)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for plot_idx, ax in enumerate(axes):
        setup_axis(ax, cfg)
        if plot_idx >= n:
            ax.axis("off")
            continue
        idx = indices[plot_idx]
        pos = states[idx, :, :2]
        mode = int(modes[idx])
        color = "tab:blue" if mode > 0 else "tab:orange"
        ax.plot(pos[:, 0], pos[:, 1], "-o", color=color, markersize=2, linewidth=1.1)
        ax.scatter(pos[0, 0], pos[0, 1], color=color, s=20)
        ax.scatter(states[idx, 0, 2], states[idx, 0, 3], color="tab:green", marker="*", s=60)
        ax.set_title("top" if mode > 0 else "bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset)
    states = data["states"]
    modes = data["modes"]
    cfg = PointObstacleConfig()
    indices = select_indices(len(states), args)

    stem = dataset.stem
    overlay_path = args.out_dir / f"{stem}_{args.split}_overlay.png"
    grid_path = args.out_dir / f"{stem}_{args.split}_grid.png"
    plot_overlay(states, modes, indices, cfg, overlay_path)
    plot_grid(states, modes, indices[: args.grid_examples], cfg, grid_path)
    print(overlay_path)
    print(grid_path)


if __name__ == "__main__":
    main()
