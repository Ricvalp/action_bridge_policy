"""Visualize toy datasets for geometry and mode sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from action_bridge.config import apply_overrides, load_config
from action_bridge.data.toy_annular import generate_annular_arrays
from action_bridge.data.toy_obstacle import generate_delayed_branch_arrays


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def common_data_config(config: Dict) -> Dict:
    data = dict(config.get("data", {}))
    for key in ["seed", "trajectory_len", "chunk_horizon", "obs_history", "action_history"]:
        if key in config:
            data[key] = config[key]
    return data


def generate_arrays(config: Dict) -> Dict[str, torch.Tensor]:
    data = common_data_config(config)
    if config.get("benchmark") == "toy_annular":
        return generate_annular_arrays(data)
    if config.get("benchmark") == "toy_delayed":
        return generate_delayed_branch_arrays(data)
    raise ValueError(f"Expected toy_delayed or toy_annular, got {config.get('benchmark')!r}.")


def draw_obstacle(ax, center: np.ndarray, radius: float, clear_radius: Optional[float] = None) -> None:
    import matplotlib.patches as patches

    ax.add_patch(patches.Circle(tuple(center), float(radius), fill=False, color="black", linewidth=1.7))
    if clear_radius is not None:
        ax.add_patch(
            patches.Circle(
                tuple(center),
                float(clear_radius),
                fill=False,
                color="0.4",
                linestyle="--",
                linewidth=1.1,
            )
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def as_numpy(arrays: Dict[str, torch.Tensor], key: str) -> np.ndarray:
    return arrays[key].detach().cpu().numpy()


def mode_colors(modes: np.ndarray) -> List[str]:
    return ["tab:blue" if int(mode) > 0 else "tab:orange" for mode in modes]


def find_pairs(starts: np.ndarray, goals: np.ndarray, modes: np.ndarray, max_pairs: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    used = set()
    for i in range(len(starts)):
        if i in used:
            continue
        for j in range(i + 1, len(starts)):
            if j in used:
                continue
            same_context = np.allclose(starts[i], starts[j], atol=1e-6) and np.allclose(goals[i], goals[j], atol=1e-6)
            opposite_modes = int(modes[i]) == -int(modes[j])
            if same_context and opposite_modes:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
        if len(pairs) >= max_pairs:
            break
    return pairs


def min_clearances(positions: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(positions - centers[:, None, :], axis=-1)
    return dist.min(axis=1) - radii


def plot_trajectory_overlay(arrays: Dict[str, torch.Tensor], config: Dict, out: Path, max_trajectories: int) -> None:
    plt = import_pyplot()
    positions = as_numpy(arrays, "positions")
    modes = as_numpy(arrays, "modes")
    starts = as_numpy(arrays, "starts")
    goals = as_numpy(arrays, "goals")
    centers = as_numpy(arrays, "obstacle_centers")
    radii = as_numpy(arrays, "obstacle_radii")
    n = min(max_trajectories, len(positions))
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for idx in range(n):
        color = "tab:blue" if int(modes[idx]) > 0 else "tab:orange"
        label = "mode +1" if int(modes[idx]) > 0 else "mode -1"
        ax.plot(positions[idx, :, 0], positions[idx, :, 1], color=color, alpha=0.38, linewidth=1.2, label=label if idx < 2 else None)
    ax.scatter(starts[:n, 0], starts[:n, 1], s=12, c="green", alpha=0.55, label="starts")
    ax.scatter(goals[:n, 0], goals[:n, 1], s=12, c="red", alpha=0.55, label="goals")
    clear = None
    if config.get("benchmark") == "toy_annular":
        clear = radii[0] + float(config.get("data", {}).get("margin", 0.08))
    draw_obstacle(ax, centers[0], radii[0], clear_radius=clear)
    ax.set_title(f"{config.get('benchmark')} full trajectories")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "01_trajectory_overlay.png", dpi=170)
    plt.close(fig)


def plot_paired_examples(arrays: Dict[str, torch.Tensor], config: Dict, out: Path, max_pairs: int) -> None:
    plt = import_pyplot()
    positions = as_numpy(arrays, "positions")
    actions = as_numpy(arrays, "actions")
    modes = as_numpy(arrays, "modes")
    starts = as_numpy(arrays, "starts")
    goals = as_numpy(arrays, "goals")
    centers = as_numpy(arrays, "obstacle_centers")
    radii = as_numpy(arrays, "obstacle_radii")
    pairs = find_pairs(starts, goals, modes, max_pairs)
    if not pairs:
        return
    cols = min(max_pairs, len(pairs))
    fig, axes = plt.subplots(1, cols, figsize=(4.4 * cols, 4.2), squeeze=False)
    prefix_steps = int(config.get("data", {}).get("shared_prefix_steps", 0)) if config.get("benchmark") == "toy_delayed" else 0
    for ax, (i, j) in zip(axes.ravel(), pairs):
        for idx in [i, j]:
            color = "tab:blue" if int(modes[idx]) > 0 else "tab:orange"
            label = "mode +1" if int(modes[idx]) > 0 else "mode -1"
            ax.plot(positions[idx, :, 0], positions[idx, :, 1], color=color, linewidth=1.7, alpha=0.85, label=label)
            if prefix_steps > 0:
                ax.plot(
                    positions[idx, : prefix_steps + 1, 0],
                    positions[idx, : prefix_steps + 1, 1],
                    color="black",
                    linewidth=2.4,
                    alpha=0.75,
                )
        draw_obstacle(ax, centers[i], radii[i])
        prefix_diff = 0.0
        if prefix_steps > 0:
            prefix_diff = float(np.abs(actions[i, :prefix_steps] - actions[j, :prefix_steps]).max())
        ax.set_title(f"paired context\nmax prefix action diff={prefix_diff:.2e}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "02_paired_mode_examples.png", dpi=170)
    plt.close(fig)


def plot_chunk_windows(arrays: Dict[str, torch.Tensor], config: Dict, out: Path, max_examples: int) -> None:
    plt = import_pyplot()
    positions = as_numpy(arrays, "positions")
    modes = as_numpy(arrays, "modes")
    centers = as_numpy(arrays, "obstacle_centers")
    radii = as_numpy(arrays, "obstacle_radii")
    horizon = int(config.get("chunk_horizon", 16))
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    start_t = max(obs_history - 1, action_history)
    candidate_times = np.linspace(start_t, positions.shape[1] - horizon - 1, num=4, dtype=int)
    rows = min(max_examples, 4)
    fig, axes = plt.subplots(rows, len(candidate_times), figsize=(3.4 * len(candidate_times), 3.0 * rows), squeeze=False)
    for r in range(rows):
        traj_idx = r
        for c, t in enumerate(candidate_times):
            ax = axes[r, c]
            ax.plot(positions[traj_idx, :, 0], positions[traj_idx, :, 1], color="0.82", linewidth=1.0)
            chunk = positions[traj_idx, t : t + horizon + 1]
            hist = positions[traj_idx, max(0, t - obs_history + 1) : t + 1]
            color = "tab:blue" if int(modes[traj_idx]) > 0 else "tab:orange"
            ax.plot(chunk[:, 0], chunk[:, 1], color=color, linewidth=2.0)
            ax.scatter(hist[:, 0], hist[:, 1], color="black", s=18)
            draw_obstacle(ax, centers[traj_idx], radii[traj_idx])
            ax.set_title(f"traj {traj_idx}, t={t}")
    fig.tight_layout()
    fig.savefig(out / "03_chunk_windows.png", dpi=170)
    plt.close(fig)


def plot_action_components(arrays: Dict[str, torch.Tensor], out: Path, max_trajectories: int) -> None:
    plt = import_pyplot()
    actions = as_numpy(arrays, "actions")
    modes = as_numpy(arrays, "modes")
    n = min(max_trajectories, len(actions))
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    t = np.arange(actions.shape[1])
    for idx in range(n):
        color = "tab:blue" if int(modes[idx]) > 0 else "tab:orange"
        axes[0].plot(t, actions[idx, :, 0], color=color, alpha=0.23, linewidth=1.0)
        axes[1].plot(t, actions[idx, :, 1], color=color, alpha=0.23, linewidth=1.0)
    for ax, name in zip(axes, ["action x", "action y"]):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("trajectory step")
    axes[0].set_title("Action components by mode")
    fig.tight_layout()
    fig.savefig(out / "04_action_components.png", dpi=170)
    plt.close(fig)


def plot_clearance_histogram(arrays: Dict[str, torch.Tensor], out: Path) -> None:
    plt = import_pyplot()
    positions = as_numpy(arrays, "positions")
    centers = as_numpy(arrays, "obstacle_centers")
    radii = as_numpy(arrays, "obstacle_radii")
    modes = as_numpy(arrays, "modes")
    clearance = min_clearances(positions, centers, radii)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(clearance[modes > 0], bins=30, alpha=0.65, color="tab:blue", label="mode +1")
    ax.hist(clearance[modes < 0], bins=30, alpha=0.65, color="tab:orange", label="mode -1")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("minimum clearance to obstacle")
    ax.set_ylabel("count")
    ax.set_title("Clearance distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "05_clearance_histogram.png", dpi=170)
    plt.close(fig)


def plot_start_goal_distribution(arrays: Dict[str, torch.Tensor], config: Dict, out: Path) -> None:
    plt = import_pyplot()
    starts = as_numpy(arrays, "starts")
    goals = as_numpy(arrays, "goals")
    modes = as_numpy(arrays, "modes")
    centers = as_numpy(arrays, "obstacle_centers")
    radii = as_numpy(arrays, "obstacle_radii")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(starts[:, 0], starts[:, 1], c=mode_colors(modes), s=16, alpha=0.55, marker="o", label="starts")
    ax.scatter(goals[:, 0], goals[:, 1], c=mode_colors(modes), s=22, alpha=0.55, marker="x", label="goals")
    clear = None
    if config.get("benchmark") == "toy_annular":
        clear = radii[0] + float(config.get("data", {}).get("margin", 0.08))
    draw_obstacle(ax, centers[0], radii[0], clear_radius=clear)
    ax.set_title("Start/goal distribution colored by sampled mode")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "06_start_goal_distribution.png", dpi=170)
    plt.close(fig)


def plot_delayed_prefix_diagnostics(arrays: Dict[str, torch.Tensor], config: Dict, out: Path) -> Dict[str, float]:
    plt = import_pyplot()
    actions = as_numpy(arrays, "actions")
    positions = as_numpy(arrays, "positions")
    starts = as_numpy(arrays, "starts")
    goals = as_numpy(arrays, "goals")
    modes = as_numpy(arrays, "modes")
    prefix = int(config.get("data", {}).get("shared_prefix_steps", 0))
    pairs = find_pairs(starts, goals, modes, max_pairs=200)
    diffs = []
    branch_y = []
    for i, j in pairs:
        if prefix > 0:
            diffs.append(float(np.abs(actions[i, :prefix] - actions[j, :prefix]).max()))
        branch_y.append(float(positions[i, prefix, 1] - positions[j, prefix, 1]))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].hist(diffs if diffs else [0.0], bins=30, color="tab:purple", alpha=0.8)
    axes[0].set_title("Paired prefix action max-diff")
    axes[0].set_xlabel("max |a_top - a_bottom|")
    axes[1].hist(branch_y if branch_y else [0.0], bins=30, color="tab:green", alpha=0.8)
    axes[1].set_title("Paired y-diff at fork")
    axes[1].set_xlabel("y_i - y_j")
    fig.tight_layout()
    fig.savefig(out / "07_delayed_prefix_diagnostics.png", dpi=170)
    plt.close(fig)
    return {
        "paired_contexts_found": float(len(pairs)),
        "paired_prefix_action_max_diff_mean": float(np.mean(diffs)) if diffs else 0.0,
        "paired_prefix_action_max_diff_max": float(np.max(diffs)) if diffs else 0.0,
    }


def plot_annular_probability_diagnostics(arrays: Dict[str, torch.Tensor], out: Path) -> Dict[str, float]:
    plt = import_pyplot()
    extra = arrays["extra_context"]
    p_ccw = extra["p_ccw_true"].detach().cpu().numpy()
    length_cw = extra["length_cw"].detach().cpu().numpy()
    length_ccw = extra["length_ccw"].detach().cpu().numpy()
    modes = as_numpy(arrays, "modes")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].hist(p_ccw, bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("true p(ccw)")
    axes[0].set_title("Mode probability")
    axes[1].scatter(length_ccw - length_cw, p_ccw, s=12, alpha=0.55)
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[1].set_xlabel("length_ccw - length_cw")
    axes[1].set_ylabel("p(ccw)")
    axes[1].set_title("Probability vs path length")
    axes[2].bar(["cw", "ccw"], [(modes < 0).mean(), (modes > 0).mean()], color=["tab:orange", "tab:blue"])
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Sampled mode rates")
    fig.tight_layout()
    fig.savefig(out / "07_annular_probability_diagnostics.png", dpi=170)
    plt.close(fig)
    return {
        "p_ccw_true_mean": float(p_ccw.mean()),
        "p_ccw_true_min": float(p_ccw.min()),
        "p_ccw_true_max": float(p_ccw.max()),
        "ccw_sample_rate": float((modes > 0).mean()),
        "cw_sample_rate": float((modes < 0).mean()),
    }


def save_summary(arrays: Dict[str, torch.Tensor], config: Dict, out: Path, extra: Dict[str, float]) -> None:
    positions = as_numpy(arrays, "positions")
    actions = as_numpy(arrays, "actions")
    centers = as_numpy(arrays, "obstacle_centers")
    radii = as_numpy(arrays, "obstacle_radii")
    modes = as_numpy(arrays, "modes")
    clearance = min_clearances(positions, centers, radii)
    speed = np.linalg.norm(actions, axis=-1)
    summary = {
        "benchmark": config.get("benchmark"),
        "num_trajectories": int(len(positions)),
        "trajectory_len": int(actions.shape[1]),
        "mode_positive_rate": float((modes > 0).mean()),
        "mode_negative_rate": float((modes < 0).mean()),
        "min_clearance_mean": float(clearance.mean()),
        "min_clearance_min": float(clearance.min()),
        "collision_rate": float((clearance < 0.0).mean()),
        "action_speed_mean": float(speed.mean()),
        "action_speed_max": float(speed.max()),
    }
    summary.update(extra)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-trajectories", type=int, default=96)
    parser.add_argument("--paired-examples", type=int, default=4)
    parser.add_argument("--chunk-examples", type=int, default=4)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config_name), args.overrides)
    out = Path(args.out_dir) if args.out_dir else Path("outputs") / "dataset_viz" / args.config_name
    out.mkdir(parents=True, exist_ok=True)
    arrays = generate_arrays(config)

    plot_trajectory_overlay(arrays, config, out, max_trajectories=args.max_trajectories)
    plot_paired_examples(arrays, config, out, max_pairs=args.paired_examples)
    plot_chunk_windows(arrays, config, out, max_examples=args.chunk_examples)
    plot_action_components(arrays, out, max_trajectories=args.max_trajectories)
    plot_clearance_histogram(arrays, out)
    plot_start_goal_distribution(arrays, config, out)

    extra: Dict[str, float] = {}
    if config.get("benchmark") == "toy_delayed":
        extra.update(plot_delayed_prefix_diagnostics(arrays, config, out))
    elif config.get("benchmark") == "toy_annular":
        extra.update(plot_annular_probability_diagnostics(arrays, out))
    save_summary(arrays, config, out, extra)
    print(out)


if __name__ == "__main__":
    main()
