"""Concept figure: SB action marginals that split into multiple modes.

This is a slide helper, not a physical Push-T simulator. It visualizes the
intended behavior of an execution-time Schrodinger bridge policy:

    previous action distribution -> future action marginals

The first few executable actions are unimodal because the robot should keep
moving coherently. Later action marginals split into two valid modes, showing
that the policy remains probabilistic when the task admits symmetric solutions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TOP_COLOR = "#00a884"
BOTTOM_COLOR = "#7b61ff"
SOURCE_COLOR = "#222222"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_action_bridge(
    num_samples: int = 180,
    horizon: int = 16,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample action sequences whose marginals bifurcate late in the chunk.

    Returns:
        actions: (N, H, 2)
        modes: (N,), +1 and -1 for the two late modes
    """

    rng = np.random.default_rng(seed)
    modes = np.ones(num_samples, dtype=np.int64)
    modes[num_samples // 2 :] = -1
    rng.shuffle(modes)

    k = np.arange(horizon)
    split = sigmoid((k - 7.2) / 1.35)
    mean_x = 0.40 - 0.045 * (k / max(1, horizon - 1))

    actions = np.zeros((num_samples, horizon, 2), dtype=np.float32)
    for i, mode in enumerate(modes):
        mode_scale = rng.normal(1.0, 0.07)
        lateral_mean = mode * mode_scale * 0.34 * split

        # Low-frequency per-sample variation gives each sampled path continuity.
        phase = rng.uniform(0.0, 2.0 * np.pi)
        smooth_x = 0.012 * np.sin(0.55 * k + phase)
        smooth_y = 0.014 * np.sin(0.40 * k + phase)
        iid = rng.normal(0.0, [0.018, 0.020], size=(horizon, 2))

        actions[i, :, 0] = mean_x + smooth_x + iid[:, 0]
        actions[i, :, 1] = lateral_mean + smooth_y + iid[:, 1]

    return actions, modes


def previous_actions(num_samples: int, seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal([0.41, 0.0], [0.026, 0.024], size=(num_samples, 2)).astype(np.float32)


def kde_density(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, bandwidth: float = 0.052) -> np.ndarray:
    """Small NumPy KDE for 2D point clouds."""

    diff_x = grid_x[..., None] - points[:, 0]
    diff_y = grid_y[..., None] - points[:, 1]
    sq = diff_x**2 + diff_y**2
    density = np.exp(-0.5 * sq / (bandwidth**2)).mean(axis=-1)
    density /= max(float(density.max()), 1e-8)
    return density


def draw_density_panel(
    ax: plt.Axes,
    points: np.ndarray,
    top_mask: np.ndarray | None,
    label: str,
    show_mode_colors: bool,
) -> None:
    x = np.linspace(0.20, 0.52, 120)
    y = np.linspace(-0.43, 0.43, 150)
    grid_x, grid_y = np.meshgrid(x, y)
    density = kde_density(points, grid_x, grid_y)
    levels = [0.07, 0.17, 0.31, 0.50, 0.72, 0.92]
    ax.contourf(grid_x, grid_y, density, levels=levels, cmap="Blues", alpha=0.90)
    ax.contour(grid_x, grid_y, density, levels=levels[1:-1], colors="white", alpha=0.72, linewidths=0.8)

    if top_mask is None or not show_mode_colors:
        ax.scatter(points[:, 0], points[:, 1], s=10, color=SOURCE_COLOR, alpha=0.28, linewidth=0)
    else:
        ax.scatter(points[top_mask, 0], points[top_mask, 1], s=10, color=TOP_COLOR, alpha=0.26, linewidth=0)
        ax.scatter(points[~top_mask, 0], points[~top_mask, 1], s=10, color=BOTTOM_COLOR, alpha=0.26, linewidth=0)

    ax.text(0.50, 0.36, label, ha="right", va="top", fontsize=11, weight="bold")
    ax.set_xlim(0.20, 0.52)
    ax.set_ylim(-0.43, 0.43)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_bifurcation_strip(ax: plt.Axes, actions: np.ndarray, modes: np.ndarray) -> None:
    horizon = actions.shape[1]
    rng = np.random.default_rng(31)
    xs = np.arange(horizon)
    for i in range(actions.shape[0]):
        color = TOP_COLOR if modes[i] > 0 else BOTTOM_COLOR
        ax.plot(xs, actions[i, :, 1], color=color, alpha=0.055, linewidth=1.0)

    top_mean = actions[modes > 0, :, 1].mean(axis=0)
    bottom_mean = actions[modes < 0, :, 1].mean(axis=0)
    all_mean = actions[:, :, 1].mean(axis=0)
    ax.plot(xs[:7], all_mean[:7], color="#333333", linewidth=3.0, alpha=0.75)
    ax.plot(xs[6:], top_mean[6:], color=TOP_COLOR, linewidth=4.0)
    ax.plot(xs[6:], bottom_mean[6:], color=BOTTOM_COLOR, linewidth=4.0)

    for k in [0, 3, 6, 9, 12, 15]:
        jitter = rng.normal(0.0, 0.035, size=actions.shape[0])
        ax.scatter(
            k + jitter,
            actions[:, k, 1],
            s=13,
            color=np.where(modes > 0, TOP_COLOR, BOTTOM_COLOR),
            alpha=0.18,
            linewidth=0,
        )

    ax.axhline(0.0, color="#cfd3dc", linewidth=1.0, zorder=0)
    ax.text(1.1, 0.18, "early actions:\none distribution", fontsize=12, color="#222222")
    text_box = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.84)
    ax.text(
        8.7,
        0.43,
        "later actions:\ntwo valid modes",
        fontsize=12,
        color=TOP_COLOR,
        weight="bold",
        bbox=text_box,
        zorder=10,
    )
    ax.text(
        8.7,
        -0.49,
        "same context,\ndifferent good choices",
        fontsize=12,
        color=BOTTOM_COLOR,
        weight="bold",
        bbox=text_box,
        zorder=10,
    )
    ax.set_xlim(-0.4, horizon - 0.4)
    ax.set_ylim(-0.55, 0.53)
    ax.axis("off")


def plot_figure(out_png: Path, out_svg: Path, seed: int, samples: int, horizon: int) -> None:
    actions, modes = sample_action_bridge(samples, horizon, seed)
    prev = previous_actions(samples, seed + 19)
    top_mask = modes > 0
    selected_steps = [-1, 0, 2, 4, 6, 8, 11, horizon - 1]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "figure.dpi": 180,
        }
    )
    fig = plt.figure(figsize=(8.6, 7.8))
    gs = fig.add_gridspec(2, len(selected_steps), height_ratios=[1.0, 1.15], hspace=0.08, wspace=0.08)

    for col, step in enumerate(selected_steps):
        ax = fig.add_subplot(gs[0, col])
        if step == -1:
            points = prev
            label = "previous action"
            mask = None
            show_modes = False
        else:
            points = actions[:, step]
            label = f"action {step}"
            mask = top_mask
            show_modes = step >= 6
        draw_density_panel(ax, points, mask, label, show_modes)

    ax_strip = fig.add_subplot(gs[1, :])
    draw_bifurcation_strip(ax_strip, actions, modes)

    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.055, top=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/sb_action_distributions/sb_action_distribution_split.png"))
    parser.add_argument("--svg", type=Path, default=Path("runs/sb_action_distributions/sb_action_distribution_split.svg"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_figure(args.out, args.svg, args.seed, args.samples, args.horizon)
    print(f"saved: {args.out}")
    print(f"saved: {args.svg}")


if __name__ == "__main__":
    main()
