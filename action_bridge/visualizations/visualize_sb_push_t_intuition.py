"""Concept figure: action-by-action Schrodinger bridge policy intuition.

The figure is deliberately artificial. It shows a Push-T-like symmetric setting
where two future action chunks are equally valid: go above or below the object.
The sampled bridge paths make bridge time coincide with execution time, so every
intermediate sample is an action that would be executed by the robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


TOP_COLOR = "#00a884"
BOTTOM_COLOR = "#7b61ff"
SOURCE_COLOR = "#222222"
STRAIGHT_COLOR = "#777777"


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate a cubic Bezier curve."""

    t = t[:, None]
    return (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t**2 * p2
        + t**3 * p3
    )


def sample_bridge_paths(
    num_samples: int = 96,
    horizon: int = 16,
    seed: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample two-mode action paths and their induced actions.

    Returns:
        positions: (N, H + 1, 2)
        actions: (N, H, 2)
        modes: (N,), +1 for top, -1 for bottom
    """

    rng = np.random.default_rng(seed)
    modes = np.ones(num_samples, dtype=np.int64)
    modes[num_samples // 2 :] = -1
    rng.shuffle(modes)

    t = np.linspace(0.0, 1.0, horizon + 1)
    positions = []
    for mode in modes:
        sign = float(mode)
        p0 = np.array([0.12, 0.50])
        p1 = np.array([0.27, 0.50 + sign * 0.32])
        p2 = np.array([0.62, 0.50 + sign * 0.30])
        p3 = np.array([0.88, 0.50 + sign * 0.05])

        # Keep paired modes visually symmetric but not perfectly identical.
        p1 = p1 + rng.normal(0.0, [0.025, 0.025])
        p2 = p2 + rng.normal(0.0, [0.025, 0.025])
        p3 = p3 + rng.normal(0.0, [0.015, 0.018])
        curve = cubic_bezier(p0, p1, p2, p3, t)

        # Small temporally smooth perturbation so the clouds look probabilistic.
        phase = rng.uniform(0.0, 2.0 * np.pi)
        smooth_noise = np.stack(
            [
                0.006 * np.sin(2.0 * np.pi * t + phase),
                0.010 * np.sin(np.pi * t + phase),
            ],
            axis=-1,
        )
        smooth_noise[0] = 0.0
        curve = np.clip(curve + smooth_noise, 0.04, 0.96)
        positions.append(curve)

    positions_arr = np.stack(positions, axis=0)
    actions = positions_arr[:, 1:] - positions_arr[:, :-1]
    actions = actions / np.maximum(np.linalg.norm(actions, axis=-1, keepdims=True), 1e-8)
    actions = 0.42 * actions
    return positions_arr, actions, modes


def draw_t_shape(ax: plt.Axes, center: tuple[float, float], alpha: float, edge: str, face: str) -> None:
    """Draw a simple unrotated Push-T-like object."""

    cx, cy = center
    bar = patches.Rectangle(
        (cx - 0.11, cy + 0.045),
        0.22,
        0.055,
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        joinstyle="round",
    )
    stem = patches.Rectangle(
        (cx - 0.028, cy - 0.095),
        0.056,
        0.14,
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        joinstyle="round",
    )
    ax.add_patch(bar)
    ax.add_patch(stem)


def add_arrow_along_path(ax: plt.Axes, path: np.ndarray, color: str, index: int) -> None:
    p0 = path[index]
    p1 = path[index + 1]
    delta = p1 - p0
    ax.arrow(
        p0[0],
        p0[1],
        delta[0] * 0.78,
        delta[1] * 0.78,
        width=0.0025,
        head_width=0.025,
        head_length=0.025,
        length_includes_head=True,
        color=color,
        alpha=0.95,
        zorder=5,
    )


def plot_workspace(ax: plt.Axes, positions: np.ndarray, modes: np.ndarray) -> None:
    ax.set_facecolor("#f7f8fb")
    ax.add_patch(
        patches.Rectangle((0.03, 0.07), 0.94, 0.86, facecolor="none", edgecolor="#d5d9e2", linewidth=2.0)
    )
    draw_t_shape(ax, (0.50, 0.50), alpha=0.90, edge="#3b4252", face="#b8c0cc")
    draw_t_shape(ax, (0.78, 0.50), alpha=0.20, edge="#3b4252", face="#8fb3ff")

    for path, mode in zip(positions, modes, strict=True):
        color = TOP_COLOR if mode > 0 else BOTTOM_COLOR
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.16, linewidth=1.3)
        ax.scatter(path[3:-1:4, 0], path[3:-1:4, 1], s=9, color=color, alpha=0.15)

    top_mean = positions[modes > 0].mean(axis=0)
    bottom_mean = positions[modes < 0].mean(axis=0)
    ax.plot(top_mean[:, 0], top_mean[:, 1], color=TOP_COLOR, linewidth=4.0, label="top mode")
    ax.plot(bottom_mean[:, 0], bottom_mean[:, 1], color=BOTTOM_COLOR, linewidth=4.0, label="bottom mode")
    add_arrow_along_path(ax, top_mean, TOP_COLOR, 5)
    add_arrow_along_path(ax, bottom_mean, BOTTOM_COLOR, 5)

    ax.scatter([0.12], [0.50], s=180, color=SOURCE_COLOR, zorder=8)
    ax.text(0.075, 0.455, "current\npusher", fontsize=11, ha="center", va="top")
    ax.text(0.47, 0.39, "T", fontsize=16, weight="bold", color="#343a46")
    ax.text(0.72, 0.38, "target", fontsize=11, color="#343a46")
    ax.text(0.29, 0.83, "sampled executable\naction paths", fontsize=12, color="#111111")
    ax.text(0.66, 0.84, "mode 1", color=TOP_COLOR, fontsize=12, weight="bold")
    ax.text(0.66, 0.15, "mode 2", color=BOTTOM_COLOR, fontsize=12, weight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.04, 0.96)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_action_marginals(ax: plt.Axes, actions: np.ndarray, modes: np.ndarray) -> None:
    rng = np.random.default_rng(123)
    horizon = actions.shape[1]
    previous = rng.normal(0.0, 0.015, size=(actions.shape[0],))
    xs_prev = -1.0 + rng.normal(0.0, 0.045, size=actions.shape[0])
    ax.scatter(xs_prev, previous, s=20, color=SOURCE_COLOR, alpha=0.45, linewidth=0)

    for k in range(horizon):
        xs = k + rng.normal(0.0, 0.045, size=actions.shape[0])
        ay = actions[:, k, 1]
        ax.scatter(
            xs[modes > 0],
            ay[modes > 0],
            s=18,
            color=TOP_COLOR,
            alpha=0.28,
            linewidth=0,
        )
        ax.scatter(
            xs[modes < 0],
            ay[modes < 0],
            s=18,
            color=BOTTOM_COLOR,
            alpha=0.28,
            linewidth=0,
        )

    top_mean = actions[modes > 0, :, 1].mean(axis=0)
    bottom_mean = actions[modes < 0, :, 1].mean(axis=0)
    ax.plot(np.arange(horizon), top_mean, color=TOP_COLOR, linewidth=3.0)
    ax.plot(np.arange(horizon), bottom_mean, color=BOTTOM_COLOR, linewidth=3.0)
    ax.plot([-1, 0], [0, top_mean[0]], color=TOP_COLOR, linewidth=2.0, alpha=0.8)
    ax.plot([-1, 0], [0, bottom_mean[0]], color=BOTTOM_COLOR, linewidth=2.0, alpha=0.8)
    ax.axhline(0.0, color="#bbbbbb", linewidth=1.2, zorder=0)
    ax.text(-1.55, 0.045, "source\nprevious action", ha="center", va="bottom", fontsize=10)
    ax.text(3.5, 0.36, "top action mode", color=TOP_COLOR, fontsize=12, weight="bold")
    ax.text(3.5, -0.39, "bottom action mode", color=BOTTOM_COLOR, fontsize=12, weight="bold")
    ax.text(8.0, 0.04, "each column is an action marginal", color="#333333", fontsize=11)
    ax.set_xlim(-1.8, horizon - 0.25)
    ax.set_ylim(-0.47, 0.47)
    ax.set_xlabel("execution step in the action chunk")
    ax.set_ylabel("vertical action component")
    ax.set_title("Bridge time = robot execution time", fontsize=15, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e5ec", linewidth=0.9)
    ax.set_xticks([-1, 0, 4, 8, 12, 15])
    ax.set_xticklabels(["prev", "0", "4", "8", "12", "15"])


def plot_figure(out_png: Path, out_svg: Path, seed: int, samples: int, horizon: int) -> None:
    positions, actions, modes = sample_bridge_paths(num_samples=samples, horizon=horizon, seed=seed)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "figure.dpi": 180,
        }
    )
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.10)
    ax_workspace = fig.add_subplot(gs[0, 0])
    ax_actions = fig.add_subplot(gs[0, 1])
    plot_workspace(ax_workspace, positions, modes)
    plot_action_marginals(ax_actions, actions, modes)
    fig.suptitle(
        "Intended SB policy behavior: sample a multimodal executable action path",
        fontsize=17,
        y=0.995,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.13, top=0.89)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/sb_push_t_intuition/sb_push_t_action_bridge.png"))
    parser.add_argument("--svg", type=Path, default=Path("runs/sb_push_t_intuition/sb_push_t_action_bridge.svg"))
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_figure(args.out, args.svg, args.seed, args.samples, args.horizon)
    print(f"saved: {args.out}")
    print(f"saved: {args.svg}")


if __name__ == "__main__":
    main()
