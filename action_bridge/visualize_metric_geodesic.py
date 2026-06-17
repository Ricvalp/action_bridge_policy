"""Visualize geodesics in a 2D cost-aware metric.

This script is a slide-figure helper for the "differential geometry as the
language of motion cost" pitch. It builds a conformal Riemannian metric

    G(x) = lambda(x) I

where lambda(x) is a mixture of Gaussian cost hills. A geodesic is computed by
optimizing the discretized path energy

    sum_k (x_{k+1} - x_k)^T G(mid_k) (x_{k+1} - x_k).

In flat space this recovers a straight line. In the learned/cost-aware metric,
the low-energy path bends around expensive regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def gaussian_cost(points: torch.Tensor) -> torch.Tensor:
    """Return scalar metric multiplier lambda(x) for points shaped (..., 2)."""

    centers = points.new_tensor(
        [
            [0.50, 0.50],
            [0.62, 0.78],
            [0.34, 0.22],
        ]
    )
    sigmas = points.new_tensor(
        [
            [0.16, 0.17],
            [0.09, 0.09],
            [0.10, 0.08],
        ]
    )
    weights = points.new_tensor([40.0, 6.0, 4.0])

    diff = (points[..., None, :] - centers) / sigmas
    exponent = -0.5 * diff.pow(2).sum(dim=-1)
    hills = weights * torch.exp(exponent)

    # Soft boundary cost keeps the geodesic from solving the problem by
    # skimming the edge of the plot.
    x = points[..., 0]
    y = points[..., 1]
    wall_width = 0.055
    wall = (
        torch.exp(-x / wall_width)
        + torch.exp(-(1.0 - x) / wall_width)
        + torch.exp(-y / wall_width)
        + torch.exp(-(1.0 - y) / wall_width)
    )
    return 1.0 + hills.sum(dim=-1) + 2.4 * wall


def path_energy(path: torch.Tensor) -> torch.Tensor:
    """Discretized conformal-metric energy for a path shaped (N, 2)."""

    deltas = path[1:] - path[:-1]
    mids = 0.5 * (path[1:] + path[:-1])
    lam = gaussian_cost(mids)
    return (lam * deltas.pow(2).sum(dim=-1)).sum()


def optimize_geodesic(
    start: torch.Tensor,
    goal: torch.Tensor,
    n_points: int = 90,
    bend: float = 0.18,
    steps: int = 2500,
    lr: float = 0.035,
    seed: int = 7,
) -> tuple[torch.Tensor, list[float]]:
    """Optimize an interior path initialized as a bent line."""

    torch.manual_seed(seed)
    t = torch.linspace(0.0, 1.0, n_points, dtype=start.dtype, device=start.device)
    straight = (1.0 - t[:, None]) * start + t[:, None] * goal

    direction = goal - start
    normal = torch.stack([-direction[1], direction[0]])
    normal = normal / normal.norm().clamp_min(1e-8)
    envelope = torch.sin(torch.pi * t)[:, None]
    init_path = straight + bend * envelope * normal
    init_path = init_path.clamp(1e-4, 1.0 - 1e-4)

    raw = torch.nn.Parameter(logit(init_path[1:-1]))
    opt = torch.optim.Adam([raw], lr=lr)
    history: list[float] = []

    for _ in range(steps):
        interior = torch.sigmoid(raw)
        path = torch.cat([start[None], interior, goal[None]], dim=0)
        energy = path_energy(path)
        opt.zero_grad()
        energy.backward()
        opt.step()
        history.append(float(energy.detach().cpu()))

    # A short LBFGS polish makes the curve visibly smoother for slides.
    lbfgs = torch.optim.LBFGS([raw], lr=0.25, max_iter=120, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        interior = torch.sigmoid(raw)
        path = torch.cat([start[None], interior, goal[None]], dim=0)
        energy = path_energy(path)
        lbfgs.zero_grad()
        energy.backward()
        return energy

    lbfgs.step(closure)
    with torch.no_grad():
        final_path = torch.cat([start[None], torch.sigmoid(raw), goal[None]], dim=0)
    return final_path.detach(), history


def make_grid(n: int = 320) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(0.0, 1.0, n)
    ys = torch.linspace(0.0, 1.0, n)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx, yy], dim=-1)
    cost = gaussian_cost(points).numpy()
    return xx.numpy(), yy.numpy(), cost


def plot_figure(path: torch.Tensor, out_png: Path, out_svg: Path | None = None) -> None:
    xx, yy, cost = make_grid()
    start = path[0].cpu().numpy()
    goal = path[-1].cpu().numpy()
    geodesic = path.cpu().numpy()
    straight = np.stack(
        [
            np.linspace(start[0], goal[0], 100),
            np.linspace(start[1], goal[1], 100),
        ],
        axis=1,
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "figure.dpi": 180,
        }
    )
    fig, ax = plt.subplots(figsize=(13.6, 3.2))
    levels = np.linspace(cost.min(), np.percentile(cost, 99.5), 24)
    ax.contourf(xx, yy, cost, levels=levels, cmap="RdYlBu_r", alpha=0.93)
    ax.contour(xx, yy, cost, levels=10, colors="white", alpha=0.28, linewidths=0.7)

    ax.plot(
        straight[:, 0],
        straight[:, 1],
        color="#585858",
        linewidth=2.6,
        linestyle="--",
        zorder=5,
    )
    ax.plot(
        geodesic[:, 0],
        geodesic[:, 1],
        color="#05a87c",
        linewidth=4.2,
        zorder=6,
    )
    ax.scatter([start[0], goal[0]], [start[1], goal[1]], s=120, color="#111111", zorder=8)
    ax.text(start[0] - 0.035, start[1] - 0.045, "A", weight="bold", fontsize=15)
    ax.text(goal[0] + 0.018, goal[1] + 0.012, "B", weight="bold", fontsize=15)

    ax.annotate(
        "high motion cost",
        xy=(0.50, 0.50),
        xytext=(0.20, 0.82),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.84),
    )
    label_idx = int(0.50 * len(geodesic))
    ax.annotate(
        "low-cost geodesic",
        xy=(geodesic[label_idx, 0], geodesic[label_idx, 1]),
        xytext=(0.63, 0.20),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.84),
        annotation_clip=False,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 1.0)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/metric_geodesic/geodesic_metric.png"))
    parser.add_argument("--svg", type=Path, default=Path("runs/metric_geodesic/geodesic_metric.svg"))
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--points", type=int, default=90)
    parser.add_argument("--bend", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.svg is not None:
        args.svg.parent.mkdir(parents=True, exist_ok=True)

    start = torch.tensor([0.08, 0.40], dtype=torch.float32)
    goal = torch.tensor([0.92, 0.62], dtype=torch.float32)
    path, history = optimize_geodesic(
        start,
        goal,
        n_points=args.points,
        bend=args.bend,
        steps=args.steps,
    )
    straight = torch.stack(
        [
            torch.linspace(start[0], goal[0], args.points),
            torch.linspace(start[1], goal[1], args.points),
        ],
        dim=1,
    )
    print(f"straight energy: {path_energy(straight).item():.4f}")
    print(f"geodesic energy: {path_energy(path).item():.4f}")
    print(f"initial optimized energy: {history[0]:.4f}")
    print(f"final optimized energy: {history[-1]:.4f}")
    plot_figure(path, args.out, args.svg)
    print(f"saved: {args.out}")
    if args.svg is not None:
        print(f"saved: {args.svg}")


if __name__ == "__main__":
    main()
