"""Matplotlib diagnostics for toy action bridge runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from action_bridge.eval.rollout import actions_to_positions


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _draw_obstacle(ax, center, radius):
    import matplotlib.patches as patches

    circle = patches.Circle((float(center[0]), float(center[1])), float(radius), fill=False, color="black", linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")


def _actions_to_positions_for_batch(base_pos: torch.Tensor, actions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    is_absolute = batch.get("action_is_absolute")
    if is_absolute is not None and bool(is_absolute.reshape(-1)[0].detach().cpu().item()):
        return torch.cat([base_pos[:, None], actions], dim=1)
    return actions_to_positions(base_pos, actions)


def _batch_goal(batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    context = batch.get("context", {})
    if not isinstance(context, dict):
        return None
    goal = context.get("goal")
    if not torch.is_tensor(goal):
        return None
    if goal.ndim == 1:
        return goal[None]
    return goal


def plot_dataset_samples(dataset, path: Path, max_items: int = 24) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for idx in range(min(max_items, len(dataset))):
        item = dataset[idx]
        positions = item["future_positions"]
        color = "tab:blue" if int(item["mode_sign"]) > 0 else "tab:orange"
        ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=0.45, linewidth=1.2)
    first = dataset[0]
    _draw_obstacle(ax, first["context"]["obstacle_center"], first["context"]["obstacle_radius"])
    ax.set_title("Dataset chunks")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_generated_samples(
    batch: Dict[str, torch.Tensor],
    generated_actions: torch.Tensor,
    path: Path,
    title: str = "Generated samples for one history",
) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    if generated_actions.ndim == 3:
        generated_actions = generated_actions[:, None]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    base_pos = batch["future_positions"][:, 0]
    sample_count = generated_actions.shape[1]
    for b in range(generated_actions.shape[0]):
        ax.plot(
            batch["future_positions"][b, :, 0].detach().cpu(),
            batch["future_positions"][b, :, 1].detach().cpu(),
            color="0.7",
            linewidth=1.0,
            alpha=0.6,
        )
        for s in range(sample_count):
            is_absolute = batch.get("action_is_absolute")
            if torch.is_tensor(is_absolute):
                is_absolute = is_absolute.reshape(-1)[b : b + 1]
            single_batch = {**batch, "action_is_absolute": is_absolute}
            pred_pos = _actions_to_positions_for_batch(base_pos[b : b + 1], generated_actions[b, s : s + 1], single_batch).squeeze(0)
            ax.plot(pred_pos[:, 0].detach().cpu(), pred_pos[:, 1].detach().cpu(), linewidth=1.2, alpha=0.8)
    goals = _batch_goal(batch)
    if goals is not None:
        ax.scatter(
            goals[:, 0].detach().cpu(),
            goals[:, 1].detach().cpu(),
            color="red",
            marker="x",
            s=48,
            linewidths=2.0,
            label="goal",
            zorder=5,
        )
    _draw_obstacle(ax, batch["context"]["obstacle_center"][0], batch["context"]["obstacle_radius"][0])
    ax.set_title(title)
    if goals is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_energy_histograms(values: Dict[str, torch.Tensor], path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(1, len(values)), figsize=(4.0 * max(1, len(values)), 3.2))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.ravel().tolist()
        except AttributeError:
            axes = [axes]
    for ax, (name, tensor) in zip(axes, values.items()):
        ax.hist(tensor.detach().cpu().flatten().numpy(), bins=30, color="tab:blue", alpha=0.8)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(true_probs: torch.Tensor, pred_probs: torch.Tensor, path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(true_probs.detach().cpu(), pred_probs.detach().cpu(), s=12, alpha=0.6)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0)
    ax.set_xlabel("true p(ccw)")
    ax.set_ylabel("model p(ccw)")
    ax.set_title("Mode probability calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latent_scatter(z: torch.Tensor, mode: torch.Tensor, path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    if z.shape[-1] < 2:
        z = torch.cat([z, torch.zeros_like(z)], dim=-1)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    colors = ["tab:orange" if int(m) < 0 else "tab:blue" for m in mode.detach().cpu()]
    ax.scatter(z[:, 0].detach().cpu(), z[:, 1].detach().cpu(), c=colors, s=18, alpha=0.75)
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("Continuous latent samples by generated mode")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_closed_loop_rollouts(
    positions: torch.Tensor,
    expert_positions: torch.Tensor,
    goals: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    modes: torch.Tensor,
    success: torch.Tensor,
    path: Path,
    max_rollouts: int = 24,
) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_rollouts, positions.shape[0])
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for idx in range(count):
        expert = expert_positions[idx].detach().cpu()
        rollout = positions[idx].detach().cpu()
        mode = int(modes[idx].detach().cpu()) if torch.is_tensor(modes) else int(modes[idx])
        color = "tab:blue" if mode > 0 else "tab:orange"
        linewidth = 2.0 if bool(success[idx]) else 1.2
        alpha = 0.9 if bool(success[idx]) else 0.42
        ax.plot(expert[:, 0], expert[:, 1], color="0.75", linewidth=1.0, alpha=0.55)
        ax.plot(rollout[:, 0], rollout[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(goals[:count, 0].detach().cpu(), goals[:count, 1].detach().cpu(), color="red", marker="x", s=28, label="goals")
    _draw_obstacle(ax, centers[0].detach().cpu(), radii[0].detach().cpu())
    ax.set_title("Closed-loop receding-horizon rollouts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
