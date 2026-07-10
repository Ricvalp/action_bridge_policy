"""Diagnostics for contact-Langevin reference policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from action_bridge.eval.rollout import actions_to_positions, generate_chunk
from action_bridge.training.common import move_to_device


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _draw_obstacle(ax, center: Optional[torch.Tensor], radius: Optional[torch.Tensor]) -> None:
    if center is not None and radius is not None:
        import matplotlib.patches as patches

        c = center.detach().cpu()
        r = float(radius.detach().cpu())
        ax.add_patch(patches.Circle((float(c[0]), float(c[1])), r, fill=False, color="black", linewidth=1.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")


def _toy_positions_from_raw_actions(batch: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
    base = batch["future_positions"][:, 0].to(actions.device, actions.dtype)
    is_absolute = batch.get("action_is_absolute")
    if is_absolute is not None and bool(is_absolute.reshape(-1)[0].detach().cpu().item()):
        return torch.cat([base[:, None], actions], dim=1)
    return actions_to_positions(base, actions)


def _q_seq_as_display_positions(batch: Dict[str, Any], q_seq: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
    if "future_positions" not in batch:
        return q_seq
    mode = batch.get("reference_coordinate_mode", None)
    if mode in {"absolute_from_delta", "absolute_action"}:
        return q_seq
    return _toy_positions_from_raw_actions(batch, raw_actions)


@torch.no_grad()
def collect_contact_diagnostics(model, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Roll out controlled and reference-only contact dynamics for a batch."""

    if not bool(getattr(model, "uses_contact_langevin", False)):
        raise ValueError("Contact diagnostics require reference.type=contact_langevin.")

    batch_device = move_to_device(batch, device)
    obs_hist = batch_device["obs_hist"]
    act_hist = batch_device["act_hist"]
    ref = model.reference_process
    adapter = model.coordinate_adapter
    h = model.encode_history(obs_hist, act_hist)
    z, z_emb = model.sample_prior_z(h, deterministic_continuous=False)
    controlled = generate_chunk(model, obs_hist, act_hist, deterministic=True, z=z, z_emb=z_emb)

    q, p = adapter.init_qp_from_history(batch_device)
    ref_q = [q]
    ref_p = [p]
    aux_values = []
    for k in range(model.chunk_horizon):
        q, p, aux = ref.reference_step(q, p, h, k)
        ref_q.append(q)
        ref_p.append(p)
        aux_values.append(aux)
    ref_q_seq = torch.stack(ref_q, dim=1)
    ref_raw_actions = adapter.decode_raw_actions(ref_q_seq)

    controlled_aux = []
    q_seq = controlled["q_seq"]
    p_seq = controlled["p_seq"]
    for k in range(model.chunk_horizon):
        _, aux = ref.force(q_seq[:, k], p_seq[:, k], h, k)
        controlled_aux.append(aux)

    def stack_aux(key: str):
        values = [item.get(key) for item in controlled_aux]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return torch.stack(values, dim=1).detach().cpu()

    controls = controlled["controls"].detach().cpu()
    control_energy_steps = 0.5 * ref.dt * controls.pow(2).sum(dim=-1)
    batch_cpu = move_to_device(batch_device, torch.device("cpu"))
    batch_cpu["reference_coordinate_mode"] = adapter.coordinate_mode
    return {
        "batch": batch_cpu,
        "controlled_actions": controlled["actions"].detach().cpu(),
        "controlled_q_seq": controlled["q_seq"].detach().cpu(),
        "controlled_p_seq": controlled["p_seq"].detach().cpu(),
        "controlled_positions": _q_seq_as_display_positions(batch_cpu, controlled["q_seq"].detach().cpu(), controlled["actions"].detach().cpu()),
        "reference_actions": ref_raw_actions.detach().cpu(),
        "reference_q_seq": ref_q_seq.detach().cpu(),
        "reference_p_seq": torch.stack(ref_p, dim=1).detach().cpu(),
        "reference_positions": _q_seq_as_display_positions(batch_cpu, ref_q_seq.detach().cpu(), ref_raw_actions.detach().cpu()),
        "controls": controls,
        "control_energy_steps": control_energy_steps,
        "gamma": stack_aux("gamma"),
        "k_diag": stack_aux("k_diag"),
        "m": stack_aux("m"),
        "grad_v": stack_aux("grad_v"),
        "dt": float(ref.dt),
        "coordinate_mode": adapter.coordinate_mode,
    }


def contact_summary_stats(diagnostics: Dict[str, Any]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for key in ["gamma", "k_diag", "control_energy_steps"]:
        value = diagnostics.get(key)
        if value is None:
            continue
        flat = value.reshape(-1).float()
        stats[f"{key}_mean"] = float(flat.mean().item())
        stats[f"{key}_min"] = float(flat.min().item())
        stats[f"{key}_max"] = float(flat.max().item())
    return stats


def plot_contact_reference_summary(diagnostics: Dict[str, Any], path: Path, example_idx: int = 0) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = diagnostics["batch"]
    idx = min(example_idx, diagnostics["controlled_positions"].shape[0] - 1)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))

    ax = axes[0, 0]
    if "future_positions" in batch:
        expert = batch["future_positions"][idx]
        ax.plot(expert[:, 0], expert[:, 1], color="0.55", linewidth=2.0, label="expert")
    controlled = diagnostics["controlled_positions"][idx]
    reference = diagnostics["reference_positions"][idx]
    ax.plot(reference[:, 0], reference[:, 1], color="tab:green", linestyle="--", linewidth=1.8, label="reference only")
    ax.plot(controlled[:, 0], controlled[:, 1], color="tab:blue", linewidth=2.0, label="controlled")
    m = diagnostics.get("m")
    if m is not None:
        attractor = m[idx]
        ax.plot(attractor[:, 0], attractor[:, 1], color="tab:red", marker="x", markersize=4, linewidth=1.2, label="attractor m")
    context = batch.get("context", {})
    center = context.get("obstacle_center")
    radius = context.get("obstacle_radius")
    _draw_obstacle(ax, center[idx] if torch.is_tensor(center) else None, radius[idx] if torch.is_tensor(radius) else None)
    ax.set_title("Paths in q/world coordinates")
    ax.legend(fontsize=8)

    t = torch.arange(diagnostics["control_energy_steps"].shape[1])
    ax = axes[0, 1]
    gamma = diagnostics.get("gamma")
    if gamma is not None:
        gamma_plot = gamma[idx]
        if gamma_plot.shape[-1] == 1:
            ax.plot(t, gamma_plot[:, 0], label="gamma")
        else:
            for dim in range(gamma_plot.shape[-1]):
                ax.plot(t, gamma_plot[:, dim], label=f"gamma{dim}")
    k_diag = diagnostics.get("k_diag")
    if k_diag is not None:
        for dim in range(k_diag.shape[-1]):
            ax.plot(t, k_diag[idx, :, dim], linestyle="--", label=f"k{dim}")
    ax.set_title("Damping and stiffness")
    ax.set_xlabel("chunk step")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    energy = diagnostics["control_energy_steps"]
    ax.plot(t, energy[idx], color="tab:purple", label="example")
    ax.plot(t, energy.mean(dim=0), color="black", linewidth=2.0, alpha=0.75, label="batch mean")
    ax.set_title("Control energy")
    ax.set_xlabel("chunk step")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    p_norm = torch.linalg.norm(diagnostics["controlled_p_seq"][idx], dim=-1)
    ref_p_norm = torch.linalg.norm(diagnostics["reference_p_seq"][idx], dim=-1)
    ax.plot(torch.arange(p_norm.numel()), p_norm, label="controlled |p|")
    ax.plot(torch.arange(ref_p_norm.numel()), ref_p_norm, label="reference |p|")
    ax.set_title("Velocity norm")
    ax.set_xlabel("chunk step")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_contact_parameter_heatmaps(diagnostics: Dict[str, Any], path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    gamma = diagnostics.get("gamma")
    k_diag = diagnostics.get("k_diag")
    energy = diagnostics["control_energy_steps"]
    if gamma is not None:
        panels.append(("gamma", gamma.mean(dim=-1)))
    if k_diag is not None:
        for dim in range(k_diag.shape[-1]):
            panels.append((f"k{dim}", k_diag[:, :, dim]))
    panels.append(("control energy", energy))
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 2.2 * len(panels)), squeeze=False)
    for ax, (name, values) in zip(axes.ravel(), panels):
        image = ax.imshow(values.float().numpy(), aspect="auto", interpolation="nearest")
        ax.set_ylabel("example")
        ax.set_title(name)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    axes[-1, 0].set_xlabel("chunk step")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _spread_steps(horizon: int, count: int = 5) -> list[int]:
    if horizon <= 1:
        return [0]
    raw = torch.linspace(0, horizon - 1, steps=min(count, horizon)).round().long().tolist()
    steps: list[int] = []
    for step in raw:
        step = int(step)
        if step not in steps:
            steps.append(step)
    return steps


def _plot_contact_overlay(ax, diagnostics: Dict[str, Any], idx: int, *, show_labels: bool = False) -> None:
    batch = diagnostics["batch"]
    context = batch.get("context", {})
    expert = batch.get("future_positions")
    if expert is not None:
        e = expert[idx]
        ax.plot(e[:, 0], e[:, 1], color="white", linewidth=1.5, alpha=0.9, label="expert" if show_labels else None)
    reference = diagnostics.get("reference_positions")
    if reference is not None:
        r = reference[idx]
        ax.plot(r[:, 0], r[:, 1], color="tab:green", linestyle="--", linewidth=1.4, alpha=0.9, label="reference" if show_labels else None)
    controlled = diagnostics["controlled_positions"][idx]
    ax.plot(controlled[:, 0], controlled[:, 1], color="tab:cyan", linewidth=1.8, label="controlled" if show_labels else None)
    m = diagnostics.get("m")
    if m is not None:
        attractor = m[idx]
        ax.plot(attractor[:, 0], attractor[:, 1], color="tab:red", marker="x", markersize=3, linewidth=1.0, alpha=0.7, label="m path" if show_labels else None)
    goal = context.get("goal")
    if torch.is_tensor(goal):
        g = goal[idx] if goal.ndim > 1 else goal
        ax.scatter([float(g[0])], [float(g[1])], color="red", marker="*", s=70, edgecolors="white", linewidths=0.5, label="goal" if show_labels else None, zorder=6)
    center = context.get("obstacle_center")
    radius = context.get("obstacle_radius")
    _draw_obstacle(ax, center[idx] if torch.is_tensor(center) else None, radius[idx] if torch.is_tensor(radius) else None)


def plot_contact_potential_contours(diagnostics: Dict[str, Any], path: Path, example_idx: int = 0, grid_size: int = 80) -> None:
    m = diagnostics.get("m")
    k_diag = diagnostics.get("k_diag")
    if m is None or k_diag is None:
        return
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = min(example_idx, m.shape[0] - 1)
    horizon = m.shape[1]
    steps = _spread_steps(horizon, count=5)
    xs = torch.linspace(0.0, 1.0, grid_size)
    ys = torch.linspace(0.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    potentials = []
    forces = []
    force_norms = []
    for step in steps:
        center_m = m[idx, step]
        stiffness = k_diag[idx, step]
        diff = grid - center_m
        potential = 0.5 * (stiffness * diff.pow(2)).sum(dim=-1)
        force = -stiffness * diff
        potentials.append(potential)
        forces.append(force)
        force_norms.append(torch.linalg.norm(force, dim=-1))

    potential_stack = torch.stack(potentials)
    force_norm_stack = torch.stack(force_norms)
    potential_max = float(potential_stack.max().item())
    force_norm_max = float(force_norm_stack.max().item())
    potential_levels = torch.linspace(0.0, max(potential_max, 1e-8), 25).numpy()
    force_levels = torch.linspace(0.0, max(force_norm_max, 1e-8), 25).numpy()

    fig, axes = plt.subplots(2, len(steps), figsize=(4.1 * len(steps), 7.7), squeeze=False, layout="constrained")
    gamma = diagnostics.get("gamma")
    for col, step in enumerate(steps):
        center_m = m[idx, step]
        stiffness = k_diag[idx, step]
        gamma_text = ""
        if gamma is not None:
            gamma_value = gamma[idx, step].float().mean()
            gamma_text = f", gamma={float(gamma_value):.3g}"

        ax = axes[0, col]
        image = ax.contourf(xx.numpy(), yy.numpy(), potentials[col].numpy(), levels=potential_levels, cmap="viridis")
        ax.scatter([float(center_m[0])], [float(center_m[1])], color="red", marker="x", s=52, linewidths=2.0, label="m" if col == 0 else None, zorder=7)
        _plot_contact_overlay(ax, diagnostics, idx, show_labels=(col == 0))
        ax.set_title(f"V, k={step}\nstiff=[{float(stiffness[0]):.2g}, {float(stiffness[1]):.2g}]{gamma_text}")

        ax = axes[1, col]
        ax.contourf(xx.numpy(), yy.numpy(), force_norms[col].numpy(), levels=force_levels, cmap="magma")
        skip = max(1, grid_size // 14)
        force = forces[col]
        if force_norm_max > 1e-12:
            quiver = force / force_norm_max * 0.08
            ax.quiver(
                xx[::skip, ::skip].numpy(),
                yy[::skip, ::skip].numpy(),
                quiver[::skip, ::skip, 0].numpy(),
                quiver[::skip, ::skip, 1].numpy(),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                color="white",
                alpha=0.8,
            )
        ax.scatter([float(center_m[0])], [float(center_m[1])], color="cyan", marker="x", s=52, linewidths=2.0, zorder=7)
        _plot_contact_overlay(ax, diagnostics, idx, show_labels=False)
        ax.set_title(f"|-grad V| and force, k={step}")

    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.colorbar(image, ax=axes[0, :], fraction=0.02, pad=0.015, label="V(q,h,k)")
    force_image = axes[1, -1].collections[0]
    fig.colorbar(force_image, ax=axes[1, :], fraction=0.02, pad=0.015, label="|-grad V|")
    fig.savefig(path, dpi=170)
    plt.close(fig)
