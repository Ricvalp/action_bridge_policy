"""Diagnostics for contact-Langevin reference policies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from action_bridge.data.pusht_adapter import (
    denormalize_actions_tensor,
    denormalize_observations_tensor,
    normalize_actions_tensor,
)
from action_bridge.eval.rollout import actions_to_positions, generate_chunk
from action_bridge.training.common import move_to_device


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _tee_polygons(pose: torch.Tensor, scale: float = 30.0) -> list[list[tuple[float, float]]]:
    x, y, theta = [float(value) for value in pose[:3]]
    length = 4.0
    local_polys = [
        [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0.0),
            (-length * scale / 2, 0.0),
        ],
        [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ],
    ]
    c = math.cos(theta)
    s = math.sin(theta)
    return [[(x + px * c - py * s, y + px * s + py * c) for px, py in poly] for poly in local_polys]


def _draw_tee(ax, pose: torch.Tensor, color: str, alpha: float, label: Optional[str] = None, linestyle: str = "-") -> None:
    from matplotlib.patches import Polygon

    for idx, poly in enumerate(_tee_polygons(pose)):
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=color if linestyle == "-" else "none",
                edgecolor=color,
                linewidth=1.3,
                linestyle=linestyle,
                alpha=alpha,
                label=label if idx == 0 else None,
            )
        )


def _draw_obstacle(ax, center: Optional[torch.Tensor], radius: Optional[torch.Tensor]) -> None:
    if center is not None and radius is not None:
        import matplotlib.patches as patches

        c = center.detach().cpu()
        r = float(radius.detach().cpu())
        ax.add_patch(patches.Circle((float(c[0]), float(c[1])), r, fill=False, color="black", linewidth=1.4))


def _is_pusht_like_batch(batch: Dict[str, Any]) -> bool:
    obs_hist = batch.get("obs_hist")
    future_actions = batch.get("future_actions")
    return (
        "future_positions" not in batch
        and torch.is_tensor(obs_hist)
        and torch.is_tensor(future_actions)
        and obs_hist.shape[-1] >= 5
        and future_actions.shape[-1] >= 2
    )


def _has_normalization_stats(stats: Any) -> bool:
    if stats is None:
        return False
    return all(key in stats for key in ("obs_mean", "obs_std", "action_mean", "action_std"))


def _normalization_stats(diagnostics: Dict[str, Any]) -> Optional[Any]:
    stats = diagnostics.get("normalization_stats")
    if _has_normalization_stats(stats):
        return stats
    return None


def _actions_for_display(tensor: torch.Tensor, diagnostics: Dict[str, Any]) -> torch.Tensor:
    stats = _normalization_stats(diagnostics)
    if stats is None:
        return tensor
    return denormalize_actions_tensor(tensor, stats)


def _observations_for_display(tensor: torch.Tensor, diagnostics: Dict[str, Any]) -> torch.Tensor:
    stats = _normalization_stats(diagnostics)
    if stats is None:
        return tensor
    return denormalize_observations_tensor(tensor, stats)


def _raw_action_grid_to_model(grid: torch.Tensor, diagnostics: Dict[str, Any]) -> torch.Tensor:
    stats = _normalization_stats(diagnostics)
    if stats is None:
        return grid
    return normalize_actions_tensor(grid, stats)


def _force_to_display(force: torch.Tensor, diagnostics: Dict[str, Any]) -> torch.Tensor:
    stats = _normalization_stats(diagnostics)
    if stats is None:
        return force
    std = torch.as_tensor(stats["action_std"], dtype=force.dtype, device=force.device)
    return force / std


def _plot_bounds(diagnostics: Dict[str, Any], example_idx: int = 0) -> tuple[float, float, float, float, bool]:
    batch = diagnostics["batch"]
    if "future_positions" in batch:
        return 0.0, 1.0, 0.0, 1.0, False
    if _is_pusht_like_batch(batch):
        return 0.0, 512.0, 0.0, 512.0, True

    values = []
    for key in ["controlled_positions", "reference_positions", "m"]:
        value = diagnostics.get(key)
        if torch.is_tensor(value):
            values.append(value[..., :2].reshape(-1, 2).float())
    for key in ["act_hist", "future_actions"]:
        value = batch.get(key)
        if torch.is_tensor(value):
            values.append(value[..., :2].reshape(-1, 2).float())
    if not values:
        return 0.0, 1.0, 0.0, 1.0, False

    cloud = torch.cat(values, dim=0)
    x_min, y_min = cloud.min(dim=0).values.tolist()
    x_max, y_max = cloud.max(dim=0).values.tolist()
    x_span = max(float(x_max - x_min), 1e-3)
    y_span = max(float(y_max - y_min), 1e-3)
    pad = 0.08 * max(x_span, y_span)
    return float(x_min - pad), float(x_max + pad), float(y_min - pad), float(y_max + pad), False


def _apply_plot_bounds(ax, bounds: tuple[float, float, float, float, bool]) -> None:
    x_min, x_max, y_min, y_max, invert_y = bounds
    ax.set_xlim(x_min, x_max)
    if invert_y:
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")


def _potential_grid(bounds: tuple[float, float, float, float, bool], grid_size: int):
    x_min, x_max, y_min, y_max, _ = bounds
    xs = torch.linspace(x_min, x_max, grid_size)
    ys = torch.linspace(y_min, y_max, grid_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy, torch.stack([xx, yy], dim=-1)


def _draw_pusht_scene(ax, batch: Dict[str, Any], idx: int, show_labels: bool = False) -> None:
    if not _is_pusht_like_batch(batch):
        return
    diagnostics = {"normalization_stats": batch.get("_normalization_stats")}
    obs_hist = _observations_for_display(batch["obs_hist"], diagnostics)
    state = obs_hist[idx, -1].detach().cpu()
    agent = state[:2]
    block_pose = state[2:5]
    goal_pose = torch.tensor([256.0, 256.0, math.pi / 4], dtype=state.dtype)
    _draw_tee(ax, goal_pose, color="tab:green", alpha=0.28, label="goal T" if show_labels else None, linestyle="--")
    _draw_tee(ax, block_pose, color="0.35", alpha=0.35, label="current T" if show_labels else None)
    ax.scatter(
        [float(agent[0])],
        [float(agent[1])],
        color="tab:purple",
        marker="o",
        s=28,
        zorder=8,
        label="agent" if show_labels else None,
    )


def _toy_positions_from_raw_actions(batch: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
    base = batch["future_positions"][:, 0].to(actions.device, actions.dtype)
    is_absolute = batch.get("action_is_absolute")
    if is_absolute is not None and bool(is_absolute.reshape(-1)[0].detach().cpu().item()):
        return torch.cat([base[:, None], actions], dim=1)
    return actions_to_positions(base, actions)


def _make_toy_obs_hist(pos_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos_hist, goal[:, None, :].expand(pos_hist.shape[0], pos_hist.shape[1], -1)], dim=-1)


def _q_seq_as_display_positions(batch: Dict[str, Any], q_seq: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
    diagnostics = {"normalization_stats": batch.get("_normalization_stats")}
    if "future_positions" not in batch:
        return _actions_for_display(q_seq, diagnostics)
    mode = batch.get("reference_coordinate_mode", None)
    if mode in {"absolute_from_delta", "absolute_action"}:
        return _actions_for_display(q_seq, diagnostics)
    return _toy_positions_from_raw_actions(batch, raw_actions)


def _unique_trajectory_ids(dataset) -> list[int]:
    seen = set()
    ids: list[int] = []
    for traj_id, _ in getattr(dataset, "indices", []):
        traj_id = int(traj_id)
        if traj_id in seen:
            continue
        ids.append(traj_id)
        seen.add(traj_id)
    return ids


def _select_trajectory_id(dataset, trajectory_id: Optional[int] = None, trajectory_fraction: float = 0.5) -> int:
    ids = _unique_trajectory_ids(dataset)
    if not ids:
        raise ValueError("Closed-loop contact diagnostics require a toy dataset with trajectory indices.")
    if trajectory_id is not None:
        if int(trajectory_id) not in ids:
            raise ValueError(f"trajectory_id={trajectory_id} is not present in this split.")
        return int(trajectory_id)
    fraction = max(0.0, min(1.0, float(trajectory_fraction)))
    return ids[round(fraction * (len(ids) - 1))]


def _select_records(records: list[Dict[str, Any]], max_panels: int) -> list[Dict[str, Any]]:
    if max_panels <= 0 or len(records) <= max_panels:
        return records
    positions = torch.linspace(0, len(records) - 1, steps=max_panels).round().long().tolist()
    selected = []
    seen = set()
    for pos in positions:
        pos = int(pos)
        if pos in seen:
            continue
        selected.append(records[pos])
        seen.add(pos)
    return selected


@torch.no_grad()
def collect_closed_loop_contact_diagnostics(
    model,
    dataset,
    config: Dict[str, Any],
    device: torch.device,
    trajectory_id: Optional[int] = None,
    trajectory_fraction: float = 0.5,
    potential_step: Optional[int] = None,
    reference_steps: Optional[int] = None,
    reference_time_mode: str = "hold_last",
) -> Dict[str, Any]:
    """Run one toy closed-loop episode and record contact-potential snapshots."""

    if not bool(getattr(model, "uses_contact_langevin", False)):
        raise ValueError("Closed-loop contact diagnostics require reference.type=contact_langevin.")
    if not all(hasattr(dataset, name) for name in ["positions", "actions", "goals", "obstacle_centers", "obstacle_radii", "modes"]):
        raise ValueError("Closed-loop contact diagnostics currently support toy datasets.")

    model.eval()
    eval_cfg = config.get("eval", {})
    traj_id = _select_trajectory_id(dataset, trajectory_id=trajectory_id, trajectory_fraction=trajectory_fraction)
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    chunk_horizon = int(config.get("chunk_horizon", 16))
    ref_steps = int(reference_steps) if reference_steps is not None else 4 * chunk_horizon
    ref_steps = max(1, ref_steps)
    if reference_time_mode not in {"hold_last", "extrapolate"}:
        raise ValueError("reference_time_mode must be 'hold_last' or 'extrapolate'.")
    warm_start = int(eval_cfg.get("closed_loop_warm_start_steps", max(obs_history - 1, action_history)))
    n_exec = int(eval_cfg.get("n_exec", config.get("inference", {}).get("n_exec", chunk_horizon)))
    n_exec = max(1, n_exec)
    local_step = min(n_exec - 1, chunk_horizon - 1) if potential_step is None else int(potential_step)
    local_step = max(0, min(local_step, chunk_horizon - 1))
    action_clip = eval_cfg.get("action_clip", None)
    if action_clip is not None:
        action_clip = float(action_clip)
    box_min = float(eval_cfg.get("box_min", 0.0))
    box_max = float(eval_cfg.get("box_max", 1.0))
    deterministic = bool(config.get("inference", {}).get("deterministic", True))
    commitment = str(config.get("inference", {}).get("latent_commitment", "chunk"))
    action_is_absolute = bool(getattr(getattr(dataset, "cfg", None), "train_absolute_actions", False))

    expert_positions = dataset.positions[traj_id : traj_id + 1].to(device)
    expert_actions = dataset.actions[traj_id : traj_id + 1].to(device)
    goal = dataset.goals[traj_id : traj_id + 1].to(device)
    center = dataset.obstacle_centers[traj_id : traj_id + 1].to(device)
    radius = dataset.obstacle_radii[traj_id : traj_id + 1].to(device)
    mode = dataset.modes[traj_id : traj_id + 1].to(device)

    warm_start = max(max(obs_history - 1, action_history), min(warm_start, expert_actions.shape[1] - 1))
    pos_hist = expert_positions[:, warm_start - obs_history + 1 : warm_start + 1]
    act_hist = expert_actions[:, warm_start - action_history : warm_start]
    current_pos = expert_positions[:, warm_start]
    rollout_prefix = expert_positions[:, : warm_start + 1]
    generated_positions = []
    generated_actions = []
    records = []
    z = None
    z_emb = None
    executed_so_far = 0
    remaining = expert_actions.shape[1] - warm_start
    replan_index = 0

    while remaining > 0:
        obs_hist = _make_toy_obs_hist(pos_hist, goal)
        if commitment == "episode" and z_emb is not None:
            pred = generate_chunk(model, obs_hist, act_hist, deterministic=deterministic, z=z, z_emb=z_emb)
        elif commitment == "sticky":
            pred = generate_chunk(
                model,
                obs_hist,
                act_hist,
                deterministic=deterministic,
                z=z,
                sticky=True,
                kappa=float(eval_cfg.get("sticky_kappa", 2.0)),
                rho_z=float(eval_cfg.get("rho_z", 1.0)),
            )
            z = pred.get("z")
        else:
            pred = generate_chunk(model, obs_hist, act_hist, deterministic=deterministic)
        if commitment == "episode" and z_emb is None:
            z = pred.get("z")
            z_emb = pred.get("z_emb")

        h_emb = model.encode_history(obs_hist, act_hist)
        q_seq = pred["q_seq"]
        p_seq = pred["p_seq"]
        m_values = []
        k_values = []
        gamma_values = []
        obs_state = obs_hist[:, -1]
        for k in range(chunk_horizon):
            _, aux = model.reference_process.force(q_seq[:, k], p_seq[:, k], h_emb, k, obs_state=obs_state)
            if aux["m"] is None or aux["k_diag"] is None:
                raise ValueError("Closed-loop contact potential plots require reference.potential_type=quadratic.")
            m_values.append(aux["m"])
            k_values.append(aux["k_diag"])
            gamma_values.append(aux["gamma"])
        m_path = torch.stack(m_values, dim=1)
        k_path = torch.stack(k_values, dim=1)
        gamma_path = torch.stack(gamma_values, dim=1)

        ref_q, ref_p = model.coordinate_adapter.init_qp_from_history({"obs_hist": obs_hist, "act_hist": act_hist})
        ref_q_values = [ref_q]
        for step in range(ref_steps):
            ref_k = min(step, chunk_horizon - 1) if reference_time_mode == "hold_last" else step
            ref_q, ref_p, _ = model.reference_process.reference_step(ref_q, ref_p, h_emb, ref_k, obs_state=obs_state)
            ref_q_values.append(ref_q)
        ref_q_long = torch.stack(ref_q_values, dim=1)
        if model.coordinate_adapter.coordinate_mode in {"absolute_from_delta", "absolute_action"}:
            reference_no_control_positions = ref_q_long
        else:
            ref_actions = model.coordinate_adapter.decode_raw_actions(ref_q_long)
            reference_no_control_positions = actions_to_positions(current_pos, ref_actions)

        records.append(
            {
                "replan_index": replan_index,
                "trajectory_time": warm_start + executed_so_far,
                "local_step": local_step,
                "current_position": current_pos[0].detach().cpu(),
                "predicted_q_seq": q_seq[0].detach().cpu(),
                "predicted_actions": pred["actions"][0].detach().cpu(),
                "m_path": m_path[0].detach().cpu(),
                "k_diag_path": k_path[0].detach().cpu(),
                "gamma_path": gamma_path[0].detach().cpu(),
                "m": m_path[0, local_step].detach().cpu(),
                "k_diag": k_path[0, local_step].detach().cpu(),
                "gamma": gamma_path[0, local_step].detach().cpu(),
                "reference_no_control_positions": reference_no_control_positions[0].detach().cpu(),
            }
        )

        actions = pred["actions"]
        if action_clip is not None:
            actions = actions.clamp(-action_clip, action_clip)
        execute = min(n_exec, remaining, actions.shape[1])
        executed = actions[:, :execute]
        generated_actions.append(executed)

        new_positions = []
        for step in range(execute):
            if action_is_absolute:
                current_pos = executed[:, step].clamp(box_min, box_max)
            else:
                current_pos = (current_pos + executed[:, step]).clamp(box_min, box_max)
            new_positions.append(current_pos)
        new_pos_tensor = torch.stack(new_positions, dim=1)
        generated_positions.append(new_pos_tensor)
        pos_hist = torch.cat([pos_hist, new_pos_tensor], dim=1)[:, -obs_history:]
        act_hist = torch.cat([act_hist, executed], dim=1)[:, -action_history:]
        executed_so_far += execute
        remaining -= execute
        replan_index += 1

    generated_positions_tensor = torch.cat(generated_positions, dim=1)
    generated_actions_tensor = torch.cat(generated_actions, dim=1)
    rollout_positions = torch.cat([rollout_prefix, generated_positions_tensor], dim=1)
    return {
        "trajectory_id": traj_id,
        "warm_start": warm_start,
        "n_exec": n_exec,
        "potential_step": local_step,
        "reference_steps": ref_steps,
        "reference_time_mode": reference_time_mode,
        "records": records,
        "rollout_positions": rollout_positions[0].detach().cpu(),
        "generated_actions": generated_actions_tensor[0].detach().cpu(),
        "expert_positions": expert_positions[0, : rollout_positions.shape[1]].detach().cpu(),
        "goal": goal[0].detach().cpu(),
        "obstacle_center": center[0].detach().cpu(),
        "obstacle_radius": radius[0].detach().cpu(),
        "mode": mode[0].detach().cpu(),
    }


@torch.no_grad()
def collect_contact_diagnostics(
    model,
    batch: Dict[str, Any],
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Roll out controlled and reference-only contact dynamics for a batch."""

    if not bool(getattr(model, "uses_contact_langevin", False)):
        raise ValueError("Contact diagnostics require reference.type=contact_langevin.")

    batch_device = move_to_device(batch, device)
    obs_hist = batch_device["obs_hist"]
    act_hist = batch_device["act_hist"]
    ref = model.reference_process
    adapter = model.coordinate_adapter
    h = model.encode_history(obs_hist, act_hist)
    obs_state = obs_hist[:, -1]
    z, z_emb = model.sample_prior_z(h, deterministic_continuous=False)
    controlled = generate_chunk(model, obs_hist, act_hist, deterministic=True, z=z, z_emb=z_emb)

    q, p = adapter.init_qp_from_history(batch_device)
    ref_q = [q]
    ref_p = [p]
    aux_values = []
    for k in range(model.chunk_horizon):
        q, p, aux = ref.reference_step(q, p, h, k, obs_state=obs_state)
        ref_q.append(q)
        ref_p.append(p)
        aux_values.append(aux)
    ref_q_seq = torch.stack(ref_q, dim=1)
    ref_raw_actions = adapter.decode_raw_actions(ref_q_seq)

    controlled_aux = []
    q_seq = controlled["q_seq"]
    p_seq = controlled["p_seq"]
    for k in range(model.chunk_horizon):
        _, aux = ref.force(q_seq[:, k], p_seq[:, k], h, k, obs_state=obs_state)
        controlled_aux.append(aux)

    def stack_aux(key: str):
        values = [item.get(key) for item in controlled_aux]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return torch.stack(values, dim=1).detach().cpu()

    controls = controlled["controls"].detach().cpu()
    control_energy_steps = 0.5 * ref.dt * controls.pow(2).sum(dim=-1)
    normalization_stats = None
    if config is not None:
        data_cfg = config.get("data", {})
        stats = data_cfg.get("normalization_stats")
        if bool(data_cfg.get("normalize", False)) and _has_normalization_stats(stats):
            normalization_stats = stats

    batch_cpu = move_to_device(batch_device, torch.device("cpu"))
    batch_cpu["reference_coordinate_mode"] = adapter.coordinate_mode
    batch_cpu["_normalization_stats"] = normalization_stats
    diagnostics = {
        "batch": batch_cpu,
        "controlled_actions": controlled["actions"].detach().cpu(),
        "controlled_q_seq": controlled["q_seq"].detach().cpu(),
        "controlled_p_seq": controlled["p_seq"].detach().cpu(),
        "reference_actions": ref_raw_actions.detach().cpu(),
        "reference_q_seq": ref_q_seq.detach().cpu(),
        "reference_p_seq": torch.stack(ref_p, dim=1).detach().cpu(),
        "controls": controls,
        "control_energy_steps": control_energy_steps,
        "gamma": stack_aux("gamma"),
        "k_diag": stack_aux("k_diag"),
        "m": stack_aux("m"),
        "grad_v": stack_aux("grad_v"),
        "dt": float(ref.dt),
        "coordinate_mode": adapter.coordinate_mode,
        "normalization_stats": normalization_stats,
    }
    diagnostics["controlled_positions"] = _q_seq_as_display_positions(
        batch_cpu,
        diagnostics["controlled_q_seq"],
        diagnostics["controlled_actions"],
    )
    diagnostics["reference_positions"] = _q_seq_as_display_positions(
        batch_cpu,
        diagnostics["reference_q_seq"],
        diagnostics["reference_actions"],
    )
    return diagnostics


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
    bounds = _plot_bounds(diagnostics, idx)
    _draw_pusht_scene(ax, batch, idx, show_labels=True)
    if "future_positions" in batch:
        expert = batch["future_positions"][idx]
        ax.plot(expert[:, 0], expert[:, 1], color="0.55", linewidth=2.0, label="expert")
    else:
        act_hist = batch.get("act_hist")
        future_actions = batch.get("future_actions")
        if torch.is_tensor(act_hist):
            h = _actions_for_display(act_hist, diagnostics)[idx]
            ax.plot(h[:, 0], h[:, 1], color="0.55", marker="o", markersize=3, linewidth=1.0, label="history")
        if torch.is_tensor(future_actions):
            f = _actions_for_display(future_actions, diagnostics)[idx]
            ax.plot(f[:, 0], f[:, 1], color="black", marker="o", markersize=3, linewidth=1.4, label="logged chunk")
    controlled = diagnostics["controlled_positions"][idx]
    reference = diagnostics["reference_positions"][idx]
    ax.plot(reference[:, 0], reference[:, 1], color="tab:green", linestyle="--", linewidth=1.8, label="reference only")
    ax.plot(controlled[:, 0], controlled[:, 1], color="tab:blue", linewidth=2.0, label="controlled")
    m = diagnostics.get("m")
    if m is not None:
        attractor = _actions_for_display(m, diagnostics)[idx]
        ax.plot(attractor[:, 0], attractor[:, 1], color="tab:red", marker="x", markersize=4, linewidth=1.2, label="attractor m")
    context = batch.get("context", {})
    center = context.get("obstacle_center")
    radius = context.get("obstacle_radius")
    _draw_obstacle(ax, center[idx] if torch.is_tensor(center) else None, radius[idx] if torch.is_tensor(radius) else None)
    _apply_plot_bounds(ax, bounds)
    ax.set_title("Paths in q/action coordinates")
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
    bounds = _plot_bounds(diagnostics, idx)
    _draw_pusht_scene(ax, batch, idx, show_labels=show_labels)
    expert = batch.get("future_positions")
    if expert is not None:
        e = expert[idx]
        ax.plot(e[:, 0], e[:, 1], color="white", linewidth=1.5, alpha=0.9, label="expert" if show_labels else None)
    else:
        act_hist = batch.get("act_hist")
        future_actions = batch.get("future_actions")
        if torch.is_tensor(act_hist):
            h = _actions_for_display(act_hist, diagnostics)[idx]
            ax.plot(h[:, 0], h[:, 1], color="0.85", marker="o", markersize=2.5, linewidth=1.0, alpha=0.85, label="history" if show_labels else None)
        if torch.is_tensor(future_actions):
            f = _actions_for_display(future_actions, diagnostics)[idx]
            ax.plot(f[:, 0], f[:, 1], color="white", marker="o", markersize=2.5, linewidth=1.3, alpha=0.9, label="logged chunk" if show_labels else None)
    reference = diagnostics.get("reference_positions")
    if reference is not None:
        r = reference[idx]
        ax.plot(r[:, 0], r[:, 1], color="tab:green", linestyle="--", linewidth=1.4, alpha=0.9, label="reference" if show_labels else None)
    controlled = diagnostics["controlled_positions"][idx]
    ax.plot(controlled[:, 0], controlled[:, 1], color="tab:cyan", linewidth=1.8, label="controlled" if show_labels else None)
    m = diagnostics.get("m")
    if m is not None:
        attractor = _actions_for_display(m, diagnostics)[idx]
        ax.plot(attractor[:, 0], attractor[:, 1], color="tab:red", marker="x", markersize=3, linewidth=1.0, alpha=0.7, label="m path" if show_labels else None)
    goal = context.get("goal")
    if torch.is_tensor(goal):
        g = goal[idx] if goal.ndim > 1 else goal
        ax.scatter([float(g[0])], [float(g[1])], color="red", marker="*", s=70, edgecolors="white", linewidths=0.5, label="goal" if show_labels else None, zorder=6)
    center = context.get("obstacle_center")
    radius = context.get("obstacle_radius")
    _draw_obstacle(ax, center[idx] if torch.is_tensor(center) else None, radius[idx] if torch.is_tensor(radius) else None)
    _apply_plot_bounds(ax, bounds)


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
    bounds = _plot_bounds(diagnostics, idx)
    xx, yy, grid = _potential_grid(bounds, grid_size)
    grid_model = _raw_action_grid_to_model(grid, diagnostics)
    m_display = _actions_for_display(m, diagnostics)
    potentials = []
    forces = []
    force_norms = []
    for step in steps:
        center_m = m[idx, step]
        stiffness = k_diag[idx, step]
        diff = grid_model - center_m
        potential = 0.5 * (stiffness * diff.pow(2)).sum(dim=-1)
        force = _force_to_display(-stiffness * diff, diagnostics)
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
    arrow_scale = 0.08 * max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    for col, step in enumerate(steps):
        center_m = m_display[idx, step]
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
            quiver = force / force_norm_max * arrow_scale
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


def plot_closed_loop_contact_potential(
    diagnostics: Dict[str, Any],
    path: Path,
    max_panels: int = 6,
    grid_size: int = 90,
) -> None:
    """Plot learned potential snapshots along one closed-loop rollout."""

    records = _select_records(list(diagnostics["records"]), int(max_panels))
    if not records:
        return
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = torch.linspace(0.0, 1.0, grid_size)
    ys = torch.linspace(0.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    potentials = []
    for record in records:
        m = record["m"]
        stiffness = record["k_diag"]
        diff = grid - m
        potential = 0.5 * (stiffness * diff.pow(2)).sum(dim=-1)
        potentials.append(potential)

    potential_max = float(torch.stack(potentials).max().item())
    levels = torch.linspace(0.0, max(potential_max, 1e-8), 25).numpy()
    fig, axes = plt.subplots(1, len(records), figsize=(5.0 * len(records), 4.4), squeeze=False, layout="constrained")
    rollout = diagnostics["rollout_positions"]
    expert = diagnostics["expert_positions"]
    goal = diagnostics["goal"]
    center = diagnostics["obstacle_center"]
    radius = diagnostics["obstacle_radius"]
    bounds = (0.0, 1.0, 0.0, 1.0, False)
    for col, (ax, record, potential) in enumerate(zip(axes.ravel(), records, potentials)):
        image = ax.contourf(xx.numpy(), yy.numpy(), potential.numpy(), levels=levels, cmap="viridis")
        replan_time = int(record["trajectory_time"])
        m_path = record["m_path"]
        q_seq = record["predicted_q_seq"]
        ref_no_control = record.get("reference_no_control_positions")
        m = record["m"]
        stiffness = record["k_diag"]
        gamma = record["gamma"].float().mean()
        ax.plot(expert[:, 0], expert[:, 1], color="white", linewidth=1.4, alpha=0.75, label="expert" if col == 0 else None)
        ax.plot(rollout[:, 0], rollout[:, 1], color="tab:cyan", linewidth=1.8, alpha=0.8, label="closed loop" if col == 0 else None)
        ax.plot(rollout[: replan_time + 1, 0], rollout[: replan_time + 1, 1], color="tab:blue", linewidth=2.4, label="executed so far" if col == 0 else None)
        ax.plot(q_seq[:, 0], q_seq[:, 1], color="tab:orange", linewidth=1.2, linestyle="--", label="planned chunk" if col == 0 else None)
        if ref_no_control is not None:
            ax.plot(
                ref_no_control[:, 0],
                ref_no_control[:, 1],
                color="tab:green",
                linewidth=1.7,
                linestyle="-.",
                alpha=0.9,
                label=f"reference no control ({diagnostics['reference_steps']} steps)" if col == 0 else None,
            )
            ax.scatter(
                [float(ref_no_control[-1, 0])],
                [float(ref_no_control[-1, 1])],
                color="tab:green",
                marker="s",
                s=30,
                zorder=8,
                label="reference end" if col == 0 else None,
            )
        ax.plot(m_path[:, 0], m_path[:, 1], color="tab:red", marker="x", markersize=3, linewidth=1.0, alpha=0.7, label="m path" if col == 0 else None)
        ax.scatter([float(m[0])], [float(m[1])], color="red", marker="x", s=64, linewidths=2.1, zorder=8, label="m at local step" if col == 0 else None)
        ax.scatter([float(record["current_position"][0])], [float(record["current_position"][1])], color="black", marker="o", s=32, zorder=8, label="current" if col == 0 else None)
        ax.scatter([float(goal[0])], [float(goal[1])], color="red", marker="*", s=72, edgecolors="white", linewidths=0.5, zorder=9, label="goal" if col == 0 else None)
        _draw_obstacle(ax, center, radius)
        _apply_plot_bounds(ax, bounds)
        ax.set_title(
            f"replan={record['replan_index']} | t={replan_time} | local step={record['local_step']}\n"
            f"stiff=[{float(stiffness[0]):.2g}, {float(stiffness[1]):.2g}] | gamma={float(gamma):.3g}",
            fontsize=9,
        )
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.colorbar(image, ax=axes[0, :], fraction=0.02, pad=0.015, label="V(q,h,k)")
    fig.savefig(path, dpi=170)
    plt.close(fig)
