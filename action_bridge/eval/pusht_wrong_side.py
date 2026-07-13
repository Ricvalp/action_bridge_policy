"""Synthetic Push-T wrong-side latent diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from action_bridge.data.pusht_adapter import denormalize_actions_tensor, normalize_actions_tensor, normalize_observations_tensor
from action_bridge.eval.rollout import generate_chunk
from action_bridge.models.action_bridge_policy import ActionBridgePolicy


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _tee_polygons(pose: np.ndarray, scale: float = 30.0) -> list[np.ndarray]:
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    length = 4.0
    local_polys = [
        np.array(
            [
                [-length * scale / 2, scale],
                [length * scale / 2, scale],
                [length * scale / 2, 0.0],
                [-length * scale / 2, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [-scale / 2, scale],
                [-scale / 2, length * scale],
                [scale / 2, length * scale],
                [scale / 2, scale],
            ],
            dtype=np.float32,
        ),
    ]
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    return [poly @ rotation.T + np.array([x, y], dtype=np.float32) for poly in local_polys]


def _draw_tee(ax, pose: np.ndarray, color: str, alpha: float, label: Optional[str] = None, linestyle: str = "-") -> None:
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


def _point_inside_tee(pose: np.ndarray, point: np.ndarray) -> bool:
    from matplotlib.path import Path as MplPath

    return any(MplPath(poly).contains_point(point) for poly in _tee_polygons(pose))


def _normalization_stats(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    stats = data_cfg.get("normalization_stats")
    if bool(data_cfg.get("normalize", False)) and stats is not None:
        return stats
    return None


def _synthetic_wrong_side_states(config: Dict[str, Any]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    eval_cfg = config.get("eval", {})
    theta = math.pi / 4
    d_cross = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    d_long = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    goal_origin = np.array([256.0, 256.0], dtype=np.float32)
    offsets = [float(value) for value in eval_cfg.get("wrong_side_offsets", [140.0, 100.0, 60.0, 20.0])]
    t_long_extent = float(eval_cfg.get("wrong_side_t_long_extent", 120.0))
    clearance = float(eval_cfg.get("wrong_side_clearance", 38.0))

    def local_to_world(origin: np.ndarray, x_local: float, y_local: float) -> np.ndarray:
        return origin + float(x_local) * d_cross + float(y_local) * d_long

    states = []
    for target_sign, side_name in [(1.0, "target_plus_dlong"), (-1.0, "target_minus_dlong")]:
        for offset in offsets:
            block_origin = goal_origin - target_sign * offset * d_long
            if target_sign > 0:
                agent_xy = local_to_world(block_origin, 0.0, t_long_extent + clearance)
                desired_push_local_y = -clearance
                pusher_local_y = t_long_extent + clearance
            else:
                agent_xy = local_to_world(block_origin, 0.0, -clearance)
                desired_push_local_y = t_long_extent + clearance
                pusher_local_y = -clearance
            state = np.array([agent_xy[0], agent_xy[1], block_origin[0], block_origin[1], theta], dtype=np.float32)
            states.append(
                {
                    "side": side_name,
                    "target_sign": float(target_sign),
                    "offset_from_goal_px": float(offset),
                    "state": state.tolist(),
                    "long_axis_direction": d_long.tolist(),
                    "cross_axis_direction": d_cross.tolist(),
                    "pusher_local_y": float(pusher_local_y),
                    "desired_push_local_y": float(desired_push_local_y),
                    "pusher_inside_current_t": _point_inside_tee(state[2:5], state[:2]),
                }
            )
    construction = {
        "theta": theta,
        "goal_origin": goal_origin.tolist(),
        "long_axis_direction": d_long.tolist(),
        "cross_axis_direction": d_cross.tolist(),
        "t_long_extent_px_for_plot_geometry": t_long_extent,
        "clearance_px": clearance,
        "description": (
            "Current and target T are long-axis aligned. The pusher is collision-free but starts on the wrong side, "
            "so it must route around the T before it can push from the useful side."
        ),
    }
    return states, construction


def _signed_cross(points: np.ndarray, origin: np.ndarray, d_cross: np.ndarray) -> np.ndarray:
    return (points[..., 0] - origin[0]) * d_cross[0] + (points[..., 1] - origin[1]) * d_cross[1]


@torch.no_grad()
def plot_wrong_side_go_around_diagnostic(
    model,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    """Plot sampled chunks for a collision-free wrong-side Push-T probe.

    The plot is intended to detect whether latent samples produce distinct
    left/right go-around modes when the pusher starts on the wrong side of an
    aligned T.
    """

    if not isinstance(model, ActionBridgePolicy):
        return {}
    if getattr(model, "latent", None) is None:
        return {}
    if not hasattr(model.latent, "prior_params"):
        return {}

    eval_cfg = config.get("eval", {})
    if not bool(eval_cfg.get("wrong_side_go_around", True)):
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = _normalization_stats(config)
    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    num_samples = int(eval_cfg.get("wrong_side_num_samples", 32))
    num_samples = max(1, num_samples)
    executed_k = min(max(0, int(eval_cfg.get("wrong_side_executed_k", 3))), int(config.get("chunk_horizon", 16)) - 1)
    states, construction = _synthetic_wrong_side_states(config)

    if any(item["pusher_inside_current_t"] for item in states):
        raise RuntimeError("Wrong-side diagnostic construction placed a pusher inside the current T.")

    was_training = model.training
    model.eval()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed = int(eval_cfg.get("wrong_side_seed", config.get("seed", 0)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        records = []
        chunks_by_state = []
        for source in states:
            raw_state = torch.tensor(source["state"], dtype=torch.float32)
            obs_raw = raw_state[None, None, :].expand(1, obs_history, -1).clone()
            act_raw = raw_state[:2][None, None, :].expand(1, action_history, -1).clone()
            obs_hist = normalize_observations_tensor(obs_raw, stats).to(device) if stats is not None else obs_raw.to(device)
            act_hist = normalize_actions_tensor(act_raw, stats).to(device) if stats is not None else act_raw.to(device)

            h_emb = model.encode_history(obs_hist, act_hist)
            mu, logvar = model.latent.prior_params(h_emb)
            std = torch.exp(0.5 * logvar)
            entropy = 0.5 * (math.log(2 * math.pi * math.e) + logvar).sum(dim=-1)

            obs_rep = obs_hist.expand(num_samples, -1, -1).contiguous()
            act_rep = act_hist.expand(num_samples, -1, -1).contiguous()
            h_rep = model.encode_history(obs_rep, act_rep)
            mu_rep, logvar_rep = model.latent.prior_params(h_rep)
            z = mu_rep + torch.exp(0.5 * logvar_rep) * torch.randn_like(mu_rep)
            sampled = generate_chunk(model, obs_rep, act_rep, deterministic=True, z=z, z_emb=model.latent.embed(z))["actions"]
            sampled_px = denormalize_actions_tensor(sampled.detach().cpu(), stats) if stats is not None else sampled.detach().cpu()

            z_mean = model.latent.embed(mu)
            mean_chunk = generate_chunk(model, obs_hist, act_hist, deterministic=True, z=mu, z_emb=z_mean)["actions"]
            mean_chunk_px = denormalize_actions_tensor(mean_chunk.detach().cpu(), stats)[0] if stats is not None else mean_chunk.detach().cpu()[0]

            sample_mean = sampled_px.mean(dim=0)
            path_dists = torch.linalg.norm(sampled_px - sample_mean[None], dim=-1)
            endpoint_dists = torch.linalg.norm(sampled_px[:, -1] - sample_mean[-1][None], dim=-1)
            executed_dists = torch.linalg.norm(sampled_px[:, executed_k] - sample_mean[executed_k][None], dim=-1)
            dist_to_mu = torch.linalg.norm(sampled_px - mean_chunk_px[None], dim=-1)

            block = np.asarray(source["state"][2:4], dtype=np.float32)
            d_cross = np.asarray(source["cross_axis_direction"], dtype=np.float32)
            arr_np = sampled_px.numpy()
            lateral_exec = _signed_cross(arr_np[:, executed_k, :], block, d_cross)
            lateral_endpoint = _signed_cross(arr_np[:, -1, :], block, d_cross)
            lateral_max_abs = np.max(np.abs(_signed_cross(arr_np, block, d_cross)), axis=1)

            record = dict(source)
            record.update(
                {
                    "sample_mean_path_spread_px": float(path_dists.mean().item()),
                    "sample_executed_k_spread_px": float(executed_dists.mean().item()),
                    "sample_endpoint_spread_px": float(endpoint_dists.mean().item()),
                    "sample_mean_dist_to_mu_chunk_px": float(dist_to_mu.mean().item()),
                    "sample_endpoint_dist_to_mu_chunk_px": float(
                        torch.linalg.norm(sampled_px[:, -1] - mean_chunk_px[-1][None], dim=-1).mean().item()
                    ),
                    "prior_mu": [float(x) for x in mu[0].detach().cpu()],
                    "prior_std": [float(x) for x in std[0].detach().cpu()],
                    "prior_entropy": float(entropy[0].detach().cpu().item()),
                    "lateral_exec_mean_px": float(np.mean(lateral_exec)),
                    "lateral_exec_std_px": float(np.std(lateral_exec)),
                    "lateral_exec_positive_fraction": float(np.mean(lateral_exec > 0.0)),
                    "lateral_endpoint_mean_px": float(np.mean(lateral_endpoint)),
                    "lateral_endpoint_std_px": float(np.std(lateral_endpoint)),
                    "lateral_endpoint_positive_fraction": float(np.mean(lateral_endpoint > 0.0)),
                    "lateral_max_abs_mean_px": float(np.mean(lateral_max_abs)),
                }
            )
            records.append(record)
            chunks_by_state.append((record, sampled_px, mean_chunk_px))

        summary = {
            "num_samples": num_samples,
            "executed_k": executed_k,
            "construction": construction,
            "states": records,
        }
        with (output_dir / "wrong_side_go_around_latent_sensitivity.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        _plot_chunks(chunks_by_state, output_dir / "wrong_side_go_around_latent_chunks.png", construction)
        _plot_lateral_summary(records, output_dir / "wrong_side_go_around_lateral_summary.png")

        endpoint_spread = [item["sample_endpoint_spread_px"] for item in records]
        lateral_std = [item["lateral_endpoint_std_px"] for item in records]
        positive_fraction = [item["lateral_endpoint_positive_fraction"] for item in records]
        return {
            "wrong_side_endpoint_spread_mean": float(np.mean(endpoint_spread)),
            "wrong_side_endpoint_spread_max": float(np.max(endpoint_spread)),
            "wrong_side_lateral_endpoint_std_mean": float(np.mean(lateral_std)),
            "wrong_side_lateral_endpoint_std_max": float(np.max(lateral_std)),
            "wrong_side_lateral_positive_fraction_mean": float(np.mean(positive_fraction)),
            "wrong_side_num_states": float(len(records)),
            "wrong_side_num_samples": float(num_samples),
        }
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            model.train()


def _plot_chunks(chunks_by_state: list[tuple[Dict[str, Any], torch.Tensor, torch.Tensor]], path: Path, construction: Dict[str, Any]) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(chunks_by_state)
    cols = min(4, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False, layout="constrained")
    theta = float(construction["theta"])
    goal_pose = np.array([256.0, 256.0, theta], dtype=np.float32)

    for ax, (record, samples, mean_chunk) in zip(axes.ravel(), chunks_by_state):
        state = np.asarray(record["state"], dtype=np.float32)
        block_pose = state[2:5]
        agent = state[:2]
        d_long = np.asarray(record["long_axis_direction"], dtype=np.float32)
        d_cross = np.asarray(record["cross_axis_direction"], dtype=np.float32)
        block_xy = state[2:4]
        long_line = np.stack([block_xy - 190.0 * d_long, block_xy + 190.0 * d_long])
        cross_line = np.stack([block_xy - 95.0 * d_cross, block_xy + 95.0 * d_cross])

        _draw_tee(ax, goal_pose, color="tab:green", alpha=0.25, label="target T", linestyle="--")
        _draw_tee(ax, block_pose, color="0.35", alpha=0.32, label="current T")
        ax.plot(long_line[:, 0], long_line[:, 1], color="0.7", linestyle=":", linewidth=1.0, label="T long-axis")
        ax.plot(cross_line[:, 0], cross_line[:, 1], color="0.82", linestyle=":", linewidth=1.0, label="left/right axis")
        ax.scatter([float(agent[0])], [float(agent[1])], c="black", s=35, zorder=6, label="pusher")
        arr = samples.numpy()
        for sample_idx in range(min(arr.shape[0], 96)):
            ax.plot(arr[sample_idx, :, 0], arr[sample_idx, :, 1], color="tab:blue", alpha=0.08, linewidth=1)
        ax.plot(mean_chunk[:, 0], mean_chunk[:, 1], color="black", linewidth=2.0, label="z=prior mean")
        ax.set_title(
            f"{record['side']}\n"
            f"offset={record['offset_from_goal_px']:.0f}px, spread={record['sample_endpoint_spread_px']:.2f}px, "
            f"left+={record['lateral_endpoint_positive_fraction']:.2f}",
            fontsize=9,
        )
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes.ravel()[count:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    fig.suptitle("wrong-side go-around latent samples", y=1.03)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_lateral_summary(records: list[Dict[str, Any]], path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{item['side'].replace('_', ' ')}\n{int(item['offset_from_goal_px'])}px" for item in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), layout="constrained")
    axes[0].bar(x, [item["sample_endpoint_spread_px"] for item in records])
    axes[0].set_ylabel("endpoint spread across z (px)")
    axes[1].bar(x, [item["lateral_endpoint_std_px"] for item in records])
    axes[1].set_ylabel("endpoint lateral std (px)")
    axes[2].bar(x, [item["lateral_endpoint_positive_fraction"] for item in records])
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("fraction on +cross side")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("wrong-side go-around multimodality metrics")
    fig.savefig(path, dpi=170)
    plt.close(fig)
