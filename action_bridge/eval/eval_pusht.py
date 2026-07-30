"""Offline Push-T checkpoint evaluation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.data.pusht_adapter import denormalize_actions_tensor, denormalize_observations_tensor
from action_bridge.eval.metrics import action_smoothness, average_metric_dicts
from action_bridge.eval.pusht_wrong_side import plot_wrong_side_go_around_diagnostic
from action_bridge.eval.rollout import predict_actions
from action_bridge.eval.visualization import plot_energy_histograms, plot_projected_demo_passive_target
from action_bridge.training.common import build_dataset, build_model, move_to_device, resolve_device, save_json


def _sync_config_dims(config, dataset) -> None:
    obs_dim = getattr(dataset, "obs_dim", None)
    action_dim = getattr(dataset, "action_dim", None)
    if obs_dim is not None:
        config["obs_dim"] = int(obs_dim)
    if action_dim is not None:
        config["action_dim"] = int(action_dim)
    normalization_stats = getattr(dataset, "normalization_stats", None)
    if normalization_stats is not None:
        config["data"]["normalization_stats"] = normalization_stats


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _normalization_stats(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    stats = data_cfg.get("normalization_stats")
    if bool(data_cfg.get("normalize", False)) and stats is not None:
        return stats
    return None


def _maybe_denormalize_actions(actions: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    stats = _normalization_stats(config)
    if stats is None:
        return actions
    return denormalize_actions_tensor(actions, stats)


def _maybe_denormalize_observations(obs: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    stats = _normalization_stats(config)
    if stats is None:
        return obs
    return denormalize_observations_tensor(obs, stats)


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def compute_pusht_batch_metrics(
    pred_actions: torch.Tensor,
    batch: Dict[str, Any],
    config: Dict[str, Any],
    path_kl_energy: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    target_actions = _maybe_denormalize_actions(batch["future_actions"], config)
    pred_actions = _maybe_denormalize_actions(pred_actions, config)
    act_hist = _maybe_denormalize_actions(batch["act_hist"], config)
    smooth_pred = action_smoothness(pred_actions, act_hist)
    smooth_target = action_smoothness(target_actions, act_hist)
    boundary = torch.linalg.norm(pred_actions[:, 0] - act_hist[:, -1], dim=-1).mean()
    return {
        "action_mse": _as_float(F.mse_loss(pred_actions, target_actions)),
        "action_l1": _as_float(F.l1_loss(pred_actions, target_actions)),
        "first_action_mse": _as_float(F.mse_loss(pred_actions[:, 0], target_actions[:, 0])),
        "final_action_mse": _as_float(F.mse_loss(pred_actions[:, -1], target_actions[:, -1])),
        "acceleration_energy": _as_float(smooth_pred["acceleration_energy"]),
        "jerk_energy": _as_float(smooth_pred["jerk_energy"]),
        "target_acceleration_energy": _as_float(smooth_target["acceleration_energy"]),
        "target_jerk_energy": _as_float(smooth_target["jerk_energy"]),
        "chunk_boundary_discontinuity": _as_float(boundary),
        "path_KL_energy": _as_float(path_kl_energy.mean()) if path_kl_energy is not None else 0.0,
        "pred_action_norm": _as_float(torch.linalg.norm(pred_actions, dim=-1).mean()),
        "target_action_norm": _as_float(torch.linalg.norm(target_actions, dim=-1).mean()),
    }


def plot_action_error_histograms(per_sample_mse: torch.Tensor, per_sample_l1: torch.Tensor, path: Path) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].hist(per_sample_mse.detach().cpu().numpy(), bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_title("chunk MSE")
    axes[1].hist(per_sample_l1.detach().cpu().numpy(), bins=30, color="tab:orange", alpha=0.8)
    axes[1].set_title("chunk L1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_action_rollout_examples(
    batch: Dict[str, Any],
    pred_actions: torch.Tensor,
    path: Path,
    config: Dict[str, Any],
    max_items: int = 6,
) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, pred_actions.shape[0])
    fig, axes = plt.subplots(count, 2, figsize=(8.0, 2.2 * count), squeeze=False, sharex=True)
    t = torch.arange(pred_actions.shape[1]).detach().cpu()
    target = _maybe_denormalize_actions(batch["future_actions"], config).detach().cpu()
    pred = _maybe_denormalize_actions(pred_actions, config).detach().cpu()
    for idx in range(count):
        for dim, name in enumerate(["action x", "action y"]):
            ax = axes[idx, dim]
            ax.plot(t, target[idx, :, dim], color="0.35", linewidth=1.4, label="logged" if idx == 0 else None)
            ax.plot(t, pred[idx, :, dim], color="tab:blue", linewidth=1.2, label="model" if idx == 0 else None)
            ax.set_ylabel(name)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("chunk step")
    axes[-1, 1].set_xlabel("chunk step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _tee_polygons(pose: np.ndarray, scale: float = 30.0):
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
        patch = Polygon(
            poly,
            closed=True,
            facecolor=color if linestyle == "-" else "none",
            edgecolor=color,
            linewidth=1.4,
            linestyle=linestyle,
            alpha=alpha,
            label=label if idx == 0 else None,
        )
        ax.add_patch(patch)


def plot_action_chunk_2d(
    batch: Dict[str, Any],
    pred_actions: torch.Tensor,
    path: Path,
    config: Dict[str, Any],
    max_items: int = 6,
) -> None:
    plt = _import_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, pred_actions.shape[0])
    cols = min(3, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    target = _maybe_denormalize_actions(batch["future_actions"], config).detach().cpu().numpy()
    pred = _maybe_denormalize_actions(pred_actions, config).detach().cpu().numpy()
    obs_hist = _maybe_denormalize_observations(batch["obs_hist"], config).detach().cpu().numpy()
    act_hist = _maybe_denormalize_actions(batch["act_hist"], config).detach().cpu().numpy()
    goal_pose = np.array([256.0, 256.0, math.pi / 4], dtype=np.float32)
    for flat_idx, ax in enumerate(axes.ravel()):
        if flat_idx >= count:
            ax.axis("off")
            continue
        state = obs_hist[flat_idx, -1]
        agent = state[:2]
        block_pose = state[2:5] if state.shape[-1] >= 5 else None
        if block_pose is not None:
            _draw_tee(ax, goal_pose, color="tab:green", alpha=0.28, label="goal T", linestyle="--")
            _draw_tee(ax, block_pose, color="0.35", alpha=0.32, label="current T")
        ax.plot(act_hist[flat_idx, :, 0], act_hist[flat_idx, :, 1], color="0.55", marker="o", markersize=3, linewidth=1.0, label="history")
        ax.scatter(agent[0], agent[1], color="tab:purple", s=28, marker="o", label="agent")
        ax.plot(target[flat_idx, :, 0], target[flat_idx, :, 1], color="black", marker="o", markersize=3, linewidth=1.6, label="logged chunk")
        ax.plot(pred[flat_idx, :, 0], pred[flat_idx, :, 1], color="tab:blue", marker="x", markersize=4, linewidth=1.4, label="pred chunk")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"chunk {flat_idx}")
        if flat_idx == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def representative_plot_indices(dataset, max_items: int) -> list[int]:
    count = min(max_items, len(dataset))
    if count <= 0:
        return []

    pairs = getattr(dataset, "indices", None)
    if pairs:
        by_episode: Dict[int, list[int]] = {}
        for dataset_idx, pair in enumerate(pairs):
            episode_id = int(pair[0])
            by_episode.setdefault(episode_id, []).append(dataset_idx)
        episodes = sorted(by_episode)
        if episodes:
            if len(episodes) >= count:
                episode_positions = np.linspace(0, len(episodes) - 1, num=count, dtype=int)
                chosen = []
                for pos in episode_positions:
                    episode_indices = by_episode[episodes[int(pos)]]
                    chosen.append(episode_indices[len(episode_indices) // 2])
                return chosen

            chosen = []
            for episode_id in episodes:
                episode_indices = by_episode[episode_id]
                chosen.append(episode_indices[len(episode_indices) // 2])
            remaining = count - len(chosen)
            if remaining > 0:
                used = set(chosen)
                global_positions = np.linspace(0, len(pairs) - 1, num=remaining + 2, dtype=int)[1:-1]
                for pos in global_positions:
                    idx = int(pos)
                    if idx not in used:
                        chosen.append(idx)
                        used.add(idx)
                    if len(chosen) >= count:
                        break
            return chosen[:count]

    return np.linspace(0, len(dataset) - 1, num=count, dtype=int).tolist()


@torch.no_grad()
def representative_plot_batch(model, dataset, config: Dict, device: torch.device) -> tuple[Optional[Dict[str, Any]], Optional[torch.Tensor]]:
    max_items = int(config.get("eval", {}).get("plot_examples", 6))
    indices = representative_plot_indices(dataset, max_items)
    if not indices:
        return None, None
    loader = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)
    batch = next(iter(loader))
    batch_device = move_to_device(batch, device)
    pred = predict_actions(model, batch_device, deterministic=bool(config.get("inference", {}).get("deterministic", True)))
    return move_to_device(batch_device, torch.device("cpu")), pred["actions"].detach().cpu()


@torch.no_grad()
def offline_receding_horizon_metrics(model, dataset, config: Dict, device: torch.device) -> Dict[str, float]:
    if not hasattr(dataset, "episode_ids") or not hasattr(dataset, "item_from_episode_time"):
        return {}
    eval_cfg = config.get("eval", {})
    max_episodes = int(eval_cfg.get("offline_rollout_episodes", eval_cfg.get("closed_loop_episodes", 64)))
    n_exec = int(eval_cfg.get("n_exec", config.get("inference", {}).get("n_exec", config.get("chunk_horizon", 16))))
    n_exec = max(1, n_exec)
    start_t = max(int(config.get("obs_history", 2)) - 1, int(config.get("action_history", 2)))
    action_errors = []
    first_errors = []
    boundaries = []
    path_kl = []
    chunks = 0
    for episode_id in list(dataset.episode_ids)[:max_episodes]:
        length = min(int(dataset.observations[episode_id].shape[0]), int(dataset.actions[episode_id].shape[0]))
        t = start_t
        while t + int(config.get("chunk_horizon", 16)) <= length:
            item = dataset.item_from_episode_time(episode_id, t)
            batch = {
                "obs_hist": item["obs_hist"][None].to(device),
                "act_hist": item["act_hist"][None].to(device),
                "future_actions": item["future_actions"][None].to(device),
                "context": {key: value[None].to(device) for key, value in item["context"].items() if torch.is_tensor(value)},
            }
            pred = predict_actions(model, batch, deterministic=bool(config.get("inference", {}).get("deterministic", True)))
            execute = min(n_exec, length - t, pred["actions"].shape[1])
            generated = pred["actions"][:, :execute]
            target = batch["future_actions"][:, :execute]
            generated_raw = _maybe_denormalize_actions(generated, config)
            target_raw = _maybe_denormalize_actions(target, config)
            act_hist_raw = _maybe_denormalize_actions(batch["act_hist"], config)
            action_errors.append((generated_raw - target_raw).pow(2).mean())
            first_errors.append((generated_raw[:, 0] - target_raw[:, 0]).pow(2).mean())
            boundaries.append(torch.linalg.norm(generated_raw[:, 0] - act_hist_raw[:, -1], dim=-1).mean())
            if pred.get("path_kl_steps") is not None:
                path_kl.append(pred["path_kl_steps"][:, :execute].sum(dim=1).mean())
            chunks += 1
            t += execute
    if not action_errors:
        return {}
    return {
        "offline_rh_action_mse": _as_float(torch.stack(action_errors).mean()),
        "offline_rh_first_action_mse": _as_float(torch.stack(first_errors).mean()),
        "offline_rh_chunk_boundary_discontinuity": _as_float(torch.stack(boundaries).mean()),
        "offline_rh_path_KL_energy": _as_float(torch.stack(path_kl).mean()) if path_kl else 0.0,
        "offline_rh_num_chunks": float(chunks),
        "offline_rh_episodes": float(min(max_episodes, len(dataset.episode_ids))),
        "offline_rh_n_exec": float(n_exec),
    }


@torch.no_grad()
def evaluate_pusht_model(
    model,
    dataset,
    config: Dict,
    device: torch.device,
    output_dir: Optional[Path] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    batch_size = int(config.get("eval", {}).get("batch_size", config.get("optim", {}).get("batch_size", 256)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    metrics = []
    all_path_kl = []
    per_sample_mse = []
    per_sample_l1 = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        pred = predict_actions(model, batch, deterministic=bool(config.get("inference", {}).get("deterministic", True)))
        target = batch["future_actions"]
        metrics.append(compute_pusht_batch_metrics(pred["actions"], batch, config, pred.get("path_kl_energy")))
        if pred.get("path_kl_energy") is not None:
            all_path_kl.append(pred["path_kl_energy"].detach().cpu())
        pred_raw = _maybe_denormalize_actions(pred["actions"], config)
        target_raw = _maybe_denormalize_actions(target, config)
        per_sample_mse.append((pred_raw - target_raw).pow(2).mean(dim=(1, 2)).detach().cpu())
        per_sample_l1.append((pred_raw - target_raw).abs().mean(dim=(1, 2)).detach().cpu())
    summary = average_metric_dicts(metrics)
    summary.update(offline_receding_horizon_metrics(model, dataset, config, device))
    if bool(config.get("eval", {}).get("sim_closed_loop", False)):
        from action_bridge.eval.pusht_sim import evaluate_pusht_sim_model

        summary.update(evaluate_pusht_sim_model(model, config, device, output_dir=output_dir))

    if output_dir is not None:
        figures = output_dir / "figures"
        plot_batch, plot_pred = representative_plot_batch(model, dataset, config, device)
        if all_path_kl:
            try:
                plot_energy_histograms({"path_KL": torch.cat(all_path_kl)}, figures / "energy_histograms.png")
            except Exception as exc:
                summary["plot_energy_error"] = str(exc)
        if per_sample_mse and per_sample_l1:
            try:
                plot_action_error_histograms(
                    torch.cat(per_sample_mse),
                    torch.cat(per_sample_l1),
                    figures / "action_error_histograms.png",
                )
            except Exception as exc:
                summary["plot_action_error_error"] = str(exc)
        if plot_batch is not None and plot_pred is not None:
            try:
                plot_action_rollout_examples(plot_batch, plot_pred, figures / "receding_horizon_action_rollout.png", config)
            except Exception as exc:
                summary["plot_action_rollout_error"] = str(exc)
            try:
                plot_action_chunk_2d(plot_batch, plot_pred, figures / "action_chunk_2d_with_t.png", config)
            except Exception as exc:
                summary["plot_action_chunk_2d_error"] = str(exc)
            try:
                plot_projected_demo_passive_target(
                    plot_batch,
                    figures / "projected_demo_passive_target.png",
                    config,
                )
            except Exception as exc:
                summary["plot_projected_demo_error"] = str(exc)
        try:
            summary.update(plot_wrong_side_go_around_diagnostic(model, config, device, figures))
        except Exception as exc:
            summary["plot_wrong_side_go_around_error"] = str(exc)
        save_json(output_dir / "metrics" / "pusht_metrics.json", summary)
    model.train()
    return summary


def load_checkpoint(checkpoint: Path, device: torch.device):
    data = torch.load(checkpoint, map_location=device)
    config = data["config"]
    model = build_model(config).to(device)
    model.load_state_dict(data["model_state"])
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        raw = torch.load(ckpt_path, map_location="cpu")
        config = apply_overrides(raw["config"], args.overrides)
        device = resolve_device(str(config.get("device", "cpu")))
        dataset = build_dataset(config, split=args.split)
        model = build_model(config).to(device)
        model.load_state_dict(raw["model_state"])
    else:
        if args.config_name is None:
            raise SystemExit("Pass --checkpoint or --config-name.")
        config = apply_overrides(load_config(args.config_name), args.overrides)
        device = resolve_device(str(config.get("device", "cpu")))
        dataset = build_dataset(config, split=args.split)
        _sync_config_dims(config, dataset)
        model = build_model(config).to(device)
    out_dir = Path(args.output_dir) if args.output_dir else None
    metrics = evaluate_pusht_model(model, dataset, config, device, output_dir=out_dir)
    if out_dir is not None:
        save_config(config, out_dir / "eval_config.json")
        save_json(out_dir / "eval_metadata.json", {"split": args.split, "device": str(device), "run_id": config.get("run_id")})
        print(f"Eval directory: {out_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
