"""Evaluate toy checkpoints and write metrics/figures."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.eval.metrics import (
    action_smoothness,
    annular_mode_metrics,
    average_metric_dicts,
    collision_stats,
    compute_toy_metrics,
    delayed_mode_metrics,
)
from action_bridge.eval.rollout import actions_to_positions, generate_chunk, predict_actions
from action_bridge.eval.visualization import (
    plot_closed_loop_rollouts,
    plot_calibration,
    plot_dataset_samples,
    plot_energy_histograms,
    plot_generated_samples,
    plot_latent_scatter,
)
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.training.common import build_dataset, build_model, move_to_device, resolve_device, save_json, slice_batch


def safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text.strip())
    return cleaned.strip("._") or "eval"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{idx:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free output directory near {path}.")


def default_eval_output_dir(config: Dict, checkpoint: Optional[Path] = None) -> Path:
    if checkpoint is not None:
        experiment = config.get("run_id") or checkpoint.parents[1].name
    else:
        experiment = config.get("run_id") or config.get("config_name") or "toy_eval"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(config.get("output_dir", "outputs")) / "eval"
    return unique_path(root / f"{safe_name(str(experiment))}_{stamp}")


def classify_generated_modes(positions: torch.Tensor, center: torch.Tensor, radius: torch.Tensor, benchmark: str) -> torch.Tensor:
    modes = []
    if benchmark == "toy_annular":
        rel = positions.detach().cpu().numpy() - center[:, None].detach().cpu().numpy()
        for sample in rel:
            import numpy as np

            theta = np.unwrap(np.arctan2(sample[:, 1], sample[:, 0]))
            modes.append(1.0 if theta[-1] - theta[0] >= 0 else -1.0)
        return torch.tensor(modes, device=positions.device, dtype=positions.dtype)
    for i in range(positions.shape[0]):
        x = positions[i, :, 0]
        y = positions[i, :, 1]
        cx = center[i, 0]
        cy = center[i, 1]
        band = radius[i] + 0.08
        mask = (x > cx - band) & (x < cx + band)
        if not bool(mask.any()):
            nearest = torch.argmin((x - cx).abs())
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[nearest] = True
        modes.append(torch.where((y[mask] - cy).mean() >= 0, torch.ones((), device=positions.device), -torch.ones((), device=positions.device)))
    return torch.stack(modes)


def delayed_hybrid_flags(positions: torch.Tensor, center: torch.Tensor, radius: torch.Tensor, collision: torch.Tensor) -> torch.Tensor:
    flags = []
    for i in range(positions.shape[0]):
        x = positions[i, :, 0]
        y = positions[i, :, 1]
        cx = center[i, 0]
        cy = center[i, 1]
        band = radius[i] + 0.08
        mask = (x > cx - band) & (x < cx + band)
        if not bool(mask.any()):
            nearest = torch.argmin((x - cx).abs())
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[nearest] = True
        signed = y[mask] - cy
        switch = bool((signed > 0.025).any() and (signed < -0.025).any())
        flags.append(torch.tensor(float(switch or bool(collision[i])), device=positions.device, dtype=positions.dtype))
    return torch.stack(flags)


def annular_hybrid_flags(positions: torch.Tensor, center: torch.Tensor, collision: torch.Tensor) -> torch.Tensor:
    flags = []
    pos_np = positions.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy()
    collision_np = collision.detach().cpu().numpy()
    for i in range(pos_np.shape[0]):
        import numpy as np

        rel = pos_np[i] - center_np[i][None]
        theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        dtheta = np.diff(theta)
        signs = np.sign(dtheta[np.abs(dtheta) > 1e-3])
        sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        flags.append(float(sign_changes > 1 or collision_np[i]))
    return torch.tensor(flags, device=positions.device, dtype=positions.dtype)


def make_obs_hist(pos_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos_hist, goal[:, None, :].expand(pos_hist.shape[0], pos_hist.shape[1], -1)], dim=-1)


@torch.no_grad()
def closed_loop_rollout(model, dataset, config: Dict, device: torch.device) -> Dict:
    eval_cfg = config.get("eval", {})
    trajectory_ids = []
    seen = set()
    for traj_id, _ in dataset.indices:
        if traj_id not in seen:
            seen.add(traj_id)
            trajectory_ids.append(traj_id)
    max_episodes = int(eval_cfg.get("closed_loop_episodes", 128))
    trajectory_ids = trajectory_ids[:max_episodes]
    if not trajectory_ids:
        return {"metrics": {}, "rollout": None}

    ids = torch.tensor(trajectory_ids, device=device, dtype=torch.long)
    expert_positions = dataset.positions.to(device)[ids]
    expert_actions = dataset.actions.to(device)[ids]
    goals = dataset.goals.to(device)[ids]
    centers = dataset.obstacle_centers.to(device)[ids]
    radii = dataset.obstacle_radii.to(device)[ids]
    modes = dataset.modes.to(device)[ids]

    obs_history = int(config.get("obs_history", 2))
    action_history = int(config.get("action_history", 2))
    warm_start = int(eval_cfg.get("closed_loop_warm_start_steps", max(obs_history - 1, action_history)))
    warm_start = max(max(obs_history - 1, action_history), min(warm_start, expert_actions.shape[1] - 1))
    n_exec = int(eval_cfg.get("n_exec", config.get("inference", {}).get("n_exec", config.get("chunk_horizon", 16))))
    n_exec = max(1, n_exec)
    success_radius = float(eval_cfg.get("success_radius", 0.08))
    action_clip = eval_cfg.get("action_clip", None)
    if action_clip is not None:
        action_clip = float(action_clip)
    box_min = float(eval_cfg.get("box_min", 0.0))
    box_max = float(eval_cfg.get("box_max", 1.0))
    deterministic = bool(config.get("inference", {}).get("deterministic", True))
    commitment = str(config.get("inference", {}).get("latent_commitment", "chunk"))

    pos_hist = expert_positions[:, warm_start - obs_history + 1 : warm_start + 1]
    act_hist = expert_actions[:, warm_start - action_history : warm_start]
    current_pos = expert_positions[:, warm_start]
    rollout_prefix = expert_positions[:, : warm_start + 1]
    generated_positions = []
    generated_actions = []
    path_kl_steps = []
    boundary_discontinuities = []
    z = None
    z_emb = None
    remaining = expert_actions.shape[1] - warm_start

    while remaining > 0:
        obs_hist = make_obs_hist(pos_hist, goals)
        if isinstance(model, ActionBridgePolicy):
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
        else:
            pred = predict_actions(model, {"obs_hist": obs_hist, "act_hist": act_hist}, deterministic=deterministic)

        actions = pred["actions"]
        if action_clip is not None:
            actions = actions.clamp(-action_clip, action_clip)
        execute = min(n_exec, remaining, actions.shape[1])
        executed = actions[:, :execute]
        generated_actions.append(executed)
        if "path_kl_steps" in pred:
            path_kl_steps.append(pred["path_kl_steps"][:, :execute])
        boundary_discontinuities.append(torch.linalg.norm(executed[:, 0] - act_hist[:, -1], dim=-1))

        new_positions = []
        for step in range(execute):
            current_pos = (current_pos + executed[:, step]).clamp(box_min, box_max)
            new_positions.append(current_pos)
        new_pos_tensor = torch.stack(new_positions, dim=1)
        generated_positions.append(new_pos_tensor)
        pos_hist = torch.cat([pos_hist, new_pos_tensor], dim=1)[:, -obs_history:]
        act_hist = torch.cat([act_hist, executed], dim=1)[:, -action_history:]
        remaining -= execute

    generated_actions_tensor = torch.cat(generated_actions, dim=1)
    generated_positions_tensor = torch.cat(generated_positions, dim=1)
    rollout_positions = torch.cat([rollout_prefix, generated_positions_tensor], dim=1)
    expert_rollout_positions = expert_positions[:, : rollout_positions.shape[1]]
    expert_future_actions = expert_actions[:, warm_start : warm_start + generated_actions_tensor.shape[1]]

    collisions = collision_stats(rollout_positions, centers, radii)
    final_error = torch.linalg.norm(rollout_positions[:, -1] - goals, dim=-1)
    success = (final_error <= success_radius) & (~collisions["collision"])
    if config.get("benchmark", "toy_delayed") == "toy_annular":
        mode_metrics = annular_mode_metrics(rollout_positions, centers, collisions["collision"])
        hybrid_flags = annular_hybrid_flags(rollout_positions, centers, collisions["collision"])
    else:
        mode_metrics = delayed_mode_metrics(rollout_positions, centers, radii, collisions["collision"])
        hybrid_flags = delayed_hybrid_flags(rollout_positions, centers, radii, collisions["collision"])
    clean_success = success & (hybrid_flags < 0.5)
    smooth = action_smoothness(generated_actions_tensor, expert_actions[:, warm_start - action_history : warm_start])
    boundary = torch.cat(boundary_discontinuities).mean() if boundary_discontinuities else generated_actions_tensor.new_zeros(())
    if path_kl_steps:
        path_kl_energy = torch.cat(path_kl_steps, dim=1).sum(dim=1).mean()
    else:
        path_kl_energy = generated_actions_tensor.new_zeros(())
    action_mse = (generated_actions_tensor - expert_future_actions).pow(2).mean()
    path_length = torch.linalg.norm(rollout_positions[:, 1:] - rollout_positions[:, :-1], dim=-1).sum(dim=1).mean()

    metrics = {
        "closed_loop_success_rate": float(success.float().mean().detach().cpu().item()),
        "closed_loop_clean_success_rate": float(clean_success.float().mean().detach().cpu().item()),
        "closed_loop_goal_error": float(final_error.mean().detach().cpu().item()),
        "closed_loop_collision_rate": float(collisions["collision_rate"].detach().cpu().item()),
        "closed_loop_min_clearance": float(collisions["min_clearance"].detach().cpu().item()),
        "closed_loop_hybrid_rate": float(hybrid_flags.mean().detach().cpu().item()),
        "closed_loop_action_mse": float(action_mse.detach().cpu().item()),
        "closed_loop_path_length": float(path_length.detach().cpu().item()),
        "closed_loop_acceleration_energy": float(smooth["acceleration_energy"].detach().cpu().item()),
        "closed_loop_jerk_energy": float(smooth["jerk_energy"].detach().cpu().item()),
        "closed_loop_chunk_boundary_discontinuity": float(boundary.detach().cpu().item()),
        "closed_loop_path_KL_energy": float(path_kl_energy.detach().cpu().item()),
        "closed_loop_episodes": float(len(trajectory_ids)),
        "closed_loop_warm_start_steps": float(warm_start),
        "closed_loop_n_exec": float(n_exec),
        "closed_loop_success_radius": float(success_radius),
    }
    metrics.update({f"closed_loop_{key}": float(value.detach().cpu().item()) for key, value in mode_metrics.items()})
    return {
        "metrics": metrics,
        "rollout": {
            "positions": rollout_positions.detach().cpu(),
            "expert_positions": expert_rollout_positions.detach().cpu(),
            "goals": goals.detach().cpu(),
            "centers": centers.detach().cpu(),
            "radii": radii.detach().cpu(),
            "modes": modes.detach().cpu(),
            "success": success.detach().cpu(),
        },
    }


@torch.no_grad()
def evaluate_toy_model(
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
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        pred = predict_actions(model, batch, deterministic=bool(config.get("inference", {}).get("deterministic", True)))
        batch_metrics = compute_toy_metrics(
            pred["actions"],
            batch,
            path_kl_energy=pred.get("path_kl_energy"),
            benchmark=config.get("benchmark", "toy_delayed"),
        )
        metrics.append(batch_metrics)
        if pred.get("path_kl_energy") is not None:
            all_path_kl.append(pred["path_kl_energy"].detach().cpu())
    summary = average_metric_dicts(metrics)
    closed_loop_result = {"metrics": {}, "rollout": None}
    if bool(config.get("eval", {}).get("closed_loop", True)) and hasattr(dataset, "positions"):
        closed_loop_result = closed_loop_rollout(model, dataset, config, device)
        summary.update(closed_loop_result["metrics"])

    if output_dir is not None:
        figures = output_dir / "figures"
        try:
            plot_dataset_samples(dataset, figures / "dataset_samples.png")
        except Exception as exc:
            summary["plot_dataset_error"] = str(exc)

        first_loader = DataLoader(dataset, batch_size=min(8, max(1, len(dataset))), shuffle=False)
        first_batch = move_to_device(next(iter(first_loader)), device)
        single = slice_batch(first_batch, slice(0, 1))
        num_samples = int(config.get("inference", {}).get("num_samples", 16))
        generated = []
        z_samples = []
        for _ in range(max(1, num_samples)):
            if isinstance(model, ActionBridgePolicy):
                out = generate_chunk(model, single["obs_hist"], single["act_hist"], mode="sample", deterministic=True)
                generated.append(out["actions"])
                if model.latent_type == "continuous" and out["z"] is not None:
                    z_samples.append(out["z"])
            else:
                generated.append(predict_actions(model, single, deterministic=True)["actions"])
        generated_actions = torch.stack(generated, dim=1)
        try:
            plot_generated_samples(single, generated_actions, figures / "generated_same_history.png")
        except Exception as exc:
            summary["plot_generated_error"] = str(exc)

        if all_path_kl:
            try:
                plot_energy_histograms({"path_KL": torch.cat(all_path_kl)}, figures / "energy_histograms.png")
            except Exception as exc:
                summary["plot_energy_error"] = str(exc)

        if isinstance(model, ActionBridgePolicy) and model.latent_type == "categorical" and config.get("benchmark") == "toy_annular":
            prior = torch.softmax(model.prior_logits(model.encode_history(first_batch["obs_hist"], first_batch["act_hist"])), dim=-1)
            true = first_batch["context"]["p_ccw_true"]
            try:
                plot_calibration(true, prior[:, 1], figures / "categorical_prior_calibration.png")
            except Exception as exc:
                summary["plot_calibration_error"] = str(exc)

        if isinstance(model, ActionBridgePolicy) and model.latent_type == "continuous" and z_samples:
            z = torch.cat(z_samples, dim=0).detach()
            flat_actions = generated_actions.reshape(-1, generated_actions.shape[-2], generated_actions.shape[-1])
            repeated = {
                "context": {k: v.repeat(flat_actions.shape[0], *([1] * (v.ndim - 1))) for k, v in single["context"].items() if torch.is_tensor(v)},
            }
            pred_pos = actions_to_positions(single["future_positions"][:, 0].repeat(flat_actions.shape[0], 1), flat_actions)
            mode = classify_generated_modes(
                pred_pos,
                repeated["context"]["obstacle_center"],
                repeated["context"]["obstacle_radius"],
                config.get("benchmark", "toy_delayed"),
            )
            try:
                plot_latent_scatter(z, mode, figures / "continuous_latent_scatter.png")
            except Exception as exc:
                summary["plot_latent_error"] = str(exc)

        if closed_loop_result.get("rollout") is not None:
            rollout = closed_loop_result["rollout"]
            try:
                plot_closed_loop_rollouts(
                    rollout["positions"],
                    rollout["expert_positions"],
                    rollout["goals"],
                    rollout["centers"],
                    rollout["radii"],
                    rollout["modes"],
                    rollout["success"],
                    figures / "closed_loop_rollouts.png",
                    max_rollouts=int(config.get("eval", {}).get("closed_loop_plot_rollouts", 24)),
                )
            except Exception as exc:
                summary["plot_closed_loop_error"] = str(exc)
            save_json(output_dir / "metrics" / "closed_loop_metrics.json", closed_loop_result["metrics"])

        save_json(output_dir / "metrics" / "test_metrics.json", summary)
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

    checkpoint_metadata = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        raw = torch.load(ckpt_path, map_location="cpu")
        config = raw["config"]
        config = apply_overrides(config, args.overrides)
        device = resolve_device(str(config.get("device", "cpu")))
        model = build_model(config).to(device)
        model.load_state_dict(raw["model_state"])
        out_dir = Path(args.output_dir) if args.output_dir else default_eval_output_dir(config, ckpt_path)
        checkpoint_metadata = {
            "checkpoint": str(ckpt_path),
            "checkpoint_name": ckpt_path.name,
            "checkpoint_step": raw.get("step"),
            "checkpoint_best_metric": raw.get("best_metric"),
        }
    else:
        if args.config_name is None:
            raise SystemExit("Pass --checkpoint or --config-name.")
        config = apply_overrides(load_config(args.config_name), args.overrides)
        device = resolve_device(str(config.get("device", "cpu")))
        model = build_model(config).to(device)
        out_dir = Path(args.output_dir) if args.output_dir else default_eval_output_dir(config)

    dataset = build_dataset(config, split=args.split)
    metrics = evaluate_toy_model(model, dataset, config, device, output_dir=out_dir)
    if out_dir is not None:
        save_config(config, out_dir / "eval_config.json")
        metadata = {
            "output_dir": str(out_dir),
            "split": args.split,
            "device": str(device),
            "config_name": config.get("config_name"),
            "run_id": config.get("run_id"),
        }
        if checkpoint_metadata is not None:
            metadata.update(checkpoint_metadata)
        save_json(out_dir / "eval_metadata.json", metadata)
        print(f"Eval directory: {out_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
