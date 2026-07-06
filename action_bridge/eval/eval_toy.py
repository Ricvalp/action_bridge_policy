"""Evaluate toy checkpoints and write metrics/figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.eval.metrics import average_metric_dicts, compute_toy_metrics
from action_bridge.eval.rollout import actions_to_positions, generate_chunk, predict_actions
from action_bridge.eval.visualization import (
    plot_calibration,
    plot_dataset_samples,
    plot_energy_histograms,
    plot_generated_samples,
    plot_latent_scatter,
)
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.training.common import build_dataset, build_model, move_to_device, resolve_device, save_json, slice_batch


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
    all_acc = []
    all_jerk = []
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
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        raw = torch.load(ckpt_path, map_location="cpu")
        config = raw["config"]
        config = apply_overrides(config, args.overrides)
        device = resolve_device(str(config.get("device", "cpu")))
        model = build_model(config).to(device)
        model.load_state_dict(raw["model_state"])
        out_dir = ckpt_path.parents[1]
    else:
        if args.config_name is None:
            raise SystemExit("Pass --checkpoint or --config-name.")
        config = apply_overrides(load_config(args.config_name), args.overrides)
        device = resolve_device(str(config.get("device", "cpu")))
        model = build_model(config).to(device)
        out_dir = None

    dataset = build_dataset(config, split=args.split)
    metrics = evaluate_toy_model(model, dataset, config, device, output_dir=out_dir)
    if out_dir is not None:
        save_config(config, out_dir / "eval_config.yaml")
    print(metrics)


if __name__ == "__main__":
    main()
