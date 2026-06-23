"""Visualize a trained policy's marginals with an overridden particle count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_collections import config_dict
import torch

from ..data import generate_dataset
from ..train import (
    dataset_path,
    evaluate_position_marginals,
    experiment_root,
    get_device,
    make_env_config,
    make_model,
)


def to_config(value):
    if isinstance(value, dict):
        cfg = config_dict.ConfigDict()
        for key, item in value.items():
            cfg[key] = to_config(item)
        return cfg
    if isinstance(value, list):
        return [to_config(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing config.json and model.pt.",
    )
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--time_slices", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/marginals_p<particles>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = experiment_root()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = to_config(json.load(f))

    cfg.device = args.device
    cfg.model.particles = args.particles
    cfg.eval.marginal_examples = args.examples
    cfg.eval.marginal_samples = args.particles
    cfg.eval.marginal_time_slices = args.time_slices

    env_cfg = make_env_config(cfg)
    data_file = dataset_path(root, cfg)
    if not data_file.exists():
        generate_dataset(
            path=data_file,
            num_trajectories=cfg.data.num_trajectories,
            trajectory_length=cfg.data.trajectory_length,
            seed=cfg.seed,
            cfg=env_cfg,
            paired_modes=cfg.data.paired_modes,
        )

    device = get_device(args.device)
    checkpoint = torch.load(run_dir / "model.pt", map_location=device)
    model = make_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = run_dir / f"marginals_p{args.particles}"
    elif not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_position_marginals(model, data_file, device, cfg, env_cfg, out_dir)
    metrics_path = out_dir / "position_marginal_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(out_dir / "position_marginals.png")
    print(metrics_path)


if __name__ == "__main__":
    main()
