"""Evaluate the hardcoded Push-T geometric reference without training."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ml_collections import ConfigDict

from action_bridge.config import apply_overrides, load_config, save_config
from action_bridge.eval.pusht_sim import evaluate_pusht_sim_model
from action_bridge.training.common import build_dataset, build_model, resolve_device, save_json, seed_everything


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_dir(name: str) -> Path:
    return Path("outputs") / "eval" / f"{timestamp()}_{name}"


def maybe_sync_dataset_stats(config: ConfigDict) -> None:
    dataset_path = config.get("data", {}).get("dataset_path", None)
    if dataset_path is None:
        return
    dataset = build_dataset(config, split="train")
    obs_dim = getattr(dataset, "obs_dim", None)
    action_dim = getattr(dataset, "action_dim", None)
    if obs_dim is not None:
        config.obs_dim = int(obs_dim)
    if action_dim is not None:
        config.action_dim = int(action_dim)
    stats = getattr(dataset, "normalization_stats", None)
    if stats is not None:
        config.data.normalization_stats = stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pusht_lowdim_geometric_reference")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--n-exec", type=int, default=8)
    parser.add_argument("--render-episodes", type=int, default=4)
    parser.add_argument("--save-gifs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config_name), args.overrides)
    config.run_id = config.get("run_id", "geometric_reference_only")
    config.obs_dim = int(config.get("obs_dim", 5))
    config.action_dim = int(config.get("action_dim", 2))
    if args.device is not None:
        config.device = args.device
    config.seed = int(args.seed)

    if "eval" not in config:
        config.eval = ConfigDict()
    config.eval.sim_closed_loop = True
    config.eval.sim_intervention = "reference_only"
    config.eval.sim_episodes = int(args.episodes)
    config.eval.sim_max_steps = int(args.max_steps)
    config.eval.sim_n_exec = int(args.n_exec)
    config.eval.sim_seed = int(args.seed)
    config.eval.sim_render_episodes = int(args.render_episodes)
    config.eval.sim_save_gifs = bool(args.save_gifs)
    config.eval.sim_gif_fps = float(args.gif_fps)
    config.eval.sim_collect_contact_diagnostics = True
    config.eval.sim_geometric_panels = 8
    config.inference.deterministic = True
    config.inference.n_exec = int(args.n_exec)
    config.inference.latent_commitment = "chunk"

    maybe_sync_dataset_stats(config)
    seed_everything(int(args.seed))
    device = resolve_device(str(config.get("device", "cpu")))
    model = build_model(config).to(device)
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(str(config.run_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_pusht_sim_model(model, config, device, output_dir=output_dir)
    save_json(output_dir / "metrics" / "geometric_reference_summary.json", metrics)
    save_config(config, output_dir / "geometric_reference_config.json")
    print(f"Geometric reference eval written to: {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
