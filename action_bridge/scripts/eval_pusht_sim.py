"""Run closed-loop Push-T simulator evaluation for a checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ml_collections import ConfigDict

from action_bridge.config import apply_overrides, save_config
from action_bridge.eval.pusht_sim import evaluate_pusht_sim_model
from action_bridge.training.common import build_model, resolve_device, save_json, seed_everything


def default_output_dir(checkpoint: Path) -> Path:
    try:
        run_dir = checkpoint.parents[1]
        if checkpoint.parent.name == "checkpoints":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return run_dir / "eval" / f"pusht_sim_{stamp}"
    except IndexError:
        pass
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "eval" / f"pusht_sim_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-exec", type=int, default=None)
    parser.add_argument("--render-episodes", type=int, default=None)
    parser.add_argument("--save-gifs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gif-fps", type=float, default=None)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    raw = torch.load(checkpoint, map_location="cpu")
    config = apply_overrides(raw["config"], args.overrides)
    if args.device is not None:
        config.device = args.device
    if "eval" not in config:
        config.eval = ConfigDict()
    config.eval.sim_closed_loop = True
    if args.episodes is not None:
        config.eval.sim_episodes = int(args.episodes)
    if args.max_steps is not None:
        config.eval.sim_max_steps = int(args.max_steps)
    if args.n_exec is not None:
        config.eval.sim_n_exec = int(args.n_exec)
    if args.render_episodes is not None:
        config.eval.sim_render_episodes = int(args.render_episodes)
    if args.save_gifs is not None:
        config.eval.sim_save_gifs = bool(args.save_gifs)
    if args.gif_fps is not None:
        config.eval.sim_gif_fps = float(args.gif_fps)
    if args.save_videos is not None:
        config.eval.sim_save_videos = bool(args.save_videos)
    if args.video_fps is not None:
        config.eval.sim_video_fps = float(args.video_fps)
    if args.seed is not None:
        config.eval.sim_seed = int(args.seed)

    seed_everything(int(config.get("eval", {}).get("sim_seed", config.get("seed", 0))))
    device = resolve_device(str(config.get("device", "cpu")))
    model = build_model(config).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_pusht_sim_model(model, config, device, output_dir=output_dir)
    save_json(output_dir / "metrics" / "pusht_sim_summary.json", metrics)
    save_config(config, output_dir / "pusht_sim_config.json")
    print(f"Push-T sim eval written to: {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
