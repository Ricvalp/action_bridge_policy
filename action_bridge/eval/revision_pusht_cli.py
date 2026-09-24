"""Shared CLI for evaluating one trusted whole-plan Push-T checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from action_bridge.eval.revision_pusht import evaluate
from action_bridge.eval.revision_pusht_parallel import evaluate_parallel
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.completion import COMPLETION_NAMES


CHECKOUT = Path(__file__).resolve().parents[2]


def main(method, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Closed-loop Push-T evaluation of a {method} EMA checkpoint. "
                    "Only load checkpoints you trust: Torch checkpoints can execute code.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1_000_000, help="First of consecutive episode seeds")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Explicit episode seeds; overrides --seed and --episodes")
    parser.add_argument("--execute", type=int, help="Actions before replanning; defaults to checkpoint setting")
    parser.add_argument("--max-steps", type=int, help="Episode step limit; defaults to checkpoint setting")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True,
                        help="Save up to two successful and two failed rollout MP4s (default: on)")
    parser.add_argument("--save-gifs", action=argparse.BooleanOptionalAction, default=False,
                        help="Also save the selected rollouts as GIFs (default: off)")
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to a timestamped workspace directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel episode processes (CPU only; default: 1)")
    parser.add_argument("--threads", type=int,
                        help="Torch threads per worker (default: 4 serial, 1 parallel)")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show episode progress, elapsed time and ETA (default: on)")
    if method != "ddim":
        parser.add_argument("--completion", choices=COMPLETION_NAMES,
                            help="Old-plan tail completion; defaults to checkpoint setting")
    args = parser.parse_args(argv)
    for name in ("episodes", "execute", "max_steps", "threads", "workers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if args.workers > 1 and torch.device(args.device).type != "cpu":
        parser.error("--workers > 1 requires --device cpu")
    if args.threads is None:
        args.threads = 1 if args.workers > 1 else 4
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    torch.set_num_threads(args.threads)
    # Read and hash the same file even if training atomically replaces latest.pt.
    # Optimizer tensors stay on CPU; only the restored EMA model uses the device.
    with checkpoint.open("rb") as stream:
        state = checkpoints.load(stream)
        stream.seek(0)
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = dict(state["config"])
    if config["method"] != method:
        parser.error(f"Expected {method}, but checkpoint contains {config['method']}")
    if method != "ddim" and config.get("protocol") != "self_source_v1":
        parser.error("This command evaluates self_source_v1 revisers only; fixed-DDIM-source checkpoints are retired")
    if "ema" not in state:
        parser.error("Checkpoint has no EMA policy weights; a reference checkpoint is not a policy")
    if method.startswith("sb_") and state.get("direction") != "forward":
        parser.error("Use an SB checkpoint saved during a forward phase, e.g. best.pt; not a reverse-phase checkpoint")
    if args.execute is not None:
        config["execute"] = args.execute
    if args.max_steps is not None:
        config["max_episode_steps"] = args.max_steps
    if not 1 <= config["execute"] <= config["horizon"]:
        parser.error("--execute must be between 1 and the checkpoint horizon")
    mode = getattr(args, "completion", None)
    completion_id = COMPLETION_NAMES.index(mode) if mode is not None else config.get("completion_id", 2)
    config["completion_id"] = completion_id
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.episodes))
    args.workers = min(args.workers, len(seeds))
    config["evaluation_seeds"] = seeds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or CHECKOUT / "workspace" / "sb_pusht" / "evaluation" / f"{method}-{stamp}").resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}. Omit --output-dir for a new timestamped directory.")
    versions = {}
    for package in ("torch", "diffusers", "gym-pusht", "pymunk"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                    checkpoint_step=state["step"], weights="ema", method=method,
                    policy_direction=state.get("direction"),
                    training_direction=state.get("training_direction", state.get("direction")),
                    forward_updates=state.get("forward_updates"),
                    config=config, metadata=state["metadata"],
                    checkpoint_runtime=state.get("runtime"), evaluation_runtime=checkpoints.runtime_identity(),
                    versions=versions, device=args.device, seeds=seeds,
                    workers=args.workers, threads=args.threads,
                    save_videos=args.save_videos, save_gifs=args.save_gifs,
                    progress=args.progress,
                    completion_id=completion_id)
    (output / "evaluation.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(f"Evaluating {method}, checkpoint step {state['step']}, "
          f"H={config['horizon']}, execute={config['execute']}, episodes={len(seeds)}, "
          f"workers={args.workers}, threads/worker={args.threads}", flush=True)
    print(f"Output: {output}", flush=True)
    if args.workers > 1:
        metrics = evaluate_parallel(state, config, args.device, output=output,
                                    completion_id=completion_id, seeds=seeds,
                                    workers=args.workers, threads=args.threads,
                                    save_videos=args.save_videos, save_gifs=args.save_gifs,
                                    progress=args.progress)
    else:
        policy = checkpoints.restore_policy(state, args.device)
        metrics = evaluate(policy, config, state["metadata"], state["dependencies"], args.device,
                           output=output, completion_id=completion_id, seeds=seeds,
                           render=args.save_videos or args.save_gifs,
                           save_videos=args.save_videos, save_gifs=args.save_gifs,
                           progress=args.progress)
    print(json.dumps(metrics, indent=2))
    return 0
