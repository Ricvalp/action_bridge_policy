"""Visualize completion and generation from a trusted Push-T EMA checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import torch

from action_bridge.eval.revision_pusht import evaluate, validate_evaluation_completion
from action_bridge.eval.revision_pusht_visualization import render_revision_process
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.completion import COMPLETION_NAMES


CHECKOUT = Path(__file__).resolve().parents[2]
METHODS = ("fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize Push-T completion, revision and execution from one EMA checkpoint. "
                    "This is a diagnostic clip, not a success-rate benchmark. "
                    "Only load trusted checkpoints: Torch checkpoints can execute code.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--execute", type=int, help="Actions before replanning; inherits checkpoint setting")
    parser.add_argument("--completion", choices=COMPLETION_NAMES,
                        help="Old-plan tail completion; inherits checkpoint setting")
    parser.add_argument("--start-replan", type=int, default=1,
                        help="First displayed zero-based decision (0 is startup; default: 1)")
    parser.add_argument("--replans", type=int, default=3,
                        help="Number of consecutive decisions to display (default: 3)")
    parser.add_argument("--fps", type=float, default=10,
                        help="Playback frame rate (default: 10)")
    parser.add_argument("--save-gif", action="store_true", help="Also save an animated GIF")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory; defaults to workspace/sb_pusht/visualizations/<method>-<timestamp>")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show rollout progress (default: on)")
    args = parser.parse_args(argv)
    for name in ("threads", "execute", "replans", "fps"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("seed", "start_replan"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

    torch.set_num_threads(args.threads)
    print(f"[1/3] Loading trusted checkpoint: {checkpoint}", flush=True)
    # Hash the same open file that was loaded, even if latest.pt is replaced.
    # The checkpoint loader leaves optimizer tensors on CPU.
    with checkpoint.open("rb") as stream:
        state = checkpoints.load(stream)
        stream.seek(0)
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = dict(state["config"])
    method = config.get("method")
    if method == "ddim":
        parser.error("DDIM has no old-plan completion/revision process; use an FM or SB reviser checkpoint")
    if method not in METHODS:
        parser.error(f"Unsupported revision method: {method!r}; expected one of {METHODS}")
    if (config.get("obs_dim"), config.get("action_dim")) != (5, 2):
        parser.error("Visualization requires the Push-T 5-state/2-target adapter")
    if config.get("protocol") != "self_source_v1":
        parser.error("Visualization requires a self_source_v1 reviser checkpoint")
    if "ema" not in state:
        parser.error("Checkpoint has no EMA policy weights; a reference checkpoint is not a policy")
    if method.startswith("sb_") and state.get("direction") != "forward":
        parser.error("Use an SB checkpoint saved during a forward phase, e.g. best.pt; not a reverse-phase checkpoint")
    if args.execute is not None:
        config["execute"] = args.execute
    if not 1 <= config["execute"] <= config["horizon"]:
        parser.error("--execute must be between 1 and the checkpoint horizon")
    if config["max_episode_steps"] < 1:
        parser.error("Checkpoint max_episode_steps must be positive")
    completion_id = (COMPLETION_NAMES.index(args.completion) if args.completion is not None
                     else config.get("completion_id", 2))
    if completion_id not in range(len(COMPLETION_NAMES)):
        parser.error(f"Checkpoint completion_id must be in [0, {len(COMPLETION_NAMES) - 1}]")
    try:
        validate_evaluation_completion(config, completion_id)
    except ValueError as error:
        parser.error(str(error))
    if completion_id == 3 and config["execute"] != state["config"]["execute"]:
        parser.error("direct_mlp --execute must match the K used to train its tail predictor")
    config["completion_id"] = completion_id
    config["evaluation_seeds"] = [args.seed]
    config["max_episode_steps"] = min(
        config["max_episode_steps"], (args.start_replan + args.replans) * config["execute"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or CHECKOUT / "workspace" / "sb_pusht" / "visualizations"
              / f"{method}-{stamp}").resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}. Choose a new path or omit --output-dir.")
    for name, value in (("SDL_VIDEODRIVER", "dummy"), ("SDL_AUDIODRIVER", "dummy"),
                        ("MPLBACKEND", "Agg"),
                        ("MPLCONFIGDIR", str(output / ".cache" / "matplotlib"))):
        os.environ.setdefault(name, value)
    print(f"Diagnostic clip, not a success-rate benchmark: {method}, EMA step {state['step']}, "
          f"H={config['horizon']}, K={config['execute']}, {COMPLETION_NAMES[completion_id]}.", flush=True)
    print(f"Output: {output}", flush=True)
    policy = checkpoints.restore_policy(state, args.device)
    print(f"[2/3] Collecting fresh trace from seed {args.seed}; "
          f"warming up to decision {args.start_replan}.", flush=True)
    evaluate(policy, config, state["metadata"], state["dependencies"], args.device,
             output=output, completion_id=completion_id, seeds=[args.seed],
             render=False, save_videos=False, save_gifs=False, progress=args.progress,
             trace_generation=True, write_metrics=False)
    episode_file = output / f"episode-seed{args.seed}.json"
    episode = json.loads(episode_file.read_text())
    print("[3/3] Rendering completion, revision and physical execution.", flush=True)
    artifacts = render_revision_process(
        episode, config, output, start_replan=args.start_replan, replans=args.replans,
        fps=args.fps, save_gif=args.save_gif)
    versions = {}
    for package in ("torch", "diffusers", "gym-pusht", "pymunk", "matplotlib", "imageio"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    identity = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        checkpoint_step=state["step"], weights="ema", method=method,
        policy_direction=state.get("direction"),
        training_direction=state.get("training_direction", state.get("direction")),
        forward_updates=state.get("forward_updates"),
        checkpoint_runtime=state.get("runtime"), visualization_runtime=checkpoints.runtime_identity(),
        versions=versions, config=config, metadata=state["metadata"],
        seed=args.seed, device=args.device, threads=args.threads,
        completion=COMPLETION_NAMES[completion_id], completion_id=completion_id,
        horizon=config["horizon"], execute=config["execute"],
        start_replan=args.start_replan, replans=args.replans,
        fps=args.fps, save_gif=args.save_gif, progress=args.progress,
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        episode=episode_file.name, diagnostic_only=True,
        phase_semantics={
            "completion": "Retain the old-plan overlap and fill only its missing tail; no robot motion.",
            "source_noise": "Perturb the completed source once before generation; no robot motion.",
            "revision": "Show actual sampler intermediates with the physical state held fixed.",
            "execution": "Execute the final prefix; show bounded commands and actual observed motion.",
            "startup": "Decision 0 repeats the observed pusher position. When K=H later decisions repeat the last executed command; no retained overlap.",
        },
        artifacts=artifacts)
    (output / "visualization.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(f"Saved diagnostic artifacts and manifest: {output / 'visualization.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
