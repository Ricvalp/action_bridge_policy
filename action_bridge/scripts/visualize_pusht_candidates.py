"""Plot revisions of one old plan at a learner-reached or synthetic Push-T state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from action_bridge.eval.revision_pusht import (
    configure_completion_velocity_weighting, evaluate, validate_evaluation_completion,
)
from action_bridge.eval.revision_pusht_candidates import (
    SOURCE_VARIANTS, candidate_statistics, context_at_replan, sample_candidate_revisions,
    source_variant_context, symmetric_context,
)
from action_bridge.eval.revision_pusht_candidate_plots import render_candidate_batch
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.completion import COMPLETION_NAMES, COMPLETION_VELOCITY_WEIGHTINGS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " Load only trusted checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", choices=("rollout", "symmetric"), default="rollout",
                        help="Actual learner rollout or a synthetic symmetric wrong-side probe")
    parser.add_argument("--source-variant", choices=SOURCE_VARIANTS, default="axial",
                        help="Synthetic scene only: change the unexecuted source, keeping histories fixed")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234, help="Simulator reset and warm-up rollout seed")
    parser.add_argument("--replan", type=int, default=6, help="Zero-based decision; 0 is startup")
    parser.add_argument("--candidate-seed", type=int, default=42)
    parser.add_argument("--source-noise", choices=("fixed", "independent", "none"),
                        help="Default: fixed for rollouts; none for the symmetric probe")
    parser.add_argument("--completion", choices=COMPLETION_NAMES)
    parser.add_argument("--completion-velocity-weighting", choices=COMPLETION_VELOCITY_WEIGHTINGS,
                        help="Learned completion velocity initialization; defaults to checkpoint setting")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.scene != "symmetric" and args.source_variant != "axial":
        parser.error("--source-variant requires --scene symmetric")
    if min(args.candidates, args.batch_size, args.threads) < 1 or min(args.seed, args.replan, args.candidate_seed) < 0:
        parser.error("Counts must be positive and seeds/replan nonnegative")
    torch.set_num_threads(args.threads)
    with args.checkpoint.open("rb") as stream:
        state = checkpoints.load(stream)
        stream.seek(0)
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = dict(state["config"])
    if config.get("protocol") not in ("self_source_v1", "self_source_k_curriculum_v1") or config["method"] not in ("fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"):
        parser.error("Use a self-source FM or SB reviser checkpoint")
    if (config["obs_dim"], config["action_dim"]) != (5, 2):
        parser.error("This visualization uses the Push-T 5-state/2-target interface")
    if config["method"].startswith("sb_") and state.get("direction") != "forward":
        parser.error("Use a forward-policy checkpoint such as best.pt")
    synthetic = args.scene == "symmetric"
    args.source_noise = args.source_noise or ("none" if synthetic else "fixed")
    default_mode = 1 if synthetic else config.get("completion_id", 2)
    mode = default_mode if args.completion is None else COMPLETION_NAMES.index(args.completion)
    validate_evaluation_completion(config, mode)
    if synthetic and mode not in (0, 1):
        parser.error("The symmetric probe needs repeat or fixed_damped completion")
    try:
        velocity_identity = configure_completion_velocity_weighting(
            config, mode, args.completion_velocity_weighting)
    except ValueError as error:
        parser.error(str(error))
    if not synthetic and args.replan * config["execute"] >= config["max_episode_steps"]:
        parser.error("Requested replan is beyond the checkpoint's episode step limit")
    config["completion_id"] = mode
    if not synthetic:
        config["max_episode_steps"] = min(config["max_episode_steps"], (args.replan + 1) * config["execute"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(__file__).resolve().parents[2] / "workspace" / "sb_pusht" / "candidate_batches"
    scene_suffix = f"-symmetric-{args.source_variant}" if synthetic else ""
    output = (args.output_dir or root / f"{config['method']}{scene_suffix}-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("SDL_VIDEODRIVER", "dummy"), ("SDL_AUDIODRIVER", "dummy"),
                        ("MPLBACKEND", "Agg"), ("MPLCONFIGDIR", str(output / ".cache" / "matplotlib"))):
        os.environ.setdefault(name, value)
    policy = checkpoints.restore_policy(state, args.device)
    if synthetic:
        print("[1/3] Building a symmetric wrong-side context; no simulator rollout.", flush=True)
        context = symmetric_context(config, state["metadata"], completion_id=mode)
        context = source_variant_context(context, config, state["metadata"],
                                         variant=args.source_variant, completion_id=mode)
    else:
        print(f"[1/3] Reaching decision {args.replan} with the policy (seed {args.seed}).", flush=True)
        evaluate(policy, config, state["metadata"], state["dependencies"], args.device,
                 output=output, completion_id=mode, seeds=[args.seed], render=False,
                 save_videos=False, save_gifs=False, progress=False, write_metrics=False, trace_generation=True)
        episode = json.loads((output / f"episode-seed{args.seed}.json").read_text())
        context = context_at_replan(episode, config, args.replan)
    print(f"[2/3] Sampling {args.candidates} candidates; scene and histories fixed.", flush=True)
    data = sample_candidate_revisions(
        policy, config, state["metadata"], state["dependencies"], context, args.device,
        candidates=args.candidates, batch_size=args.batch_size, seed=args.candidate_seed,
        completion_id=mode, source_noise=args.source_noise,
    )
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output / "candidates.npz", **arrays)
    print("[3/3] Plotting command targets (not simulated candidate outcomes).", flush=True)
    context_label = context["source_label"] if synthetic else f"seed {args.seed}"
    artifacts = render_candidate_batch(
        data, output, title=f"{config['method']} | EMA step {state['step']:,} | {context_label}")
    statistics = candidate_statistics(data, config["execute"])
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": state["step"], "weights": "ema", "config": config,
        **velocity_identity,
        "metadata": state["metadata"], "checkpoint_runtime": state.get("runtime"),
        "visualization_runtime": checkpoints.runtime_identity(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "replan": context["replan"], "robot_step": context["robot_step"], "execute": config["execute"],
        "old_executed": context["executed"], "has_previous_plan": context["has_previous_plan"],
        "scene": ("synthetic symmetric wrong-side state; not learner-reached or physically simulated" if synthetic
                  else "learner-reached state; identical observation/action histories and old plan for every candidate"),
        "candidate_semantics": "unclipped absolute xy commands; not physical trajectories; no scoring or selection",
        "source_noise": args.source_noise,
        "physics_snapshot": False, "statistics": statistics, "artifacts": artifacts,
        "arrays": "candidates.npz", "warmup_episode": None if synthetic else f"episode-seed{args.seed}.json",
    }
    if synthetic:
        manifest["construction"] = {
            "state_raw": context["state_raw"].tolist(), "goal_pose_raw": context["goal_pose_raw"].tolist(),
            "symmetry_axis_raw": context["symmetry_axis_raw"].tolist(),
            "lateral_axis_raw": context["lateral_axis_raw"].tolist(),
            "history": context["synthetic_history"],
            "retreat_px_per_action": context["synthetic_speed_px_per_action"],
            "completion": COMPLETION_NAMES[mode],
            "source_variant": context["source_variant"], "source_label": context["source_label"],
            "note": "Scene geometry and histories are symmetric; the source may be deliberately right-biased",
        }
        lateral = (data["candidates_raw"] - context["state_raw"][2:4]) @ context["lateral_axis_raw"]
        manifest["lateral_command_statistics"] = {
            name: {"mean_px": float(lateral[:, index].mean()), "std_px": float(lateral[:, index].std()),
                   "negative_fraction": float((lateral[:, index] < 0).mean()),
                   "positive_fraction": float((lateral[:, index] > 0).mean())}
            for name, index in (("command_K", config["execute"] - 1), ("command_H", config["horizon"] - 1))
        }
    (output / "candidates.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Output: {output}\n{json.dumps(statistics, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
