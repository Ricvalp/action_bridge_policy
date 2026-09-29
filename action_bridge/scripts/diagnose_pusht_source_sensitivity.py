"""Vary a source chunk at one fixed Push-T context, using matched sampler noise.

Only the short context-acquisition rollout uses the simulator. Intervened
sources and their revisions are command targets, not simulated trajectories.
Load only checkpoints you trust (Torch checkpoints can contain executable code).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from action_bridge.eval.revision_pusht import evaluate, validate_evaluation_completion
from action_bridge.eval.revision_pusht_candidates import context_at_replan
from action_bridge.eval.revision_pusht_source_sensitivity import (
    sample_source_sensitivity, sensitivity_statistics,
)
from action_bridge.eval.revision_pusht_source_sensitivity_plots import render_source_sensitivity
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.completion import COMPLETION_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="Simulator reset and warm-up rollout seed")
    parser.add_argument("--replan", type=int, default=6, help="Zero-based decision; 0 probes the startup source")
    parser.add_argument("--samples", type=int, default=64, help="Paired noise realizations per source")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[-40., -15., -5., 5., 15., 40.],
                        help="Signed maximum source displacements in pixels; baseline is always included")
    parser.add_argument("--families", nargs="+", default=["translate_x", "translate_y", "bow_x", "bow_y"],
                        choices=("translate_x", "translate_y", "bow_x", "bow_y"))
    parser.add_argument("--region", choices=("all", "retained", "tail"), default="all",
                        help="Intervene directly on the completed sampler input, after fixed source noise")
    parser.add_argument("--completion", choices=COMPLETION_NAMES,
                        help="Completion mode for the warm-up rollout; defaults to the checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.samples, args.batch_size, args.threads) < 1:
        parser.error("--samples, --batch-size and --threads must be positive")
    if min(args.seed, args.replan, args.sampling_seed) < 0:
        parser.error("Seeds and --replan must be nonnegative")
    if not np.isfinite(args.amplitudes).all() or not any(value != 0 for value in args.amplitudes):
        parser.error("--amplitudes must be finite and include a nonzero displacement")
    if len(set(args.amplitudes)) != len(args.amplitudes) or len(set(args.families)) != len(args.families):
        parser.error("Amplitude and family lists must not contain duplicates")
    torch.set_num_threads(args.threads)
    with args.checkpoint.open("rb") as stream:
        state = checkpoints.load(stream)
        stream.seek(0)
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = dict(state["config"])
    if config.get("protocol") not in ("self_source_v1", "self_source_k_curriculum_v1"):
        parser.error("Use a self_source_v1 or self_source_k_curriculum_v1 checkpoint")
    if config["method"] not in ("sb_ou", "sb_kinetic", "fm_paired", "fm_local_ot"):
        parser.error("This diagnostic requires an SB or FM reviser, not DDIM")
    if (config["obs_dim"], config["action_dim"]) != (5, 2) or state["metadata"]["codec"]["units"] != "pixels":
        parser.error("This visualization requires the Push-T 5-state/2-target pixel interface")
    if config["method"].startswith("sb_") and state.get("direction") != "forward":
        parser.error("Use a forward-policy checkpoint, such as best.pt")
    mode = config.get("completion_id", 2) if args.completion is None else COMPLETION_NAMES.index(args.completion)
    validate_evaluation_completion(config, mode)
    if args.replan * config["execute"] >= config["max_episode_steps"]:
        parser.error("Requested replan exceeds the checkpoint episode step limit")
    if args.region == "retained" and (args.replan == 0 or config["execute"] == config["horizon"]):
        parser.error("There is no retained region at startup or when K=H")
    config["completion_id"] = mode
    warmup_config = config | {"max_episode_steps": min(config["max_episode_steps"], (args.replan + 1) * config["execute"])}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    default_root = Path(__file__).resolve().parents[2] / "workspace/sb_pusht/source_sensitivity"
    output = (args.output_dir or default_root / f"{config['method']}-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("SDL_VIDEODRIVER", "dummy"), ("SDL_AUDIODRIVER", "dummy"),
                        ("MPLBACKEND", "Agg"), ("MPLCONFIGDIR", str(output / ".cache/matplotlib"))):
        os.environ.setdefault(name, value)
    policy = checkpoints.restore_policy(state, args.device)
    print(f"[1/3] Reaching decision {args.replan}, seed {args.seed}; then freezing the context.", flush=True)
    evaluate(policy, warmup_config, state["metadata"], state["dependencies"], args.device,
             output=output, completion_id=mode, seeds=[args.seed], render=False,
             save_videos=False, save_gifs=False, progress=args.progress,
             write_metrics=False, trace_generation=True)
    episode = json.loads((output / f"episode-seed{args.seed}.json").read_text())
    try:
        context = context_at_replan(episode, config, args.replan)
    except ValueError as error:
        parser.error(str(error))
    print("[2/3] Varying only the source; pairing the same random draws across interventions.", flush=True)
    data = sample_source_sensitivity(
        policy, config, state["metadata"], state["dependencies"], context, args.device,
        samples=args.samples, batch_size=args.batch_size, seed=args.sampling_seed,
        amplitudes_px=args.amplitudes, families=args.families, region=args.region,
        completion_id=mode, progress=args.progress,
    )
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    arrays.update(variant_labels=np.asarray(data["variant_labels"]), families=np.asarray(data["families"]))
    np.savez_compressed(output / "source_sensitivity.npz", **arrays)
    statistics = sensitivity_statistics(data)
    print("[3/3] Plotting sources, paired revisions, response curves and revision-time sensitivity.", flush=True)
    artifacts = render_source_sensitivity(
        data, output, title=f"{config['method']} | EMA step {state['step']:,} | seed {args.seed}, replan {args.replan}")
    report = dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_step=state["step"], weights="ema", config=config,
        checkpoint_runtime=state.get("runtime"), diagnostic_runtime=checkpoints.runtime_identity(),
        metadata=state["metadata"], training_curriculum=state.get("curriculum"),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        replan=args.replan, robot_step=context["robot_step"], execute=config["execute"],
        elapsed_count=context["executed"], has_previous_plan=context["has_previous_plan"],
        variant_labels=data["variant_labels"], families=data["families"],
        amplitudes_px=np.asarray(data["amplitudes_px"]).tolist(), intervention_region=args.region,
        source_noise="same recorded source perturbation retained for every intervention; not redrawn",
        sampler_noise="common random numbers across source variants; independent second baseline also sampled",
        intervention="modify the completed source directly; no re-completion, conditioning change or clipping",
        fixed=["observation history", "executed-command history", "K", "completion ID", "startup bit", "reference coefficients"],
        statistics=statistics, artifacts=artifacts, arrays="source_sensitivity.npz",
        warmup_episode=f"episode-seed{args.seed}.json", sample_seconds=data.get("sample_seconds"),
        interpretation=("Paired changes measure pathwise source sensitivity under common sampling noise, "
                        "not success or a formal test of equality of output distributions. Small mean shifts "
                        "can hide multimodality. Small responses can be legitimate bridge contraction, "
                        "not proof of a faulty policy. Large interventions may be out of distribution. "
                        "Repeat across contexts/seeds before drawing conclusions."),
        physics_snapshot=False, simulated_candidates=False,
    )
    (output / "source_sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
