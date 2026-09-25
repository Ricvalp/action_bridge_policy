"""Offline state/old-plan gaps for a trusted pretrained Push-T reviser."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from action_bridge.eval.revision_pusht_gaps import diagnose_offline_sources
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.completion import COMPLETION_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " Only load checkpoints/windows you trust.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True, help="Matching prepared windows.pt")
    parser.add_argument("--episodes", type=int, default=8, help="Number of fixed held-out episode prefixes")
    parser.add_argument("--replans", type=int, default=16, help="Maximum replans per episode, including startup")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--completion", choices=COMPLETION_NAMES)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.replans < 2 or args.threads < 1 or args.seed < 0:
        parser.error("Require positive episodes/threads, replans >= 2, and seed >= 0")
    torch.set_num_threads(args.threads)
    # Hash the exact file read, even if a trainer replaces latest.pt concurrently.
    with args.checkpoint.open("rb") as stream:
        state = checkpoints.load(stream)
        stream.seek(0)
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = state["config"]
    if config["method"] not in ("fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"):
        parser.error("Use an FM/SB policy checkpoint")
    if config.get("protocol") != "self_source_v1":
        parser.error("Use a self_source_v1 checkpoint")
    mode = config.get("completion_id", 2) if args.completion is None else COMPLETION_NAMES.index(args.completion)
    with args.windows.open("rb") as stream:
        windows = checkpoints.load(stream)
        stream.seek(0)
        windows_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(__file__).resolve().parents[2] / "workspace" / "sb_pusht" / "source_gaps"
    output = args.output_dir or root / f"{config['method']}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    policy = checkpoints.restore_policy(state, args.device)
    report = diagnose_offline_sources(policy, config, state["metadata"], state["dependencies"],
                                     windows, args.device, episodes=args.episodes, replans=args.replans,
                                     seed=args.seed, completion_id=mode)
    report["identity"] = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                          "windows": str(args.windows.resolve()), "windows_sha256": windows_hash,
                          "checkpoint_step": state["step"], "weights": "ema.forward" if config["method"].startswith("sb_") else "ema",
                          "training_direction": state.get("training_direction", state.get("direction")),
                          "device": args.device, "threads": args.threads,
                          "config": config, "checkpoint_runtime": state.get("runtime"),
                          "diagnostic_runtime": checkpoints.runtime_identity()}
    (output / "source_gaps.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Offline source gaps (no simulation): {output / 'source_gaps.json'}")
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
