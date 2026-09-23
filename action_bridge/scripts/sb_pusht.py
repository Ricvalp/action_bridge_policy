"""Resumable, five-job Push-T experiment. See docs/SB_PUSHT.md for commands."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

import torch

from action_bridge.configs.sb_pusht import COMPLETIONS, METHODS, get_config
from action_bridge.data.revision_pusht import load_windows, make_local_pairer, neighbor_blocks
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.data import build_source_pairs
from action_bridge.plan_revision.tracking import Tracker, TrackingOptions
from action_bridge.plan_revision.training import fit_completion, restore_completion, train


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def dataset_digest(path):
    if path.is_file():
        return checkpoints.digest(path)
    return checkpoints.object_digest({str(file.relative_to(path)): checkpoints.digest(file)
                                      for file in sorted(path.rglob("*")) if file.is_file()})


def discover_implementations(checkout):
    """Audit Python sources without requiring an external search executable."""
    pattern = re.compile(r"DDIMScheduler|DDPMScheduler|flow_matching|class .*EMA")
    paths = []
    for directory, directories, files in os.walk(checkout):
        directories[:] = [name for name in directories
                          if name not in {".git", "site-packages", ".native", "node_modules"}
                          and not name.startswith((".venv", ".uv-cache"))]
        for name in files:
            path = Path(directory) / name
            if path.suffix != ".py" or path.is_symlink():
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                paths.append(str(path))
    return sorted(paths)


def audit(root, dataset=None):
    checkout = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()
    versions = {}
    for package in ("torch", "diffusers", "gym-pusht", "pymunk", "numpy", "zarr"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    paths = discover_implementations(checkout)
    checkpoints_found = []
    for directory in ("workspace/checkpoints", "workspace/experiments", "workspace/runs"):
        for path in (checkout / directory).rglob("*.pt"):
            if any(word in str(path).lower() for word in ("pusht", "push_t", "diffusion")):
                checkpoints_found.append(str(path))
    report = dict(commit=git("rev-parse", "HEAD"), worktree=git("status", "--short"),
                  patch_sha256=checkpoints.object_digest(git("diff", "HEAD")),
                  source_files_sha256=checkpoints.runtime_identity()["source_files_sha256"],
                  lock_sha256=checkpoints.digest(checkout / "uv.lock"), versions=versions,
                  discovered_implementation_paths=paths,
                  discovered_checkpoint_paths=checkpoints_found,
                  reused=["action_bridge/models/diffusion_policy.py", "action_bridge/models/encoders.py",
                          "action_bridge/data/pusht_adapter.py", "action_bridge/eval/pusht_sim.py"],
                  gaps=["No compatible trained Push-T DDIM or standalone frozen reference was found in the initial audit"],
                  dataset_path=None if dataset is None else str(dataset.resolve()),
                  dataset_exists=dataset is not None and dataset.exists(),
                  methods=list(METHODS), completion_evaluations=["sb_kinetic/repeat", "sb_kinetic/fixed_damped"],
                  training_seed=0, status="implementation; no full-budget result implied")
    write_json(root / "audit.json", report)
    return report


def chosen(root, method):
    best = root / method / "best.pt"
    return best if best.exists() else root / method / "latest.pt"


def window_spec(config):
    # Noise settings are metadata, rather than random changes to stored labels.
    # Include them so that cached pixel-scale annotations cannot become stale.
    keys = ("horizon", "execute", "obs_history", "action_history", "source_std", "endpoint_std")
    return {key: config[key] for key in keys}


def source_spec(config):
    """Only source/cache semantics, not the learner's optimizer or method."""
    keys = ("obs_dim", "action_dim", "horizon", "execute", "obs_history", "action_history",
            "robot_dt", "prior_ridge", "max_rate", "mobility_smoothing",
            "proposal_seed", "proposals_per_history", "batch_size", "ot_block_size",
            "source_std", "endpoint_std")
    return {"format": "frozen_ddim_pairs_v1", **{key: config[key] for key in keys}}


def validate_sources(state, metadata, config):
    if state["metadata"] != metadata:
        raise ValueError("Source cache/proposal/reference mismatch")
    if state.get("source_spec") != source_spec(config):
        raise ValueError("Source cache configuration mismatch; use a new run root for changed source semantics")


def validate_dependencies(reference_state, proposal_state, metadata, config):
    """A checkpoint hash alone does not establish dataset/coordinate compatibility."""
    for name, state in (("Reference", reference_state), ("DDIM proposal", proposal_state)):
        if state["metadata"] != metadata:
            raise ValueError(f"{name} checkpoint belongs to another dataset, split or normalizer")
        for key in ("obs_dim", "action_dim", "horizon", "obs_history", "action_history"):
            if state["config"][key] != config[key]:
                raise ValueError(f"{name} checkpoint has incompatible {key}")
    if proposal_state["config"]["method"] != "ddim":
        raise ValueError("The frozen proposal must be a DDIM checkpoint")
    if reference_state["completion_config"]["robot_dt"] != config["robot_dt"]:
        raise ValueError("Reference checkpoint uses a different robot-index dt")


def validate_evaluation_sources(state, checkpoint_hash, sources, source_hash):
    """Keep the offline probe and every bootstrap tied to the trained source law."""
    method = state["config"]["method"]
    metadata = sources["metadata"]
    expected = (metadata | {"source_hash": source_hash}) if method != "ddim" else {
        key: value for key, value in metadata.items() if key not in {"reference_hash", "proposal_hash"}}
    if state["metadata"] != expected:
        raise ValueError("Evaluation checkpoint and source cache have different dataset/dependency provenance")
    validate_sources(sources, metadata, state["config"])
    if method == "ddim":
        if checkpoint_hash != metadata["proposal_hash"]:
            raise ValueError("DDIM evaluation must use the same checkpoint as the frozen bootstrap proposal")
    else:
        for dependency in ("proposal", "reference"):
            if state["dependencies"][dependency + "_sha256"] != metadata[dependency + "_hash"]:
                raise ValueError(f"Evaluation uses a different frozen {dependency}")
        if method.startswith("sb_") and state["direction"] != "forward":
            raise ValueError("Deploy a completed forward DSBM phase, not a reverse-only checkpoint")


def stage_tracking(stage, root, records, config, metadata, device, options, dependencies=None):
    """Keep tracking options out of checkpoint/cache compatibility checks."""
    options = options or TrackingOptions()
    tracker = Tracker(root / stage, config | {"stage": stage}, metadata, options,
                      group=root.name, name=f"{root.name}-{stage}")
    visualize = None
    if options.enabled and options.image_count:
        from action_bridge.eval.revision_pusht_plots import make_action_chunk_plotter
        visualize = make_action_chunk_plotter(records, config, metadata, root / stage, device,
                                              dependencies=dependencies, count=options.image_count,
                                              reference=stage == "reference")
    return tracker, visualize


def stage_evaluation(output, config, options):
    if options is False:
        return None
    from action_bridge.eval.revision_pusht_async import AsyncEvaluation

    def factory(on_result):
        return AsyncEvaluation(output, config, on_result, **(options or {}))
    return factory


def run_stage(stage, root, dataset, config, device, *, tracking=None, evaluation=None):
    root.mkdir(parents=True, exist_ok=True)
    if stage == "audit":
        return audit(root, dataset)
    if stage == "report":
        from action_bridge.plan_revision.reporting import report
        return report(root)
    if stage == "evaluate":
        from action_bridge.eval.revision_pusht import evaluate
        from action_bridge.plan_revision.reporting import revision_probe
        sources = checkpoints.load(root / "sources.pt")
        source_hash = checkpoints.digest(root / "sources.pt")
        for method in METHODS:
            if not checkpoints.load(root / method / "latest.pt").get("complete"):
                raise ValueError(f"Finish {method} training before the primary closed-loop comparison")
            checkpoint = chosen(root, method)
            state = checkpoints.load(checkpoint, device)
            checkpoint_hash = checkpoints.digest(checkpoint)
            validate_evaluation_sources(state, checkpoint_hash, sources, source_hash)
            policy = checkpoints.restore_policy(state, device)
            for mode in ([2, 1, 0] if method == "sb_kinetic" else [2]):
                output = root / "evaluation" / f"{method}-{COMPLETIONS[mode]}"
                result_path = output / "result.json"
                identity = {"checkpoint_sha256": checkpoint_hash, "source_sha256": source_hash,
                            "completion_id": mode, "seeds": config["evaluation_seeds"],
                            "normalizer_id": state["metadata"]["normalizer_id"],
                            "max_episode_steps": state["config"]["max_episode_steps"],
                            "execute": state["config"]["execute"]}
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if previous["identity"] != identity:
                        raise ValueError(f"Evaluation {output} belongs to another checkpoint")
                    continue
                metrics = evaluate(policy, state["config"], state["metadata"], state["dependencies"], device,
                                   output=output, completion_id=mode, seeds=config["evaluation_seeds"], render=True)
                probe = revision_probe(policy, sources["records"]["test"], sources["records"]["val"],
                                       state["config"], state["dependencies"] or sources["dependencies"], device, mode)
                write_json(result_path, dict(identity=identity, metrics=metrics, probe=probe,
                                             training_seconds=state["training_seconds"],
                                             cache_seconds=state["cache_seconds"], parameters=state["parameters"]))
        return run_stage("report", root, dataset, config, device)
    if dataset is None or not dataset.exists():
        raise FileNotFoundError("Supply --dataset with the real local Push-T replay dataset; the driver never substitutes toy data")
    windows_path = root / "windows.pt"
    data_spec = window_spec(config)
    if not windows_path.exists():
        audit_report = audit(root, dataset)
        windows, metadata = load_windows(dataset.resolve(), config)
        metadata.update(dataset_sha256=dataset_digest(dataset), policy_commit=audit_report["commit"],
                        lock_sha256=audit_report["lock_sha256"],
                        physics_versions={key: audit_report["versions"][key] for key in ("gym-pusht", "pymunk")})
        checkpoints.save(windows_path, {"records": windows, "metadata": metadata, "data_spec": data_spec})
    data = checkpoints.load(windows_path)
    if data["data_spec"] != data_spec or data["metadata"]["dataset_path"] != str(dataset.resolve()):
        raise ValueError("Window cache belongs to another dataset or shape configuration")
    if data["metadata"]["dataset_sha256"] != dataset_digest(dataset):
        raise ValueError("Dataset changed since window preparation")
    metadata, windows = data["metadata"], data["records"]
    if stage == "prepare":
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not checkpoints.same_training_config(manifest["config"], config) or manifest["metadata"] != metadata:
                raise ValueError("Existing experiment manifest has different configuration/provenance")
            manifest["config"] = config
            write_json(manifest_path, manifest)
        else:
            write_json(manifest_path, {"config": config, "metadata": metadata,
                       "jobs": {method: "pending" for method in METHODS},
                       "completion_evaluations": ["sb_kinetic/repeat", "sb_kinetic/fixed_damped"]})
        return metadata
    if stage == "reference":
        tracker, visualize = stage_tracking(stage, root, windows["val"], config, metadata, device, tracking)
        with tracker:
            return fit_completion(windows["train"], windows["val"], config,
                                  root / "reference", metadata, device, tracker=tracker, visualize=visualize)
    if stage == "ddim":
        tracker, visualize = stage_tracking(stage, root, windows["val"], config, metadata, device, tracking)
        with tracker:
            return train(windows["train"], config, root / "ddim", metadata, {}, device,
                         tracker=tracker, visualize=visualize,
                         evaluation_factory=stage_evaluation(root / stage, config, evaluation))
    reference_path = root / "reference" / "latest.pt"
    proposal_path = chosen(root, "ddim")
    reference_state = checkpoints.load(reference_path)
    if not reference_state.get("complete"):
        raise ValueError("Finish standalone reference pretraining before freezing it")
    if not checkpoints.load(root / "ddim" / "latest.pt").get("complete"):
        raise ValueError("Finish the DDIM baseline before freezing the proposal")
    proposal_state = checkpoints.load(proposal_path)
    validate_dependencies(reference_state, proposal_state, metadata, config)
    dependencies = {key: reference_state[key] for key in
                    ("completion_config", "completion_state", "innovation_variance")}
    dependencies.update(proposal_config=proposal_state["config"], proposal_state=proposal_state["ema"],
                        reference_sha256=checkpoints.digest(reference_path),
                        proposal_sha256=checkpoints.digest(proposal_path))
    source_metadata = metadata | {"reference_hash": dependencies["reference_sha256"],
                                  "proposal_hash": dependencies["proposal_sha256"]}
    source_path = root / "sources.pt"
    if stage == "sources":
        if source_path.exists():
            cached = checkpoints.load(source_path)
            validate_sources(cached, source_metadata, config)
            return cached
        start = time.perf_counter()
        completion = restore_completion(dependencies, device)
        proposal = checkpoints.restore_policy(proposal_state, device)
        records = {split: build_source_pairs(value, proposal, completion,
                                              dependencies["innovation_variance"], config, device)
                   for split, value in windows.items()}
        neighbors = neighbor_blocks(records["train"], metadata["normalization"], config["ot_block_size"])
        checkpoints.save(source_path, dict(records=records, metadata=source_metadata, dependencies=dependencies,
                                           neighborhood=neighbors, config=config,
                                           source_spec=source_spec(config),
                                           cache_seconds=time.perf_counter() - start))
        return {"source_hash": checkpoints.digest(source_path)}
    if stage not in METHODS:
        raise ValueError(f"Unknown stage: {stage}")
    sources = checkpoints.load(source_path)
    validate_sources(sources, source_metadata, config)
    metadata = source_metadata | {"source_hash": checkpoints.digest(source_path)}
    pairer = make_local_pairer(sources["records"]["train"], sources["neighborhood"], config) if stage == "fm_local_ot" else None
    tracker, visualize = stage_tracking(stage, root, sources["records"]["val"], config,
                                       metadata, device, tracking, dependencies)
    with tracker:
        return train(sources["records"]["train"], config, root / stage, metadata,
                     dependencies, device, pairer=pairer, tracker=tracker, visualize=visualize,
                     evaluation_factory=stage_evaluation(root / stage, config, evaluation))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "prepare", "reference", "sources", *METHODS, "evaluate", "report", "all"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, help="Optional JSON overrides, recorded in every checkpoint")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int,
                        help="Asynchronous closed-loop interval; overrides validation_every (default 10000)")
    parser.add_argument("--eval-device", default="cpu", help="Separate evaluator's policy device")
    parser.add_argument("--eval-threads", type=int, default=2)
    parser.add_argument("--sim-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-videos", action=argparse.BooleanOptionalAction, default=True,
                        help="Save selected evaluation MP4s locally, never upload them to W&B")
    parser.add_argument("--wandb", action="store_true", help="Log training metrics and action-chunk images")
    parser.add_argument("--wandb-project", default="action-bridge-policy")
    parser.add_argument("--wandb-entity", help="W&B user or team (defaults to your W&B settings)")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-images-every", type=int, default=5000,
                        help="Action-chunk image interval in optimizer updates (forward phases for SB)")
    parser.add_argument("--wandb-image-count", type=int, default=3, help="Fixed validation examples; 0 disables images")
    args = parser.parse_args(argv)
    if args.wandb_images_every < 1 or args.wandb_image_count < 0:
        parser.error("--wandb-images-every must be positive and --wandb-image-count nonnegative")
    if args.eval_threads < 1 or (args.eval_every is not None and args.eval_every < 1):
        parser.error("--eval-threads and --eval-every must be positive")
    evaluation = dict(device=args.eval_device, threads=args.eval_threads,
                      save_videos=args.eval_videos) if args.sim_eval else False
    tracking = TrackingOptions(enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
                               mode=args.wandb_mode, images_every=args.wandb_images_every,
                               image_count=args.wandb_image_count)
    torch.set_num_threads(args.threads)
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if args.eval_every is not None:
        overrides["validation_every"] = args.eval_every
    stages = ("prepare", "reference", "ddim", "sources", *METHODS[1:], "evaluate", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        config = get_config(stage if stage in METHODS else "ddim") | overrides
        config["method"] = stage if stage in METHODS else "ddim"
        output = args.run_root.resolve()
        print(f"Stage: {stage}; artifacts: {output}", flush=True)
        run_stage(stage, output, args.dataset, config, args.device, tracking=tracking, evaluation=evaluation)
        if (output / "manifest.json").exists():
            manifest = json.loads((output / "manifest.json").read_text())
            manifest.setdefault("stages", {})[stage] = "complete"
            if stage in METHODS:
                manifest["jobs"][stage] = "complete"
            write_json(output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
