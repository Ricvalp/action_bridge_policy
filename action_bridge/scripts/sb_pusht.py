"""Resumable, five-job Push-T experiment. See docs/SB_PUSHT.md for commands."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

import torch

from action_bridge.configs.sb_pusht import COMPLETIONS, METHODS, PROTOCOL, get_config
from action_bridge.data.revision_pusht import load_windows, make_local_pairer, neighbor_blocks
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.tracking import Tracker, TrackingOptions
from action_bridge.plan_revision.training import fit_completion, train


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def record_stage_completion(root, stage):
    """Merge separate Slurm jobs' results without losing concurrent updates.

    Lock a sibling file, not the manifest itself: replacing the manifest changes
    its inode. Atomic replacement also keeps readers from seeing partial JSON.
    """
    import fcntl

    path = Path(root) / "manifest.json"
    if not path.exists():
        return
    with (path.parent / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(path.read_text())
        manifest.setdefault("stages", {})[stage] = "complete"
        if stage in METHODS:
            manifest["jobs"][stage] = "complete"
        temporary = path.with_suffix(".json.tmp")
        write_json(temporary, manifest)
        temporary.replace(path)


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
    for directory in ("workspace/checkpoints", "workspace/experiments", "workspace/runs", "workspace/sb_pusht"):
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
                  protocol=PROTOCOL,
                  artifact_reuse="No automatic reuse of legacy checkpoints; discovered paths are not compatibility checks.",
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
    keys = ("protocol", "horizon", "execute", "obs_history", "action_history", "source_std", "endpoint_std")
    return {key: config[key] for key in keys}


def validate_reference(reference_state, metadata, config):
    """A checkpoint hash alone does not establish dataset/coordinate compatibility."""
    if reference_state["metadata"] != metadata:
        raise ValueError("Reference checkpoint belongs to another dataset, split or normalizer")
    for key in ("obs_dim", "action_dim", "horizon", "obs_history", "action_history"):
        if reference_state["config"][key] != config[key]:
            raise ValueError(f"Reference checkpoint has incompatible {key}")
    if reference_state["completion_config"]["robot_dt"] != config["robot_dt"]:
        raise ValueError("Reference checkpoint uses a different robot-index dt")


def experiment_manifest(config):
    """The sole active comparison; no external proposal job or factorial sweep."""
    return {"protocol": PROTOCOL, "config": config,
            "jobs": {method: "pending" for method in METHODS},
            "auxiliary": {"reference": "shared frozen supervised fit"},
            "dependencies": {method: ([] if method == "ddim" else ["reference"]) for method in METHODS},
            "source_curriculum": config["source_self_probabilities"],
            "completion_evaluations": ["sb_kinetic/repeat", "sb_kinetic/fixed_damped"],
            "main_completion": "learned_dissipative", "artifact_reuse": []}


def validate_evaluation(state, metadata, config):
    if state["config"].get("protocol") != PROTOCOL:
        raise ValueError("Legacy results are not part of self_source_v1; use a new run root")
    if state["metadata"] != metadata:
        raise ValueError("Evaluation checkpoint belongs to another dataset or normalizer")
    if state["config"]["updates"] != config["updates"]:
        raise ValueError("Comparison checkpoints must have the configured optimizer budget")
    if "proposal_state" in state["dependencies"]:
        raise ValueError("External proposal checkpoints are forbidden in self_source_v1")
    if state["config"]["method"].startswith("sb_") and state["direction"] != "forward":
        raise ValueError("Deploy a forward DSBM checkpoint")


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
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Only self_source_v1 is active; legacy runs remain untouched")
    if stage == "sources":
        raise ValueError("The shared sources stage is retired; each reviser builds its own block sources")
    root.mkdir(parents=True, exist_ok=True)
    if stage == "audit":
        return audit(root, dataset)
    if stage == "report":
        from action_bridge.plan_revision.reporting import report
        return report(root)
    if stage == "evaluate":
        from action_bridge.eval.revision_pusht import evaluate
        from action_bridge.plan_revision.reporting import common_source_probe
        data = checkpoints.load(root / "windows.pt")
        policies, configs, reference_hash = {}, {}, None
        for method in METHODS:
            latest = checkpoints.load(root / method / "latest.pt")
            if not latest.get("complete"):
                raise ValueError(f"Finish {method} training before the primary closed-loop comparison")
            validate_evaluation(latest, data["metadata"], config)
            checkpoint = chosen(root, method)
            state = checkpoints.load(checkpoint, device)
            checkpoint_hash = checkpoints.digest(checkpoint)
            validate_evaluation(state, data["metadata"], config)
            policy = checkpoints.restore_policy(state, device)
            if method != "ddim":
                shared_hash = state["dependencies"]["reference_sha256"]
                if reference_hash is not None and shared_hash != reference_hash:
                    raise ValueError("All revisers must share the frozen reference")
                reference_hash = shared_hash
                dependencies = state["dependencies"]
                policies[method], configs[method] = policy, state["config"]
            for mode in ([2, 1, 0] if method == "sb_kinetic" else [2]):
                output = root / "evaluation" / f"{method}-{COMPLETIONS[mode]}"
                result_path = output / "result.json"
                identity = {"protocol": PROTOCOL, "checkpoint_sha256": checkpoint_hash,
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
                write_json(result_path, dict(identity=identity, metrics=metrics,
                                             checkpoint_step=state["step"],
                                             optimizer_updates=latest["step"],
                                             training_seconds=latest["training_seconds"],
                                             source_cache_seconds=latest.get("source_cache_seconds", 0.),
                                             cache_seconds=latest["cache_seconds"], parameters=state["parameters"]))
        common_source_probe(policies, {"validation": data["records"]["val"], "test": data["records"]["test"]},
                            configs, dependencies, device, root / "evaluation" / "common_source_probe",
                            count=config["probe_records_per_model"])
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
            write_json(manifest_path, experiment_manifest(config) | {"metadata": metadata})
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
    reference_state = checkpoints.load(reference_path)
    if not reference_state.get("complete"):
        raise ValueError("Finish standalone reference pretraining before freezing it")
    validate_reference(reference_state, metadata, config)
    dependencies = {key: reference_state[key] for key in
                    ("completion_config", "completion_state", "innovation_variance")}
    dependencies.update(reference_sha256=checkpoints.digest(reference_path),
                        reference_training_seconds=reference_state.get("training_seconds", 0.),
                        completion_encoder_spec=reference_state.get("completion_encoder_spec", {"kind": "HistoryEncoder"}))
    if stage not in METHODS:
        raise ValueError(f"Unknown stage: {stage}")
    def pairer_factory(records):
        neighborhood = neighbor_blocks(records, metadata["normalization"], config["ot_block_size"])
        return make_local_pairer(records, neighborhood, config)

    tracker, visualize = stage_tracking(stage, root, windows["val"], config,
                                       metadata, device, tracking, dependencies)
    with tracker:
        return train(windows["train"], config, root / stage, metadata,
                     dependencies, device, pairer_factory=pairer_factory if stage == "fm_local_ot" else None,
                     tracker=tracker, visualize=visualize,
                     evaluation_factory=stage_evaluation(root / stage, config, evaluation))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "prepare", "reference", *METHODS, "evaluate", "report", "all"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, help="Optional JSON overrides, recorded in every checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print the active manifest without training or writing files")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int,
                        help="Asynchronous closed-loop interval; overrides validation_every (default 10000)")
    parser.add_argument("--eval-device", default="cpu", help="Separate evaluator's policy device")
    parser.add_argument("--eval-threads", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int,
                        help="Training-time evaluation episodes, using consecutive seeds from the first "
                             "validation seed; default: the configured validation_seeds")
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
    if args.eval_episodes is not None and args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    evaluation = dict(device=args.eval_device, threads=args.eval_threads,
                      save_videos=args.eval_videos) if args.sim_eval else False
    if args.sim_eval and args.eval_episodes is not None:
        evaluation["episodes"] = args.eval_episodes
    tracking = TrackingOptions(enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
                               mode=args.wandb_mode, images_every=args.wandb_images_every,
                               image_count=args.wandb_image_count)
    torch.set_num_threads(args.threads)
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if args.eval_every is not None:
        overrides["validation_every"] = args.eval_every
    if args.dry_run:
        print(json.dumps(experiment_manifest(get_config() | overrides), indent=2))
        return 0
    stages = ("prepare", "reference", *METHODS, "evaluate", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        config = get_config(stage if stage in METHODS else "ddim") | overrides
        config["method"] = stage if stage in METHODS else "ddim"
        output = args.run_root.resolve()
        print(f"Stage: {stage}; artifacts: {output}", flush=True)
        run_stage(stage, output, args.dataset, config, args.device, tracking=tracking, evaluation=evaluation)
        record_stage_completion(output, stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
