"""Reload a checkpoint and measure ID/OOD behavior through the same simulator."""

import argparse
import json
from pathlib import Path

import torch

from .artifacts import new_run, sha256, write_json
from .data import load_episodes
from .env import Context, sample_context
from .evaluation import evaluate_policy
from .plotting import plot_comparison
from .train import load_checkpoint


SUITES = {
    "id": [0.],
    "unseen_orientation": [-65., -45., 45., 65.],
    "mass": [.5, 2.],
    "stiffness": [.5, 3.],
    "damping": [.5, 2.],
    "friction": [0., 3.],
    "gains": [.5, 2.],
    "normal_impulse_sweep": [-.3, -.15, 0., .15, .3],
}


def evaluate_checkpoint(checkpoint, output=None, *, suites=None, episodes=8,
                        device="cpu", seed=1000000, reference_only=False,
                        n_exec=None, gif=False, dataset=None, contexts=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    model, normalizer, payload = load_checkpoint(checkpoint, device, reference_only)
    config = payload["config"]
    execution_count = config["n_exec"] if n_exec is None else n_exec
    if not 1 <= execution_count <= config["horizon"]:
        raise ValueError("n_exec must be between one and the trained horizon")
    method = "reference_only" if reference_only else config["method"]
    output = Path(output) if output else new_run(f"eval-{method}")
    output.mkdir(parents=True, exist_ok=False)
    # All suites start from identical held-out context seeds. Changing physics
    # or orientation therefore does not secretly change goals/initial samples.
    if contexts is None:
        test = load_episodes(dataset or config["dataset"], "test")
        if episodes > len(test):
            raise ValueError(f"Dataset has {len(test)} ID test contexts; requested {episodes}")
        contexts = [Context.from_dict(ep["context"]) for ep in test[:episodes]]
    records = []
    for suite in suites or ["id"]:
        for value in SUITES[suite]:
            selected = contexts if suite in ("id", "normal_impulse_sweep") else [
                sample_context(context.seed, suite=suite, value=value) for context in contexts]
            impulse = value if suite == "normal_impulse_sweep" else 0.
            label = f"{suite}-{value:g}"
            summary = evaluate_policy(
                model, normalizer, selected, output / label,
                n_exec=execution_count, seed=seed, impulse=impulse,
                plots=True, label=method, gif=gif and suite == "id",
            )
            aggregate = {key: item for key, item in summary.items()
                         if isinstance(item, (int, float)) or item is None}
            record = dict(aggregate, method=method, train_episodes=config["train_episodes"],
                          seed=config["seed"], evaluation_seed=seed, suite=suite, value=value,
                          checkpoint_step=payload["step"], updates=config["steps"],
                          parameter_count=payload["provenance"]["parameter_count"],
                          denoising_steps=config["num_inference_steps"] if method.startswith("diffusion") else 0,
                          prediction_type=config.get("prediction_type", "epsilon") if method.startswith("diffusion") else None,
                          n_exec=execution_count, horizon=config["horizon"])
            record.update({key: payload["provenance"][key] for key in (
                "trainable_parameter_count", "reference_pretraining_updates", "reference_selected_step",
                "reference_pretraining_parameters", "frozen_reference_parameters") if key in payload["provenance"]})
            records.append(record)
            print(json.dumps({key: record.get(key) for key in (
                "method", "suite", "value", "num_episodes", "task_success_rate",
                "clean_success_rate", "peak_normal_force", "recovery_time")}))
    write_json(output / "comparison.json", records)
    write_json(output / "evaluation_config.json", dict(
        checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint),
        policy_config=config, normalizer=payload["normalizer"],
        suites=suites or ["id"], episodes=len(contexts), sampling_seed=seed,
        n_exec=execution_count, method=method, device=device,
        contexts=[context.to_dict() for context in contexts],
    ))
    plot_comparison(records, output / "figures")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path, help="Override the dataset path after copying a checkpoint")
    parser.add_argument("--suites", nargs="+", choices=[*SUITES, "all"], default=["id"])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-exec", type=int)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    suites = list(SUITES) if "all" in args.suites else args.suites
    evaluate_checkpoint(args.checkpoint, args.output, suites=suites, episodes=args.episodes,
                        device=args.device, seed=args.seed, reference_only=args.reference_only,
                        n_exec=args.n_exec, gif=args.gif, dataset=args.dataset)


if __name__ == "__main__":
    main()
