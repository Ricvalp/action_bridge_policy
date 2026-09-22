"""Matched seeds/data/updates, validation-only selection, then held-out evaluation."""

import argparse
import json
from pathlib import Path

from .artifacts import new_run, write_json
from .config import METHODS, method_config
from .evaluate import SUITES, evaluate_checkpoint
from .plotting import plot_comparison
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--n-exec", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--grid", type=Path, help="JSON mapping method to a list of config overrides; selected ONLY by val MSE")
    args = parser.parse_args()
    output = args.output or new_run("sweep")
    output.mkdir(parents=True, exist_ok=False)
    grid = json.loads(args.grid.read_text()) if args.grid else {}
    shared_fields = {"horizon", "n_exec", "obs_history", "action_history", "batch_size",
                     "seed", "steps", "train_episodes", "method", "dataset", "val_windows"}
    for candidates in grid.values():
        if not candidates:
            raise ValueError("Each grid needs at least one candidate")
        for candidate in candidates:
            if shared_fields.intersection(candidate):
                raise ValueError("Grid overrides must not change shared data, histories, horizon or execution/training budgets")
    methods = list(args.methods)
    full = "action_bridge_contact_frame_full"
    if any(method in methods for method in ("diffusion_residual", "reference_only")) and full not in methods:
        raise ValueError("Include action_bridge_contact_frame_full to supply the matched reference")
    # The exact selected full reference is reused by BOTH reference-only and
    # residual diffusion. Its extra training cost is recorded, not hidden.
    methods.sort(key=lambda method: method != full)
    results, selections = [], []
    for size in args.train_sizes:
        for seed in args.seeds:
            reference_checkpoint = None
            for method in methods:
                name = f"{method}-n{size}-seed{seed}"
                if method == "reference_only":
                    checkpoint = reference_checkpoint
                else:
                    candidates = []
                    for index, overrides in enumerate(grid.get(method, [{}])):
                        config = method_config(method) | overrides | dict(
                            steps=args.steps, train_episodes=size, seed=seed,
                            horizon=args.horizon, n_exec=args.n_exec, batch_size=args.batch_size,
                            device=args.device, threads=args.threads)
                        run, summary = train(config, args.dataset, output / f"{name}-candidate{index}",
                                             reference_checkpoint=reference_checkpoint)
                        candidates.append((summary["best_val_action_mse"], run))
                    score, chosen = min(candidates, key=lambda item: item[0])
                    checkpoint = chosen / "best.pt"
                    selections.append(dict(method=method, train_episodes=size, seed=seed,
                                           val_action_mse=score, checkpoint=str(checkpoint),
                                           candidates=len(candidates), updates_per_candidate=args.steps))
                    if method == full:
                        reference_checkpoint = checkpoint
                records = evaluate_checkpoint(checkpoint, output / f"{name}-eval",
                    suites=args.suites, episodes=args.eval_episodes, device=args.device,
                    reference_only=method == "reference_only")
                results.extend(records)
                write_json(output / "comparison.json", results)
                write_json(output / "selection.json", selections)
                plot_comparison(results, output / "figures")
    print(f"Comparison and figures: {output}")


if __name__ == "__main__":
    main()
