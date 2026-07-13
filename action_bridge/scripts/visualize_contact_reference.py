"""Visualize learned contact-Langevin damping, potential, and controls."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from action_bridge.config import apply_overrides, save_config
from action_bridge.eval.contact_visualization import (
    collect_contact_diagnostics,
    contact_summary_stats,
    plot_contact_parameter_heatmaps,
    plot_contact_potential_contours,
    plot_contact_reference_summary,
)
from action_bridge.eval.selection import history_index_metadata, representative_history_indices
from action_bridge.training.common import build_dataset, build_model, resolve_device, save_json, seed_everything


def parse_fraction_list(raw: str | None):
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def default_output_dir(checkpoint: Path) -> Path:
    try:
        run_dir = checkpoint.parents[1]
        if checkpoint.parent.name == "checkpoints":
            return run_dir / "figures" / "contact_reference"
    except IndexError:
        pass
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "eval" / f"contact_reference_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--time-fractions", default="0.2,0.5,0.8")
    parser.add_argument("--trajectory-fractions", default=None)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    seed_everything(args.seed)
    checkpoint = Path(args.checkpoint)
    raw = torch.load(checkpoint, map_location="cpu")
    config = apply_overrides(raw["config"], args.overrides)
    if args.device is not None:
        config.device = args.device
    device = resolve_device(str(config.get("device", "cpu")))

    model = build_model(config).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()
    if not bool(getattr(model, "uses_contact_langevin", False)):
        raise SystemExit("This checkpoint does not use reference.type=contact_langevin.")

    dataset = build_dataset(config, split=args.split)
    if args.dataset_index is not None:
        indices = [int(args.dataset_index)]
    else:
        indices = representative_history_indices(
            dataset,
            args.num_examples,
            time_fractions=parse_fraction_list(args.time_fractions) or [0.5],
            trajectory_fractions=parse_fraction_list(args.trajectory_fractions),
        )
    subset = Subset(dataset, indices)
    batch = next(iter(DataLoader(subset, batch_size=len(indices), shuffle=False)))

    diagnostics = collect_contact_diagnostics(model, batch, device, config=config)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_contact_reference_summary(diagnostics, output_dir / "contact_reference_summary.png", example_idx=args.example_idx)
    plot_contact_parameter_heatmaps(diagnostics, output_dir / "contact_parameter_heatmaps.png")
    plot_contact_potential_contours(
        diagnostics,
        output_dir / "contact_potential_contours.png",
        example_idx=args.example_idx,
        grid_size=args.grid_size,
    )

    save_json(
        output_dir / "contact_diagnostics_summary.json",
        {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "indices": indices,
            "selected_histories": history_index_metadata(dataset, indices),
            "time_fractions": parse_fraction_list(args.time_fractions),
            "trajectory_fractions": parse_fraction_list(args.trajectory_fractions),
            "coordinate_mode": diagnostics["coordinate_mode"],
            "grid_size": args.grid_size,
            "summary": contact_summary_stats(diagnostics),
        },
    )
    save_config(config, output_dir / "contact_diagnostics_config.json")
    print(f"Contact diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
