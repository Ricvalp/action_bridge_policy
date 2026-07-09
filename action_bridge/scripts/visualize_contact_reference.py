"""Visualize learned contact-Langevin damping, potential, and controls."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

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
from action_bridge.training.common import build_dataset, build_model, resolve_device, save_json, seed_everything


def representative_indices(dataset, count: int) -> List[int]:
    count = min(count, len(dataset))
    if count <= 0:
        return []
    selected: List[int] = []
    seen_modes = set()
    for idx in range(len(dataset)):
        item = dataset[idx]
        mode = item.get("mode_sign")
        if mode is None:
            continue
        mode_value = int(mode)
        if mode_value in seen_modes:
            continue
        selected.append(idx)
        seen_modes.add(mode_value)
        if len(selected) >= count:
            return selected
    if len(selected) < count:
        positions = torch.linspace(0, max(0, len(dataset) - 1), steps=count + 2).long().tolist()[1:-1]
        for idx in positions:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= count:
                break
    idx = 0
    while len(selected) < count:
        if idx not in selected:
            selected.append(idx)
        idx += 1
    return selected


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
    indices = representative_indices(dataset, args.num_examples)
    subset = Subset(dataset, indices)
    batch = next(iter(DataLoader(subset, batch_size=len(indices), shuffle=False)))

    diagnostics = collect_contact_diagnostics(model, batch, device)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_contact_reference_summary(diagnostics, output_dir / "contact_reference_summary.png", example_idx=args.example_idx)
    plot_contact_parameter_heatmaps(diagnostics, output_dir / "contact_parameter_heatmaps.png")
    plot_contact_potential_contours(diagnostics, output_dir / "contact_potential_contours.png", example_idx=args.example_idx)

    save_json(
        output_dir / "contact_diagnostics_summary.json",
        {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "indices": indices,
            "coordinate_mode": diagnostics["coordinate_mode"],
            "summary": contact_summary_stats(diagnostics),
        },
    )
    save_config(config, output_dir / "contact_diagnostics_config.json")
    print(f"Contact diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
