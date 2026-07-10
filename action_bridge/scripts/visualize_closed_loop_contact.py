"""Visualize learned contact potential along a closed-loop toy rollout."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from action_bridge.config import apply_overrides, save_config
from action_bridge.eval.contact_visualization import (
    collect_closed_loop_contact_diagnostics,
    plot_closed_loop_contact_potential,
)
from action_bridge.training.common import build_dataset, build_model, resolve_device, save_json, seed_everything


def default_output_dir(checkpoint: Path) -> Path:
    try:
        run_dir = checkpoint.parents[1]
        if checkpoint.parent.name == "checkpoints":
            return run_dir / "figures" / "closed_loop_contact"
    except IndexError:
        pass
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "eval" / f"closed_loop_contact_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trajectory-id", type=int, default=None)
    parser.add_argument("--trajectory-fraction", type=float, default=0.5)
    parser.add_argument("--potential-step", type=int, default=None)
    parser.add_argument("--reference-steps", type=int, default=None)
    parser.add_argument("--reference-time-mode", choices=["hold_last", "extrapolate"], default="hold_last")
    parser.add_argument("--max-panels", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=90)
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
    diagnostics = collect_closed_loop_contact_diagnostics(
        model,
        dataset,
        config,
        device,
        trajectory_id=args.trajectory_id,
        trajectory_fraction=args.trajectory_fraction,
        potential_step=args.potential_step,
        reference_steps=args.reference_steps,
        reference_time_mode=args.reference_time_mode,
    )

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_closed_loop_contact_potential(
        diagnostics,
        output_dir / "closed_loop_contact_potential.png",
        max_panels=args.max_panels,
        grid_size=args.grid_size,
    )
    save_json(
        output_dir / "closed_loop_contact_diagnostics.json",
        {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "trajectory_id": diagnostics["trajectory_id"],
            "warm_start": diagnostics["warm_start"],
            "n_exec": diagnostics["n_exec"],
            "potential_step": diagnostics["potential_step"],
            "reference_steps": diagnostics["reference_steps"],
            "reference_time_mode": diagnostics["reference_time_mode"],
            "num_replans": len(diagnostics["records"]),
            "records": [
                {
                    "replan_index": item["replan_index"],
                    "trajectory_time": item["trajectory_time"],
                    "local_step": item["local_step"],
                    "m": item["m"].tolist(),
                    "k_diag": item["k_diag"].tolist(),
                    "gamma": item["gamma"].reshape(-1).tolist(),
                    "reference_no_control_final_position": item["reference_no_control_positions"][-1].tolist(),
                }
                for item in diagnostics["records"]
            ],
        },
    )
    save_config(config, output_dir / "closed_loop_contact_config.json")
    print(f"Closed-loop contact diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
