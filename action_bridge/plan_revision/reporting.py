"""Offline revision diagnostics and compact, honest experiment tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.contracts import take
from action_bridge.plan_revision.training import restore_completion


@torch.no_grad()
def revision_probe(policy, heldout, validation, config, dependencies, device, mode=2, count=256):
    """One sampled prediction per fixed record; never select best-of-N plans."""
    completion = restore_completion(dependencies, device)
    generator = torch.Generator(device=device).manual_seed(12345)
    def source(batch):
        plan = complete_plan(batch["old_actions"], config["execute"], batch["obs_hist"],
                             batch["act_hist"], mode, completion, robot_dt=config["robot_dt"])
        return plan + config["source_std"] * torch.randn(plan.shape, device=device, generator=generator)
    val = take(validation, slice(0, count), device)
    threshold = float((source(val) - val["future_actions"]).square().mean((1, 2)).median())
    batch = take(heldout, slice(0, count), device)
    initial = source(batch)
    reference = reference_for(batch, config) if config["method"].startswith("sb_") else None
    final, _ = policy.sample(batch["obs_hist"], batch["act_hist"], mode,
                             source_actions=initial, reference=reference, generator=generator)
    before = (initial - batch["future_actions"]).square()
    after = (final - batch["future_actions"]).square()
    split = config["horizon"] - config["execute"]
    result = {"threshold_from_validation_mse": threshold, "records": len(initial),
              "coordinate_units": "normalized action units", "selection": "first fixed held-out records"}
    for name, selected in (("low", before.mean((1, 2)) <= threshold),
                           ("high", before.mean((1, 2)) > threshold)):
        row = {"count": int(selected.sum())}
        if bool(selected.any()):
            for part, index in (("overlap", slice(0, split)), ("tail", slice(split, None))):
                row[part + "_before_mse"] = float(before[selected, index].mean())
                row[part + "_after_mse"] = float(after[selected, index].mean())
            row["revision_rms"] = float((final[selected] - initial[selected]).square().mean().sqrt())
        result[name] = row
    return result


def report(root):
    root = Path(root)
    rows = []
    for path in sorted((root / "evaluation").glob("*/result.json")):
        item = json.loads(path.read_text())
        row = {"name": path.parent.name, "training_seconds": item["training_seconds"],
               "cache_seconds": item["cache_seconds"], "parameters": item["parameters"]}
        row.update({key: value for key, value in item["metrics"].items()
                    if isinstance(value, (float, int))})
        rows.append(row)
    if rows:
        columns = sorted(set().union(*(row.keys() for row in rows)))
        with (root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        figure, ax = plt.subplots(figsize=(9, 4))
        names = [row["name"] for row in rows]
        ax.bar(names, [row["success_rate"] for row in rows])
        ax.set(ylim=(0, 1), ylabel="Success fraction (one training seed)")
        ax.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        figure.savefig(root / "success.png")
        plt.close(figure)
    lines = ["# Whole-plan Push-T experiment", "",
             f"Completed evaluation rows: {len(rows)} / 7.", "",
             "One training seed; these results do not establish seed robustness or exact SB optimality.",
             "Source plans during deployment come from the reviser itself, unlike the frozen DDIM source cache.", ""]
    if not rows:
        lines.append("No substantive closed-loop comparison has been run. Unit tests are not experimental results.")
    else:
        lines += ["| Method / completion | Success | Max coverage | Final coverage |",
                  "| --- | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(f"| {row['name']} | {row['success_rate']:.3f} | "
                         f"{row.get('max_coverage', float('nan')):.3f} | {row.get('final_coverage', float('nan')):.3f} |")
        lines += ["", "Full diagnostics are in `results.csv`; fixed-record low/high-error probes and video paths are in each evaluation directory."]
    for stage in ("reference", "ddim", "fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"):
        path = root / stage / "latest.pt"
        if path.exists():
            from action_bridge.plan_revision.checkpoints import load
            state = load(path)
            lines.append(f"\n{stage}: {state['step']} updates; complete={state.get('complete', False)}; "
                         f"reported training time {state.get('training_seconds', 0):.1f}s.")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return rows
