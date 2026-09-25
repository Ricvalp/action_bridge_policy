"""Offline revision diagnostics and compact, honest experiment tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.contracts import take, tree_map
from action_bridge.plan_revision.data import build_self_sources, immutable_snapshot
from action_bridge.plan_revision.training import restore_completion


REVISERS = ("fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic")


def _probe_windows(windows, count):
    """Fixed early histories with their complete recorded replay prefixes."""
    indices, ordinary = [], 0
    for episode in windows["episode_id"].unique(sorted=True):
        rows = (windows["episode_id"] == episode).nonzero().flatten()
        rows = rows[windows["time_index"][rows].argsort()]
        for offset, index in enumerate(rows.tolist()):
            indices.append(index)
            ordinary += int(offset > 0)
            if ordinary >= count:
                return take(windows, torch.tensor(indices))
    if ordinary < count:
        raise ValueError(f"Common-source probe needs {count} ordinary histories; found {ordinary}")


def _concatenate(batches):
    return {key: (_concatenate([batch[key] for batch in batches])
                  if isinstance(batches[0][key], dict)
                  else torch.cat([batch[key] for batch in batches]))
            for key in batches[0] if isinstance(batches[0][key], (torch.Tensor, dict))}


def _probe_metrics(initial, final, target, threshold, split):
    before = (initial - target).square()
    after = (final - target).square()
    result = {}
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


def _build_probe_pools(policies, windows_by_split, configs, dependencies, completion, device, count):
    pools, replay_info = {}, {}
    for split, seed in (("validation", 73001), ("test", 73002)):
        windows = _probe_windows(windows_by_split[split], count)
        batches, expected_keys = [], None
        for origin, method in enumerate(REVISERS):
            records, info = build_self_sources(
                windows, immutable_snapshot(policies[method]), completion,
                torch.as_tensor(dependencies["innovation_variance"], device=device),
                configs[method], device, block=3, seed=seed + origin, p_self=1.,
                modes=(configs[method].get("completion_id", 2),))
            indices = records["has_previous_plan"].nonzero().flatten()[:count]
            if len(indices) != count:
                raise ValueError("Insufficient non-startup proposals for an equal common-source pool")
            records = take(records, indices)
            keys = torch.stack([records["episode_id"], records["time_index"]], dim=1)
            if expected_keys is not None and not torch.equal(keys, expected_keys):
                raise ValueError("Probe origins do not share the same recorded history keys")
            expected_keys = keys
            records["origin_model_id"] = torch.full((count,), origin, dtype=torch.long)
            batches.append(records)
            replay_info[f"{split}/{method}"] = info
        pools[split] = _concatenate(batches)
    return pools, replay_info


@torch.no_grad()
def common_source_probe(policies, windows_by_split, configs, dependencies, device, output, *, count=64):
    """Freeze an equally weighted pool, then revise identical conditioned sources.

    Each of the four origins supplies ``count`` ordinary previous plans at the
    same held-out history keys. No proposal is ever moved to another context.
    Only validation labels determine the low/high error threshold.
    """
    if set(policies) != set(REVISERS) or set(configs) != set(REVISERS):
        raise ValueError("The common-source diagnostic requires the four trained revisers")
    if count < 1:
        raise ValueError("Probe count must be positive")
    for config in configs.values():
        if config.get("protocol") != "self_source_v1":
            raise ValueError("Legacy source laws cannot enter the replacement probe")
    if len({config.get("completion_id", 2) for config in configs.values()}) != 1:
        raise ValueError("A common-source probe requires the same completion mode across revisers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(config["execute"] == config["horizon"] for config in configs.values()):
        result = {"protocol": "self_source_v1", "status": "not_applicable",
                  "reason": "K=H exhausts every plan; there are no retained old-plan overlaps to probe."}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    completion = restore_completion(dependencies, device)
    provenance = {"snapshot_sha256": {method: checkpoints.content_digest(model.state_dict())
                                      for method, model in policies.items()},
                  "shared_components_sha256": checkpoints.content_digest(dependencies),
                  "logged_windows_sha256": checkpoints.content_digest(windows_by_split),
                  "sampler_configs": configs}
    # This file is written before querying any reviser for diagnostic outputs.
    # It contains completed, already perturbed sources: no resampling per model.
    artifact = output / "common_sources.pt"
    if artifact.exists():
        saved = torch.load(artifact, map_location="cpu", weights_only=False)
        if (saved["provenance"] != provenance or saved["count_per_origin"] != count
                or saved["protocol"] != "self_source_v1"):
            raise ValueError("Existing frozen common-source probe has different data, models or settings")
        pools, threshold = saved["pools"], saved["threshold_from_validation_mse"]
    else:
        pools, replay_info = _build_probe_pools(policies, windows_by_split, configs, dependencies,
                                               completion, device, count)
        validation = pools["validation"]
        threshold = float((validation["source_actions"] - validation["future_actions"]).square().mean((1, 2)).median())
        torch.save({"protocol": "self_source_v1", "pools": tree_map(lambda x: x.detach().cpu(), pools),
                    "origin_methods": REVISERS, "count_per_origin": count,
                    "threshold_from_validation_mse": threshold, "replay": replay_info,
                    "provenance": provenance}, artifact)
    batch = take(pools["test"], slice(None), device)
    result = {"protocol": "self_source_v1", "source_artifact": artifact.name,
              "source_sha256": checkpoints.digest(artifact), "records": len(batch["source_actions"]),
              "count_per_origin": count, "origin_methods": list(REVISERS),
              "threshold_from_validation_mse": threshold, "coordinate_units": "normalized action units",
              "selection": "same fixed held-out histories, equal self-only proposals per trained reviser",
              "limitation": "One logged expert future is not the only valid behavior; closed-loop success is primary.",
              "methods": {}}
    for method in REVISERS:
        config = configs[method]
        model = policies[method].eval()
        reference = reference_for(batch, config) if method.startswith("sb_") else None
        final, _ = model.sample(batch["obs_hist"], batch["act_hist"], batch["completion_id"],
            source_actions=batch["source_actions"], reference=reference,
            generator=torch.Generator(device=device).manual_seed(73003),
            has_previous_plan=batch["has_previous_plan"])
        if not bool(final.isfinite().all()):
            raise ValueError(f"Nonfinite common-source predictions from {method}")
        result["methods"][method] = _probe_metrics(batch["source_actions"], final,
            batch["future_actions"], threshold, config["horizon"] - config["execute"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(root):
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    main_mode = manifest.get("config", {}).get("completion_id", 2)
    rows = []
    for path in sorted((root / "evaluation").glob("*/result.json")):
        if path.parent.name == "common_source_probe":
            continue
        item = json.loads(path.read_text())
        protocol = item.get("protocol", item.get("identity", {}).get("protocol",
                            item.get("metrics", {}).get("protocol")))
        if protocol != "self_source_v1":
            continue
        row = {"name": path.parent.name, "training_seconds": item["training_seconds"],
               "cache_seconds": item["cache_seconds"], "parameters": item["parameters"],
               "source_cache_seconds": item.get("source_cache_seconds", 0.),
               "optimizer_updates": item.get("optimizer_updates"),
               "checkpoint_step": item.get("checkpoint_step")}
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
    lines = ["# Whole-plan Push-T experiment — self_source_v1", "",
             f"Completed evaluation rows: {len(rows)}.", "",
             "One training seed; these results do not establish seed robustness or exact SB optimality.",
             "Each reviser learns from its own frozen EMA replay with early expert-old-plan mixing; "
             "the final half uses self-only plans on logged states, not on-policy state training.",
             "Changing source boundaries between SB rounds fits successive bridge problems; "
             "fixed-boundary convergence guarantees do not automatically apply.", ""]
    if not rows:
        lines.append("No substantive closed-loop comparison has been run. Unit tests are not experimental results.")
    else:
        if any(row.get("optimizer_updates") not in (None, 300_000) for row in rows):
            lines += ["**Reduced-budget results: not the prescribed 300k-update comparison.**", ""]
        def table(title, selected):
            lines.extend([f"## {title}", "", "| Method / completion | Success | Max coverage | Final coverage |",
                          "| --- | ---: | ---: | ---: |"])
            for row in selected:
                label = "ddim (completion N/A)" if row["name"].startswith("ddim-") else row["name"]
                lines.append(f"| {label} | {row['success_rate']:.3f} | "
                             f"{row.get('max_coverage', float('nan')):.3f} | {row.get('final_coverage', float('nan')):.3f} |")
            lines.append("")
        table("Main comparison", [row for row in rows if not row["name"].startswith("sb_kinetic")
                                 or row.get("completion_id") == main_mode])
        table("Same kinetic checkpoint, three completion modes", [row for row in rows
                                                                 if row["name"].startswith("sb_kinetic")])
        lines += ["Full diagnostics are in `results.csv`; rollout overlays are saved with each evaluation.",
                  "The shared low/high-error revision diagnostic is in `evaluation/common_source_probe/result.json`. "
                  "Its frozen proposal pool is identical across revisers; logged MSE is not a unique test of valid behavior."]
    for stage in ("reference", "direct_tail", "ddim", "fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"):
        path = root / stage / "latest.pt"
        if path.exists():
            from action_bridge.plan_revision.checkpoints import load
            state = load(path)
            if state["config"].get("protocol") != "self_source_v1":
                continue
            lines.append(f"\n{stage}: {state['step']} updates; complete={state.get('complete', False)}; "
                         f"reported training time {state.get('training_seconds', 0):.1f}s.")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return rows
