"""Fixed synthetic Push-T multimodality probe for training (no physics rollout).

The old plan retreats along the scene's symmetry axis. SB starts from its clean,
fixed-damped completion; DDIM uses the same observed history but no source plan.
The fixed deployment K and sampling seed make successive training images useful
comparisons even while the curriculum's training K changes.
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from action_bridge.data.pusht_adapter import normalize_observations_np
from action_bridge.eval.revision_pusht_candidate_plots import _limits, _set_axes
from action_bridge.eval.revision_pusht_candidates import (
    candidate_statistics, sample_candidate_revisions, symmetric_context,
)
from action_bridge.eval.visualization import _draw_tee, _import_pyplot, _tee_polygons
from action_bridge.plan_revision.checkpoints import frozen_copy
from action_bridge.plan_revision.contracts import ActionCodec
from action_bridge.plan_revision.tracking import preserve_rng


def _sample_ddim(model, config, metadata, context, device, *, candidates, batch_size, seed):
    """Generate genuine DDIM draws, without passing a fictitious source chunk."""
    sampler = frozen_copy(model).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    codec = ActionCodec(**metadata["codec"])
    obs = torch.as_tensor(normalize_observations_np(
        context["obs_history_raw"], metadata["normalization"]), device=device)[None]
    actions = codec.encode(torch.as_tensor(context["action_history_raw"], device=device))[None]
    plans = []
    for start in range(0, candidates, batch_size):
        count = min(batch_size, candidates - start)
        generated, _ = sampler.sample(obs.expand(count, -1, -1), actions.expand(count, -1, -1),
                                      generator=generator)
        if generated.shape != (count, config["horizon"], 2) or not torch.isfinite(generated).all():
            raise ValueError("DDIM probe returned nonfinite or incorrectly shaped command targets")
        plans.append(codec.decode(generated).cpu().numpy())
    # DDIM has no old/source plan: keep only its actual context and scene geometry.
    source_fields = {"old_plan_raw", "completed_raw", "fixed_source_raw", "executed", "has_previous_plan"}
    return {key: value for key, value in context.items() if key not in source_fields} | {
        "candidates_raw": np.concatenate(plans), "execute": config["execute"],
        "source_noise_mode": "not_applicable",
    }


def _render(data, output, *, method, step):
    """Overlay all unranked samples and plot signed offsets from the scene axis."""
    from matplotlib.collections import LineCollection

    plt = _import_pyplot()
    plans = data["candidates_raw"]
    count, horizon, _ = plans.shape
    execute = int(data["execute"])
    state, goal = data["state_raw"], data["goal_pose_raw"]
    source = data.get("fixed_source_raw")
    source_points = [] if source is None else [source]
    limits = _limits([plans, state[:2], *_tee_polygons(state[2:]), *_tee_polygons(goal),
                      *source_points], workspace=True)
    focus = _limits([plans[:, :execute], state[:2], *source_points])
    title = f"{method} | step {step} | {count} draws | H={horizon}, fixed deployment K={execute}"
    note = "Synthetic symmetric scene; command targets, NOT simulated paths or success. No candidate selection."
    source_label = ("Clean axial source; fixed-damped completion (ID 1), no source noise"
                    if source is not None else "DDIM: conditioned on histories only; no source chunk")
    paths = [output / "symmetric_overlay.png", output / "symmetric_lateral.png"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    try:
        for ax, length, bounds, heading in zip(axes, (horizon, execute), (limits, focus),
                                              ("All generated commands", f"First K={execute} commands (zoom)")):
            _draw_tee(ax, goal, "tab:green", .35, label="Goal T", linestyle="--")
            _draw_tee(ax, state[2:], ".45", .3, label="Current T")
            ax.axline(goal[:2], goal[:2] + data["symmetry_axis_raw"], color="tab:brown",
                      linestyle="--", linewidth=1., label="Symmetry axis")
            ax.scatter(*state[:2], s=65, color="tab:cyan", edgecolor="black", zorder=6, label="Pusher")
            if source is not None:
                retained = data["old_plan_raw"][int(data["executed"]):]
                ax.plot(*retained.T, "o--", color="black", markersize=3, linewidth=1.5,
                        label="Unexecuted previous chunk", zorder=5)
                tail = source[len(retained) - 1:]
                ax.plot(*tail.T, "s--", color="tab:purple", markersize=3, linewidth=1.3,
                        label="Fixed-damped completion tail", zorder=5)
            ax.add_collection(LineCollection(plans[:, :length], colors="tab:blue", linewidths=.65,
                                            alpha=max(.015, min(.4, 4 / count)), label="Generated chunks"))
            marker_alpha = max(.04, min(.6, 8 / count))
            ax.scatter(*plans[:, execute - 1].T, s=12, marker="D", color="tab:orange",
                       alpha=marker_alpha, zorder=4, label=f"Kth command ({execute})")
            if length == horizon:
                ax.scatter(*plans[:, -1].T, s=14, marker="x", color="tab:red",
                           alpha=marker_alpha, zorder=4, label=f"Last command ({horizon})")
            _set_axes(ax, bounds, heading)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .07),
                            ncol=4, fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.)
        fig.suptitle(title + "\n" + source_label, fontsize=11)
        fig.text(.5, .025, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .21, 1, .92))
        fig.savefig(paths[0], dpi=140)
    finally:
        plt.close(fig)

    offsets = (plans[:, [execute - 1, horizon - 1]] - goal[:2]) @ data["lateral_axis_raw"]
    radius = max(float(np.abs(offsets).max()), 1.) * 1.05
    bins = np.linspace(-radius, radius, 51)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    try:
        for index, (ax, name, color) in enumerate(zip(
                axes, (f"Kth command ({execute})", f"Last command ({horizon})"), ("tab:orange", "tab:red"))):
            ax.hist(offsets[:, index], bins=bins, color=color, alpha=.8)
            ax.axvline(0., color="tab:brown", linestyle="--")
            ax.set(title=name, xlabel="Signed offset from symmetry axis (pixels)", ylabel="Draws")
        fig.suptitle(title + "\nLateral command distribution", fontsize=11)
        fig.text(.5, .025, "Negative / positive are the two sides of the scene axis, not screen-left / right.\n"
                 "Side occupancy alone establishes neither multimodality nor task success.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .9))
        fig.savefig(paths[1], dpi=140)
    finally:
        plt.close(fig)
    return paths


def make_symmetry_plotter(config, metadata, output, device, *, dependencies=None,
                         candidates=1000, batch_size=64, seed=42):
    """Return ``plot(model, step) -> [overlay PNG, lateral histogram PNG]``.

    Disabled for zero samples, non-Push-T profiles, unsupported methods, H=K,
    SB without a reference, or SB not trained with completion mode 1. The latter
    avoids inventing an unseen conditioning value merely to preserve symmetry.
    Fixed damping is a diagnostic intervention, not the deployment completion.
    ``config`` must retain deployment ``execute``, not a curriculum-stage override.
    Sampling never executes commands and does not change the trainer's model/RNG.
    """
    if candidates < 0 or batch_size < 1 or not 0 <= seed < 2**63:
        raise ValueError("Need nonnegative candidates, positive batch_size, and a 63-bit nonnegative seed")
    method = config.get("method")
    codec = metadata.get("codec", {})
    if (not candidates or method not in ("sb_ou", "sb_kinetic", "ddim")
            or config.get("obs_dim") != 5 or config.get("action_dim") != 2
            or codec.get("units") != "pixels"
            or codec.get("semantics", "absolute_target") != "absolute_target"):
        return None
    if not 0 < config.get("execute", 0) < config.get("horizon", 0):
        warnings.warn("Skipping symmetric previous-chunk probe: requires 0 < deployment K < H.", stacklevel=2)
        return None
    if method != "ddim" and (dependencies is None or 1 not in config.get("training_completion_modes", (0, 1, 2))):
        warnings.warn("Skipping symmetric SB probe: requires a reference and trained fixed_damped mode (ID 1).",
                      stacklevel=2)
        return None
    # Capture the deployment context once; advancing the curriculum must not
    # silently move this probe or alter its histories, source, K, or random seed.
    config, metadata = copy.deepcopy(config), copy.deepcopy(metadata)
    with preserve_rng():
        context = symmetric_context(config, metadata, completion_id=1)
    output = Path(output)

    @torch.no_grad()
    def plot(model, step):
        with preserve_rng():
            if method == "ddim":
                data = _sample_ddim(model, config, metadata, context, device, candidates=candidates,
                                    batch_size=batch_size, seed=seed)
            else:
                data = sample_candidate_revisions(
                    model, config, metadata, dependencies, context, device, candidates=candidates,
                    batch_size=batch_size, seed=seed, completion_id=1, source_noise="none")
            folder = output / "symmetric_probe" / f"step_{step:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            plans = data["candidates_raw"]
            arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
            arrays["lateral_offsets_px"] = (
                plans[:, [config["execute"] - 1, config["horizon"] - 1]] - context["goal_pose_raw"][:2]
            ) @ context["lateral_axis_raw"]
            np.savez_compressed(folder / "samples.npz", **arrays)
            report = {
                "step": int(step), "method": method, "protocol": config.get("protocol"),
                "candidates": candidates, "batch_size": batch_size, "sampling_seed": seed,
                "horizon": config["horizon"], "deployment_k": config["execute"],
                "num_inference_steps": config["num_inference_steps"],
                "condition_on_k": method != "ddim" and config.get("condition_on_k", False),
                "completion_id": 1 if method != "ddim" else None,
                "completion": "fixed_damped" if method != "ddim" else None,
                "source_noise": "none" if method != "ddim" else "not_applicable",
                "generation_randomness": ("DDIM initial Gaussian noise; deterministic eta=0 denoising"
                    if method == "ddim" else "SB dynamics noise plus initial auxiliary velocity"
                    if method == "sb_kinetic" else "SB dynamics noise"),
                "action_units": "pixels", "command_semantics": "absolute_target",
                "scene": "synthetic_symmetric", "simulated_candidates": False,
                "history": context["synthetic_history"],
                "statistics": candidate_statistics(data, config["execute"]),
                "interpretation": "Command diversity, not physical paths, a mode-count estimate, or success.",
                "note": "Source is a clean symmetric fixed-damped intervention, not learned completion."
                        if method != "ddim" else "DDIM uses observed histories only, with no source plan.",
            }
            (folder / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
            return _render(data, folder, method=method, step=int(step))

    return plot
