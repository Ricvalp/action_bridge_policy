"""Visualize source interventions with fixed conditioning and matched sampler noise."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from action_bridge.eval.revision_pusht_candidate_plots import _limits, _set_axes
from action_bridge.eval.visualization import _draw_tee, _import_pyplot, _tee_polygons


FAMILY_LABELS = {
    "translate_x": "Translate x", "translate_y": "Translate y",
    "bow_x": "Bend x", "bow_y": "Bend y",
}


def _validate(data):
    sources = np.asarray(data["sources_raw"])
    chunks = np.asarray(data["candidates_raw"])
    traces = np.asarray(data["revision_states_raw"])
    times = np.asarray(data["revision_times"])
    if sources.ndim != 3 or sources.shape[-1] != 2:
        raise ValueError("sources_raw must have shape (variants, H, 2)")
    variants, horizon, _ = sources.shape
    if chunks.ndim != 4 or chunks.shape[0] != variants or chunks.shape[2:] != (horizon, 2):
        raise ValueError("candidates_raw must have shape (variants, samples, H, 2)")
    samples = chunks.shape[1]
    if not variants or not samples or not 1 <= int(data["execute"]) <= horizon:
        raise ValueError("Need variants, samples, and 1 <= K <= H")
    if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("revision_times must be a strictly increasing grid")
    if traces.shape != (variants, samples, len(times), horizon, 2):
        raise ValueError("revision_states_raw must have shape (variants, samples, times, H, 2)")
    for name in ("repeat_baseline_raw", "independent_baseline_raw"):
        if np.asarray(data[name]).shape != chunks.shape[1:]:
            raise ValueError(f"{name} must have shape (samples, H, 2)")
    if np.asarray(data["state_raw"]).shape != (5,):
        raise ValueError("state_raw must contain the pusher position and T pose")
    if any(len(data[name]) != variants for name in ("variant_labels", "families", "amplitudes_px")):
        raise ValueError("Variant labels, families, and amplitudes must match sources")
    if data["families"][0] != "baseline" or float(data["amplitudes_px"][0]) != 0:
        raise ValueError("The first source must be the zero-amplitude baseline")
    families = list(dict.fromkeys(family for family in data["families"] if family != "baseline"))
    if not families:
        raise ValueError("Need at least one source intervention family")
    names = ("sources_raw", "candidates_raw", "revision_states_raw", "revision_times",
             "repeat_baseline_raw", "independent_baseline_raw", "amplitudes_px", "state_raw",
             "old_plan_raw", "completed_raw", "fixed_source_raw", "goal_pose_raw")
    if any(not np.isfinite(np.asarray(data[name])).all() for name in names if data.get(name) is not None):
        raise ValueError("Cannot plot nonfinite states, sources, samples, or revision traces")
    return sources, chunks, traces, times, families


def _rms(displacement):
    """RMS Euclidean distance in pixels, averaged over all points supplied."""
    return float(np.sqrt(np.mean(np.sum(np.square(displacement), axis=-1))))


def _scene(ax, data):
    from matplotlib.patches import Rectangle

    state = np.asarray(data["state_raw"])
    goal = np.asarray(data.get("goal_pose_raw", [256., 256., math.pi / 4]))
    _draw_tee(ax, goal, "tab:green", .3, label="Goal T", linestyle="--")
    _draw_tee(ax, state[2:5], "0.45", .25, label="Current T")
    ax.add_patch(Rectangle((0, 0), 512, 512, fill=False, color="0.6", linestyle=":", linewidth=.7))
    ax.scatter(*state[:2], s=55, color="tab:cyan", edgecolors="black", zorder=8,
               label="Current pusher")


def _commands(ax, plan, color, execute, *, linestyle="-", alpha=1., label=None):
    ax.plot(*plan.T, color=color, linestyle=linestyle, linewidth=1.6, alpha=alpha, label=label)
    ax.scatter(*plan[execute - 1], marker="D", s=22, color=color, alpha=alpha, zorder=6)
    ax.scatter(*plan[-1], marker="x", s=25, color=color, alpha=alpha, zorder=6)


def _family_indices(data, family):
    indices = [index for index, name in enumerate(data["families"]) if name == family]
    return sorted(indices, key=lambda index: data["amplitudes_px"][index])


def _save(fig, output, name, paths):
    for extension in ("png", "svg"):
        relative = f"{name}.{extension}"
        fig.savefig(output / relative, dpi=140)
        paths[f"{name}_{extension}"] = relative


def render_source_sensitivity(data, output, *, title=""):
    """Save four fixed-condition diagnostics; return paths relative to ``output``.

    Index zero is the unmodified source. Sample index n uses identical sampler
    noise for every intervention, so paired differences isolate the source
    change. Coordinates are decoded command targets, not simulated trajectories.
    No input array is changed, and out-of-workspace commands are never clipped.
    """
    sources, chunks, traces, times, families = _validate(data)
    plt = _import_pyplot()
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants, samples, horizon, _ = chunks.shape
    execute = int(data["execute"])
    amplitudes = np.asarray(data["amplitudes_px"], dtype=float)
    radius = max(1., float(np.abs(amplitudes).max()))
    norm = Normalize(-radius, radius)
    cmap = plt.get_cmap("coolwarm")
    colors = ["black" if index == 0 else cmap(norm(value)) for index, value in enumerate(amplitudes)]
    mean_chunks = chunks.mean(axis=1)
    state = np.asarray(data["state_raw"])
    goal = np.asarray(data.get("goal_pose_raw", [256., 256., math.pi / 4]))
    limits = _limits([sources, chunks, state[:2], *_tee_polygons(state[2:5]), *_tee_polygons(goal)])
    heading = (f"{title + ' | ' if title else ''}Source sensitivity: H={horizon}, K={execute}, {samples} matched draws"
               f" | region: {data.get('intervention_region', 'whole')}")
    fixed_note = "Fixed observation / action history and reference; common sampler noise across sources."
    target_note = "Unclipped command targets, NOT physical trajectories or task success."
    marker_handles = [Line2D([], [], color="black", marker="D", linestyle="none", label=f"Command K={execute}"),
                      Line2D([], [], color="black", marker="x", linestyle="none", label=f"Last command H={horizon}")]
    paths = {}

    fig, axes = plt.subplots(2, len(families), figsize=(max(9., 4.5 * len(families)), 10.), squeeze=False)
    try:
        for column, family in enumerate(families):
            indices = _family_indices(data, family)
            for row in range(2):
                ax = axes[row, column]
                _scene(ax, data)
                if row == 1:
                    for sample in np.linspace(0, samples - 1, min(samples, 24), dtype=int):
                        ax.plot(*chunks[0, sample].T, color="0.6", alpha=.16, linewidth=.7)
                for index in [*indices, 0]:
                    plan = sources[index] if row == 0 else mean_chunks[index]
                    _commands(ax, plan, colors[index], execute)
                label = FAMILY_LABELS.get(family, family)
                _set_axes(ax, limits, f"{label}: {'input sources' if row == 0 else 'generated mean chunks'}")
        scene_handles, _ = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles=[*scene_handles, Line2D([], [], color="black", label="Unmodified source / output"), *marker_handles],
                   loc="lower center", bbox_to_anchor=(.5, .115), ncol=3, fontsize=9)
        fig.suptitle(heading, fontsize=13)
        fig.text(.5, .079, "Mean chunks can lie between modes; inspect matched draws in paired_paths.png.", ha="center", fontsize=10)
        fig.text(.5, .049, fixed_note, ha="center", fontsize=9)
        fig.text(.5, .021, target_note, ha="center", fontsize=9)
        fig.subplots_adjust(left=.05, right=.92, bottom=.23, top=.91, hspace=.35, wspace=.25)
        color_ax = fig.add_axes((.945, .3, .012, .45))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                     label="Signed peak source displacement (pixels)")
        _save(fig, output, "source_response", paths)
    finally:
        plt.close(fig)

    selected_draws = np.linspace(0, samples - 1, min(samples, 2), dtype=int)
    fig, axes = plt.subplots(len(selected_draws), len(families),
                             figsize=(max(9., 4.5 * len(families)), 4.5 * len(selected_draws) + 1.5), squeeze=False)
    try:
        for column, family in enumerate(families):
            indices = _family_indices(data, family)
            extremes = list(dict.fromkeys([indices[0], indices[-1]]))
            for row, sample in enumerate(selected_draws):
                ax = axes[row, column]
                _scene(ax, data)
                for index in [*extremes, 0]:
                    ax.plot(*sources[index].T, ":", color=colors[index], linewidth=1.3, alpha=.8)
                    _commands(ax, chunks[index, sample], colors[index], execute)
                amplitude_label = ", ".join(f"{amplitudes[index]:+g}" for index in extremes)
                _set_axes(ax, limits, f"{FAMILY_LABELS.get(family, family)}: draw #{sample + 1}\n"
                                     f"Source shifts: {amplitude_label} px vs baseline")
        handles = [Line2D([], [], color="black", linestyle=":", label="Source (dotted)"),
                   Line2D([], [], color="black", label="Generated chunk (solid)"), *marker_handles]
        fig.legend(handles, [item.get_label() for item in handles], loc="lower center",
                   bbox_to_anchor=(.5, .064), ncol=4, fontsize=9)
        fig.suptitle(f"{title + ' | ' if title else ''}Matched individual draws: baseline black, negative blue, positive red\n"
                     "Same draw ID means the same sampler noise in every panel.", fontsize=12)
        fig.text(.5, .025, target_note, ha="center", fontsize=9)
        fig.subplots_adjust(left=.05, right=.98, bottom=.16, top=.84, hspace=.45, wspace=.25)
        _save(fig, output, "paired_paths", paths)
    finally:
        plt.close(fig)

    input_rms = np.asarray([_rms(source - sources[0]) for source in sources])
    output_rms = np.asarray([_rms(chunk - chunks[0]) for chunk in chunks])
    mean_rms = np.asarray([_rms(chunk - mean_chunks[0]) for chunk in mean_chunks])
    independent_rms = _rms(np.asarray(data["independent_baseline_raw"]) - chunks[0])
    repeat_rms = _rms(np.asarray(data["repeat_baseline_raw"]) - chunks[0])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    try:
        for column, family in enumerate(families):
            indices = sorted([0, *_family_indices(data, family)], key=lambda index: amplitudes[index])
            color = plt.get_cmap("tab10")(column % 10)
            axes[0].plot(amplitudes[indices], output_rms[indices], "o-", color=color,
                         label=FAMILY_LABELS.get(family, family))
            axes[0].plot(amplitudes[indices], mean_rms[indices], "--", color=color, alpha=.75)
            nonzero = [index for index in indices if input_rms[index] > 0]
            axes[1].plot(amplitudes[nonzero], output_rms[nonzero] / input_rms[nonzero],
                         "o-", color=color, label=FAMILY_LABELS.get(family, family))
        axes[0].axhline(independent_rms, color="0.35", linestyle=":", label="Same source, independent noise")
        axes[0].axhline(repeat_rms, color="black", linestyle="-.", linewidth=.8,
                       label="Same source, repeated seed")
        axes[1].axhline(1., color="0.5", linestyle=":", label="Unit gain")
        for ax in axes:
            ax.set_xlabel("Signed peak source displacement (pixels)")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set(title="Output response (solid); mean shift (dashed)", ylabel="RMS Euclidean command change (pixels)")
        axes[1].set(title="Matched-noise sensitivity gain", ylabel="Output RMS / input RMS")
        fig.suptitle(heading, fontsize=12)
        fig.text(.5, .047, "RMS averages squared Euclidean distances over draws and all H commands; zero-input gain is undefined.",
                 ha="center", fontsize=9)
        fig.text(.5, .015, "Small gain here indicates local source insensitivity, not evidence of task failure or global independence.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .095, 1, .94))
        _save(fig, output, "sensitivity_curves", paths)
    finally:
        plt.close(fig)

    # Columns occupy their actual (possibly nonuniform) revision-time intervals.
    trace_rms = np.sqrt(np.mean(np.sum(np.square(traces - traces[0:1]), axis=-1), axis=(1, 3)))
    row_order = [0, *(index for family in families for index in _family_indices(data, family))]
    time_edges = np.concatenate(([times[0]], (times[:-1] + times[1:]) / 2, [times[-1]]))
    fig, ax = plt.subplots(figsize=(12., max(4.8, .28 * variants + 2.2)))
    try:
        mesh = ax.pcolormesh(time_edges, np.arange(len(row_order) + 1) - .5,
                             trace_rms[row_order], shading="flat", cmap="magma", vmin=0)
        ax.set_yticks(np.arange(len(row_order)), [data["variant_labels"][index] for index in row_order], fontsize=8)
        ax.set_ylim(len(row_order) - .5, -.5)
        ax.set(xlabel="Revision time (actual sampler grid)", ylabel="Source intervention",
               title=f"{title + ' | ' if title else ''}Does source dependence persist through revision?")
        fig.colorbar(mesh, ax=ax, label="Matched-noise RMS change from baseline (pixels)")
        fig.text(.5, .055, "Each row compares the same-noise revision states against the unmodified-source states.", ha="center", fontsize=9)
        fig.text(.5, .025, "Fading differences indicate contraction under these interventions; this is not a task-success metric.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .095, 1, .98))
        _save(fig, output, "revision_sensitivity", paths)
    finally:
        plt.close(fig)
    return paths
