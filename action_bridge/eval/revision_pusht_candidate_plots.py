"""Plot alternative command chunks sampled at one fixed Push-T state."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from action_bridge.eval.visualization import _draw_tee, _import_pyplot, _tee_polygons


def _retained(data):
    old = data.get("old_plan_raw")
    if not data["has_previous_plan"] or old is None:
        return np.empty((0, 2))
    return np.asarray(old)[int(data["executed"]):]


def _limits(points, *, workspace=False):
    """Include every supplied target, including targets outside the workspace."""
    points = np.concatenate([np.asarray(p).reshape(-1, 2) for p in points])
    if workspace:
        points = np.concatenate((points, [[0., 0.], [512., 512.]]))
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    half_span = max(float((high - low).max()) / 2 + 18., 45.)
    return (center[0] - half_span, center[0] + half_span), (center[1] + half_span, center[1] - half_span)


def _goal_pose(data):
    return np.asarray(data.get("goal_pose_raw", [256., 256., math.pi / 4]))


def _scene(ax, data):
    from matplotlib.patches import Rectangle

    state = np.asarray(data["state_raw"])
    goal = _goal_pose(data)
    _draw_tee(ax, goal, "tab:green", .45,
              label="Goal T", linestyle="--")
    _draw_tee(ax, state[2:5], "0.45", .35, label="Current T")
    if "symmetry_axis_raw" in data:
        ax.axline(goal[:2], goal[:2] + np.asarray(data["symmetry_axis_raw"]),
                  color="tab:brown", linestyle="--", linewidth=1., alpha=.7,
                  label="Symmetry axis", zorder=0)
    ax.add_patch(Rectangle((0, 0), 512, 512, fill=False, color="0.6", linestyle=":", linewidth=.8))
    ax.scatter(*state[:2], s=75, color="tab:cyan", edgecolors="black", zorder=7, label="Current pusher")
    retained = _retained(data)
    completed = np.asarray(data["completed_raw"])
    if len(retained):
        tail = completed[len(retained) - 1:]
        ax.plot(*tail.T, "s--", color="tab:purple", linewidth=1.3, markersize=3,
                label="Appended completion tail", zorder=4)
        ax.plot(*retained.T, "o--", color="black", markerfacecolor="none", markersize=5,
                linewidth=1.5, label="Unexecuted previous chunk", zorder=5)
    else:
        label = "Startup anchor" if not data["has_previous_plan"] else "Completed source (no retained actions)"
        ax.plot(*completed.T, "s--", color="tab:purple", linewidth=1.3, markersize=3, label=label)
    return ax


def _commands(ax, candidates, *, executed, sources=None, all_actions=True):
    horizon = candidates.shape[1]
    shown = horizon if all_actions else executed
    alpha = max(.008, min(.65, 4 / len(candidates)))
    if sources is not None:
        for index, source in enumerate(sources):
            ax.plot(*source[:shown].T, ":", color="0.65", linewidth=.6, alpha=alpha,
                    label="Source supplied to sampler" if index == 0 else None, zorder=1)
    for index, candidate in enumerate(candidates):
        ax.plot(*candidate[:shown].T, "-", color="tab:blue", linewidth=1., alpha=alpha,
                label="Revised command chunks" if index == 0 else None, zorder=2)
    marker_alpha = max(.03, alpha)
    ax.scatter(*candidates[:, 0].T, s=17, color="tab:green", alpha=marker_alpha, zorder=6,
               label="First command")
    ax.scatter(*candidates[:, executed - 1].T, s=20, marker="D", color="tab:orange",
               alpha=marker_alpha, zorder=6, label=f"Command K={executed}")
    if all_actions and executed != horizon:
        ax.scatter(*candidates[:, -1].T, s=23, marker="x", color="tab:red",
                   alpha=marker_alpha, zorder=6, label=f"Last command H={horizon}")


def _set_axes(ax, limits, title):
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="x (pixels)", ylabel="y (pixels)", title=title)
    ax.set_aspect("equal", adjustable="box")


def _lateral_offsets(data, command):
    """Signed command displacement from the supplied scene's symmetry axis."""
    commands = np.asarray(data["candidates_raw"])[:, command]
    return (commands - _goal_pose(data)[:2]) @ np.asarray(data["lateral_axis_raw"])


def _render_lateral_distribution(data, output, *, executed, title):
    plt = _import_pyplot()
    horizon = np.asarray(data["candidates_raw"]).shape[1]
    offsets = [_lateral_offsets(data, executed - 1), _lateral_offsets(data, -1)]
    radius = max(1., *(float(np.abs(values).max()) for values in offsets))
    bins = np.linspace(-radius * 1.05, radius * 1.05, 51)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    try:
        for ax, values, label, color in zip(axes, offsets,
                                           (f"Command K={executed}", f"Last command H={horizon}"),
                                           ("tab:orange", "tab:red")):
            ax.hist(values, bins=bins, color=color, alpha=.7)
            ax.axvline(0., color="tab:brown", linestyle="--", label="Symmetry axis")
            ax.set(title=label, xlabel="Signed lateral command offset (pixels)", ylabel="Candidates")
            ax.legend()
        context_label = data.get("context_label", "synthetic symmetric scene")
        source_label = data.get("source_label", "")
        heading = " | ".join(label for label in (title, source_label) if label)
        fig.suptitle(f"{heading + ' | ' if heading else ''}{context_label}: lateral command distribution")
        fig.text(.5, .035, "Offsets are command targets, not simulated pusher positions or task outcomes.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .92))
        fig.savefig(output / "candidate_lateral.png", dpi=150)
    finally:
        plt.close(fig)


def render_candidate_batch(data, output, *, title=""):
    """Save overlays and up to 16 unranked individual draws; return relative paths.

    All coordinates are already decoded pixels. Plotting neither clips commands
    nor moves the scene: every candidate shares exactly the supplied state.
    ``executed`` counts consumed commands in the previous plan (zero at startup).
    ``execute`` is K, the number planned for execution after selecting a chunk.
    """
    plt = _import_pyplot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = np.asarray(data["candidates_raw"])
    sources = np.asarray(data["sources_raw"])
    completed = np.asarray(data["completed_raw"])
    state = np.asarray(data["state_raw"])
    count, horizon, _ = candidates.shape
    executed = int(data.get("execute", data["executed"]))
    if count == 0 or not 1 <= executed <= horizon:
        raise ValueError("Need at least one candidate and 1 <= executed <= horizon")
    if not all(np.isfinite(array).all() for array in (candidates, sources, completed, state)):
        raise ValueError("Cannot plot nonfinite states or command targets")

    retained = _retained(data)
    synthetic = data.get("scene_kind") == "synthetic_symmetric"
    scene_points = [state[:2], *_tee_polygons(state[2:5]), *_tee_polygons(_goal_pose(data))]
    full_limits = _limits([*scene_points, candidates, sources, completed, retained], workspace=True)
    grid_limits = _limits([state[:2], candidates, sources, completed, retained])
    focus_limits = _limits([state[:2], candidates[:, :executed], sources[:, :executed],
                            completed[:executed], retained[:executed]])
    outside = ((candidates < 0) | (candidates > 512)).any(axis=-1)
    out_of_bounds = int(outside.any(axis=-1).sum())
    context_label = data.get("context_label", "synthetic symmetric scene" if synthetic else "learner-reached state")
    location = ("manually specified, not a simulator rollout" if synthetic else
                f"replan {data['replan']}, robot step {data['robot_step']}")
    heading = (f"{title}\n" if title else "") + (
        f"{count} revisions of ONE {context_label} | H={horizon}, K={executed}\n{location}")
    if data.get("source_label"):
        heading += f" | {data['source_label']}"
    note = ("Unclipped command targets, NOT simulated pusher paths. Fixed scene; candidates are neither executed nor ranked.\n"
            f"Full view includes every target; {out_of_bounds}/{count} chunks contain targets outside [0, 512]. "
            f"Source noise: {data.get('source_noise_mode', 'as supplied')}.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    try:
        for ax in axes:
            _scene(ax, data)
        _commands(axes[0], candidates, executed=executed, sources=sources)
        _commands(axes[1], candidates, executed=executed, sources=sources, all_actions=False)
        _set_axes(axes[0], full_limits, "Whole plans + full workspace")
        _set_axes(axes[1], focus_limits, f"Next {executed} commands (automatic zoom)")
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=5, fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.)
        fig.suptitle(heading, fontsize=12)
        fig.text(.5, .015, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .21, 1, .98))
        fig.savefig(output / "candidate_overlay.png", dpi=150)
        fig.savefig(output / "candidate_overlay.svg")
    finally:
        plt.close(fig)

    indices = np.linspace(0, count - 1, min(count, 16), dtype=int)
    columns = min(4, len(indices))
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.3 * rows + 1.3), squeeze=False)
    try:
        for ax, index in zip(axes.flat, indices):
            _scene(ax, data)
            _commands(ax, candidates[index:index + 1], executed=executed, sources=sources[index:index + 1])
            _set_axes(ax, grid_limits, f"Draw #{index + 1} (not ranked)")
        for ax in list(axes.flat)[len(indices):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .025),
                            ncol=min(5, 2 * columns), fontsize=7)
        for handle in legend.legend_handles:
            handle.set_alpha(1.)
        fig.suptitle(f"{title + ' | ' if title else ''}Individual revisions: evenly spaced draw IDs, shared command zoom\n"
                     f"Same fixed {context_label}; curves are command targets, not physical trajectories.", fontsize=11)
        fig.tight_layout(rect=(0, .1 if rows > 1 else .2, 1, .93 if rows > 1 else .88))
        fig.savefig(output / "candidate_grid.png", dpi=130)
    finally:
        plt.close(fig)

    paths = {"overlay_png": "candidate_overlay.png", "overlay_svg": "candidate_overlay.svg",
             "grid_png": "candidate_grid.png"}
    if synthetic:
        _render_lateral_distribution(data, output, executed=executed, title=title)
        paths["lateral_png"] = "candidate_lateral.png"
    return paths


def render_source_comparison(conditions, output, *, title=""):
    """Compare fixed input sources in one scene, using shared spatial/histogram axes.

    ``conditions`` maps descriptive labels to candidate-batch dictionaries.
    Each condition supplies one fixed source, repeated in ``sources_raw``.
    No command sequence is executed or scored by this visualization.
    """
    if not conditions:
        raise ValueError("Need at least one source condition")
    plt = _import_pyplot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first = next(iter(conditions.values()))
    horizon = np.asarray(first["candidates_raw"]).shape[1]
    executed = int(first.get("execute", first["executed"]))
    scene_points, lateral_points = [], []
    for data in conditions.values():
        state = np.asarray(data["state_raw"])
        candidates = np.asarray(data["candidates_raw"])
        source = np.asarray(data["sources_raw"])[0]
        if candidates.shape[1] != horizon or int(data.get("execute", data["executed"])) != executed:
            raise ValueError("Source comparisons need the same H and K")
        scene_points.extend([state[:2], *_tee_polygons(state[2:5]), *_tee_polygons(_goal_pose(data)),
                             candidates, source, data["completed_raw"], _retained(data)])
        lateral_points.extend([_lateral_offsets(data, executed - 1), _lateral_offsets(data, -1),
                               (source - _goal_pose(data)[:2]) @ np.asarray(data["lateral_axis_raw"])])
    limits = _limits(scene_points)
    radius = max(1., *(float(np.abs(values).max()) for values in lateral_points)) * 1.05
    bins = np.linspace(-radius, radius, 61)

    fig, axes = plt.subplots(2, len(conditions), figsize=(5.4 * len(conditions), 10.4),
                             squeeze=False, gridspec_kw={"height_ratios": [1.5, 1.]})
    try:
        for column, (label, data) in enumerate(conditions.items()):
            candidates = np.asarray(data["candidates_raw"])
            source = np.asarray(data["sources_raw"])[0]
            top, bottom = axes[:, column]
            _scene(top, data)
            _commands(top, candidates, executed=executed)
            top.plot(*source.T, "-", color="tab:purple", linewidth=3.,
                     label="Fixed input source", zorder=8)
            top.scatter(*source[-1], marker="s", color="tab:purple", s=45, zorder=8)
            _set_axes(top, limits, f"{label}\n{len(candidates):,} revisions, H={horizon}, K={executed}")
            for command, name, color in ((executed - 1, f"Command K={executed}", "tab:orange"),
                                         (-1, f"Last command H={horizon}", "tab:red")):
                bottom.hist(_lateral_offsets(data, command), bins=bins, histtype="stepfilled",
                            color=color, alpha=.4, label=name)
            source_end = float((source[-1] - _goal_pose(data)[:2]) @ np.asarray(data["lateral_axis_raw"]))
            bottom.axvline(0., color="tab:brown", linestyle="--", label="Scene symmetry axis")
            bottom.axvline(source_end, color="tab:purple", linewidth=2., label="Input source endpoint")
            bottom.set(xlim=(-radius, radius), xlabel="Lateral offset (pixels; + = right)", ylabel="Candidates")
            bottom.legend(fontsize=8, loc="upper left")
        # Equal histogram scales make the concentration comparable between conditions.
        maximum = max(ax.get_ylim()[1] for ax in axes[1])
        for ax in axes[1]:
            ax.set_ylim(0., maximum)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045),
                            ncol=min(6, 2 * len(conditions)), fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.)
        fig.suptitle((f"{title}\n" if title else "") + "Source intervention: fixed scene, different input command chunks",
                     fontsize=14)
        fig.text(.5, .013, "Purple = input source. Revised curves are command targets, NOT simulated physical paths. "
                 "No candidate execution, task scoring or selection.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .14, 1, .94), h_pad=3.)
        fig.savefig(output / "source_comparison.png", dpi=150)
        fig.savefig(output / "source_comparison.svg")
    finally:
        plt.close(fig)
    return {"comparison_png": "source_comparison.png", "comparison_svg": "source_comparison.svg"}
