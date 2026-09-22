"""Small, headless plotting helpers; no simulator renderer or GPU is needed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _finish(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _trace_arrays(trace):
    observations = np.asarray(trace["observations"])
    normal = observations[0, 4:6]
    tangent = np.array([normal[1], -normal[0]])
    return observations, np.asarray(trace["actions"]), normal, tangent


def _scene(ax, observations, actions, normal, tangent):
    offset, goal = observations[0, 6], observations[0, 8]
    surface_origin = offset * normal
    target = surface_origin + goal * tangent
    start_projection = surface_origin + (observations[0, :2] @ tangent) * tangent
    # Include the surface and goal even when a failed rollout never approaches.
    points = np.concatenate([observations[:, :2], actions, target[None], start_projection[None]], axis=0)
    lower, upper = points.min(axis=0) - 0.1, points.max(axis=0) + 0.1
    extent = max(1.0, np.linalg.norm(upper - lower))
    line = surface_origin + np.array([-extent, extent])[:, None] * tangent
    ax.plot(line[:, 0], line[:, 1], color="0.4", linewidth=2, label="surface")
    ax.scatter(*target, marker="*", s=130, color="tab:green", label="tangential goal")
    ax.quiver(*((target + start_projection) / 2), *normal, angles="xy", scale_units="xy", scale=6, color="0.5")
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel="world x [m]", ylabel="world y [m]")
    ax.set_aspect("equal", adjustable="box")


def plot_episode(trace: dict, context, output: str | Path, *, title: str = "Surface contact", reference_trace=None) -> None:
    """Physical trajectory, target trajectory, reference parameters and forces.

    Force decomposition uses normalized command-space acceleration units, not
    measured physical Newtons. Keeping these in separate panels avoids implying
    that the learned command dynamics are a passive physical controller.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observations, actions, normal, tangent = _trace_arrays(trace)
    times, action_times = observations[:, 10], observations[:-1, 10]
    infos = trace.get("infos", {})
    force = np.asarray(trace.get("normal_force", infos.get("normal_force", observations[:, 9])))
    normal_velocity, tangent_velocity = observations[:, 2:4] @ normal, observations[:, 2:4] @ tangent
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    _scene(axes[0, 0], observations, actions, normal, tangent)
    axes[0, 0].plot(observations[:, 0], observations[:, 1], label="physical position")
    axes[0, 0].plot(actions[:, 0], actions[:, 1], alpha=0.65, linestyle="--", label="command target")
    if reference_trace is not None:
        reference_obs = np.asarray(reference_trace["observations"])
        axes[0, 0].plot(reference_obs[:, 0], reference_obs[:, 1], label="reference only", color="tab:purple")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(times, observations[:, 7], label="physical distance")
    axes[0, 1].plot(action_times, actions @ normal - observations[0, 6], "--", label="target distance")
    axes[0, 1].axhline(0, color="0.5", linewidth=1)
    axes[0, 1].set(ylabel="signed normal distance [m]")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(times, force)
    axes[1, 0].set(ylabel="physical normal force [N]")
    axes[1, 1].plot(times, normal_velocity, label="normal")
    axes[1, 1].plot(times, tangent_velocity, label="tangential")
    axes[1, 1].set(ylabel="physical velocity [m/s]")
    axes[1, 1].legend(fontsize=8)
    axes[2, 0].plot(times, observations[:, :2] @ tangent, label="physical position")
    axes[2, 0].plot(action_times, actions @ tangent, "--", label="command target")
    axes[2, 0].axhline(observations[0, 8], color="tab:green", linestyle=":", label="goal")
    axes[2, 0].set(ylabel="tangential position [m]")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(action_times, actions[:, 0], label="target world x")
    axes[2, 1].plot(action_times, actions[:, 1], label="target world y")
    axes[2, 1].set(ylabel="absolute Cartesian targets [m]")
    axes[2, 1].legend(fontsize=8)
    impulses = np.asarray(trace.get("impulse", np.zeros(len(actions))))
    for ax in axes.flat[1:]:
        ax.set(xlabel="time [s]")
        for time in action_times[impulses != 0]:
            ax.axvline(time, color="tab:red", linestyle=":", alpha=0.6)
    fig.suptitle(title)
    _finish(fig, output / "rollout.png")

    if "gamma_normal" in trace:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0, 0].plot(action_times, trace["gamma_normal"], label="normal damping")
        axes[0, 0].plot(action_times, trace["gamma_tangent"], label="tangential damping")
        axes[0, 0].set(ylabel="learned command damping")
        axes[0, 0].legend(fontsize=8)
        difference = np.asarray(trace["gamma_normal"]) - np.asarray(trace["gamma_tangent"])
        axes[0, 1].plot(action_times, difference)
        axes[0, 1].axhline(0, color="0.5", linewidth=1)
        axes[0, 1].set(ylabel="normal − tangential damping")
        axes[1, 0].plot(action_times, trace["stiffness"])
        axes[1, 0].set(ylabel="learned normal command stiffness")
        axes[1, 1].plot(action_times, trace["desired_offset"])
        axes[1, 1].set(ylabel="learned target offset [m]")
        for ax in axes.flat:
            ax.set(xlabel="time [s]")
        fig.suptitle(f"{title}: learned reference, no damping ordering imposed")
        _finish(fig, output / "learned_reference.png")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, coordinate in zip(axes, ("normal", "tangent")):
            ax.plot(action_times, trace[f"ref_{coordinate}"], label="reference")
            ax.plot(action_times, trace[f"residual_{coordinate}"], label="residual acceleration σu")
            ax.set(ylabel=f"{coordinate} command acceleration")
            ax.legend(fontsize=8)
        axes[-1].set(xlabel="time [s]")
        fig.suptitle("Reference / residual decomposition (normalized command units, not N)")
        _finish(fig, output / "reference_residual.png")


def plot_reference_overlay(trace: dict, reference_trace: dict, output: str | Path) -> None:
    """Compare a controlled rollout to a residual-disabled rollout, same context."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for current, label in ((trace, "controlled"), (reference_trace, "reference only")):
        observations, actions, normal, tangent = _trace_arrays(current)
        times = observations[:, 10]
        axes[0].plot(times, observations[:, 7], label=label)
        axes[1].plot(times, observations[:, :2] @ tangent, label=label)
        axes[2].plot(times[:-1], actions @ normal - observations[0, 6], label=label)
    for ax, ylabel in zip(axes, ("physical normal distance [m]", "physical tangential position [m]", "target normal distance [m]")):
        ax.set(xlabel="time [s]", ylabel=ylabel)
        ax.legend(fontsize=8)
    _finish(fig, output)


def save_rollout_gif(trace: dict, context, output: str | Path, *, fps: int = 15) -> None:
    """A representative CPU-rendered animation; targets and physical point differ."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    observations, actions, normal, tangent = _trace_arrays(trace)
    fig, ax = plt.subplots(figsize=(6, 5))
    _scene(ax, observations, actions, normal, tangent)
    physical, = ax.plot([], [], "o", color="tab:blue", markersize=8, label="physical point")
    target, = ax.plot([], [], "x", color="tab:orange", markersize=8, label="command target")
    trail, = ax.plot([], [], color="tab:blue", alpha=0.4)
    title = ax.set_title("")
    ax.legend(fontsize=8)
    stride = max(1, round(1 / (context.physics.control_dt * fps)))
    frames = list(range(0, len(observations), stride))

    def update(index):
        physical.set_data([observations[index, 0]], [observations[index, 1]])
        q = actions[max(0, index - 1)]
        target.set_data([q[0]], [q[1]])
        trail.set_data(observations[:index + 1, 0], observations[:index + 1, 1])
        title.set_text(f"t = {observations[index, 10]:.2f} s  |  normal force = {observations[index, 9]:.2f} N")
        return physical, target, trail, title

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _number(record: dict, name: str):
    value = record.get(name)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _grouped_curve(records: list[dict], x_key: str, y_key: str):
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        x, y = _number(record, x_key), _number(record, y_key)
        if x is not None and y is not None:
            groups[record["method"]][x].append(y)
    return groups


def _curve(ax, records: list[dict], x_key: str, y_key: str, xlabel: str, ylabel: str):
    groups = _grouped_curve(records, x_key, y_key)
    for method, values in sorted(groups.items()):
        xs = sorted(values)
        means = np.array([np.mean(values[x]) for x in xs])
        deviations = np.array([np.std(values[x]) for x in xs])
        ax.plot(xs, means, "o-", label=method)
        ax.fill_between(xs, means - deviations, means + deviations, alpha=0.15)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if groups:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Not evaluated", ha="center", va="center", transform=ax.transAxes)


def plot_comparison(records: list[dict], output: str | Path) -> None:
    """Plot flat sweep rows, averaging seed replicates with ±1 seed SD bands.

    Rows have ``method, train_episodes, seed, suite, value`` and aggregate metric
    keys. Do not pool different data budgets in robustness curves: those curves
    use only the largest budget present. ID plots include all budgets explicitly.
    Missing suites are labeled as unevaluated, never synthesized as results.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    id_rows = [row for row in records if row.get("suite") == "id"]
    budgets = [row["train_episodes"] for row in records if "train_episodes" in row]
    largest_budget = max(budgets) if budgets else None
    full_rows = [row for row in records if largest_budget is None or row.get("train_episodes") == largest_budget]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in sorted({row["method"] for row in id_rows}):
        rows = [row for row in id_rows if row["method"] == method]
        for ax, force_key in zip(axes, ("peak_normal_force", "rms_normal_force")):
            valid = [row for row in rows if _number(row, force_key) is not None and _number(row, "task_success_rate") is not None]
            ax.scatter([row[force_key] for row in valid], [row["task_success_rate"] for row in valid], label=method, alpha=0.7)
            ax.set(xlabel=force_key.replace("_", " ") + " [N]", ylabel="task success rate", ylim=(-0.03, 1.03))
            if valid:
                ax.legend(fontsize=7)
    fig.suptitle("ID task / force trade-off (individual seeds and data budgets)")
    _finish(fig, output / "task_force_pareto.png")

    impulse_rows = [row for row in full_rows if row.get("suite") in ("normal_impulse_sweep", "impulse")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _curve(axes[0], impulse_rows, "value", "task_success_rate", "normal impulse [N s]", "task success rate")
    _curve(axes[1], impulse_rows, "value", "clean_success_rate", "normal impulse [N s]", "clean success rate")
    _finish(fig, output / "impulse_success.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _curve(axes[0], impulse_rows, "value", "recovery_time", "normal impulse [N s]", "recovery time, recovered episodes only [s]")
    _curve(axes[1], impulse_rows, "value", "recovery_success_rate", "normal impulse [N s]", "recovery rate, eligible episodes only")
    _curve(axes[2], impulse_rows, "value", "recovery_eligible", "normal impulse [N s]", "eligible fraction (contact before impulse)")
    _finish(fig, output / "impulse_recovery.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _curve(axes[0], id_rows, "train_episodes", "task_success_rate", "training demonstrations", "ID task success rate")
    _curve(axes[1], id_rows, "train_episodes", "clean_success_rate", "training demonstrations", "ID clean success rate")
    _finish(fig, output / "low_data.png")
    orientation_rows = [row for row in full_rows if row.get("suite") == "unseen_orientation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _curve(axes[0], orientation_rows, "value", "task_success_rate", "unseen surface orientation [degrees]", "task success rate")
    _curve(axes[1], orientation_rows, "value", "clean_success_rate", "unseen surface orientation [degrees]", "clean success rate")
    _finish(fig, output / "unseen_orientation.png")

    physics_suites = [suite for suite in ("mass", "stiffness", "damping", "friction", "gains", "contact_physics_shift")
                     if any(row.get("suite") == suite for row in full_rows)]
    if physics_suites:
        fig, axes = plt.subplots(len(physics_suites), 2, figsize=(12, 3 * len(physics_suites)), squeeze=False)
        for row_axes, suite in zip(axes, physics_suites):
            selected = [row for row in full_rows if row.get("suite") == suite]
            xlabel = suite.replace("_", " ") + " multiplier"
            _curve(row_axes[0], selected, "value", "task_success_rate", xlabel, "task success rate")
            _curve(row_axes[1], selected, "value", "clean_success_rate", xlabel, "clean success rate")
        _finish(fig, output / "physics_shift.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare independently evaluated policies from comparison.json files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", help="Optional one label per input, e.g. different checkpoints of one method")
    args = parser.parse_args(argv)
    if args.labels is not None and len(args.labels) != len(args.inputs):
        parser.error("--labels must contain one label per input file")
    records = []
    for index, path in enumerate(args.inputs):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) and "method" in row for row in rows):
            parser.error(f"{path} must contain the list of records written to comparison.json")
        records.extend([dict(row, method=args.labels[index]) for row in rows] if args.labels else rows)
    # Re-evaluating one seed is not an independent training-seed replicate.
    keys = [(row["method"], row.get("train_episodes"), row.get("seed"), row.get("suite"), row.get("value"))
            for row in records]
    if len(set(keys)) != len(keys):
        parser.error("Duplicate method/seed/condition records; use --labels to distinguish runs, or remove duplicate inputs")
    plot_comparison(records, args.output)
    print(f"Plotted {len(records)} evaluation records from {len(args.inputs)} files in {args.output}")


if __name__ == "__main__":
    main()
