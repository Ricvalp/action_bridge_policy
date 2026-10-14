"""Figures separating offline training tuples from their simulated consequences.

Inputs are ordinary JSON-serializable records produced by the mismatch diagnostic.
All positions and command targets are in the original 512-pixel Push-T frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from action_bridge.eval.visualization import _draw_tee, _import_pyplot


DEMO = "#2378b5"
POLICY = "#e07a21"
TAIL = "#8a4fb2"
CONTROL = "#218c81"
GOAL = np.array([256.0, 256.0, np.pi / 4])


def _line(ax, points, *, color, label, physical=False, **kwargs):
    points = np.asarray(points, dtype=float)
    if len(points):
        ax.plot(
            points[:, 0], points[:, 1], color=color, label=label,
            linestyle="-" if physical else "--", linewidth=2.0,
            marker="." if physical else "o", markersize=3.5, **kwargs,
        )


def _pusher(ax, position, color, *, label=None, hollow=False):
    from matplotlib.patches import Circle

    ax.add_patch(Circle(
        position, radius=15, edgecolor=color,
        facecolor="white" if hollow else color, linewidth=2,
        alpha=0.9, zorder=7, label=label,
    ))
    ax.scatter(*position, color=color if hollow else "white", s=12, zorder=8)


def _scene(ax, title, positions, poses):
    from matplotlib.patches import Rectangle

    _draw_tee(ax, GOAL, color="#5d8660", alpha=0.45, linestyle="--", label="Goal T")
    ax.add_patch(Rectangle((0, 0), 512, 512, fill=False, edgecolor="0.75", linewidth=1))
    # Keep any out-of-arena commands visible instead of silently clipping them.
    extent = [np.array([[0.0, 0.0], [512.0, 512.0]])]
    extent.extend(np.asarray(p).reshape(-1, 2) for p in positions if len(p))
    for pose in poses:
        extent.append(np.asarray(pose[:2])[None] + np.array([[-130, -130], [130, 130]]))
    all_points = np.concatenate(extent)
    low, high = np.nanmin(all_points, axis=0) - 12, np.nanmax(all_points, axis=0) + 12
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(high[1], low[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=10)
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    ax.grid(alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)


def _legend(ax, **kwargs):
    ax.legend(fontsize=8.5, framealpha=0.93, handlelength=2, borderpad=0.6, **kwargs)


def plot_mismatch(record: dict[str, Any], path: str | Path) -> Path:
    """Plot one offline training tuple and two matched physics rollouts.

    A simulated expert-command rollout is a reset/replay control: its deviation
    from the recorded pusher exposes the pose-only reset's own reconstruction
    error. A policy rollout is not an exact clone of the demonstration dynamics.
    """
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    demo = np.asarray(record["demo_states"], dtype=float)
    expert_commands = np.asarray(record["demo_commands"], dtype=float)
    old = np.asarray(record["old_plan"], dtype=float)
    source = np.asarray(record["completed_source"], dtype=float)
    noisy = np.asarray(record["noisy_source"], dtype=float)
    target = np.asarray(record["current_target"], dtype=float)
    simulated = np.asarray(record["simulated_policy"]["states"], dtype=float)
    control = np.asarray(record["simulated_expert"]["states"], dtype=float)
    execute = int(record["execute"])
    retained = old[execute:]
    tail = source[len(retained):]
    metrics = record["metrics"]
    start, step = int(record["start_step"]), int(record["step"])
    split = str(record.get("split", "train"))
    split_label = "TRAIN demo" if split == "train" else f"HELD-OUT {split.upper()} demo"
    tuple_label = "Training tuple" if split == "train" else "Held-out diagnostic tuple"
    history_label = "recorded training histories" if split == "train" else "recorded held-out histories (not training data)"

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.7))
    fig.suptitle(
        f"Offline source / physical-state mismatch  ·  {split_label} {record['episode_id']}  ·  "
        f"t = {step}  ·  H = {len(old)}, K = {execute}",
        fontsize=16, fontweight="semibold", y=0.985,
    )

    ax = axes[0, 0]
    _scene(ax, f"1. At t = {start}: two alternative command sequences",
           [demo[:, :2], expert_commands, old], [demo[0, 2:5]])
    _draw_tee(ax, demo[0, 2:5], DEMO, 0.16, label="Recorded starting T")
    _line(ax, expert_commands, color=DEMO, label=f"Expert commands: next {execute}")
    _line(ax, old[:execute], color=POLICY, label=f"EMA commands: next {execute}")
    _line(ax, old[max(0, execute - 1):], color=POLICY,
          label="EMA suffix: not executed yet", alpha=0.45)
    _pusher(ax, demo[0, :2], DEMO, label="Same starting pusher pose")
    _legend(ax, loc="best")

    ax = axes[0, 1]
    _scene(ax, f"2. Execute {execute} actions: physical pusher trajectories",
           [demo[:, :2], simulated[:, :2], control[:, :2]],
           [demo[0, 2:5], demo[-1, 2:5], simulated[-1, 2:5], control[-1, 2:5]])
    _draw_tee(ax, demo[0, 2:5], "0.5", 0.11)
    _draw_tee(ax, demo[-1, 2:5], DEMO, 0.24)
    _draw_tee(ax, simulated[-1, 2:5], POLICY, 0.8, linestyle="--")
    _draw_tee(ax, control[-1, 2:5], CONTROL, 0.6, linestyle=":")
    _line(ax, demo[:, :2], color=DEMO, label="Recorded pusher", physical=True, zorder=4)
    _line(ax, simulated[:, :2], color=POLICY, label="Simulated: EMA commands", physical=True, zorder=5)
    _line(ax, control[:, :2], color=CONTROL, label="Simulated: expert commands (control)", physical=True, zorder=3)
    _pusher(ax, demo[-1, :2], DEMO)
    _pusher(ax, control[-1, :2], CONTROL, hollow=True)
    _pusher(ax, simulated[-1, :2], POLICY, hollow=True)
    ax.plot([demo[-1, 0], simulated[-1, 0]], [demo[-1, 1], simulated[-1, 1]],
            color="0.25", linestyle=":", linewidth=1.5, zorder=2)
    ax.text(0.03, 0.03,
            f"EMA vs recorded pusher: {metrics['pusher_counterfactual_gap_px']:.1f} px\n"
            f"Expert replay vs recorded: {metrics['expert_replay_gap_px']:.1f} px\n"
            "Colored T outlines = final physical T poses",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.85", "pad": 5})
    _legend(ax, loc="upper left")
    physical_points = np.concatenate([demo[:, :2], simulated[:, :2], control[:, :2]])
    if np.ptp(physical_points, axis=0).max() < 250:
        # Keep the arena view, but use point markers in the inset: true 15-pixel
        # pusher discs obscure small physical differences rather than reveal them.
        center = (physical_points.min(axis=0) + physical_points.max(axis=0)) / 2
        half_width = max(20.0, float(np.ptp(physical_points, axis=0).max()) * 0.6)
        inset_x = 0.04 if center[0] > 256 else 0.51
        inset = ax.inset_axes([inset_x, 0.17, 0.45, 0.35], zorder=20)
        _draw_tee(inset, demo[-1, 2:5], DEMO, 0.1)
        for states, color in [(demo, DEMO), (control, CONTROL), (simulated, POLICY)]:
            inset.plot(states[:, 0], states[:, 1], ".-", color=color,
                       linewidth=2.8 if color == DEMO else 1.3, markersize=3)
            inset.scatter(*states[-1, :2], color=color, s=15, zorder=5)
        inset.set_xlim(center[0] - half_width, center[0] + half_width)
        inset.set_ylim(center[1] + half_width, center[1] - half_width)
        inset.set_aspect("equal", adjustable="box")
        inset.set_title("Physical paths: zoom [px]", fontsize=8, pad=4)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.2)

    ax = axes[1, 0]
    _scene(ax, f"3. {tuple_label} at t = {step}: recorded state",
           [source, noisy, target, demo[-1:, :2], simulated[-1:, :2]], [demo[-1, 2:5]])
    _draw_tee(ax, demo[-1, 2:5], DEMO, 0.17, label="Conditioning: recorded T")
    _line(ax, target, color=DEMO, label="Supervision target: expert future commands", alpha=0.8)
    _line(ax, retained, color=POLICY, label="Source: retained EMA suffix", zorder=5)
    tail_connected = np.concatenate([retained[-1:], tail])
    _line(ax, tail_connected, color=TAIL, label="Source: appended completion", zorder=4)
    if len(retained):
        ax.scatter(*retained[0], marker="D", s=55, color=POLICY, edgecolors="white", zorder=6)
        for pusher, color in [(demo[-1, :2], DEMO), (simulated[-1, :2], POLICY)]:
            ax.plot([pusher[0], retained[0, 0]], [pusher[1], retained[0, 1]],
                    color=color, linestyle=":", linewidth=1.2, alpha=0.65)
        ax.annotate("First retained command", retained[0], xytext=(10, 12),
                    textcoords="offset points", fontsize=8.5, color=POLICY,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2})
    if not np.allclose(source, noisy):
        ax.scatter(noisy[:, 0], noisy[:, 1], marker="x", s=25, color="0.35",
                   alpha=0.65, linewidths=1, label="Actual source after noise", zorder=3)
    _pusher(ax, demo[-1, :2], DEMO, label="Conditioning: recorded pusher")
    _pusher(ax, simulated[-1, :2], POLICY, hollow=True)
    ax.annotate("Counterfactual pusher\nNOT used as conditioning",
                simulated[-1, :2], xytext=(12, -28), textcoords="offset points",
                color=POLICY, fontsize=8.5, zorder=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2})
    _legend(ax, loc="best")

    ax = axes[1, 1]
    ticks = np.arange(len(demo))
    policy_gap = np.linalg.norm(simulated[:, :2] - demo[:, :2], axis=-1)
    control_gap = np.linalg.norm(control[:, :2] - demo[:, :2], axis=-1)
    action_effect = np.linalg.norm(simulated[:, :2] - control[:, :2], axis=-1)
    ax.plot(ticks, policy_gap, "o-", color=POLICY, label="EMA simulation vs recorded pusher", linewidth=2.2)
    ax.plot(ticks, control_gap, "o-", color=CONTROL, label="Expert replay vs recorded (reset control)", linewidth=2)
    ax.plot(ticks, action_effect, "--", color="0.4", label="EMA simulation vs expert replay", linewidth=1.7)
    ax.set(title="4. How the physical mismatch accumulates", xlabel="Commands executed since the pose reset",
           ylabel="Pusher position difference [pixels]", xlim=(0, execute))
    ax.title.set_fontweight("semibold")
    ax.title.set_fontsize(12)
    ax.set_ylim(bottom=0)
    if execute <= 16:
        ax.set_xticks(ticks)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _legend(ax, loc="upper left")
    if len(retained):
        ax.text(0.03, 0.03,
                "First retained command → current pusher distance\n"
                f"Recorded: {metrics['retained_to_demo_pusher_gap_px']:.1f} px    "
                f"Simulated EMA: {metrics['retained_to_simulated_pusher_gap_px']:.1f} px",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.93, "pad": 6})

    fig.text(0.5, 0.035,
             "Solid curves are physical pusher paths; dashed curves / dots are command targets, not physical positions.\n"
             f"Offline replay reconstructed with checkpoint EMA and {history_label} — not the original historical source cache.\n"
             "Simulations reset the same recorded pose; demonstration velocities are unobserved. Expert replay measures reset error. This is not a success evaluation.",
             ha="center", va="center", fontsize=9.2, color="0.3", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.962), h_pad=3, w_pad=2.5)
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)
    return path


def plot_gap_summary(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Aggregate paired counterfactual / reset-control gaps over replay tuples."""
    if not records:
        raise ValueError("The mismatch summary needs at least one replay record")
    plt = _import_pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def values(key):
        return np.asarray([record["metrics"][key] for record in records], dtype=float)

    gap = values("pusher_counterfactual_gap_px")
    control = values("expert_replay_gap_px")
    demo_retained = values("retained_to_demo_pusher_gap_px")
    sim_retained = values("retained_to_simulated_pusher_gap_px")
    split = str(records[0].get("split", "train"))
    tuple_label = "TRAIN replay tuples" if split == "train" else f"held-out {split.upper()} diagnostic tuples"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.9))
    fig.suptitle(f"Offline replay mismatch across {len(records)} {tuple_label}",
                 fontsize=16, fontweight="semibold", y=0.97)

    ax = axes[0]
    for data, color, label in [(gap, POLICY, "EMA commands"), (control, CONTROL, "Expert commands (reset control)")]:
        ordered = np.sort(data)
        ax.step(np.r_[0, ordered], np.r_[0, np.arange(1, len(data) + 1) / len(data)], where="post",
                color=color, linewidth=2.2, label=f"{label}\nmedian {np.median(data):.1f} px; 90th pct. {np.quantile(data, 0.9):.1f} px")
    ax.set(title="Simulation vs recorded pusher", xlabel="Pusher difference at replan [pixels]",
           ylabel="Fraction of replay tuples", xlim=(0, None), ylim=(0, 1.02))
    _legend(ax, loc="lower right")

    ax = axes[1]
    ax.scatter(control, gap, color=POLICY, s=30, alpha=0.62, edgecolors="none")
    top = max(1.0, float(max(control.max(), gap.max()))) * 1.05
    ax.plot([0, top], [0, top], ":", color="0.45", label="Equal gap")
    ax.set(title="Policy mismatch vs reset error", xlabel="Expert replay vs recorded [pixels]",
           ylabel="EMA simulation vs recorded [pixels]", xlim=(0, top), ylim=(0, top))
    ax.set_aspect("equal", adjustable="box")
    _legend(ax, loc="lower right")

    ax = axes[2]
    finite = np.isfinite(demo_retained) & np.isfinite(sim_retained)
    if finite.any():
        x, y = demo_retained[finite], sim_retained[finite]
        ax.scatter(x, y, color=TAIL, s=30, alpha=0.62, edgecolors="none")
        top = max(1.0, float(max(x.max(), y.max()))) * 1.05
        ax.plot([0, top], [0, top], ":", color="0.45", label="Equal distance")
        ax.set(xlim=(0, top), ylim=(0, top))
        ax.text(0.04, 0.96, "Below diagonal: retained command is closer\nto simulated EMA pusher than recorded pusher",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        _legend(ax, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No retained suffix (K = H)", transform=ax.transAxes, ha="center")
    ax.set(title="First retained command → pusher", xlabel="Distance to recorded pusher [pixels]",
           ylabel="Distance to simulated EMA pusher [pixels]")
    ax.set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.grid(alpha=0.17)
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(12)
    fig.text(0.5, 0.045,
             "Each point is an offline replay tuple, not an independent rollout. Commands are replayed from the same recorded pose, with unobserved velocities.\n"
             "Expert-command replay controls for reset/reconstruction mismatch; it cannot remove all state-distribution differences.",
             ha="center", fontsize=9.2, color="0.3", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.115, 1, 0.93), w_pad=3)
    fig.savefig(path, dpi=165, facecolor="white")
    plt.close(fig)
    return path
