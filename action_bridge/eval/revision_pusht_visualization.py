"""Slow, phase-by-phase Push-T plan revision, using recorded data only.

Robot motion uses simulator states and bounded commands. All other phases keep
the physical scene frozen: their lines are *unclipped targets*, not motion. The
two clocks (robot steps and sampler time) deliberately never advance together.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from action_bridge.eval.visualization import _draw_tee, _import_pyplot


_COMPLETIONS = {0: "repeat", 1: "fixed damped", 2: "learned dissipative", 3: "direct MLP"}
_GOAL = np.array([256., 256., math.pi / 4])


@dataclass(frozen=True)
class _Frame:
    trace: dict
    phase: str
    step: int
    count: int = 0
    revision: int = 0
    origin: int = 0
    seconds: float = .35


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}; got {value.shape}")
    return value


def _prepare(episode, config, start_replan, replans):
    """Validate full recorded traces; never invent missing sampler iterations."""
    horizon, execute = int(config["horizon"]), int(config["execute"])
    if not 1 <= execute <= horizon:
        raise ValueError("execute must satisfy 1 <= K <= H")
    if start_replan < 0 or replans < 1:
        raise ValueError("start_replan must be nonnegative and replans must be positive")
    actions = np.asarray(episode["actions_raw"], dtype=float).reshape(-1, 2)
    actions = _array(actions, (len(actions), 2), "actions_raw")
    states = _array(episode["states_raw"], (len(actions) + 1, 5), "states_raw")
    records = {int(trace.get("replan", index)): trace
               for index, trace in enumerate(episode.get("plan_traces", []))}
    if start_replan not in records:
        raise ValueError(f"Requested start_replan={start_replan} has no recorded trace; "
                         f"available replans: {sorted(records)}")
    traces = []
    for replan in range(start_replan, start_replan + replans):
        if replan not in records:
            break
        trace = dict(records[replan], replan=replan)
        step = int(trace["step"])
        if step >= len(states):
            break  # A bounded/early-ended episode has no physical state here.
        if step < 0 or (traces and step <= traces[-1]["step"]):
            raise ValueError("Recorded replan steps must be nonnegative and strictly increasing")
        trace["step"] = step
        trace["state_raw"] = _array(trace["state_raw"], (5,), "state_raw")
        if not np.allclose(trace["state_raw"], states[step], rtol=1e-5, atol=1e-4):
            raise ValueError("Trace state_raw does not match the recorded decision state")
        for key in ("completed_raw", "perturbed_source_raw", "final_raw"):
            if trace.get(key) is None:
                raise ValueError(f"Revision-process rendering requires full traces with {key}")
            trace[key] = _array(trace[key], (horizon, 2), key)
        old = trace.get("old_plan_raw")
        trace["old_plan_raw"] = None if old is None else _array(old, (horizon, 2), "old_plan_raw")
        consumed = int(trace.get("old_executed", 0))
        if not 0 <= consumed <= min(horizon, step):
            raise ValueError("old_executed exceeds the recorded physical history or horizon")
        if consumed and old is None:
            raise ValueError("old_executed requires old_plan_raw")
        trace["old_executed"] = consumed
        aligned = trace.get("aligned_old_raw")
        trace["aligned_old_raw"] = (None if aligned is None else
                                    _array(aligned, (horizon - consumed, 2), "aligned_old_raw"))
        startup = bool(trace.get("startup", False)) or not bool(trace.get("has_previous_plan", False))
        trace["startup"] = startup
        if not startup and aligned is None:
            raise ValueError("A non-startup revision requires aligned_old_raw")
        candidates = np.asarray(trace.get("revision_candidate_plans_raw"), dtype=float)
        if candidates.ndim != 3 or candidates.shape[1:] != (horizon, 2) or len(candidates) < 2:
            raise ValueError("Full revision_candidate_plans_raw must include source and final [S,H,2]")
        trace["revision_candidate_plans_raw"] = _array(candidates, candidates.shape, "revision candidates")
        times = _array(trace.get("revision_times"), (len(candidates),), "revision_times")
        if np.any(np.diff(times) < 0):
            raise ValueError("revision_times must be in nondecreasing sampler order")
        trace["revision_times"] = times
        if not np.allclose(candidates[0], trace["perturbed_source_raw"], rtol=1e-5, atol=1e-4):
            raise ValueError("The first revision candidate must be the actual perturbed source")
        if not np.allclose(candidates[-1], trace["final_raw"], rtol=1e-5, atol=1e-4):
            raise ValueError("The last revision candidate must be the final plan")
        anchor = trace.get("startup_anchor_raw", trace["completed_raw"][0])
        trace["startup_anchor_raw"] = _array(anchor, (2,), "startup_anchor_raw")
        traces.append(trace)
    if not traces:
        raise ValueError(f"start_replan={start_replan} is beyond the recorded episode states")
    return states, actions, traces


def _phase_frames(trace, horizon):
    """Yield semantic frames (holds are applied only when writing the movie)."""
    step, consumed = trace["step"], trace["old_executed"]
    if consumed:
        for current in range(step - consumed, step + 1):
            yield _Frame(trace, "EXECUTE", current, origin=step - consumed, seconds=.25)
    retained = 0 if trace["startup"] else len(trace["aligned_old_raw"])
    yield _Frame(trace, "STARTUP" if trace["startup"] else "ALIGN", step,
                 count=retained, seconds=.8)
    # Each added target has a robot-horizon index; this is not elapsed wall time.
    for count in range(retained + 1, horizon + 1):
        yield _Frame(trace, "COMPLETE", step, count=count, seconds=.3)
    yield _Frame(trace, "COMPLETE", step, count=horizon, seconds=.8)
    yield _Frame(trace, "SOURCE_NOISE", step, count=horizon, seconds=.8)
    for index in range(len(trace["revision_times"])):
        yield _Frame(trace, "REVISE", step, revision=index, seconds=.3)
    yield _Frame(trace, "READY", step, count=horizon, seconds=.9)


def _layers(frame, execute):
    """(targets, index offset, color, caption, linestyle) without clipping."""
    trace, phase = frame.trace, frame.phase
    old, completed = trace["old_plan_raw"], trace["completed_raw"]
    if phase in ("EXECUTE", "EXECUTE_FINAL"):
        plan = old if phase == "EXECUTE" else trace["final_raw"]
        return [(plan, 0, "0.55", "Cached plan (unclipped)", "--")]
    layers = []
    if old is not None and not trace["startup"]:
        layers.append((old, -trace["old_executed"], "0.78", "Old plan, shifted index", ":"))
    retained = 0 if trace["startup"] else len(trace["aligned_old_raw"])
    if retained:
        layers.append((trace["aligned_old_raw"], 0, "0.4", "Retained old targets", "-"))
    count = frame.count if phase in ("STARTUP", "ALIGN", "COMPLETE") else len(completed)
    if count > retained:
        layers.append((completed[retained:count], retained, "tab:orange",
                       "Startup anchor targets" if trace["startup"] else "Appended completion", "-"))
    if phase == "STARTUP":
        layers.append((trace["startup_anchor_raw"][None], 0, "tab:orange", "Observed command anchor", "-"))
    if phase in ("SOURCE_NOISE", "REVISE", "READY"):
        layers.append((trace["perturbed_source_raw"], 0, "#168b8b", "Actual noisy source", "--"))
    if phase in ("REVISE", "READY"):
        plan = (trace["revision_candidate_plans_raw"][frame.revision]
                if phase == "REVISE" else trace["final_raw"])
        layers.append((plan, 0, "#8438b0", "Revised targets", "-"))
    if phase == "READY":
        layers.append((trace["final_raw"][:execute], 0, "#1a994e", "Next K targets (unclipped)", "-"))
    return layers


def _caption(frame, config):
    trace, phase = frame.trace, frame.phase
    t, consumed = trace["step"], trace["old_executed"]
    if phase in ("EXECUTE", "EXECUTE_FINAL"):
        return (f"Recorded physical step {frame.step}. "
                "Black: actual pusher path. Cyan squares: issued, clipped commands.")
    if phase == "STARTUP" or (phase == "COMPLETE" and trace["startup"]):
        return (f"Physical scene frozen at step {t}. Observed command anchor startup: "
                "no old-tail completion (no overlap).")
    if phase == "ALIGN":
        return (f"Physical scene frozen at step {t}. Drop {consumed} executed targets; "
                f"old index {consumed} becomes new index 0. Retain H-K targets.")
    if phase == "COMPLETE":
        mode = _COMPLETIONS.get(int(trace["completion_id"]), str(trace["completion_id"]))
        return (f"Physical scene frozen at step {t}. {mode} completion: target index "
                f"{frame.count - 1} / {config['horizon'] - 1} (robot-horizon index, not wall time).")
    if phase == "SOURCE_NOISE":
        return (f"Physical scene frozen at step {t}. Actual perturbed source after noise "
                f"(configured source std={config.get('source_std', 'unknown')}); completion is orange.")
    if phase == "REVISE":
        index, times = frame.revision, trace["revision_times"]
        nfe = int(trace.get("nfe", 0))
        accounting = (f"Midpoint accepted steps {len(times) - 1}; 2 NFE/step, total NFE={nfe}"
                      if str(config["method"]).startswith("fm_") else f"total NFE={nfe}")
        return (f"Physical step {t} FROZEN; revision tau={times[index]:.5g} "
                f"({index}/{len(times) - 1} accepted updates). {accounting}.")
    return (f"Physical scene frozen at step {t}. Final plan ready: next {config['execute']} targets "
            "highlighted green; clipping happens only when issuing commands.")


def _bounds(traces, states):
    points = [np.array([[0., 0.], [512., 512.]]), states[:, :2]]
    for trace in traces:
        for key in ("old_plan_raw", "completed_raw", "perturbed_source_raw", "final_raw",
                    "revision_candidate_plans_raw"):
            if trace[key] is not None:
                points.append(trace[key].reshape(-1, 2))
    points = np.concatenate(points)
    low, high = points.min(axis=0), points.max(axis=0)
    # The simulator rectangle stays explicit even when an unclipped target is outside it.
    pad = np.maximum(high - low, 512.) * .04
    return low - pad, high + pad


def _draw_panel(axes, frame, states, actions, config, bounds, *, compact=False):
    from matplotlib.patches import Circle, Rectangle

    scene, xaxis, yaxis = axes
    for axis in axes:
        axis.clear()
    trace, step = frame.trace, frame.step
    state = states[step] if frame.phase.startswith("EXECUTE") else trace["state_raw"]
    _draw_tee(scene, _GOAL, "tab:green", .65, label="Goal T", linestyle="--")
    _draw_tee(scene, state[2:5], "0.35", .5, label="Recorded T")
    scene.add_patch(Rectangle((0, 0), 512, 512, fill=False, linestyle=":", edgecolor="0.55"))
    scene.add_patch(Circle(state[:2], 15, color="#2878bd", ec="black", label="Recorded pusher", zorder=8))
    trail_start = max(0, trace["step"] - trace["old_executed"])
    trail = states[trail_start:step + 1, :2]
    if len(trail):
        scene.plot(*trail.T, color="black", linewidth=2.2, label="Actual pusher path", zorder=7)
    for plan, offset, color, label, style in _layers(frame, int(config["execute"])):
        if plan is None or not len(plan):
            continue
        indices = np.arange(len(plan)) + offset
        scene.plot(*plan.T, marker="o", markersize=3, linewidth=1.6, linestyle=style,
                   color=color, label=label, alpha=.9)
        for dim, axis in enumerate((xaxis, yaxis)):
            axis.plot(indices, plan[:, dim], marker="o", markersize=3, linewidth=1.6,
                      linestyle=style, color=color)
    if frame.phase.startswith("EXECUTE"):
        issued = actions[frame.origin:step]
        if len(issued):
            scene.scatter(*issued.T, s=25, marker="s", color="#00acc7", label="Issued clipped commands", zorder=9)
            for dim, axis in enumerate((xaxis, yaxis)):
                axis.scatter(np.arange(len(issued)), issued[:, dim], s=18, marker="s", color="#00acc7", zorder=9)
    low, high = bounds
    scene.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]), xlabel="x (pixels)", ylabel="y (pixels)")
    scene.set_aspect("equal", adjustable="box")
    labels = {"ALIGN": "ALIGN / AFTER EXECUTION", "STARTUP": "OBSERVED-ANCHOR STARTUP",
              "COMPLETE": "ALIGN / COMPLETE", "SOURCE_NOISE": "SOURCE NOISE",
              "EXECUTE_FINAL": "EXECUTE FINAL CHUNK"}
    scene.set_title(labels.get(frame.phase, frame.phase), fontsize=11 if compact else 14, weight="bold")
    for dim, axis in enumerate((xaxis, yaxis)):
        axis.axhspan(0, 512, color="0.95", zorder=-5)
        axis.axvline(-.5, color="0.55", linestyle=":", linewidth=1)
        axis.axvline(config["execute"] - .5, color="#1a994e", linestyle=":", linewidth=1)
        offset = -trace["old_executed"] if not frame.phase.startswith("EXECUTE") else 0
        axis.set(xlim=(min(-1, offset - .5), config["horizon"] - .5),
                 ylim=(low[dim], high[dim]), ylabel=f"{'xy'[dim]} target (px)")
        axis.grid(alpha=.2)
        axis.tick_params(labelsize=8)
    yaxis.set_xlabel("Target index in robot horizon (not revision time)", fontsize=8 if compact else 10)
    handles, labels = scene.get_legend_handles_labels()
    scene.legend(handles, labels, loc="upper left", fontsize=6 if compact else 8,
                 framealpha=.85, ncol=2 if compact else 1)


def _storyboard(path, trace, states, actions, config, bounds):
    plt = _import_pyplot()
    horizon = config["horizon"]
    retained = 0 if trace["startup"] else len(trace["aligned_old_raw"])
    panels = [
        _Frame(trace, "STARTUP" if trace["startup"] else "ALIGN", trace["step"], count=retained),
        _Frame(trace, "COMPLETE", trace["step"], count=horizon),
        _Frame(trace, "REVISE", trace["step"], revision=(len(trace["revision_times"]) - 1) // 2),
        _Frame(trace, "READY", trace["step"], count=horizon),
    ]
    fig = plt.figure(figsize=(20, 9), dpi=110)
    try:
        grid = fig.add_gridspec(3, 4, height_ratios=(3.5, 1, 1), left=.045, right=.99,
                               top=.85, bottom=.1, hspace=.35, wspace=.3)
        for column, (frame, name) in enumerate(zip(panels, ("After execution", "Completed", "Mid revision", "Final"))):
            axes = [fig.add_subplot(grid[row, column]) for row in range(3)]
            _draw_panel(axes, frame, states, actions, config, bounds, compact=True)
            tau = (f" | tau={trace['revision_times'][frame.revision]:.4g}" if frame.phase == "REVISE" else "")
            axes[0].set_title(name + tau, fontsize=12, weight="bold")
        status = ("Observed command anchor startup; no old-tail completion. " if trace["startup"]
                  else f"Drop {trace['old_executed']} old targets, retain H-K, append {_COMPLETIONS.get(int(trace['completion_id']), 'unknown')} tail. ")
        fig.suptitle(f"{config['method']} | replan {trace['replan']} at physical step {trace['step']}\n"
                     f"{status}All four scenes are frozen at the recorded decision state.", fontsize=15)
        fig.text(.5, .025, "Gray: retained targets | orange: completion | teal: actual noisy source | purple: revised positions | green: next K.\n"
                 "Targets are unclipped, including outside the dotted simulator arena; physical motion is not inferred from plans.",
                 ha="center", fontsize=10)
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)


def render_revision_process(episode, config, output, *, start_replan=1, replans=3,
                            fps=10, save_gif=False):
    """Write a streamed MP4 and one four-panel PNG per consecutive replan.

    ``output`` is an explicit directory or an MP4 filename. Playback holds are
    illustrative, not measurements of generation latency. Rendering never opens
    a simulator, samples a policy, modifies a plan, or buffers movie frames unless
    ``save_gif`` is requested. Returned paths and metadata are JSON serializable.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(config["robot_dt"]) or config["robot_dt"] <= 0:
        raise ValueError("robot_dt must be positive and finite")
    states, actions, traces = _prepare(episode, config, int(start_replan), int(replans))
    output = Path(output)
    video = output if output.suffix.lower() == ".mp4" else output / "revision-process.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    bounds = _bounds(traces, states)
    storyboards = []
    for trace in traces:
        path = video.with_name(f"{video.stem}-replan{trace['replan']:03d}.png")
        _storyboard(path, trace, states, actions, config, bounds)
        storyboards.append(str(path))

    def frames():
        for trace in traces:
            yield from _phase_frames(trace, int(config["horizon"]))
        last = traces[-1]
        # The last ready plan also gets actual execution, bounded by early termination.
        end = min(len(actions), last["step"] + int(config["execute"]))
        for step in range(last["step"] + 1, end + 1):
            yield _Frame(last, "EXECUTE_FINAL", step, origin=last["step"], seconds=.3)

    import imageio.v2 as imageio

    plt = _import_pyplot()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), left=.06, right=.98,
                           bottom=.2, top=.79, wspace=.3, hspace=.2)
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
    title = fig.suptitle("", fontsize=16, y=.96)
    subtitle = fig.text(.5, .865, "", ha="center", fontsize=10)
    caption = fig.text(.5, .09, "", ha="center", fontsize=10, wrap=True)
    fig.text(.5, .035, "Illustrative slow playback / phase holds, not measured wall time. "
             "Plans are target positions, never simulated motion or kinetic velocities.", ha="center", fontsize=9, color="0.35")
    gif_frames = [] if save_gif else None
    frame_count = 0
    phases = []
    try:
        with imageio.get_writer(video, format="FFMPEG", fps=fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=2,
                                ffmpeg_params=["-threads", "1"]) as writer:
            for frame in frames():
                _draw_panel(axes, frame, states, actions, config, bounds)
                title.set_text(f"{config['method']} | replan {frame.trace['replan']} | {frame.phase.replace('_', ' ')}")
                subtitle.set_text(f"H={config['horizon']} targets | K={config['execute']} executed / replan | "
                                  f"physical step {frame.step} | {_COMPLETIONS.get(int(frame.trace['completion_id']), 'unknown')} completion")
                caption.set_text(_caption(frame, config))
                fig.canvas.draw()
                pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                repeats = max(1, round(fps * frame.seconds))
                for _ in range(repeats):
                    writer.append_data(pixels)
                    if gif_frames is not None:
                        gif_frames.append(pixels)
                frame_count += repeats
                if not phases or phases[-1] != frame.phase:
                    phases.append(frame.phase)
    finally:
        plt.close(fig)
    gif = None
    if gif_frames is not None:
        gif = video.with_suffix(".gif")
        imageio.mimsave(gif, gif_frames, format="GIF", duration=1000 / fps, loop=0)
    return {"video": str(video), "storyboards": storyboards,
            "gif": None if gif is None else str(gif),
            "replans": [trace["replan"] for trace in traces], "frame_count": frame_count,
            "fps": float(fps), "phases": phases, "illustrative_timing": True,
            "first_physical_step": traces[0]["step"] - traces[0]["old_executed"],
            "last_physical_step": min(len(actions), traces[-1]["step"] + int(config["execute"]))}
