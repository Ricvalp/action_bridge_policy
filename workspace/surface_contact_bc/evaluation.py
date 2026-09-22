"""One real-state, receding-horizon evaluator for every surface-contact policy.

All thresholds below are evaluation criteria, never training losses. Physical
position ``x`` and the policy's absolute Cartesian target ``q`` stay distinct.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .env import APPROACH_END, ENGAGE_END, SLIDE_END, Context, SurfaceContactEnv


@dataclass(frozen=True)
class MetricThresholds:
    tangent_error: float = 0.025
    terminal_speed: float = 0.08
    contact_tail_seconds: float = 0.5
    contact_tail_fraction: float = 0.9
    clean_penetration: float = 0.02
    clean_peak_force: float = 30.0
    clean_contact_losses: int = 1
    chatter_velocity: float = 0.01
    recovery_distance: float = 0.003
    recovery_normal_speed: float = 0.04
    recovery_hold_seconds: float = 0.2


def _history(values: list[np.ndarray], length: int) -> np.ndarray:
    """Repeat the first real value when a history has not filled yet."""
    return np.stack(([values[0]] * max(0, length - len(values)) + values)[-length:])


def rollout(
    policy,
    normalizer,
    context: Context,
    *,
    n_exec: int = 2,
    seed: int = 0,
    impulse: float = 0.0,
    impulse_time: float = 3.0,
) -> dict[str, np.ndarray]:
    """Execute targets through the shared PD controller, replanning on real state.

    The first action-history entries are the initial physical position (a hold
    target), not zero targets or an expert action. A fixed random seed makes
    diffusion's initial noise reproducible; no candidate selection is used.
    Positive impulses point into free space and are measured in N s.
    """
    if not 1 <= n_exec <= policy.chunk_horizon:
        raise ValueError("n_exec must be between 1 and the policy chunk horizon")
    if impulse and not ENGAGE_END <= impulse_time < min(SLIDE_END, context.duration):
        raise ValueError("apply the impulse during sliding: 1.8 <= time < 4.8 seconds")
    env = SurfaceContactEnv(context)
    observations = [np.asarray(env.observe(), dtype=np.float64)]
    action_history = [env.x.copy()]
    targets: list[np.ndarray] = []
    infos = [dict(env.last_info)]
    diagnostics: dict[str, list[np.ndarray]] = {}
    replans, impulses, inference_seconds = [], [], []
    impulse_applied = False
    device = next(policy.parameters(), torch.empty(0)).device
    generator = torch.Generator(device=device).manual_seed(seed)
    policy.eval()
    total_steps = int(np.ceil(context.duration / context.physics.control_dt))
    while len(targets) < total_steps:
        obs_hist = normalizer.normalize_obs(_history(observations, policy.obs_history))
        act_hist = normalizer.normalize_action(_history(action_history, policy.action_history))
        obs_tensor = torch.as_tensor(obs_hist, dtype=torch.float32, device=device)[None]
        act_tensor = torch.as_tensor(act_hist, dtype=torch.float32, device=device)[None]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        with torch.no_grad():
            predicted, auxiliary = policy.generate(
                obs_tensor, act_tensor, generator=generator, diagnostics=True
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds.append(perf_counter() - start)
        chunk = normalizer.denormalize_action(predicted[0].detach().cpu().numpy())
        if not np.isfinite(chunk).all():
            raise FloatingPointError("policy generated a non-finite Cartesian target")
        chunk_info = {key: value[0].detach().cpu().numpy() for key, value in auxiliary.items()}
        count = min(n_exec, total_steps - len(targets))
        for offset in range(count):
            apply_impulse = bool(impulse and not impulse_applied and env.time + 1e-9 >= impulse_time)
            value = impulse if apply_impulse else 0.0
            target = np.asarray(chunk[offset], dtype=np.float64)
            observation = env.step(target, impulse=value)
            impulse_applied |= apply_impulse
            observations.append(np.asarray(observation, dtype=np.float64))
            action_history.append(target)
            targets.append(target)
            infos.append(dict(env.last_info))
            replans.append(offset == 0)
            impulses.append(value)
            for key, series in chunk_info.items():
                diagnostics.setdefault(key, []).append(np.asarray(series[offset]))
    trace = {
        "observations": np.stack(observations),
        "actions": np.stack(targets),
        "replan": np.asarray(replans, dtype=bool),
        "impulse": np.asarray(impulses),
        "inference_seconds": np.asarray(inference_seconds),
    }
    for key in infos[0]:
        trace[key] = np.asarray([info[key] for info in infos])
    trace.update({key: np.stack(values) for key, values in diagnostics.items()})
    return trace


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _sustained_start(mask: np.ndarray, count: int) -> int | None:
    """First start index of ``count`` consecutive true samples."""
    if len(mask) < count:
        return None
    candidates = np.flatnonzero(np.convolve(mask.astype(int), np.ones(count, int), "valid") >= count)
    return int(candidates[0]) if len(candidates) else None


def _series(trace: dict, key: str, fallback) -> np.ndarray:
    # Expert datasets keep their simulator diagnostics grouped under "infos".
    return np.asarray(trace.get(key, trace.get("infos", {}).get(key, fallback)))


def episode_metrics(
    trace: dict,
    context: Context,
    *,
    thresholds: MetricThresholds | None = None,
    counterfactual: dict | None = None,
) -> dict:
    """Task, physical contact, and command-path metrics, in separate fields.

    Recovery time is conditional on actually recovering, accompanied by a
    success flag and eligibility flag. Progress loss needs a paired rollout
    without an impulse. Missing measurements are JSON null, not fabricated 0.
    """
    limits = thresholds or MetricThresholds()
    obs, targets = np.asarray(trace["observations"]), np.asarray(trace["actions"])
    dt = context.physics.control_dt
    times = obs[:, 10]
    normal, tangent = obs[0, 4:6], np.array([obs[0, 5], -obs[0, 4]])
    distance, goal = obs[:, 7], float(obs[0, 8])
    tangent_position = obs[:, :2] @ tangent
    normal_velocity = obs[:, 2:4] @ normal
    speed = np.linalg.norm(obs[:, 2:4], axis=-1)
    force = _series(trace, "normal_force", obs[:, 9]).astype(float)
    peak_force = _series(trace, "peak_normal_force", force).astype(float)
    penetration = np.maximum(-distance, 0.0)
    peak_penetration = _series(trace, "peak_penetration", penetration).astype(float)
    contact = _series(trace, "contact", (distance < 0) & (force > 0)).astype(bool)
    tail_count = min(len(obs), max(1, round(limits.contact_tail_seconds / dt)))
    terminal_error = abs(tangent_position[-1] - goal)
    task_at_time = (np.abs(tangent_position - goal) <= limits.tangent_error) & (speed <= limits.terminal_speed)
    task_success = bool(task_at_time[-1])
    contact_fraction = float(contact[-tail_count:].mean())
    contact_success = contact_fraction >= limits.contact_tail_fraction
    engaged = np.flatnonzero(contact)
    after_engagement = contact[engaged[0]:] if len(engaged) else np.ones(0, bool)
    contact_losses = int(np.count_nonzero(after_engagement[:-1] & ~after_engagement[1:]))
    contact_loss_duration = float(np.count_nonzero(~after_engagement) * dt)
    clean = bool(
        task_success and contact_success
        and peak_penetration.max() <= limits.clean_penetration
        and peak_force.max() <= limits.clean_peak_force
        and contact_losses <= limits.clean_contact_losses
    )
    # A completion time is reported only if the final task condition persists.
    completion_time = None
    if task_success:
        not_reached = np.flatnonzero(~task_at_time)
        completion_time = float(times[not_reached[-1] + 1] if len(not_reached) else times[0])
    # Ignore approach and near-zero numerical velocity flips when counting chatter.
    near_contact = np.arange(len(obs)) >= (engaged[0] if len(engaged) else len(obs))
    significant = normal_velocity[near_contact & (np.abs(normal_velocity) > limits.chatter_velocity)]
    chatter = int(np.count_nonzero(significant[1:] * significant[:-1] < 0))
    velocity = np.diff(np.vstack([obs[0, :2], targets]), axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    target_jumps = np.linalg.norm(np.diff(targets, axis=0), axis=-1)
    boundaries = np.asarray(trace.get("replan", np.zeros(len(targets), bool)))[1:]
    boundary_mean = float(target_jumps[boundaries].mean()) if np.any(boundaries) else None
    ordinary_mean = float(target_jumps[~boundaries].mean()) if np.any(~boundaries) else None
    result = {
        "task_success": task_success,
        "contact_success": bool(contact_success),
        "clean_success": clean,
        "tangential_terminal_error": float(terminal_error),
        "tangential_tracking_rmse": None,
        "completion_time": completion_time,
        "terminal_speed": float(speed[-1]),
        "peak_normal_force": float(peak_force.max()),
        "rms_normal_force": _rms(force),
        "max_penetration": float(peak_penetration.max()),
        "contact_tail_fraction": contact_fraction,
        "contact_loss_count": contact_losses,
        "contact_loss_duration": contact_loss_duration,
        "normal_velocity_energy": float(np.sum(normal_velocity[1:] ** 2) * dt),
        "normal_velocity_energy_after_engagement": float(np.sum(normal_velocity[near_contact] ** 2) * dt),
        "normal_velocity_chatter": chatter,
        "target_velocity_rms": _rms(np.linalg.norm(velocity, axis=-1)),
        "target_acceleration_rms": _rms(np.linalg.norm(acceleration, axis=-1)),
        "target_jerk_rms": _rms(np.linalg.norm(jerk, axis=-1)),
        "replan_target_jump_mean": boundary_mean,
        "within_chunk_target_jump_mean": ordinary_mean,
        "chunk_boundary_discontinuity": (
            boundary_mean - ordinary_mean if boundary_mean is not None and ordinary_mean is not None else None
        ),
        "inference_seconds_per_replan": (
            float(np.mean(trace["inference_seconds"])) if len(trace.get("inference_seconds", [])) else None
        ),
        "replans": int(len(trace.get("inference_seconds", []))),
        "recovery_eligible": None,
        "recovery_success": None,
        "recovery_time": None,
        "post_impulse_tangential_progress_loss": None,
    }
    if "tangential_reference" in trace:
        result["tangential_tracking_rmse"] = _rms(tangent_position - trace["tangential_reference"])
    impulses = np.asarray(trace.get("impulse", np.zeros(len(targets))))
    impulse_indices = np.flatnonzero(impulses)
    result["impulse"] = float(impulses[impulse_indices[0]]) if len(impulse_indices) else 0.0
    result["impulse_time"] = None
    if len(impulse_indices):
        index = int(impulse_indices[0])
        result["impulse_time"] = float(times[index])
        start = max(0, index - round(0.2 / dt))
        reference_distance = float(np.median(distance[start:index + 1]))
        eligible = bool(contact[start:index + 1].mean() >= 0.8)
        result["recovery_eligible"] = eligible
        stable = (
            contact[index + 1:]
            & (np.abs(distance[index + 1:] - reference_distance) <= limits.recovery_distance)
            & (np.abs(normal_velocity[index + 1:]) <= limits.recovery_normal_speed)
        )
        first = _sustained_start(stable, max(1, round(limits.recovery_hold_seconds / dt))) if eligible else None
        result["recovery_success"] = (first is not None) if eligible else None
        if first is not None:
            result["recovery_time"] = float(times[index + 1 + first] - times[index])
        if counterfactual is not None:
            baseline_position = np.asarray(counterfactual["observations"])[:, :2] @ tangent
            direction = float(np.sign(goal - tangent_position[0]))
            baseline_progress = baseline_position[-1] - baseline_position[index]
            progress = tangent_position[-1] - tangent_position[index]
            result["post_impulse_tangential_progress_loss"] = float(direction * (baseline_progress - progress))
    phase_time = times[:-1]
    phases = {"approach": phase_time < APPROACH_END,
              "engage": (phase_time >= APPROACH_END) & (phase_time < ENGAGE_END),
              "slide": (phase_time >= ENGAGE_END) & (phase_time < SLIDE_END),
              "stop": phase_time >= SLIDE_END}
    for name in ("gamma_normal", "gamma_tangent", "stiffness", "desired_offset", "control_ref_ratio"):
        if name not in trace:
            continue
        values = np.asarray(trace[name])
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
        for phase, mask in phases.items():
            result[f"{name}_{phase}_mean"] = float(values[mask].mean()) if mask.any() else None
    if "gamma_normal" in trace:
        result["gamma_normal_minus_tangent_mean"] = float(
            np.mean(np.asarray(trace["gamma_normal"]) - np.asarray(trace["gamma_tangent"]))
        )
    for name in ("ref_normal", "ref_tangent", "residual_normal", "residual_tangent", "control_normal", "control_tangent"):
        if name in trace:
            result[f"{name}_energy"] = float(np.sum(np.asarray(trace[name]) ** 2))
    if "path_kl" in trace:
        result["path_kl_executed_sum"] = float(np.sum(trace["path_kl"]))
    return result


def aggregate_metrics(episodes: list[dict]) -> dict:
    """Means across episodes; null values are excluded and counts are retained."""
    if not episodes:
        raise ValueError("evaluation needs at least one episode")
    summary = {"num_episodes": len(episodes)}
    for key in episodes[0]:
        if key in ("episode", "context_seed"):
            continue
        name = key.replace("_success", "_success_rate") if key.endswith("_success") else key
        values = [row[key] for row in episodes if row.get(key) is not None and isinstance(row[key], (int, float, bool))]
        summary[f"{name}_count"] = len(values)
        if not values:
            summary[name] = None
            continue
        summary[name] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
    return summary


def evaluate_policy(
    policy,
    normalizer,
    contexts: list[Context],
    output: str | Path,
    *,
    n_exec: int = 2,
    seed: int = 0,
    impulse: float = 0.0,
    impulse_time: float = 3.0,
    plots: bool = True,
    label: str | None = None,
    gif: bool = False,
    thresholds: MetricThresholds | None = None,
) -> dict:
    """Save per-episode JSON/NPZ plus aggregate metrics and representative plots.

    Disturbed episodes are paired with the same context, noise seed, and policy
    without the impulse. Thus progress loss measures a disturbance effect, not
    variation between unrelated tasks. No task metric selects a sampled action.
    """
    from .plotting import plot_episode, plot_reference_overlay, save_rollout_gif

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace_directory = output / "traces"
    trace_directory.mkdir(exist_ok=True)
    episodes, first_trace = [], None
    for index, context in enumerate(contexts):
        trace = rollout(policy, normalizer, context, n_exec=n_exec, seed=seed + index,
                        impulse=impulse, impulse_time=impulse_time)
        counterfactual = rollout(policy, normalizer, context, n_exec=n_exec, seed=seed + index) if impulse else None
        metrics = episode_metrics(trace, context, thresholds=thresholds, counterfactual=counterfactual)
        metrics.update({"episode": index, "context_seed": context.seed})
        episodes.append(metrics)
        np.savez_compressed(trace_directory / f"episode_{index:03d}.npz", **trace)
        if counterfactual is not None:
            np.savez_compressed(trace_directory / f"episode_{index:03d}_no_impulse.npz", **counterfactual)
        if first_trace is None:
            first_trace = trace
    summary = aggregate_metrics(episodes)
    summary.update({"label": label, "n_exec": n_exec, "evaluation_seed": seed,
                    "thresholds": asdict(thresholds or MetricThresholds()),
                    "contexts": [context.to_dict() for context in contexts]})
    (output / "episodes.json").write_text(json.dumps(episodes, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if first_trace is not None and plots:
        plot_episode(first_trace, contexts[0], output, title=label or "Surface-contact rollout")
        if hasattr(policy, "reference_only") and not policy.reference_only:
            reference_policy = deepcopy(policy)
            reference_policy.reference_only = True
            reference_trace = rollout(reference_policy, normalizer, contexts[0], n_exec=n_exec,
                                      seed=seed, impulse=impulse, impulse_time=impulse_time)
            np.savez_compressed(trace_directory / "episode_000_reference_only.npz", **reference_trace)
            plot_reference_overlay(first_trace, reference_trace, output / "reference_only_overlay.png")
    if first_trace is not None and gif:
        save_rollout_gif(first_trace, contexts[0], output / "rollout.gif")
    return summary
