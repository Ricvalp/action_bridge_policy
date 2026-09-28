"""Many candidate revisions at one fixed Push-T decision or synthetic context.

Sampling holds the physical scene and policy context fixed. It does not simulate,
score, select or execute any candidate, nor claim to clone full physics state.
"""
from __future__ import annotations

import numpy as np
import torch

from action_bridge.data.pusht_adapter import normalize_observations_np
from action_bridge.eval.revision_pusht import validate_evaluation_completion
from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.checkpoints import frozen_copy
from action_bridge.plan_revision.completion import complete_plan
from action_bridge.plan_revision.contracts import ActionCodec
from action_bridge.plan_revision.tracking import preserve_rng
from action_bridge.plan_revision.training import restore_completion


SOURCE_VARIANTS = ("axial", "right_tilt_15", "right_turn_90", "right_route")


def context_at_replan(episode, config, replan):
    """Reconstruct exactly the histories used by the existing closed-loop runner.

    Histories use measured states and issued (clipped) commands, not predicted
    positions or unclipped commands. Early histories use the initial-state pad.
    """
    traces = {trace["replan"]: trace for trace in episode["plan_traces"]}
    if replan not in traces:
        raise ValueError(f"Episode did not reach replan {replan}; available: {sorted(traces)}")
    trace = traces[replan]
    step = int(trace["step"])
    states = np.asarray(episode["states_raw"], dtype=np.float32)
    commands = np.asarray(episode["actions_raw"], dtype=np.float32).reshape(-1, 2)
    obs_indices = np.maximum(np.arange(step - config["obs_history"] + 1, step + 1), 0)
    command_history = [commands[index] if index >= 0 else states[0, :2]
                       for index in range(step - config["action_history"], step)]
    return {
        "state_raw": states[step].copy(), "obs_history_raw": states[obs_indices].copy(),
        "action_history_raw": np.stack(command_history),
        "old_plan_raw": None if trace["old_plan_raw"] is None else np.asarray(trace["old_plan_raw"], dtype=np.float32),
        "completed_raw": np.asarray(trace["completed_raw"], dtype=np.float32),
        "fixed_source_raw": np.asarray(trace["perturbed_source_raw"], dtype=np.float32),
        "executed": int(trace["old_executed"]), "has_previous_plan": bool(trace["has_previous_plan"]),
        "replan": replan, "robot_step": step,
    }


@torch.no_grad()
def symmetric_context(config, metadata, *, completion_id=1):
    """A hand-built wrong-side probe, not a simulated or learner-reached state.

    T and goal have the same orientation and share their reflection axis. The
    pusher is on the goal-facing side of the crossbar: pushing toward the goal
    requires going around the T. Its old targets retreat along that axis.
    Histories assume ideal target tracking; the T stays still. Fixed damping or
    repeat completion preserves symmetry without altering any learned weights.
    """
    horizon, executed = config["horizon"], config["execute"]
    if not 0 < executed < horizon:
        raise ValueError("The symmetric previous-chunk probe requires 0 < K < H")
    if completion_id not in (0, 1):
        raise ValueError("Use repeat or fixed_damped completion to keep the source symmetric")
    validate_evaluation_completion(config, completion_id)
    theta = np.pi / 4
    axis = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    lateral = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    goal = np.array([256., 256., theta], dtype=np.float32)
    block = goal[:2] + 140. * axis
    pusher = block - 40. * axis
    # Keep even longer synthetic histories outside the crossbar (radius 15).
    longest_past = max(executed - 1, config["obs_history"] - 1, config["action_history"] - 1, 1)
    speed = min(2., 14. / longest_past)

    def targets(offsets):
        return pusher - speed * np.asarray(offsets, dtype=np.float32)[:, None] * axis

    old = targets(np.arange(horizon) - executed + 1)
    actions = targets(np.arange(1 - config["action_history"], 1))
    state = np.r_[pusher, block, np.float32(theta)].astype(np.float32)
    observations = np.repeat(state[None], config["obs_history"], axis=0)
    observations[:, :2] = targets(np.arange(1 - config["obs_history"], 1))
    codec = ActionCodec(**metadata["codec"])
    completed = complete_plan(
        codec.encode(torch.from_numpy(old))[None], executed,
        torch.from_numpy(normalize_observations_np(observations, metadata["normalization"]))[None],
        codec.encode(torch.from_numpy(actions))[None], completion_id,
        robot_dt=config["robot_dt"],
    )
    completed = codec.decode(completed)[0].numpy()
    return {
        "state_raw": state, "obs_history_raw": observations, "action_history_raw": actions,
        "old_plan_raw": old, "completed_raw": completed, "fixed_source_raw": completed.copy(),
        "executed": executed, "has_previous_plan": True, "replan": None, "robot_step": None,
        "scene_kind": "synthetic_symmetric", "symmetry_axis_raw": axis,
        "lateral_axis_raw": lateral, "goal_pose_raw": goal,
        "synthetic_history": "Stationary T; ideal pusher tracking of axial retreat commands, not simulated",
        "synthetic_speed_px_per_action": speed,
    }


@torch.no_grad()
def source_variant_context(context, config, metadata, *, variant="axial", completion_id=1):
    """Change only the synthetic old plan's unexecuted suffix, then complete it.

    Right means +lateral_axis_raw, not horizontal screen-right. Rotations keep
    the original radial distances from the pusher. The longer right_route is a
    hand-built go-around template, designed for H16/K8. Neither executed history
    nor the frozen reference changes; no plan is physically executed here.
    """
    if variant not in SOURCE_VARIANTS:
        raise ValueError(f"Unknown source variant: {variant}")
    if context.get("scene_kind") != "synthetic_symmetric" or completion_id not in (0, 1):
        raise ValueError("Source interventions require the synthetic scene and repeat/fixed_damped completion")
    labels = {"axial": "Axial source", "right_tilt_15": "15 deg right tilt",
              "right_turn_90": "90 deg right turn", "right_route": "Right-side route"}
    result = {**context, "source_variant": variant, "source_label": labels[variant]}
    if variant == "axial":
        return result
    old = context["old_plan_raw"].copy()
    executed = context["executed"]
    pusher, block = context["state_raw"][:2], context["state_raw"][2:4]
    axis, lateral = context["symmetry_axis_raw"], context["lateral_axis_raw"]
    if variant == "right_route":
        # Local (lateral, longitudinal) coordinates: crossbar [-60,60]x[0,30],
        # stem [-15,15]x[30,120]. This route goes around the positive-x side.
        route = np.array([[0., -40.], [15., -40.], [40., -35.], [78., -22.],
                          [82., 10.], [72., 55.], [50., 100.], [40., 130.], [35., 134.]], dtype=np.float32)
        sample_times = np.linspace(0., 1., len(old) - executed + 1)[1:]
        local = np.stack([np.interp(sample_times, np.linspace(0., 1., len(route)), route[:, j])
                          for j in range(2)], axis=-1).astype(np.float32)
        old[executed:] = block + local[:, :1] * lateral + local[:, 1:] * axis
    else:
        angle = np.deg2rad(15. if variant == "right_tilt_15" else 90.)
        distance = (old[executed:] - pusher) @ (-axis)
        direction = -np.cos(angle) * axis + np.sin(angle) * lateral
        old[executed:] = pusher + distance[:, None] * direction
    codec = ActionCodec(**metadata["codec"])
    completed = complete_plan(
        codec.encode(torch.from_numpy(old))[None], executed,
        torch.from_numpy(normalize_observations_np(context["obs_history_raw"], metadata["normalization"]))[None],
        codec.encode(torch.from_numpy(context["action_history_raw"]))[None], completion_id,
        robot_dt=config["robot_dt"],
    )
    completed = codec.decode(completed)[0].numpy()
    return {**result, "old_plan_raw": old, "completed_raw": completed, "fixed_source_raw": completed.copy()}


@torch.no_grad()
def sample_candidate_revisions(policy, config, metadata, dependencies, context, device="cpu", *,
                               candidates=200, batch_size=32, seed=42,
                               completion_id=2, source_noise="fixed"):
    """One fixed context shared by every candidate, with fresh generation noise.

    fixed: repeat the actual noisy source of the recorded decision, isolating
           SB sampler randomness (and kinetic auxiliary-velocity randomness).
    independent: add a fresh source perturbation to the same completed old plan.
    none: use the clean completed plan; SB stochastic dynamics remain active.

    FM is deterministic for a fixed source: use independent source noise to
    inspect FM diversity. Reproducibility includes batch_size, which affects
    how random draws are assigned to candidates.
    """
    if candidates < 1 or batch_size < 1 or seed < 0:
        raise ValueError("Require positive candidates/batch_size and nonnegative seed")
    if source_noise not in ("fixed", "independent", "none"):
        raise ValueError("source_noise must be fixed, independent or none")
    if config["method"] not in ("fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"):
        raise ValueError("Candidate revisions require an FM or SB checkpoint")
    validate_evaluation_completion(config, completion_id)
    codec = ActionCodec(**metadata["codec"])
    with preserve_rng():
        sampler = frozen_copy(policy).to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        obs = torch.as_tensor(normalize_observations_np(context["obs_history_raw"], metadata["normalization"]),
                              dtype=torch.float32, device=device)[None]
        actions = codec.encode(torch.as_tensor(context["action_history_raw"], device=device))[None]
        completed = codec.encode(torch.as_tensor(context["completed_raw"], device=device))[None]
        fixed_source = codec.encode(torch.as_tensor(context["fixed_source_raw"], device=device))[None]
        prior = None
        if config["method"].startswith("sb_"):
            completion = restore_completion(dependencies, device)
            prior = completion.prior(
                obs, actions, torch.as_tensor(dependencies["innovation_variance"], device=device),
                ridge=config["prior_ridge"], max_rate=config["max_rate"],
                mobility_smoothing=config["mobility_smoothing"],
            )
        plans, sources = [], []
        for offset in range(0, candidates, batch_size):
            count = min(batch_size, candidates - offset)
            source = (fixed_source if source_noise == "fixed" else completed).expand(count, -1, -1).clone()
            if source_noise == "independent":
                source += config["source_std"] * torch.randn(source.shape, device=device, generator=generator)
            reference = None
            if prior is not None:
                reference_batch = {"precision": prior["precision"].expand(count, -1, -1),
                                   "prior_mean": prior["mean"].expand(count, -1)}
                if config["mobility_smoothing"]:
                    reference_batch["mobility"] = prior["mobility"]
                reference = reference_for(reference_batch, config)
            generated, _ = sampler.sample(
                obs.expand(count, -1, -1), actions.expand(count, -1, -1), completion_id,
                source_actions=source, reference=reference, generator=generator,
                has_previous_plan=torch.full((count,), context["has_previous_plan"], device=device, dtype=torch.bool),
            )
            if generated.shape != (count, config["horizon"], 2) or not torch.isfinite(generated).all():
                raise ValueError("Candidate sampler returned nonfinite or incorrectly shaped targets")
            plans.append(codec.decode(generated).cpu().numpy())
            sources.append(codec.decode(source).cpu().numpy())
    return {**context, "candidates_raw": np.concatenate(plans), "sources_raw": np.concatenate(sources),
            "source_noise_mode": source_noise, "execute": config["execute"]}


def candidate_statistics(data, execute):
    """Spread of commands, not physical outcomes or evidence of useful diversity."""
    plans = data["candidates_raw"].astype(np.float64)
    center = plans.mean(axis=0)
    spread = np.linalg.norm(plans - center[None], axis=-1)
    return {
        "candidates": len(plans),
        "plan_rms_spread_px": float(np.sqrt(np.mean(spread**2))),
        "first_command_rms_spread_px": float(np.sqrt(np.mean(spread[:, 0]**2))),
        "kth_command_rms_spread_px": float(np.sqrt(np.mean(spread[:, execute - 1]**2))),
        "last_command_rms_spread_px": float(np.sqrt(np.mean(spread[:, -1]**2))),
        "outside_workspace_target_fraction": float(((plans < 0) | (plans > 512)).any(axis=-1).mean()),
    }
