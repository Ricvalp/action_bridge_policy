"""Push-T attachment for the generic chunk samplers and execution cache.

No demonstration labels are accepted here. Histories contain actual observations
and executed, bounded commands; cached plans are predictions, not action history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from action_bridge.data.pusht_adapter import normalize_observations_np
from action_bridge.eval.pusht_sim import (
    _action_bounds, _info_success, _make_pusht_env, _obs_to_state, _reset_env,
)
from action_bridge.plan_revision.cache import reference_for
from action_bridge.plan_revision.completion import COMPLETION_NAMES, complete_plan
from action_bridge.plan_revision.contracts import ActionCodec, PlanCache
from action_bridge.plan_revision.training import restore_completion


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _overlay(frame, aligned, completed, final, states, low, high, label):
    """Plan lines are candidates; the white-on-black trail is physical motion."""
    from PIL import Image, ImageDraw
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    size = np.asarray(image.size, dtype=np.float32)

    def points(path):
        path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
        return [tuple(point) for point in ((path - low) / (high - low) * size)]

    for path, color in ((aligned, "gray"), (completed, "orange"), (final, "purple")):
        if path is not None and len(path):
            coordinates = points(path)
            if len(coordinates) > 1:
                draw.line(coordinates, fill=color, width=2)
            for x, y in coordinates:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    if len(states) > 1:
        coordinates = points(np.asarray(states)[:, :2])
        draw.line(coordinates, fill="black", width=4)
        draw.line(coordinates, fill="white", width=2)
    draw.rectangle((0, 0, image.width, 34), fill="white")
    draw.text((4, 2), label, fill="black")
    draw.text((4, 18), "gray old | orange completed | purple revised | white pusher", fill="black")
    return image


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_video(frames, path):
    """Write a 10 Hz Push-T rollout as a browser-playable MP4."""
    import imageio.v2 as imageio

    # Stream frames instead of creating another full copy of the rollout.
    with imageio.get_writer(path, format="FFMPEG", fps=10, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=2,
                            ffmpeg_params=["-threads", "1"]) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


@torch.no_grad()
def evaluate(policy, config, metadata, dependencies, device, output=None,
             completion_id=2, seeds=None, render=False, save_videos=True,
             save_gifs=False):
    """Evaluate one sample per seed; save first two successes/failures as MP4s.

    Uses the legacy benchmark success definition (termination, explicit success,
    or reward >= .95). ``env_success_rate`` also reports the stricter native
    environment flag, while coverage metrics use actual ``info['coverage']``.
    Revisers generate their first chunk from an observed anchor, without any
    external policy or future labels. Later sources use their own old plans.
    ``render=False`` disables all frame collection, regardless of media flags.
    GIFs are optional; media stays in ``output`` and is not uploaded to W&B.
    """
    device = torch.device(device)
    if (config["obs_dim"], config["action_dim"]) != (5, 2):
        raise ValueError("This evaluator is the Push-T 5-state/2-target adapter")
    if completion_id not in range(3):
        raise ValueError("completion_id must be 0, 1 or 2")
    horizon, executed = int(config["horizon"]), int(config["execute"])
    if not 1 <= executed <= horizon:
        raise ValueError("execute must satisfy 1 <= K <= H")
    if config["max_episode_steps"] < 1:
        raise ValueError("max_episode_steps must be positive")
    seeds = list(config["evaluation_seeds"] if seeds is None else seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if render and output is None:
        raise ValueError("Rendering requires an explicit output directory")
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    codec = ActionCodec(**metadata["codec"])
    stats = metadata["normalization"]
    reviser = config["method"] != "ddim"
    completion = restore_completion(dependencies, device) if reviser else None
    policy.eval()
    env = _make_pusht_env(render_mode="rgb_array", obs_type="state")
    cache = PlanCache()
    episodes, saved = [], {"success": 0, "failure": 0}
    try:
        low, high = _action_bounds(env)
        if not (np.allclose(low, codec.low) and np.allclose(high, codec.high)):
            raise ValueError("Saved codec bounds differ from the Push-T environment")
        for seed in seeds:
            cache.reset()
            generator = torch.Generator(device=device).manual_seed(int(seed) + 123)
            state, _ = _reset_env(env, int(seed))
            obs_hist = np.repeat(state[None], config["obs_history"], axis=0)
            act_hist = np.repeat(state[None, :2], config["action_history"], axis=0)
            actions, preclipped, rewards, coverages, states = [], [], [], [], [state.copy()]
            boundaries, revisions, timings, energies, coefficient_rows = [], [], [], [], []
            tail_speeds, tail_accelerations = [], []
            traces, frames, nfes = [], [], []
            startup_nfe, startups, clipped_count = 0, 0, 0
            terminated = truncated = False
            info = {}
            collect_frames = (render and (save_videos or save_gifs)
                              and (saved["success"] < 2 or saved["failure"] < 2))
            while len(actions) < config["max_episode_steps"]:
                obs_tensor = torch.as_tensor(normalize_observations_np(obs_hist, stats)[None],
                                             device=device, dtype=torch.float32)
                act_tensor = codec.encode(torch.as_tensor(act_hist[None], device=device, dtype=torch.float32))
                has_previous_plan = cache.plan is not None and cache.executed < horizon
                startup = reviser and not has_previous_plan
                aligned = completed = source = reference = None
                aligned_raw = completed_raw = None
                prior = None
                _synchronize(device)
                start = time.perf_counter()
                # Preserve the exact overlap before replacing the cache. DDIM
                # is also measured against its previous independently drawn plan.
                if has_previous_plan:
                    aligned = cache.plan[:, cache.executed:].clone()
                    aligned_raw = _array(codec.decode(aligned)[0])
                if reviser:
                    if startup:
                        # This is the absolute-target action adapter's startup
                        # rule. Before the first command act_hist is initialized
                        # to the observed pusher; later it contains real commands.
                        completed = act_tensor[:, -1:].expand(-1, horizon, -1).clone()
                    else:
                        completed = complete_plan(cache.plan, cache.executed, obs_tensor, act_tensor,
                                                  completion_id, completion, robot_dt=config["robot_dt"])
                        # Include the last two old targets so tail acceleration
                        # measures the boundary as well as later continuation.
                        tail_commands = np.concatenate([_array(codec.decode(cache.plan)[0, -2:]),
                                                        _array(codec.decode(completed)[0, horizon - cache.executed:])])
                        velocity = np.diff(tail_commands, axis=0) / config["robot_dt"]
                        tail_speeds.append(float(np.sqrt(np.mean(np.sum(velocity[1:]**2, axis=-1)))))
                        acceleration = np.diff(velocity, axis=0) / config["robot_dt"]
                        tail_accelerations.append(float(np.sqrt(np.mean(np.sum(acceleration**2, axis=-1)))))
                    completed_raw = _array(codec.decode(completed)[0])
                    source = completed + config["source_std"] * torch.randn(
                        completed.shape, device=device, dtype=completed.dtype, generator=generator)
                    if config["method"].startswith("sb_"):
                        prior = completion.prior(obs_tensor, act_tensor,
                            torch.as_tensor(dependencies["innovation_variance"], device=device),
                            ridge=config["prior_ridge"], max_rate=config["max_rate"],
                            mobility_smoothing=config["mobility_smoothing"])
                        reference_batch = {"precision": prior["precision"], "prior_mean": prior["mean"]}
                        if config["mobility_smoothing"]:
                            reference_batch["mobility"] = prior["mobility"]
                        reference = reference_for(reference_batch, config)
                        _, stiffness, damping = completion.coefficients(obs_tensor, act_tensor)
                        coefficient_rows.append({"stiffness": float(stiffness.mean()),
                            "damping": float(damping.mean()), "rate_min": float(prior["rate_min"].mean()),
                            "rate_max": float(prior["rate_max"].mean()),
                            "stabilized_fraction": float(prior["stabilized_fraction"].mean())})
                    generated, diagnostics = policy.sample(obs_tensor, act_tensor, completion_id,
                        source_actions=source, reference=reference, generator=generator,
                        has_previous_plan=torch.tensor([has_previous_plan], device=device))
                    if startup:
                        startups += 1
                        startup_nfe += int(diagnostics.get("nfe", 0))
                else:
                    generated, diagnostics = policy.sample(obs_tensor, act_tensor, generator=generator)
                _synchronize(device)
                timings.append(time.perf_counter() - start)
                if generated.shape != (1, horizon, 2) or not bool(generated.isfinite().all()):
                    raise ValueError("Policy produced a nonfinite or incorrectly shaped action chunk")
                raw = _array(codec.decode(generated)[0])
                if aligned is not None:
                    overlap = raw[:aligned.shape[1]] - aligned_raw
                    revisions.append(float(np.sqrt(np.mean(np.sum(overlap**2, axis=-1)))))
                nfes.append(int(diagnostics.get("nfe", 0)))
                if "control_energy" in diagnostics:
                    energies.append(float(torch.as_tensor(diagnostics["control_energy"]).mean()))
                if len(traces) < 3:
                    trace = {"step": len(actions), "startup": startup,
                             "has_previous_plan": has_previous_plan, "completion_id": completion_id,
                             "aligned_old_raw": aligned_raw, "completed_raw": completed_raw,
                             "perturbed_source_raw": None if source is None else _array(codec.decode(source)[0]),
                             "final_raw": raw, "nfe": nfes[-1]}
                    if "revision_states" in diagnostics:
                        candidates = torch.as_tensor(diagnostics["revision_states"], device=device).reshape(-1, horizon, 2)
                        trace["revision_candidate_plans_raw"] = _array(codec.decode(candidates))
                    traces.append(trace)
                cache.store(generated)
                execute_now = min(executed, config["max_episode_steps"] - len(actions))
                for index, command in enumerate(raw[:execute_now]):
                    clipped = np.clip(command, low, high).astype(np.float32)
                    clipped_count += int(np.any(command != clipped))
                    if index == 0:
                        boundaries.append(float(np.linalg.norm(clipped - act_hist[-1])))
                    obs, reward, terminated, truncated, info = env.step(clipped)
                    state = _obs_to_state(obs)
                    actions.append(clipped.copy())
                    preclipped.append(command.copy())
                    rewards.append(float(reward))
                    coverages.append(float(info.get("coverage", reward)))
                    states.append(state.copy())
                    cache.advance()
                    obs_hist = np.concatenate([obs_hist[1:], state[None]], axis=0)
                    act_hist = np.concatenate([act_hist[1:], clipped[None]], axis=0)
                    if collect_frames:
                        frame = env.render()
                        if frame is not None:
                            frames.append(_overlay(frame, aligned_raw, completed_raw, raw, states, low, high,
                                f"{config['method']} | {COMPLETION_NAMES[completion_id]} | step {len(actions)}"))
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
            success = bool(terminated) or max(rewards, default=0.) >= .95 or _info_success(info)
            env_success = bool(terminated) or _info_success(info)
            action_array = np.asarray(actions, dtype=np.float32)
            acceleration = np.diff(action_array, n=2, axis=0) / config["robot_dt"]**2
            jerk = np.diff(action_array, n=3, axis=0) / config["robot_dt"]**3
            episode = {"seed": int(seed), "success": success, "env_success": env_success,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "max_coverage": max(coverages, default=0.), "final_coverage": coverages[-1] if coverages else 0.,
                "episode_length": len(actions), "revision_rms": _mean(revisions),
                "boundary_jump": _mean(boundaries),
                "command_acceleration": _mean(np.linalg.norm(acceleration, axis=-1)),
                "command_jerk": _mean(np.linalg.norm(jerk, axis=-1)),
                "clipping_rate": clipped_count / max(1, len(actions)),
                "inference_seconds_per_replan": _mean(timings), "nfe_per_replan": _mean(nfes),
                "startup_nfe": startup_nfe, "startups": startups, "replans": len(nfes),
                "control_energy": _mean(energies), "reference_coefficients": coefficient_rows,
                "completed_tail_speed": _mean(tail_speeds),
                "completed_tail_acceleration": _mean(tail_accelerations),
                "actions_raw": action_array, "actions_preclipped_raw": np.asarray(preclipped),
                "states_raw": np.asarray(states), "rewards": rewards, "coverage": coverages,
                "plan_traces": traces}
            if output is not None:
                bucket = "success" if success else "failure"
                if frames and saved[bucket] < 2:
                    stem = f"{bucket}-seed{seed}"
                    if save_videos:
                        filename = f"{stem}.mp4"
                        _save_video(frames, output / filename)
                        episode["video"] = filename
                    if save_gifs:
                        filename = f"{stem}.gif"
                        frames[0].save(output / filename, save_all=True, append_images=frames[1:],
                                       duration=100, loop=0)
                        episode["gif"] = filename
                    saved[bucket] += 1
                with (output / f"episode-seed{seed}.json").open("w") as stream:
                    json.dump(_jsonable(episode), stream)
            episodes.append(episode)
    finally:
        cache.reset()
        env.close()
    scalar_keys = ("max_coverage", "final_coverage", "episode_length", "revision_rms", "boundary_jump",
                   "command_acceleration", "command_jerk", "clipping_rate", "inference_seconds_per_replan",
                   "nfe_per_replan", "startup_nfe", "startups", "control_energy",
                   "completed_tail_speed", "completed_tail_acceleration")
    metrics = {key: _mean([episode[key] for episode in episodes]) for key in scalar_keys}
    metrics.update(success_rate=_mean([episode["success"] for episode in episodes]),
                   sim_success_rate=_mean([episode["success"] for episode in episodes]),
                   env_success_rate=_mean([episode["env_success"] for episode in episodes]),
                   episodes=len(episodes), protocol=config.get("protocol", "ordinary_ddim"),
                   external_bootstrap=False, completion_id=int(completion_id),
                   completion=COMPLETION_NAMES[completion_id], method=config["method"], seeds=seeds,
                   horizon=horizon, execute=executed, command_units=codec.units, robot_dt=config["robot_dt"],
                   success_definition="legacy wrapper: terminated OR info success OR max reward >= 0.95")
    coefficient_rows = [row for episode in episodes for row in episode["reference_coefficients"]]
    if coefficient_rows:
        metrics.update({f"reference_{key}": _mean([row[key] for row in coefficient_rows])
                        for key in coefficient_rows[0]})
    if output is not None:
        with (output / "metrics.json").open("w") as stream:
            json.dump(metrics, stream, indent=2)
    return metrics
