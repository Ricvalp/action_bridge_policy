"""Pose-reset Push-T counterfactuals for offline-source diagnostics.

The recorded 5D state contains poses, not simulator velocities/contact caches.
These are short pose-reset experiments, not exact clones of the demonstration.
Only public Gym reset/step APIs are used; simulator code is not modified.
"""
from __future__ import annotations

import numpy as np


def _state(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError("Push-T pose must be a finite [pusher x/y, T x/y/angle] vector")
    return result.copy()


def reset_to_pose(env, state, seed):
    """Return ``(observed_pose, diagnostics)`` after correcting legacy reset offset.

    gym-pusht sets the T position before its angle, rotating around an offset
    center of mass. Its public reset therefore does not initially return the
    requested pose. Correct that translation using the returned observations,
    and reject cases that cannot reproduce the requested pose in five resets.
    """
    desired = _state(state)
    requested = desired.copy()
    errors = []
    for attempt in range(5):
        observed, info = env.reset(seed=int(seed), options={"reset_to_state": requested.copy()})
        observed = _state(observed)
        error = desired - observed
        error[4] = np.arctan2(np.sin(error[4]), np.cos(error[4]))
        errors.append({"position_max_abs_pixels": float(np.abs(error[:4]).max()),
                       "angle_abs_radians": float(abs(error[4]))})
        if errors[-1]["position_max_abs_pixels"] < 1e-3 and abs(error[4]) < 1e-5:
            velocity = info.get("vel_agent")
            return observed, {
                "seed": int(seed), "reset_count": attempt + 1,
                "requested_pose": desired.tolist(), "observed_pose": observed.tolist(),
                "reset_errors": errors,
                "pusher_velocity_after_reset": (None if velocity is None else
                                                 np.asarray(velocity, dtype=float).tolist()),
                "state_semantics": "pose reset; demo velocities and contact state are unavailable",
            }
        requested += error
    raise ValueError(f"Public Push-T reset could not reproduce the requested pose after 5 resets: {errors[-1]}")


def _make_env():
    import gymnasium as gym
    import gym_pusht  # noqa: F401 -- registers the public environment ID

    return gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")


def simulate_commands(state, commands, seed=0, env_factory=None):
    """Execute targets from a reconstructed pose, stopping at terminal states.

    Returns JSON-serializable trajectories. ``states`` includes the initial
    pose; ``commands`` contains only actions actually executed. A shorter path
    is explicitly indicated by ``completed_steps`` and terminal flags.
    """
    desired = _state(state)
    commands = np.asarray(commands, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 2 or not np.isfinite(commands).all():
        raise ValueError("Commands must be a finite [K, 2] array of absolute xy targets")
    env = (env_factory or _make_env)()
    try:
        # The pinned Push-T environment uses its public 10 Hz metadata as the
        # control frequency. This is separate from completion's robot_dt=1.
        control_hz = float(env.metadata["render_fps"])
        if control_hz != 10.:
            raise ValueError(f"Expected Push-T's 10 Hz control rate, got {control_hz}")
        low = np.asarray(env.action_space.low, dtype=np.float64)
        high = np.asarray(env.action_space.high, dtype=np.float64)
        if (low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all()
                or np.any(low >= high)):
            raise ValueError("Expected finite ordered 2D Push-T action bounds")
        observed, reset = reset_to_pose(env, desired, seed)
        states, executed = [observed.tolist()], []
        terminated = truncated = False
        clipped_steps = 0
        for command in commands:
            bounded = np.clip(command, low, high)
            clipped_steps += int(np.any(bounded != command))
            actual = bounded.astype(np.float32)
            observed, _, terminated, truncated, _ = env.step(actual)
            states.append(_state(observed).tolist())
            executed.append(actual.tolist())
            if terminated or truncated:
                break
        return {
            "states": states, "commands": executed,
            "requested_steps": len(commands), "completed_steps": len(executed),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "clipped_steps": clipped_steps, "control_hz": control_hz,
            "control_dt_seconds": 1. / control_hz, "reset": reset,
        }
    finally:
        env.close()
