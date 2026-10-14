"""Same-pose public reset, real command execution, and terminal handling."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from action_bridge.eval.revision_pusht_mismatch_sim import reset_to_pose, simulate_commands


class FakeEnv:
    metadata = {"render_fps": 10}
    action_space = SimpleNamespace(low=np.zeros(2), high=np.full(2, 512.))

    def __init__(self, *, stuck=False, terminal_step=None, truncated=False, fail_step=False):
        self.stuck, self.terminal_step = stuck, terminal_step
        self.truncate, self.fail_step = truncated, fail_step
        self.reset_calls, self.actions = [], []
        self.closed = False

    def reset(self, *, seed, options):
        self.reset_calls.append((seed, np.asarray(options["reset_to_state"]).copy()))
        self.pose = np.asarray(options["reset_to_state"], dtype=float).copy()
        if self.stuck:
            self.pose = np.zeros(5)
        else:
            self.pose[2:4] += [21., -33.]
        self.actions = []
        return self.pose.copy(), {"vel_agent": np.zeros(2)}

    def step(self, command):
        if self.fail_step:
            raise RuntimeError("simulation failed")
        assert command.dtype == np.float32
        self.actions.append(command.copy())
        self.pose[:2] = command
        done = len(self.actions) == self.terminal_step
        return self.pose.copy(), 0., done and not self.truncate, done and self.truncate, {}

    def close(self):
        self.closed = True


POSE = np.array([110., 120., 250., 260., .6])


def test_public_reset_corrects_legacy_translation_and_preserves_input():
    env = FakeEnv()
    original = POSE.copy()
    observed, details = reset_to_pose(env, original, 321)
    np.testing.assert_allclose(observed, original)
    np.testing.assert_array_equal(original, POSE)
    assert details["reset_count"] == 2
    assert details["reset_errors"][0]["position_max_abs_pixels"] == 33.
    assert details["reset_errors"][-1]["position_max_abs_pixels"] == 0.
    assert all(seed == 321 for seed, _ in env.reset_calls)
    json.dumps(details, allow_nan=False)


def test_bad_reset_fails_after_five_attempts():
    env = FakeEnv(stuck=True)
    with pytest.raises(ValueError, match="after 5 resets"):
        reset_to_pose(env, POSE, 0)
    assert len(env.reset_calls) == 5


def test_commands_are_clipped_and_paths_include_initial_state():
    env = FakeEnv()
    result = simulate_commands(POSE, [[-20., 600.], [123., 456.]], env_factory=lambda: env)
    assert result["commands"] == [[0., 512.], [123., 456.]]
    np.testing.assert_allclose(result["states"][0], POSE)
    assert len(result["states"]) == 3
    assert result["completed_steps"] == result["requested_steps"] == 2
    assert result["clipped_steps"] == 1
    assert result["control_dt_seconds"] == .1
    assert env.closed
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("truncated", [False, True])
def test_stop_at_terminal_or_truncated_state(truncated):
    env = FakeEnv(terminal_step=1, truncated=truncated)
    result = simulate_commands(POSE, [[10., 20.], [30., 40.]], env_factory=lambda: env)
    assert len(env.actions) == result["completed_steps"] == 1
    assert len(result["states"]) == 2 and result["requested_steps"] == 2
    assert result["terminated"] is not truncated
    assert result["truncated"] is truncated
    assert env.closed


@pytest.mark.parametrize("failure", ["step", "reset", "frequency"])
def test_environment_closes_on_error(failure):
    env = FakeEnv(stuck=failure == "reset", fail_step=failure == "step")
    if failure == "frequency":
        env.metadata = {"render_fps": 20}
    with pytest.raises((ValueError, RuntimeError)):
        simulate_commands(POSE, [[10., 20.]], env_factory=lambda: env)
    assert env.closed


@pytest.mark.parametrize("state,commands", [(POSE * np.nan, [[1., 2.]]),
    (POSE, [[1., np.inf]]), (POSE, [1., 2.]), ([1., 2.], [[1., 2.]])])
def test_nonfinite_or_malformed_inputs_rejected_before_env_creation(state, commands):
    def no_env():
        pytest.fail("Invalid inputs must not start a simulator")
    with pytest.raises(ValueError):
        simulate_commands(state, commands, env_factory=no_env)


def test_real_gym_pose_reset_when_available(monkeypatch):
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("gym_pusht")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    # A real recorded pose, not an initial reset parameter. Legacy reset moves
    # its T almost 90 pixels before the public translation correction.
    pose = [222., 97., 222.9938201904297, 381.5990295410156, 3.0079994201660156]
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    try:
        observed, details = reset_to_pose(env, pose, 123)
        np.testing.assert_allclose(observed, pose, atol=1e-3, rtol=0)
        assert details["reset_count"] == 2
        assert details["pusher_velocity_after_reset"] == [0., 0.]
    finally:
        env.close()
