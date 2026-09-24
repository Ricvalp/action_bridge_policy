"""Opt-in generation traces preserve the sampler and expose its actual states."""

import math

import pytest
import torch

from action_bridge.plan_revision.gaussian import GaussianReference
from action_bridge.plan_revision.models import build_policy


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(method, nfe=6):
    cfg = dict(method=method, horizon=4, action_dim=2, obs_dim=3,
               obs_history=2, action_history=1, channels=[8, 16],
               history_dim=16, hidden_dim=16, time_dim=8,
               num_inference_steps=nfe)
    generator = torch.Generator().manual_seed(91)
    inputs = dict(
        obs_hist=torch.randn(2, 2, 3, generator=generator),
        act_hist=torch.randn(2, 1, 2, generator=generator),
        mode=torch.tensor([0, 2]),
        source_actions=torch.randn(2, 4, 2, generator=generator),
        has_previous_plan=torch.tensor([False, True]),
    )
    if method.startswith("sb_"):
        inputs["reference"] = GaussianReference(
            torch.eye(8).expand(2, 8, 8), torch.zeros(2, 8),
            kind="kinetic" if method == "sb_kinetic" else "ou",
        )
    return build_policy(cfg).eval(), inputs


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_trace_preserves_outputs_rng_and_default_diagnostics(method):
    policy, inputs = example(method)
    generators = [torch.Generator().manual_seed(43) for _ in range(3)]
    default, default_info = policy.sample(**inputs, generator=generators[0])
    untraced, untraced_info = policy.sample(**inputs, generator=generators[1], trace=False)
    traced, traced_info = policy.sample(**inputs, generator=generators[2], trace=True)

    assert_exact(default, untraced)
    assert_exact(default, traced)
    assert_exact(generators[0].get_state(), generators[1].get_state())
    assert_exact(generators[0].get_state(), generators[2].get_state())
    assert default_info.keys() == untraced_info.keys()
    assert "revision_times" not in default_info
    assert default_info["nfe"] == untraced_info["nfe"] == traced_info["nfe"] == 6
    if method.startswith("fm_"):
        assert default_info == untraced_info == {"nfe": 6}
        count = 3
        expected_times = torch.linspace(0, 1, count + 1)
    else:
        count = 6
        expected_times = .5 - .5 * torch.cos(torch.linspace(0, math.pi, count + 1))
        assert_exact(default_info["control_energy"], untraced_info["control_energy"])
        assert_exact(default_info["control_energy"], traced_info["control_energy"])
        assert_exact(default_info["revision_states"], untraced_info["revision_states"])
        # Preserve exactly the initial state and the original sparse selections.
        indices = [0] + [i + 1 for i in range(count)
                         if i in (count // 3, 2 * count // 3, count - 1)]
        assert_exact(default_info["revision_states"], traced_info["revision_states"][:, indices])
    states, times = traced_info["revision_states"], traced_info["revision_times"]
    assert states.shape == (2, count + 1, 8)
    assert times.shape == (count + 1,)
    assert_exact(times, expected_times)
    assert_exact(states[:, 0], inputs["source_actions"].flatten(1))
    assert_exact(states[:, -1], traced.flatten(1))
    assert not states.requires_grad
    assert states.dtype == times.dtype == traced.dtype
    assert states.device == times.device == traced.device


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot"])
def test_fm_trace_contains_accepted_steps_not_midpoint_evaluations(method, monkeypatch):
    policy, inputs = example(method, nfe=8)
    evaluations = []

    def constant_velocity(state, tau, conditioning):
        evaluations.append((state.clone(), float(tau)))
        return torch.ones_like(state)

    monkeypatch.setattr(policy.field, "conditioned", constant_velocity)
    actions, info = policy.sample(**inputs, trace=True)
    source = inputs["source_actions"].flatten(1)
    expected = source[:, None] + torch.linspace(0, 1, 5)[None, :, None]
    torch.testing.assert_close(info["revision_states"], expected)
    torch.testing.assert_close(actions, inputs["source_actions"] + 1)
    assert info["nfe"] == len(evaluations) == 8
    assert [tau for _, tau in evaluations] == [i / 8 for i in range(8)]
    for step in range(4):
        assert_exact(info["revision_states"][:, step], evaluations[2 * step][0])


def test_fm_trace_follows_nonlinear_midpoint_updates_not_endpoint_interpolation(monkeypatch):
    policy, inputs = example("fm_paired", nfe=8)

    def velocity(state, tau, conditioning):
        return state.square() * .1 + tau

    monkeypatch.setattr(policy.field, "conditioned", velocity)
    actions, info = policy.sample(**inputs, trace=True)
    state = inputs["source_actions"].flatten(1)
    expected = [state]
    for step in range(4):
        tau = step / 4
        midpoint = state + .125 * velocity(state, tau, None)
        state = state + .25 * velocity(midpoint, tau + .125, None)
        expected.append(state)
    assert_exact(info["revision_states"], torch.stack(expected, dim=1))
    linear = torch.lerp(inputs["source_actions"].flatten(1), actions.flatten(1), .5)
    assert not torch.allclose(info["revision_states"][:, 2], linear)


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("steps", [1, 5])
def test_bridge_rollout_trace_records_every_reference_step(method, reverse, steps, monkeypatch):
    policy, inputs = example(method)
    reference = inputs["reference"]
    state = reference.augment(inputs["source_actions"].flatten(1),
                              generator=torch.Generator().manual_seed(10))
    original_step = reference.step
    observed = [reference.positions(state).clone()]
    durations = []

    def observed_step(state, control, dt, *, reverse, generator):
        next_state = original_step(state, control, dt, reverse=reverse, generator=generator)
        observed.append(reference.positions(next_state).clone())
        durations.append(dt.clone())
        return next_state

    monkeypatch.setattr(reference, "step", observed_step)
    arguments = {key: inputs[key] for key in ("obs_hist", "act_hist", "mode", "has_previous_plan")}
    traced_generator = torch.Generator().manual_seed(19)
    final_state, info = policy.rollout(
        state, reference=reference, **arguments, reverse=reverse, steps=steps,
        generator=traced_generator, trace=True,
    )
    expected_times = .5 - .5 * torch.cos(torch.linspace(0, math.pi, steps + 1))
    assert info["nfe"] == steps == len(durations)
    assert info["revision_states"].shape == (2, steps + 1, 8)
    assert_exact(info["revision_states"], torch.stack(observed, dim=1))
    assert_exact(info["revision_times"], expected_times)
    assert_exact(torch.stack(durations), expected_times.diff())
    assert_exact(info["revision_states"][:, -1], reference.positions(final_state))

    monkeypatch.setattr(reference, "step", original_step)
    default_generator = torch.Generator().manual_seed(19)
    default_state, default_info = policy.rollout(
        state, reference=reference, **arguments, reverse=reverse, steps=steps,
        generator=default_generator,
    )
    assert_exact(final_state, default_state)
    assert_exact(info["control_energy"], default_info["control_energy"])
    assert_exact(traced_generator.get_state(), default_generator.get_state())
    indices = [0] + [i + 1 for i in range(steps)
                     if i in (steps // 3, 2 * steps // 3, steps - 1)]
    assert_exact(default_info["revision_states"], info["revision_states"][:, indices])
