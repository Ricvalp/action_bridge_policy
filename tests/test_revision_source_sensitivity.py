"""Changing a completed source must not also change context or sampling noise."""
import copy
import random

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht_source_sensitivity as sensitivity
from action_bridge.plan_revision.checkpoints import content_digest
from action_bridge.plan_revision.models import build_policy
from test_revision_generation_trace import scenario


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def context(horizon=4, executed=2, previous=True):
    source = np.stack((np.linspace(100., 130., horizon), np.linspace(110., 180., horizon)), axis=-1).astype(np.float32)
    state = np.array([98., 105., 250., 250., .5], dtype=np.float32)
    return {"state_raw": state, "obs_history_raw": np.repeat(state[None], 2, axis=0),
            "action_history_raw": np.array([[90., 100.], [95., 102.]], dtype=np.float32),
            "old_plan_raw": source - 2., "completed_raw": source - .5,
            "fixed_source_raw": source, "has_previous_plan": previous,
            "executed": executed, "replan": 1, "robot_step": executed}


def test_translations_and_bows_are_in_raw_units_and_leave_context_intact():
    original = context(horizon=8, executed=4)
    before = copy.deepcopy(original)
    data = sensitivity.build_source_interventions(original, {"horizon": 8, "execute": 4},
                                                  amplitudes_px=(-10., 0., 10.))
    assert data["sources_raw"].shape == (9, 8, 2)
    assert data["baseline_index"] == 0 and data["variant_labels"][0] == "baseline"
    np.testing.assert_array_equal(data["sources_raw"][0], original["fixed_source_raw"])
    for index, family in enumerate(data["families"][1:], start=1):
        delta = data["sources_raw"][index] - data["sources_raw"][0]
        dimension = 0 if family.endswith("_x") else 1
        np.testing.assert_array_equal(delta[:, 1 - dimension], 0.)
        if family.startswith("translate_"):
            np.testing.assert_allclose(delta[:, dimension], data["amplitudes_px"][index], atol=1e-5)
        else:
            np.testing.assert_array_equal(delta[[0, -1]], 0.)
            assert np.abs(delta).max() == pytest.approx(10., abs=1e-5)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(original[key], value)
            np.testing.assert_array_equal(data[key], value)


@pytest.mark.parametrize("region,selected", [("retained", slice(0, 6)), ("tail", slice(6, 8))])
def test_region_uses_actual_previous_execution_not_deployment_k(region, selected):
    original = context(horizon=8, executed=2)
    data = sensitivity.build_source_interventions(original, {"horizon": 8, "execute": 4},
                                                  amplitudes_px=(20.,), families=("translate_x",), region=region)
    expected = np.zeros((8, 2))
    expected[selected, 0] = 20.
    np.testing.assert_allclose(data["sources_raw"][1] - data["sources_raw"][0], expected, atol=1e-5)
    assert data["retained_count"] == 6


def test_startup_has_no_retained_source_and_all_of_it_is_tail():
    original = context(horizon=8, executed=0, previous=False)
    config = {"horizon": 8, "execute": 4}
    with pytest.raises(ValueError, match="retained region is empty"):
        sensitivity.build_source_interventions(original, config, region="retained")
    all_data = sensitivity.build_source_interventions(original, config)
    tail_data = sensitivity.build_source_interventions(original, config, region="tail")
    np.testing.assert_array_equal(all_data["sources_raw"], tail_data["sources_raw"])
    assert tail_data["retained_count"] == 0


@pytest.mark.parametrize("kwargs,match", [
    ({"amplitudes_px": (0.,)}, "nonzero"),
    ({"amplitudes_px": (np.nan,)}, "finite"),
    ({"amplitudes_px": (np.inf,)}, "finite"),
    ({"families": ("made_up",)}, "families"),
    ({"families": ("bow_x",), "region": "tail"}, "three targets"),
    ({"region": "old"}, "region"),
])
def test_invalid_interventions_fail_clearly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sensitivity.build_source_interventions(context(), {"horizon": 4, "execute": 2}, **kwargs)


def test_large_intervention_is_not_clipped_to_workspace():
    data = sensitivity.build_source_interventions(context(), {"horizon": 4, "execute": 2},
                                                  families=("translate_x",), amplitudes_px=(1000.,))
    np.testing.assert_array_equal(data["sources_raw"][1, :, 0] - data["sources_raw"][0, :, 0], 1000.)
    assert (data["sources_raw"][1, :, 0] > 512.).all()


class ToySampler(torch.nn.Module):
    """Known transfer functions, with fresh but explicitly paired random draws."""
    calls = []

    def __init__(self, gain):
        super().__init__()
        self.gain = gain
        self.anchor = torch.nn.Parameter(torch.tensor(0.))

    def sample(self, obs_hist, act_hist, mode, *, source_actions, generator, trace, **kwargs):
        self.calls.append((obs_hist.clone(), act_hist.clone(), mode, kwargs.get("execution_k")))
        noise = torch.randn(source_actions.shape, generator=generator, device=source_actions.device)
        generated = self.gain * source_actions + noise + self.anchor
        states = torch.stack((source_actions, generated), dim=1).flatten(2)
        return generated, {"revision_states": states, "revision_times": source_actions.new_tensor([0., 1.])}


@pytest.mark.parametrize("gain", [0., 1.])
def test_known_source_response_and_noise_controls(monkeypatch, gain):
    _, config, metadata, _ = scenario("fm_paired", monkeypatch)
    config["condition_on_k"] = True
    ToySampler.calls = []
    sampler = ToySampler(gain).train()
    original = context()
    before = copy.deepcopy(original)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    data = sensitivity.sample_source_sensitivity(sampler, config, metadata, {}, original,
                                                 samples=5, batch_size=2, seed=19,
                                                 families=("translate_x", "bow_y"),
                                                 amplitudes_px=(-10., 10.), progress=False)
    stats = sensitivity.sensitivity_statistics(data)
    assert data["candidates_raw"].shape == (5, 5, 4, 2)
    assert data["revision_states_raw"].shape == (5, 5, 2, 4, 2)
    np.testing.assert_array_equal(data["revision_times"], [0., 1.])
    assert stats["repeat_same_noise_rms_px"] == 0.
    assert stats["same_source_independent_noise_rms_px"] > 0.
    assert stats["variants"][0]["gain"] is None
    for variant in stats["variants"][1:]:
        assert variant["gain"] == pytest.approx(gain, abs=2e-6)
        if gain == 0:
            assert variant["paired_output_rms_shift_px"] == 0.
    # First target time is really the intervened source, not an interpolation.
    np.testing.assert_allclose(data["revision_states_raw"][:, 0, 0], data["sources_raw"], atol=2e-5)
    assert sampler.training and sampler.anchor.requires_grad and sampler.anchor.item() == 0.
    assert torch.equal(torch_state, torch.get_rng_state())
    assert random.getstate() == python_state
    now_numpy = np.random.get_state()
    assert now_numpy[0] == numpy_state[0] and now_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(now_numpy[1], numpy_state[1])
    for key in ("obs_history_raw", "action_history_raw", "fixed_source_raw", "old_plan_raw"):
        np.testing.assert_array_equal(original[key], before[key])
    first_obs, first_act, _, _ = ToySampler.calls[0]
    for obs, actions, completion_id, execution_k in ToySampler.calls:
        torch.testing.assert_close(obs, first_obs[:1].expand_as(obs))
        torch.testing.assert_close(actions, first_act[:1].expand_as(actions))
        assert completion_id == 2 and (execution_k == config["execute"]).all()


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic", "fm_paired", "fm_local_ot"])
@pytest.mark.parametrize("condition_on_k", [False, True])
def test_actual_samplers_have_paired_traces_and_frozen_reference(monkeypatch, method, condition_on_k):
    policy, config, metadata, dependencies = scenario(method, monkeypatch)
    if condition_on_k:
        config.update(condition_on_k=True, model_schema="k_conditioned_v1",
                      protocol="self_source_k_curriculum_v1")
        policy = build_policy(config)
    policy.train()
    before = content_digest(policy.state_dict())
    reference_before = content_digest(dependencies)
    prior_calls = []
    restore = sensitivity.restore_completion

    def restore_counted(*args, **kwargs):
        completion = restore(*args, **kwargs)
        prior = completion.prior

        def counted(*args, **kwargs):
            prior_calls.append(1)
            return prior(*args, **kwargs)

        completion.prior = counted
        return completion

    monkeypatch.setattr(sensitivity, "restore_completion", restore_counted)
    data = sensitivity.sample_source_sensitivity(policy, config, metadata, dependencies, context(),
                                                 samples=3, batch_size=2, seed=7,
                                                 amplitudes_px=(-5., 5.), families=("translate_x",),
                                                 progress=False)
    expected_times = 5 if method.startswith("sb_") else 3
    assert data["revision_states_raw"].shape == (3, 3, expected_times, 4, 2)
    assert data["candidates_raw"].shape == (3, 3, 4, 2)
    np.testing.assert_array_equal(data["revision_states_raw"][:, :, -1], data["candidates_raw"])
    np.testing.assert_array_equal(data["repeat_baseline_raw"], data["candidates_raw"][0])
    assert np.isfinite(data["revision_states_raw"]).all()
    assert content_digest(policy.state_dict()) == before and policy.training
    assert content_digest(dependencies) == reference_before
    stats = sensitivity.sensitivity_statistics(data)
    if method.startswith("sb_"):
        assert len(prior_calls) == 1
        assert stats["same_source_independent_noise_rms_px"] > 0.
    else:
        assert prior_calls == []
        assert stats["same_source_independent_noise_rms_px"] == 0.


def test_ddim_is_rejected_before_sampler_use():
    with pytest.raises(ValueError, match="FM or SB"):
        sensitivity.sample_source_sensitivity(None, {"method": "ddim"}, {}, {}, context())


def test_statistics_average_euclidean_squared_distances_over_samples_and_targets():
    baseline = np.zeros((2, 4, 2))
    baseline[0, :, 0], baseline[1, :, 0] = -1., 1.
    delta = np.zeros_like(baseline)
    delta[0, :, 0], delta[1, :, 0] = [1., 2., 3., 4.], [3., 4., 5., 6.]
    sources = np.zeros((2, 4, 2))
    sources[1, :, 0] = 4.
    independent = baseline.copy()
    independent[:, :, 1] += 4.
    data = {"candidates_raw": np.stack((baseline, baseline + delta)), "sources_raw": sources,
            "baseline_index": 0, "variant_labels": ["baseline", "translate_x +4px"],
            "families": ["baseline", "translate_x"], "amplitudes_px": [0., 4.],
            "execute": 2, "samples": 2, "repeat_baseline_raw": baseline.copy(),
            "independent_baseline_raw": independent}
    stats = sensitivity.sensitivity_statistics(data)
    variant = stats["variants"][1]
    assert variant["source_rms_shift_px"] == 4.
    assert variant["paired_output_rms_shift_px"] == pytest.approx(np.sqrt(14.5))
    assert variant["gain"] == pytest.approx(np.sqrt(14.5) / 4.)
    assert variant["mean_output_shift_px"] == pytest.approx(np.sqrt(13.5))
    assert variant["executed_prefix_output_rms_shift_px"] == pytest.approx(np.sqrt(7.5))
    assert variant["kth_output_rms_shift_px"] == pytest.approx(np.sqrt(10.))
    assert variant["last_output_rms_shift_px"] == pytest.approx(np.sqrt(26.))
    assert variant["output_rms_spread_px"] == 2.
    assert stats["baseline_output_rms_spread_px"] == 1.
    assert stats["repeat_same_noise_rms_px"] == 0.
    assert stats["same_source_independent_noise_rms_px"] == 4.


def test_malformed_sampler_trace_is_rejected(monkeypatch):
    _, config, metadata, _ = scenario("fm_paired", monkeypatch)

    class Broken(ToySampler):
        def sample(self, *args, **kwargs):
            result, diagnostics = super().sample(*args, **kwargs)
            diagnostics["revision_states"] = diagnostics["revision_states"].repeat(1, 1, 2)
            return result, diagnostics

    with pytest.raises(ValueError, match="Invalid revision trace"):
        sensitivity.sample_source_sensitivity(Broken(1.), config, metadata, {}, context(),
                                               samples=1, amplitudes_px=(5.,),
                                               families=("translate_x",), progress=False)
