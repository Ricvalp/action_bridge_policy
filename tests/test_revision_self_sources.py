"""Self-source replay checks, not extra training experiments."""
import copy

import pytest
import torch
from torch import nn

from action_bridge.plan_revision.cache import draw_records, refresh_coupling
from action_bridge.plan_revision.completion import LearnedCompletion
from action_bridge.plan_revision.contracts import take
from action_bridge.plan_revision.data import SourceReplayError, build_self_sources, immutable_snapshot
from action_bridge.plan_revision.models import build_policy


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(episodes=3, replans=4, *, action_dim=2, mapping=False):
    config = dict(method="fm_paired", horizon=4, execute=2, action_dim=action_dim,
                  obs_dim=3, obs_history=2, action_history=2, channels=[8, 16],
                  history_dim=16, hidden_dim=16, time_dim=8, num_inference_steps=2,
                  batch_size=16, source_std=.01, endpoint_std=.001, robot_dt=1.,
                  prior_ridge=.05, max_rate=4., mobility_smoothing=0.,
                  temperature=.05, revision_gamma=2., coupling_steps=2)
    count = episodes * replans
    g = torch.Generator().manual_seed(51)
    windows = dict(obs_hist=torch.randn(count, 2, 3, generator=g),
                   act_hist=torch.randn(count, 2, action_dim, generator=g),
                   future_actions=torch.randn(count, 4, action_dim, generator=g),
                   valid_mask=torch.ones(count, 4, dtype=torch.bool),
                   episode_id=torch.arange(episodes).repeat_interleave(replans),
                   time_index=torch.arange(replans).repeat(episodes) * 2)
    windows["startup_actions"] = windows["act_hist"][:, -1:, :].repeat(1, 4, 1)
    encoder = None
    if mapping:
        windows["obs_hist"] = {"state": windows["obs_hist"]}
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 + action_dim, 8)
            def forward(self, obs, actions):
                return self.linear(torch.cat((obs["state"][:, -1], actions[:, -1]), -1))
        encoder = Encoder()
    completion = LearnedCompletion(3, action_dim, 2, 2, 4, hidden_dim=8,
                                   encoder=encoder).eval().requires_grad_(False)
    return config, windows, completion


class ObservedOnlySampler(nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))
        self.register_buffer("offset", torch.tensor(.1))
        self.calls = []
        self.fail = fail

    def sample(self, obs, act_hist, mode, *, source_actions, reference, generator, has_previous_plan):
        assert not torch.is_grad_enabled()
        state = obs["state"] if isinstance(obs, dict) else obs
        self.calls.append(dict(obs=state.clone(), act_hist=act_hist.clone(),
                               mode=mode.clone(), has_previous_plan=has_previous_plan.clone()))
        output = source_actions + self.weight * state[:, -1, :1, None] + self.offset
        if self.fail:
            output[:] = float("nan")
        return output, {"nfe": 1}


def replay(windows, config, completion, *, p_self=1., snapshot=None, **kwargs):
    snapshot = immutable_snapshot(ObservedOnlySampler()) if snapshot is None else snapshot
    return build_self_sources(windows, snapshot, completion, torch.ones(config["action_dim"]),
                              config, "cpu", block=2, seed=73, p_self=p_self, **kwargs)


def test_self_only_replay_cannot_read_future_labels_and_is_reproducible():
    config, windows, completion = fixture()
    first, diagnostics = replay(windows, config, completion)
    altered = dict(windows, future_actions=torch.full_like(windows["future_actions"], 1e8))
    second, _ = replay(altered, config, completion)
    for key in ("source_actions", "old_actions", "generated_actions", "precision", "prior_mean"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert diagnostics["expert_records"] == 0
    assert diagnostics["startup_records"] == 9
    assert diagnostics["self_records"] == 27
    assert not first["source_actions"].requires_grad
    assert not first["precision"].requires_grad


def test_recorded_histories_never_change_and_generated_plan_replays_with_episode_reset():
    config, windows, completion = fixture(mapping=True, action_dim=6)
    original = copy.deepcopy(windows)
    snapshot = immutable_snapshot(ObservedOnlySampler())
    records, _ = replay(windows, config, completion, snapshot=snapshot)
    for index, (episode, time, mode) in enumerate(zip(records["episode_id"], records["time_index"],
                                                   records["completion_id"])):
        source_index = torch.where((windows["episode_id"] == episode) &
                                   (windows["time_index"] == time))[0].item()
        torch.testing.assert_close(records["act_hist"][index], original["act_hist"][source_index])
        torch.testing.assert_close(records["obs_hist"]["state"][index], original["obs_hist"]["state"][source_index])
        if time == 0:
            assert not records["has_previous_plan"][index]
            torch.testing.assert_close(records["old_actions"][index], windows["startup_actions"][source_index])
        else:
            previous = torch.where((records["episode_id"] == episode) &
                                   (records["time_index"] == time - config["execute"]) &
                                   (records["completion_id"] == mode))[0].item()
            torch.testing.assert_close(records["old_actions"][index], records["generated_actions"][previous])
    torch.testing.assert_close(windows["act_hist"], original["act_hist"])
    torch.testing.assert_close(windows["obs_hist"]["state"], original["obs_hist"]["state"])


@pytest.mark.parametrize("probability", [.1, .5, 1.])
def test_whole_old_chunk_curriculum_rates_and_previous_expert_only(probability):
    config, windows, completion = fixture(episodes=200, replans=4)
    config["source_std"] = 0.
    records, diagnostics = replay(windows, config, completion, p_self=probability)
    ordinary = records["has_previous_plan"]
    rate = (records["source_origin"][ordinary] == 2).float().mean().item()
    assert abs(rate - probability) < .04
    assert (records["source_origin"][~ordinary] == 0).all()
    assert set(records["completion_id"].tolist()) == {0, 1, 2}
    torch.testing.assert_close(records["source_actions"][ordinary, :2], records["old_actions"][ordinary, 2:])
    for index in (records["source_origin"] == 1).nonzero(as_tuple=True)[0]:
        previous = torch.where((windows["episode_id"] == records["episode_id"][index]) &
                               (windows["time_index"] == records["time_index"][index] - 2))[0].item()
        torch.testing.assert_close(records["old_actions"][index], windows["future_actions"][previous])
    assert diagnostics["numerical_failures"] == 0


def test_snapshot_has_independent_parameters_buffers_and_encoder_storage():
    student = ObservedOnlySampler()
    snapshot = immutable_snapshot(student)
    with torch.no_grad():
        student.weight.add_(100)
        student.offset.add_(100)
    assert snapshot.weight.item() == pytest.approx(.2)
    assert snapshot.offset.item() == pytest.approx(.1)
    assert not snapshot.training and not snapshot.weight.requires_grad


def test_draw_preserves_completed_perturbed_population_and_conditions():
    config, windows, completion = fixture()
    records, _ = replay(windows, config, completion)
    drawn = draw_records(records, 100, "cpu", source_std=1000., endpoint_std=0.)
    for key in ("source_actions", "completion_id", "has_previous_plan", "source_origin", "generated_actions"):
        torch.testing.assert_close(drawn[key], records[key][drawn["record_id"]], rtol=0, atol=0)


def test_numerical_failure_is_counted_and_stops_replay_without_fallback():
    config, windows, completion = fixture()
    with pytest.raises(SourceReplayError, match="numerical_failures=3") as caught:
        replay(windows, config, completion, snapshot=immutable_snapshot(ObservedOnlySampler(fail=True)))
    assert caught.value.diagnostics["numerical_failures"] == 3


def test_missing_startup_or_skipped_grid_is_rejected():
    config, windows, completion = fixture()
    rows = windows["time_index"] > 0
    with pytest.raises(ValueError, match="start at time zero"):
        replay(take(windows, rows), config, completion)


def test_fully_executed_chunk_restarts_from_current_observed_anchor():
    config, windows, completion = fixture(episodes=2, replans=3)
    config.update(execute=4, source_std=0.)
    windows["time_index"] *= 2
    records, diagnostics = replay(windows, config, completion, p_self=0.)
    assert not records["has_previous_plan"].any()
    assert not records["source_origin"].any()
    torch.testing.assert_close(records["source_actions"], records["startup_actions"])
    assert diagnostics["startup_records"] == len(records["source_actions"])


def test_default_probability_reads_configured_block_schedule():
    config, windows, completion = fixture()
    config["source_self_probabilities"] = [.2, .3, 0., 1.]
    records, diagnostics = build_self_sources(
        windows, immutable_snapshot(ObservedOnlySampler()), completion,
        torch.ones(config["action_dim"]), config, "cpu", block=2, seed=73)
    assert diagnostics["p_self"] == 0.
    assert (records["source_origin"][records["has_previous_plan"]] == 1).all()


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_reviser_startup_has_explicit_conditioning_and_no_external_model(method):
    config, windows, completion = fixture(episodes=2, replans=2)
    config["method"] = method
    snapshot = immutable_snapshot(build_policy(config))
    records, diagnostics = replay(windows, config, completion, snapshot=snapshot)
    assert torch.isfinite(records["generated_actions"]).all()
    assert diagnostics["startup_records"] == 6
    field = snapshot.field if method.startswith("fm_") else snapshot.forward_field
    context = take(windows, slice(0, 2))
    startup = field.conditioning(context["obs_hist"], context["act_hist"], 0, False)
    ordinary = field.conditioning(context["obs_hist"], context["act_hist"], 0, True)
    torch.testing.assert_close(startup[:, :-1], ordinary[:, :-1])
    assert (startup[:, -1] == 0).all() and (ordinary[:, -1] == 1).all()


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_kinetic_coupling_retains_source_population_context_and_sampled_velocity(direction):
    config, windows, completion = fixture(episodes=2, replans=2)
    config["method"] = "sb_kinetic"
    snapshot = immutable_snapshot(build_policy(config))
    records, _ = replay(windows, config, completion, snapshot=snapshot)
    original = records["source_actions"].clone()
    coupling, diagnostics = refresh_coupling(records, 8, "cpu", config, completion, snapshot, direction)
    for key in ("source_actions", "completion_id", "has_previous_plan", "episode_id", "time_index"):
        torch.testing.assert_close(coupling[key], records[key][coupling["record_id"]])
    torch.testing.assert_close(records["source_actions"], original, rtol=0, atol=0)
    n = config["horizon"] * config["action_dim"]
    assert coupling["x0"].shape[1] == 2 * n
    assert coupling["x0"][:, n:].abs().sum() > 0
    assert coupling["x1"][:, n:].abs().sum() > 0
    assert diagnostics["generated_endpoint"] == ("source" if direction == "forward" else "target")
