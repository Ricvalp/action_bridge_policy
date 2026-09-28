"""Small deterministic source/schedule checks; no learning benchmark."""
import copy

import pytest
import torch
from torch import nn

from action_bridge.plan_revision.completion import LearnedCompletion, complete_plan
from action_bridge.plan_revision.curriculum import (
    CURRICULUM_PROTOCOL, scheduled_k, stage_config, stage_windows, validate_curriculum,
)
from action_bridge.plan_revision.data import build_self_sources, immutable_snapshot


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(horizon=16, dim=3, length=19):
    config = dict(protocol=CURRICULUM_PROTOCOL, method="fm_paired", horizon=horizon, execute=8,
                  k_values_by_block=[1, 2, 4, 8], training_blocks=4, updates=8,
                  source_self_probabilities=[.1, .5, 1., 1.], condition_on_k=True,
                  model_schema="k_conditioned_v1", batch_size=3, robot_dt=.5,
                  source_std=0., prior_ridge=.05, max_rate=4., mobility_smoothing=0.,
                  completion_velocity_weighting="last_pair")
    count = 2 * length
    times = torch.arange(length).repeat(2)
    episode = torch.arange(2).repeat_interleave(length)
    actions = (times[:, None, None] + torch.arange(horizon)[None, :, None]).float().expand(-1, -1, dim).clone()
    windows = dict(episode_id=episode, time_index=times, future_actions=actions,
                   valid_mask=torch.ones(count, horizon, dtype=torch.bool),
                   obs_hist=times[:, None, None].float().expand(-1, 2, 4).clone(),
                   act_hist=(times[:, None, None] - torch.tensor([2, 1])[None, :, None]).float().expand(-1, -1, dim).clone(),
                   startup_actions=torch.zeros_like(actions))
    completion = LearnedCompletion(4, dim, 2, 2, horizon, hidden_dim=8, robot_dt=.5).eval().requires_grad_(False)
    return config, windows, completion


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.register_buffer("offset", torch.tensor(.5))
        self.calls = []

    def sample(self, obs, history, modes, *, source_actions, reference, generator, has_previous_plan,
               execution_k):
        self.calls.append((history.clone(), execution_k.clone()))
        assert not torch.is_grad_enabled()
        return source_actions + self.weight + self.offset, {}


def replay(config, windows, completion, block, **kwargs):
    snapshot = immutable_snapshot(Recorder())
    result, diagnostics = build_self_sources(windows, snapshot, completion,
        torch.ones(completion.action_dim), config, "cpu", block=block, seed=11,
        p_self=kwargs.pop("p_self", 1.), modes=kwargs.pop("modes", (0, 1, 2)), **kwargs)
    return result, diagnostics, snapshot


@pytest.mark.parametrize("block,k", list(enumerate([1, 2, 4, 8])))
def test_stage_rebuilds_dense_record_grid_and_never_subsamples_histories(block, k):
    config, windows, completion = fixture()
    before = copy.deepcopy(windows)
    records, diagnostics, sampler = replay(config, windows, completion, block)
    assert records["source_actions"].shape[1:] == (16, 3)
    assert records["execution_k"].unique().tolist() == [k]
    assert not (records["time_index"] % k).any()
    ordinary = records["has_previous_plan"]
    assert (records["overlap_length"][ordinary] == 16 - k).all()
    assert (records["completion_length"][ordinary] == k).all()
    assert not records["elapsed_count"][~ordinary].any()
    assert (records["elapsed_count"][ordinary] == k).all()
    assert not records["overlap_mask"][~ordinary].any()
    for index, (ep, time, mode) in enumerate(zip(records["episode_id"], records["time_index"], records["completion_id"])):
        original = ((windows["episode_id"] == ep) & (windows["time_index"] == time)).nonzero()[0, 0]
        assert torch.equal(records["act_hist"][index], before["act_hist"][original])
        assert torch.equal(records["obs_hist"][index], before["obs_hist"][original])
        if time:
            prev = ((records["episode_id"] == ep) & (records["time_index"] == time - k) & (records["completion_id"] == mode)).nonzero()[0, 0]
            assert torch.equal(records["old_actions"][index], records["generated_actions"][prev])
            assert torch.equal(records["source_actions"][index, :16-k], records["generated_actions"][prev, k:])
    assert diagnostics["source_replay_calls"] == len(sampler.calls)
    assert diagnostics["source_replay_examples"] == len(records["source_actions"])
    assert diagnostics["queried_tail_coefficients"] == list(range(16-k, 16))
    assert diagnostics["source_build_seconds"] >= 0
    for key in before:
        assert torch.equal(windows[key], before[key])


@pytest.mark.parametrize("k", [1, 2, 4, 8, 9, 10])
def test_non_pusht_shapes_full_completion_and_exact_expert_alignment(k):
    config, windows, completion = fixture(horizon=10, dim=5, length=23)
    config.update(execute=k, k_values_by_block=[k]*4)
    records, _, _ = replay(config, windows, completion, 0, p_self=0., modes=(0,))
    ordinary = records["has_previous_plan"]
    assert ordinary.any(), "K=H must retain real previous-plan history"
    source = records["source_actions"][ordinary]
    assert source.shape[1:] == (10, 5)
    assert torch.equal(source[:, :10-k], records["future_actions"][ordinary, :10-k])
    assert (records["completion_length"][ordinary] == k).all()
    assert (records["alignment_shift"][ordinary] == k).all()


def test_curriculum_completion_edge_velocity_from_full_old_plan_and_coefficient_indices():
    class Indexed:
        horizon, action_dim, robot_dt = 6, 3, .5
        def coefficients(self, obs, history):
            shape = (len(history), 6, 3)
            return torch.zeros(shape), torch.zeros(shape), torch.arange(6)[None, :, None].expand(shape) * .1
    old = torch.arange(6.)[None, :, None].expand(2, -1, 3).clone()
    reference = Indexed()
    for k in (1, 2, 5, 6):
        got = complete_plan(old, k, torch.zeros(2, 2, 4), torch.zeros(2, 2, 3), 2,
                            reference, robot_dt=.5, velocity_weighting="linear", allow_full_completion=True)
        q, p = old[:, -1].clone(), (old[:, -1] - old[:, -2]) / .5
        tail = []
        for j in range(6-k, 6):
            p = p * (1 - .5 * j * .1)
            q = q + .5 * p
            tail.append(q)
        expected = torch.cat([old[:, k:], torch.stack(tail, 1)], 1)
        torch.testing.assert_close(got, expected)
    with pytest.raises(ValueError, match="bootstrap"):
        complete_plan(old, 6, None, None, 0)


def test_noise_once_startup_observed_only_future_independence_and_snapshot_immutability():
    config, windows, completion = fixture(length=7)
    config["source_std"] = .2
    records, _, snapshot = replay(config, windows, completion, 1, modes=(0,))
    changed = dict(windows, future_actions=windows["future_actions"] + 1000.)
    other, _, _ = replay(config, changed, completion, 1, modes=(0,))
    assert torch.equal(records["source_actions"], other["source_actions"])
    assert torch.equal(records["generated_actions"], other["generated_actions"])
    generator = torch.Generator().manual_seed(11)
    startup = records["time_index"] == 0
    noise = torch.randn(2, 16, 3, generator=generator) * .2
    torch.testing.assert_close(records["source_actions"][startup], noise, rtol=0, atol=0)
    assert snapshot.weight.item() == pytest.approx(.1)
    assert snapshot.offset.item() == .5
    assert not snapshot.training and not snapshot.weight.requires_grad


def test_stage_grid_validates_dense_input_and_schedule_does_not_mutate_deployment():
    config, windows, _ = fixture()
    active = stage_config(config, 1)
    assert active["active_k"] == 2 and active["execute"] == 8 and "active_k" not in config
    sparse = {key: value[::2] for key, value in windows.items()}
    with pytest.raises(ValueError, match="dense contiguous"):
        stage_windows(sparse, active)
    fixed = dict(config, protocol="self_source_v1")
    assert stage_windows(windows, fixed) is windows
    assert scheduled_k(fixed, 100) == 8


@pytest.mark.parametrize("change", [
    {"k_values_by_block": [1, 4, 2, 8]}, {"k_values_by_block": [True, 2, 4, 8]},
    {"k_values_by_block": [1, 2, 4, 17]}, {"k_values_by_block": [1, 2, 4]},
    {"execute": 4}, {"condition_on_k": False}, {"model_schema": "legacy"},
    {"updates": 9}, {"startup_sampling_fraction": 0}, {"training_completion_modes": [3]},
    {"method": "sb_ou", "rounds": 4, "phase_updates": 2},
    {"curriculum_mode": "gated"},
])
def test_invalid_curricula_are_rejected(change):
    config, _, _ = fixture()
    config.update(change)
    with pytest.raises(ValueError):
        validate_curriculum(config)


@pytest.mark.parametrize("alpha", [0., .8, 1.])
@pytest.mark.parametrize("velocity", [0., 2.])
def test_damping_recurrence_hand_calculations(alpha, velocity):
    class Damping:
        horizon, action_dim, robot_dt = 7, 4, .25
        def coefficients(self, obs, history):
            shape = (2, self.horizon, self.action_dim)
            return torch.zeros(shape), torch.zeros(shape), torch.full(shape, (1-alpha)/self.robot_dt)
    model = Damping()
    old = (torch.arange(7.) * velocity * .25)[None, :, None].expand(2, -1, 4)
    for k in (1, 6, 7):
        result = complete_plan(old, k, None, None, 2, model, robot_dt=.25, allow_full_completion=True)
        offsets = torch.tensor([sum(alpha**j for j in range(1, i+1)) for i in range(1, k+1)])
        expected = old[:, -1:] + offsets[None, :, None] * velocity * .25
        torch.testing.assert_close(result[:, -k:], expected)


def test_adapter_dense_windows_preserve_fixed_split_and_normalization(tmp_path):
    import numpy as np
    from action_bridge.configs.sb_pusht import get_config
    from action_bridge.data.revision_pusht import load_windows
    obs = np.arange(10 * 30 * 5, dtype=np.float32).reshape(10, 30, 5)
    actions = np.arange(10 * 30 * 2, dtype=np.float32).reshape(10, 30, 2)
    path = tmp_path / "episodes.npz"
    np.savez(path, obs=obs, actions=actions)
    base = get_config("fm_paired")
    fixed, fixed_meta = load_windows(path, base)
    current, current_meta = load_windows(path, dict(base, protocol=CURRICULUM_PROTOCOL))
    assert current_meta["replan_grid"] == {"start": 0, "stride": 1}
    assert current_meta["normalization"] == fixed_meta["normalization"]
    assert current_meta["splits"] == fixed_meta["splits"]
    selected = current["train"]["time_index"] % 8 == 0
    for key in fixed["train"]:
        assert torch.equal(fixed["train"][key], current["train"][key][selected])


def test_local_ot_rejects_incompatible_k_and_overlap(monkeypatch):
    import action_bridge.plan_revision.ot as ot
    from action_bridge.data.revision_pusht import make_local_pairer
    from action_bridge.plan_revision.contracts import take
    records = dict(source_actions=torch.randn(4, 6, 3), future_actions=torch.randn(4, 6, 3),
                   completion_id=torch.zeros(4, dtype=torch.long), has_previous_plan=torch.ones(4, dtype=torch.bool),
                   execution_k=torch.tensor([1, 2, 1, 1]),
                   overlap_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0],
                                              [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool))
    neighborhood = dict(neighbors=torch.arange(4).expand(4, -1), features=torch.zeros(4, 2), radius=0.)
    config = dict(ot_block_size=4, endpoint_std=0., ot_entropy=.1, ot_context_weight=1.)
    def inspect(source, target, distance, compatible, **kwargs):
        assert compatible[0].tolist() == [[True, False, True, False], [False, True, False, False],
                                          [True, False, True, False], [False, False, False, True]]
        ids = torch.zeros(1, 1, dtype=torch.long)
        return ids, ids, {"context_displacement": torch.zeros(1)}
    monkeypatch.setattr(ot, "batched_local_ot_pair", inspect)
    batch = take(records, torch.tensor([0]))
    batch["record_id"] = torch.tensor([0])
    make_local_pairer(records, neighborhood, config)(batch, None, "cpu")
