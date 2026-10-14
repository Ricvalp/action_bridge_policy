"""The mismatch diagnostic must not replace data histories with generated ones."""
import copy

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht_mismatch as mismatch
from action_bridge.eval.revision_pusht_mismatch_plots import plot_mismatch, plot_gap_summary
from test_revision_plots import FakePolicy, FakeReference, replay_examples


def case(monkeypatch):
    config, windows, metadata = replay_examples()
    config.update(method='fm_paired', protocol='self_source_v1')
    windows['act_hist'].zero_()
    windows['startup_actions'].zero_()
    observations = [np.tile([60., 100., 230., 260., .5], (12, 1)) for _ in range(2)]
    actions = [np.tile([100., 200.], (12, 1)) for _ in range(2)]
    monkeypatch.setattr(mismatch, 'restore_completion', lambda *args: FakeReference().eval())
    return config, windows, metadata, observations, actions


def fake_simulation(state, commands, seed=0):
    # Ideal tracking is a test fixture, never used by the real diagnostic.
    states = np.tile(state, (len(commands) + 1, 1))
    states[1:, :2] = commands
    return dict(states=states.tolist(), commands=np.asarray(commands).tolist())


def run(monkeypatch, mutate_future=False, simulate=fake_simulation):
    config, windows, metadata, obs, actions = case(monkeypatch)
    if mutate_future:
        windows['future_actions'] += 10.
    return mismatch.reconstruct_mismatch(
        FakePolicy(), config, metadata, {'innovation_variance': torch.ones(2)},
        windows, obs, actions, episodes=2, replans=4, seed=31,
        progress=False, simulate=simulate)


def test_training_history_stays_recorded_and_source_is_actual_previous_prediction(monkeypatch):
    before = torch.get_rng_state().clone()
    result = run(monkeypatch)
    assert torch.equal(before, torch.get_rng_state())
    assert len(result['records']) == 6
    assert not result['skipped']
    for row in result['records']:
        assert row['step'] - row['start_step'] == 2
        np.testing.assert_allclose(row['demo_states'][-1][:2], [60., 100.])
        np.testing.assert_allclose(row['demo_commands'], [[100., 200.]] * 2)
        np.testing.assert_allclose(row['old_plan'], [[120., 240.]] * 4)
        np.testing.assert_allclose(row['simulated_policy']['states'][-1][:2], [120., 240.])
        np.testing.assert_allclose(row['completed_source'][:2], row['old_plan'][2:])
        assert row['metrics']['retained_to_demo_pusher_gap_px'] == pytest.approx(np.hypot(60., 140.))
        assert row['metrics']['retained_to_simulated_pusher_gap_px'] == 0.


def test_future_labels_do_not_affect_ema_source_or_simulation(monkeypatch):
    first, changed = run(monkeypatch), run(monkeypatch, mutate_future=True)
    for left, right in zip(first['records'], changed['records']):
        for key in ('old_plan', 'noisy_source', 'current_prediction', 'simulated_policy', 'metrics'):
            assert left[key] == right[key]
        assert left['current_target'] != right['current_target']


def test_early_terminal_branch_is_counted_and_not_falsely_time_aligned(monkeypatch):
    count = 0
    def terminate_first_pair(state, commands, seed=0):
        nonlocal count
        count += 1
        result = fake_simulation(state, commands, seed)
        if count == 1:
            result['states'] = result['states'][:2]
        return result
    result = run(monkeypatch, simulate=terminate_first_pair)
    assert len(result['records']) == 5
    assert len(result['skipped']) == 1
    assert result['skipped'][0]['policy_steps'] == 1


def test_prefix_selection_preserves_startup_and_never_mutates_windows(monkeypatch):
    config, windows, _, _, _ = case(monkeypatch)
    original = copy.deepcopy(windows)
    prefix, ids = mismatch.select_prefixes(windows, config, episodes=1, replans=3, seed=9)
    assert len(ids) == 1
    assert prefix['time_index'].tolist() == [0, 2, 4]
    for key in windows:
        torch.testing.assert_close(windows[key], original[key])
    with pytest.raises(ValueError, match='not in'):
        mismatch.select_prefixes(windows, config, episodes=1, replans=3, seed=9, episode=999)


def test_curriculum_prefix_keeps_dense_intermediate_histories(monkeypatch):
    config, windows, _, _, _ = case(monkeypatch)
    config['protocol'] = 'self_source_k_curriculum_v1'
    windows['time_index'] //= 2
    prefix, _ = mismatch.select_prefixes(windows, config, episodes=1, replans=2, seed=9)
    assert prefix['time_index'].tolist() == [0, 1, 2]


def test_real_plots_render_and_selection_is_explicit(tmp_path, monkeypatch):
    result = run(monkeypatch)
    selected = mismatch.representative_examples(result['records'])
    assert [name for name, _ in selected] == ['median_gap', 'p90_gap']
    plot_mismatch(selected[0][1], tmp_path / 'example.png')
    plot_gap_summary(result['records'], tmp_path / 'summary.png')
    from PIL import Image
    for name in ('example.png', 'summary.png'):
        with Image.open(tmp_path / name) as image:
            assert image.format == 'PNG'
            assert image.size[0] > 1000


def test_reject_unaligned_physical_paths():
    with pytest.raises(ValueError, match='aligned'):
        mismatch.compare_paths(np.zeros((3, 5)), np.zeros((2, 2)), np.zeros((4, 2)),
                               np.zeros((2, 5)), np.zeros((3, 5)), 2)
