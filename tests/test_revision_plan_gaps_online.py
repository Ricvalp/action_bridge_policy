"""Plan consistency is measured in clean pixel coordinates before replanning."""

import copy
import json

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.eval import revision_pusht_parallel as parallel
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.contracts import ActionCodec
from test_revision_parallel_eval import tiny_state
from test_revision_rollout import Proposal, Revision, setup


@pytest.mark.parametrize("method", ["ddim", "fm_paired"])
def test_plan_gaps_use_actual_state_clipped_command_and_clean_decoded_plan(
        monkeypatch, tmp_path, method):
    env, config, metadata, dependencies, _ = setup(monkeypatch, method=method)
    config.update(max_episode_steps=3, source_std=.5)
    metadata["codec"].update(mean=[50., 20.], std=[10., 5.])
    codec = ActionCodec(**metadata["codec"])
    raw_plan = torch.tensor([[[100., 100.], [600., 100.], [620., 100.], [630., 100.]]])

    class FixedPlan(torch.nn.Module):
        def sample(self, *args, **kwargs):
            return codec.encode(raw_plan), {"nfe": 1}

    original_step = env.step

    def lagging_step(command):
        state, reward, terminated, truncated, info = original_step(command)
        state[0] -= 10.
        env.state = state.copy()
        return state, reward, terminated, truncated, info

    monkeypatch.setattr(env, "step", lagging_step)
    metrics = evaluator.evaluate(FixedPlan(), config, metadata, dependencies, "cpu",
                                 output=tmp_path, seeds=[1], completion_id=0, progress=False)
    episode = json.loads((tmp_path / "episode-seed1.json").read_text())
    assert episode["plan_gaps"] == [{
        "robot_step": 2,
        "retained_pusher_gap_px": 118.,
        "previous_command_mismatch_px": 88.,
        "command_tracking_gap_px": 10.,
        "retained_plan_jump_px": 20.,
    }]
    assert metrics["gap_samples"] == 1
    assert metrics["retained_pusher_gap_px_mean"] == 118.
    assert metrics["previous_command_mismatch_px_max"] == 88.


@pytest.mark.parametrize("execute,max_steps", [(4, 5), (2, 2), (2, 1)])
def test_startup_and_no_retained_chunk_have_no_gap_samples(
        monkeypatch, tmp_path, execute, max_steps):
    _, config, metadata, dependencies, _ = setup(monkeypatch)
    config.update(execute=execute, max_episode_steps=max_steps)
    metrics = evaluator.evaluate(Revision(), config, metadata, dependencies, "cpu",
                                 output=tmp_path, seeds=[1], completion_id=0, progress=False)
    episode = json.loads((tmp_path / "episode-seed1.json").read_text())
    assert episode["plan_gaps"] == []
    assert metrics["gap_samples"] == 0
    assert "retained_pusher_gap_px_mean" not in metrics


def test_diagnostics_do_not_change_stochastic_commands_or_physical_trajectory(monkeypatch, tmp_path):
    _, config, metadata, dependencies, _ = setup(monkeypatch)
    config["source_std"] = .2
    episodes = []
    for label in ("measured", "disabled"):
        if label == "disabled":
            monkeypatch.setattr(evaluator, "gap_record", lambda *args: {})
            monkeypatch.setattr(evaluator, "summarize_gaps", lambda rows: {"gap_samples": len(rows)})
        path = tmp_path / label
        evaluator.evaluate(Revision(), config, metadata, dependencies, "cpu",
                           output=path, seeds=[17], completion_id=0, progress=False)
        episodes.append(json.loads((path / "episode-seed17.json").read_text()))
    measured, disabled = episodes
    for key in measured:
        if key not in ("plan_gaps", "inference_seconds_per_replan"):
            assert measured[key] == disabled[key], key


def test_summary_pools_replans_not_episode_means_and_accepts_missing_diagnostics(monkeypatch, tmp_path):
    _, config, metadata, dependencies, _ = setup(monkeypatch, method="ddim")
    evaluator.evaluate(Proposal(), config, metadata, dependencies, "cpu",
                       output=tmp_path, seeds=[1], progress=False)
    first = json.loads((tmp_path / "episode-seed1.json").read_text())
    second = copy.deepcopy(first)
    second["seed"] = 2
    keys = [key for key in first["plan_gaps"][0] if key != "robot_step"]
    first["plan_gaps"] = [{key: 1. for key in keys}]
    second["plan_gaps"] = [{key: 10. for key in keys}] * 2
    metrics = evaluator.summarize_episodes([first, second], config, metadata)
    assert metrics["gap_samples"] == 3
    assert metrics["retained_pusher_gap_px_mean"] == 7.
    assert metrics["retained_pusher_gap_px_median"] == 10.
    assert metrics["retained_pusher_gap_px_p95"] == 10.
    del first["plan_gaps"]
    assert evaluator.summarize_episodes([first], config, metadata)["gap_samples"] == 0


def test_parallel_and_serial_pool_identical_nonempty_gap_metrics(tmp_path):
    pytest.importorskip("gym_pusht")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = tiny_state("ddim", weights=True)
        state["config"]["max_episode_steps"] = 5
        seeds = state["config"]["evaluation_seeds"]
        expected = evaluator.evaluate(
            checkpoints.restore_policy(state, "cpu"), state["config"], state["metadata"],
            state["dependencies"], "cpu", output=tmp_path / "serial", seeds=seeds,
            save_videos=False, progress=False,
        )
        actual = parallel.evaluate_parallel(
            state, state["config"], "cpu", output=tmp_path / "parallel", seeds=seeds,
            workers=2, threads=1, save_videos=False, progress=False,
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert actual["gap_samples"] == expected["gap_samples"] == 6
    for key, value in expected.items():
        if "_gap_px_" in key or "_mismatch_px_" in key or "_jump_px_" in key:
            assert actual[key] == pytest.approx(value, rel=1e-6, abs=1e-7)
    for seed in seeds:
        serial = json.loads((tmp_path / "serial" / f"episode-seed{seed}.json").read_text())
        spawned = json.loads((tmp_path / "parallel" / f"episode-seed{seed}.json").read_text())
        assert serial["plan_gaps"] == spawned["plan_gaps"]
        np.testing.assert_array_equal(serial["actions_raw"], spawned["actions_raw"])
        np.testing.assert_array_equal(serial["states_raw"], spawned["states_raw"])
