"""Offline gap diagnostics use matched held-out contexts and raw pixel geometry."""
import copy
import json

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht_gaps as gaps
from action_bridge.scripts import diagnose_pusht_sources as cli
from test_revision_plots import FakePolicy, FakeReference, replay_examples


def case(monkeypatch):
    config, records, metadata = replay_examples()
    config.update(method="sb_ou", protocol="self_source_v1", obs_dim=5, action_dim=2,
                  obs_history=2, action_history=2)
    payload = {"metadata": copy.deepcopy(metadata), "records": {"val": records},
               "data_spec": {key: config[key] for key in ("horizon", "execute", "obs_history", "action_history")}}

    def restore(*args):
        torch.rand(7)  # Initialization must not disturb the caller's random stream.
        return FakeReference().eval()

    monkeypatch.setattr(gaps, "restore_completion", restore)
    return config, records, metadata, payload


def test_gap_geometry_uses_next_retained_target_not_final_executed_target():
    result = gaps.gap_record([0, 0], [3, 4], [[0, 0], [3, 4], [6, 8], [9, 12]], 2)
    assert result == dict(retained_pusher_gap_px=10., previous_command_mismatch_px=0.,
                          command_tracking_gap_px=5., retained_plan_jump_px=5.)
    assert gaps.summarize_gaps([]) == {"gap_samples": 0}
    summary = gaps.summarize_gaps([result, {key: 2 * value for key, value in result.items()}])
    assert summary["gap_samples"] == 2
    assert summary["retained_pusher_gap_px_mean"] == 15
    assert summary["retained_pusher_gap_px_p95"] == 19.5


@pytest.mark.parametrize("executed", [0, 4, 5])
def test_no_fake_zero_gap_for_missing_retained_plan(executed):
    with pytest.raises(ValueError, match="retained-plan"):
        gaps.gap_record([0, 0], [0, 0], np.zeros((4, 2)), executed)


def test_matched_panels_decode_once_exclude_startup_and_preserve_rng(monkeypatch):
    config, records, metadata, payload = case(monkeypatch)
    before = copy.deepcopy(records)
    policy = FakePolicy()
    policy.child.eval()
    rng = torch.get_rng_state().clone()
    report = gaps.diagnose_offline_sources(policy, config, metadata, {"innovation_variance": torch.ones(2)},
                                           payload, "cpu", episodes=2, replans=4)
    assert torch.equal(rng, torch.get_rng_state())
    assert policy.training and not policy.child.training and not policy.calls
    assert report["metrics"]["offline_self_gap_samples"] == 6
    assert report["metrics"]["offline_expert_gap_samples"] == 6
    self_rows, expert_rows = report["records"]["self"], report["records"]["expert"]
    assert [(row["episode_id"], row["robot_step"]) for row in self_rows] == [
        (row["episode_id"], row["robot_step"]) for row in expert_rows]
    for generated, expert in zip(self_rows, expert_rows):
        assert generated["robot_step"] > 0
        np.testing.assert_allclose(generated["pusher_xy"], [60., 100.])
        np.testing.assert_allclose(generated["retained_first_xy"], [120., 240.])
        np.testing.assert_allclose(expert["retained_first_xy"], [100., 200.])
        assert generated["retained_pusher_gap_px"] == pytest.approx(np.hypot(60, 140))
        assert expert["retained_pusher_gap_px"] == pytest.approx(np.hypot(40, 100))
    for key in records:
        torch.testing.assert_close(records[key], before[key], rtol=0, atol=0)


def test_self_panel_does_not_use_expert_future_labels(monkeypatch):
    config, records, metadata, payload = case(monkeypatch)
    args = FakePolicy(), config, metadata, {"innovation_variance": torch.ones(2)}
    original = gaps.diagnose_offline_sources(*args, payload, "cpu", episodes=1, replans=3)
    changed = copy.deepcopy(payload)
    changed["records"]["val"]["future_actions"] += 20
    second = gaps.diagnose_offline_sources(*args, changed, "cpu", episodes=1, replans=3)
    assert original["records"]["self"] == second["records"]["self"]
    assert original["records"]["expert"] != second["records"]["expert"]


def test_seeded_episode_prefix_selection_is_repeatable(monkeypatch):
    config, _, metadata, payload = case(monkeypatch)
    args = FakePolicy(stochastic=True), config, metadata, {"innovation_variance": torch.ones(2)}, payload, "cpu"
    first = gaps.diagnose_offline_sources(*args, episodes=1, replans=3, seed=31)
    torch.rand(23)
    second = gaps.diagnose_offline_sources(*args, episodes=1, replans=3, seed=31)
    assert first == second
    assert first["metrics"]["offline_self_gap_samples"] == 2


@pytest.mark.parametrize("key", ["codec", "normalization", "splits", "dataset_sha256"])
def test_reject_mismatched_windows(monkeypatch, key):
    config, _, metadata, payload = case(monkeypatch)
    payload["metadata"][key] = "wrong"
    with pytest.raises(ValueError, match=key):
        gaps.diagnose_offline_sources(FakePolicy(), config, metadata, {}, payload, "cpu")


def test_allow_dataset_relocation_but_not_execute_override(monkeypatch):
    config, _, metadata, payload = case(monkeypatch)
    metadata["dataset_path"] = "/hpc/data.zarr"
    payload["metadata"]["dataset_path"] = "/workstation/data.zarr"
    gaps.diagnose_offline_sources(FakePolicy(), config, metadata, {"innovation_variance": torch.ones(2)}, payload, "cpu")
    with pytest.raises(ValueError, match="execute"):
        gaps.diagnose_offline_sources(FakePolicy(), config | {"execute": 1}, metadata, {}, payload, "cpu")


def test_k_equals_h_returns_only_zero_counts_without_generation(monkeypatch):
    config, _, metadata, payload = case(monkeypatch)
    config["execute"] = payload["data_spec"]["execute"] = 4
    payload["records"]["val"]["time_index"] *= 2
    report = gaps.diagnose_offline_sources(FakePolicy(), config, metadata, {}, payload, "cpu")
    assert report["metrics"] == {"offline_self_gap_samples": 0, "offline_expert_gap_samples": 0}
    assert report["selection"]["no_retained_plan"]


def test_reject_untrained_completion_mode(monkeypatch):
    config, _, metadata, payload = case(monkeypatch)
    config["training_completion_modes"] = [0, 1, 3]
    with pytest.raises(ValueError, match="not trained"):
        gaps.diagnose_offline_sources(FakePolicy(), config, metadata, {}, payload, "cpu", completion_id=2)


def test_reject_old_windows_without_startup(monkeypatch):
    config, _, metadata, payload = case(monkeypatch)
    del payload["records"]["val"]["startup_actions"]
    with pytest.raises(ValueError, match="prepare fresh windows"):
        gaps.diagnose_offline_sources(FakePolicy(), config, metadata, {}, payload, "cpu")


def test_offline_cli_writes_report_without_simulator(tmp_path, monkeypatch):
    config, _, metadata, payload = case(monkeypatch)
    checkpoint, windows, output = tmp_path / "policy.pt", tmp_path / "windows.pt", tmp_path / "diagnostic"
    torch.save({"format": "plan_revision_v1", "config": config, "metadata": metadata,
                "dependencies": {"innovation_variance": torch.ones(2)}, "step": 250,
                "direction": "forward"}, checkpoint)
    torch.save({"format": "plan_revision_v1", **payload}, windows)
    monkeypatch.setattr(cli.checkpoints, "restore_policy", lambda *args: FakePolicy())
    previous_threads = torch.get_num_threads()
    try:
        assert cli.main(["--checkpoint", str(checkpoint), "--windows", str(windows),
                         "--output-dir", str(output), "--episodes", "1", "--replans", "3"]) == 0
    finally:
        torch.set_num_threads(previous_threads)
    report = json.loads((output / "source_gaps.json").read_text())
    assert report["metrics"]["offline_self_gap_samples"] == 2
    assert report["identity"]["checkpoint_step"] == 250
    assert len(report["identity"]["checkpoint_sha256"]) == 64
    assert len(report["identity"]["windows_sha256"]) == 64
    with pytest.raises(FileExistsError):
        cli.main(["--checkpoint", str(checkpoint), "--windows", str(windows), "--output-dir", str(output)])
