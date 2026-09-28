"""Candidate diversity changes randomness, never the learner-reached context."""
import copy
import json

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.eval.revision_pusht_candidates import (
    SOURCE_VARIANTS, candidate_statistics, context_at_replan, sample_candidate_revisions,
    source_variant_context, symmetric_context,
)
from action_bridge.plan_revision import checkpoints
from action_bridge.scripts import visualize_pusht_candidates as cli
from test_revision_generation_trace import scenario


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def collect(tmp_path, monkeypatch, method="sb_ou"):
    policy, config, metadata, dependencies = scenario(method, monkeypatch)
    evaluator.evaluate(policy, config, metadata, dependencies, "cpu", output=tmp_path,
                       seeds=[71], progress=False, trace_generation=True,
                       render=False, save_videos=False)
    episode = json.loads((tmp_path / "episode-seed71.json").read_text())
    return policy, config, metadata, dependencies, episode


def test_histories_at_startup_and_replan_use_only_past_executed_commands(tmp_path, monkeypatch):
    _, config, _, _, episode = collect(tmp_path, monkeypatch)
    startup = context_at_replan(episode, config, 0)
    np.testing.assert_array_equal(startup["obs_history_raw"], [episode["states_raw"][0]] * 2)
    np.testing.assert_array_equal(startup["action_history_raw"], [episode["states_raw"][0][:2]] * 2)
    assert not startup["has_previous_plan"] and startup["old_plan_raw"] is None
    assert startup["executed"] == 0
    context = context_at_replan(episode, config, 2)
    assert context["robot_step"] == 4 and context["executed"] == 2
    np.testing.assert_array_equal(context["obs_history_raw"], episode["states_raw"][3:5])
    np.testing.assert_array_equal(context["action_history_raw"], episode["actions_raw"][2:4])
    np.testing.assert_array_equal(context["old_plan_raw"], episode["plan_traces"][1]["final_raw"])
    # Future commands and future physics must not affect the reconstructed input.
    changed = copy.deepcopy(episode)
    changed["actions_raw"][4:] = [[-100, -100]] * (len(changed["actions_raw"]) - 4)
    changed["states_raw"][5:] = [[-100] * 5] * (len(changed["states_raw"]) - 5)
    again = context_at_replan(changed, config, 2)
    for key in ("state_raw", "obs_history_raw", "action_history_raw"):
        np.testing.assert_array_equal(again[key], context[key])
    with pytest.raises(ValueError, match="did not reach"):
        context_at_replan(episode, config, 999)


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic", "fm_paired", "fm_local_ot"])
def test_fixed_source_only_sb_has_sampler_diversity(tmp_path, monkeypatch, method):
    policy, config, metadata, dependencies, episode = collect(tmp_path, monkeypatch, method)
    context = context_at_replan(episode, config, 1)
    original = copy.deepcopy(policy.state_dict())
    before = torch.get_rng_state().clone()
    policy.train()
    data = sample_candidate_revisions(policy, config, metadata, dependencies, context,
                                      candidates=7, batch_size=3, seed=22)
    assert data["candidates_raw"].shape == (7, 4, 2)
    assert np.isfinite(data["candidates_raw"]).all()
    np.testing.assert_allclose(data["sources_raw"], np.repeat(context["fixed_source_raw"][None], 7, axis=0), atol=3e-5)
    assert torch.equal(before, torch.get_rng_state()) and policy.training
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
    if method.startswith("sb_"):
        assert candidate_statistics(data, 2)["plan_rms_spread_px"] > 0
        assert not np.allclose(data["candidates_raw"][0], data["candidates_raw"][1])
    else:
        np.testing.assert_allclose(data["candidates_raw"], np.repeat(data["candidates_raw"][:1], 7, axis=0), atol=1e-3)
    repeated = sample_candidate_revisions(policy, config, metadata, dependencies, context,
                                          candidates=7, batch_size=3, seed=22)
    np.testing.assert_array_equal(repeated["candidates_raw"], data["candidates_raw"])


def test_independent_source_noise_and_clean_source_are_explicit(tmp_path, monkeypatch):
    policy, config, metadata, dependencies, episode = collect(tmp_path, monkeypatch, "fm_paired")
    context = context_at_replan(episode, config, 1)
    args = policy, config, metadata, dependencies, context
    varied = sample_candidate_revisions(*args, candidates=5, source_noise="independent")
    clean = sample_candidate_revisions(*args, candidates=1, source_noise="none")
    assert not np.array_equal(varied["sources_raw"][0], varied["sources_raw"][1])
    assert not np.array_equal(varied["candidates_raw"][0], varied["candidates_raw"][1])
    np.testing.assert_allclose(clean["sources_raw"][0], context["completed_raw"], atol=3e-5)
    assert candidate_statistics(clean, 2)["plan_rms_spread_px"] == 0


def test_cli_saves_reached_context_all_candidates_and_identity(tmp_path, monkeypatch):
    policy, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    checkpoint = tmp_path / "best.pt"
    checkpoints.save(checkpoint, {"config": config, "metadata": metadata, "dependencies": dependencies,
                                  "ema": policy.state_dict(), "direction": "forward", "step": 50})
    before = checkpoint.read_bytes()
    monkeypatch.setattr(cli, "render_candidate_batch", lambda *args, **kwargs: {"overlay_png": "test.png"})
    for name in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER", "MPLBACKEND", "MPLCONFIGDIR"):
        monkeypatch.delenv(name, raising=False)
    args = ["--checkpoint", str(checkpoint), "--candidates", "5", "--batch-size", "2",
            "--replan", "1", "--seed", "71", "--output-dir", str(tmp_path / "result")]
    assert cli.main(args) == 0
    report = json.loads((tmp_path / "result/candidates.json").read_text())
    assert report["robot_step"] == 2 and report["statistics"]["candidates"] == 5
    assert report["physics_snapshot"] is False and report["source_noise"] == "fixed"
    assert len(report["checkpoint_sha256"]) == 64 and checkpoint.read_bytes() == before
    with np.load(tmp_path / "result/candidates.npz", allow_pickle=False) as arrays:
        assert arrays["candidates_raw"].shape == (5, 4, 2)
        assert arrays["obs_history_raw"].shape == (2, 5)
    with pytest.raises(FileExistsError):
        cli.main(args)


@pytest.mark.parametrize("mode", [0, 1])
def test_synthetic_probe_is_symmetric_collision_free_and_has_consistent_history(monkeypatch, mode):
    _, config, metadata, _ = scenario("sb_ou", monkeypatch)
    config.update(horizon=16, execute=8)
    context = symmetric_context(config, metadata, completion_id=mode)
    state = context["state_raw"]
    axis, lateral = context["symmetry_axis_raw"], context["lateral_axis_raw"]
    block = state[2:4]
    assert context["scene_kind"] == "synthetic_symmetric"
    assert context["replan"] is None and context["robot_step"] is None
    np.testing.assert_allclose(context["old_plan_raw"][7], state[:2])
    np.testing.assert_allclose(context["action_history_raw"][-1], state[:2])
    np.testing.assert_allclose(context["obs_history_raw"][-1], state)
    np.testing.assert_allclose(context["old_plan_raw"][6:8], context["action_history_raw"])
    np.testing.assert_allclose(context["completed_raw"][:8], context["old_plan_raw"][8:], atol=3e-5)
    np.testing.assert_allclose((context["goal_pose_raw"][:2] - block) @ axis, -140., atol=3e-5)
    np.testing.assert_allclose((state[:2] - block) @ axis, -40., atol=3e-5)
    # All supplied commands stay on the symmetry axis, ahead of the crossbar
    # (local y<0), with at least the pusher radius (15px) of clearance.
    for key in ("old_plan_raw", "completed_raw", "fixed_source_raw", "action_history_raw"):
        points = context[key]
        np.testing.assert_allclose((points - block) @ lateral, 0., atol=4e-5)
        assert ((points - block) @ axis < -15.).all()
        assert ((points >= 0) & (points <= 512)).all()
    np.testing.assert_allclose((context["obs_history_raw"][:, :2] - block) @ lateral, 0., atol=3e-5)
    # Reflection across x+y=512 leaves the entire T, goal and square workspace unchanged.
    from action_bridge.eval.visualization import _tee_polygons
    for pose in (state[2:], context["goal_pose_raw"]):
        for polygon in _tee_polygons(pose):
            reflected = 512. - polygon[:, ::-1]
            distances = np.linalg.norm(reflected[:, None] - polygon[None], axis=-1)
            assert distances.min(axis=1).max() < 1e-4


def test_symmetric_probe_rejects_asymmetric_completion_and_fully_executed_old_plan(monkeypatch):
    _, config, metadata, _ = scenario("sb_ou", monkeypatch)
    with pytest.raises(ValueError, match="keep the source symmetric"):
        symmetric_context(config, metadata, completion_id=2)
    with pytest.raises(ValueError, match="0 < K < H"):
        symmetric_context(config | {"execute": config["horizon"]}, metadata)


def test_cli_synthetic_context_does_not_run_simulation(tmp_path, monkeypatch):
    policy, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    checkpoint = tmp_path / "best.pt"
    checkpoints.save(checkpoint, {"config": config, "metadata": metadata, "dependencies": dependencies,
                                  "ema": policy.state_dict(), "direction": "forward", "step": 50})
    monkeypatch.setattr(cli, "evaluate", lambda *a, **kw: pytest.fail("Synthetic probe must not run simulation"))
    monkeypatch.setattr(cli, "render_candidate_batch", lambda *a, **kw: {})
    assert cli.main(["--checkpoint", str(checkpoint), "--scene", "symmetric", "--candidates", "7",
                     "--output-dir", str(tmp_path / "result")]) == 0
    report = json.loads((tmp_path / "result/candidates.json").read_text())
    assert report["replan"] is None and report["robot_step"] is None and report["warmup_episode"] is None
    assert report["source_noise"] == "none" and report["construction"]["completion"] == "fixed_damped"
    assert report["statistics"]["candidates"] == 7
    assert "not learner-reached" in report["scene"]
    assert "command_K" in report["lateral_command_statistics"]
    with np.load(tmp_path / "result/candidates.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["sources_raw"], np.repeat(arrays["completed_raw"][None], 7, axis=0), atol=3e-5)
        assert arrays["candidates_raw"].shape == (7, 4, 2)


@pytest.mark.parametrize("variant", SOURCE_VARIANTS)
def test_source_interventions_preserve_scene_history_and_executed_prefix(monkeypatch, variant):
    _, config, metadata, _ = scenario("sb_ou", monkeypatch)
    config.update(horizon=16, execute=8)
    original = symmetric_context(config, metadata)
    before = copy.deepcopy(original)
    changed = source_variant_context(original, config, metadata, variant=variant)
    for key in ("state_raw", "obs_history_raw", "action_history_raw", "goal_pose_raw"):
        np.testing.assert_array_equal(changed[key], original[key])
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])
    np.testing.assert_array_equal(changed["old_plan_raw"][:8], original["old_plan_raw"][:8])
    np.testing.assert_allclose(changed["completed_raw"][:8], changed["old_plan_raw"][8:], atol=3e-5)
    np.testing.assert_array_equal(changed["fixed_source_raw"], changed["completed_raw"])
    if variant != "axial":
        lateral = (changed["completed_raw"] - changed["state_raw"][2:4]) @ changed["lateral_axis_raw"]
        assert (lateral > 0).all()
    if variant in ("right_tilt_15", "right_turn_90"):
        for key in ("old_plan_raw", "completed_raw"):
            old_radii = np.linalg.norm(original[key] - original["state_raw"][:2], axis=-1)
            new_radii = np.linalg.norm(changed[key] - original["state_raw"][:2], axis=-1)
            np.testing.assert_allclose(new_radii, old_radii, atol=5e-5)


def test_right_route_is_geometrically_clear_of_t_and_walls_at_h16_k8(monkeypatch):
    _, config, metadata, _ = scenario("sb_ou", monkeypatch)
    config.update(horizon=16, execute=8)
    context = source_variant_context(symmetric_context(config, metadata), config, metadata, variant="right_route")
    route = np.vstack((context["state_raw"][:2], context["completed_raw"]))
    t = np.linspace(0., 1., 1001)[None, :, None]
    dense = (route[:-1, None] * (1. - t) + route[1:, None] * t).reshape(-1, 2)
    local = np.stack([(dense - context["state_raw"][2:4]) @ context[axis]
                      for axis in ("lateral_axis_raw", "symmetry_axis_raw")], axis=-1)
    for low, high in (([-60., 0.], [60., 30.]), ([-15., 30.], [15., 120.])):
        outside = np.maximum(np.maximum(np.array(low) - local, local - np.array(high)), 0.)
        assert np.linalg.norm(outside, axis=-1).min() > 15.
    assert ((dense > 15.) & (dense < 512. - 15.)).all()
    # Finishes past the stem tip, still on its positive-lateral side.
    assert local[-1, 0] > 0. and local[-1, 1] > 135.


def test_nonaxial_source_cli_requires_synthetic_scene():
    with pytest.raises(SystemExit):
        cli.main(["--checkpoint", "unused.pt", "--source-variant", "right_turn_90"])
