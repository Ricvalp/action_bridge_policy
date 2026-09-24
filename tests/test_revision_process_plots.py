"""Revision movies distinguish recorded motion from frozen-scene generation."""
from __future__ import annotations

import json
import numpy as np
from PIL import Image
import pytest

from action_bridge.eval import revision_pusht_visualization as plots


def _example(*, execute=2):
    horizon = 4
    config = {"method": "fm_paired", "horizon": horizon, "execute": execute,
              "robot_dt": .1, "source_std": .03}
    length = 2 * execute + 1  # Last execution is interrupted by the episode end.
    states = np.array([[80 + 4 * i, 100 + 2 * i, 240 + i, 230., .1 * i]
                       for i in range(length + 1)])
    commands, traces = [], []
    old = None
    for replan, step in enumerate(range(0, length, execute)):
        startup = old is None or execute == horizon
        anchor = states[0, :2] if not commands else commands[-1]
        retained = None if startup else old[execute:].copy()
        completed = (np.repeat(anchor[None], horizon, axis=0) if startup else
                     np.concatenate((retained, np.array([[220 + 5 * i, 260 + i]
                                                         for i in range(execute)]))))
        source = completed + np.array([10., -15.])
        final = np.array([[600., 110.], [160., 140.], [170., 160.], [180., 180.]]) + replan
        times = np.array([0., .13, .7, 1.])
        candidates = np.array([(1 - tau) * source + tau * final for tau in times])
        traces.append({"replan": replan, "step": step, "state_raw": states[step].copy(),
                       "old_plan_raw": old, "old_executed": 0 if old is None else execute,
                       "startup": startup, "has_previous_plan": not startup,
                       "startup_anchor_raw": anchor.copy(), "completion_id": 2,
                       "aligned_old_raw": retained, "completed_raw": completed,
                       "perturbed_source_raw": source, "final_raw": final, "nfe": 6,
                       "revision_candidate_plans_raw": candidates, "revision_times": times})
        commands.extend(np.clip(final[:min(execute, length - step)], 0, 512))
        old = final.copy()
    return {"states_raw": states, "actions_raw": np.array(commands), "plan_traces": traces}, config


def test_phase_order_exact_candidates_and_physical_states():
    episode, config = _example()
    states, actions, traces = plots._prepare(episode, config, 1, 3)
    assert [trace["replan"] for trace in traces] == [1, 2]
    trace = traces[0]
    frames = list(plots._phase_frames(trace, config["horizon"]))
    phases = [frame.phase for frame in frames]
    assert phases[:3] == ["EXECUTE"] * 3
    assert [frame.step for frame in frames[:3]] == [0, 1, 2]
    assert all(frame.step == trace["step"] for frame in frames[3:])
    assert phases[3:8] == ["ALIGN", "COMPLETE", "COMPLETE", "COMPLETE", "SOURCE_NOISE"]
    assert phases[8:] == ["REVISE"] * 4 + ["READY"]
    revision = [frame for frame in frames if frame.phase == "REVISE"]
    assert [frame.trace["revision_times"][frame.revision] for frame in revision] == [0., .13, .7, 1.]
    for frame, candidate in zip(revision, trace["revision_candidate_plans_raw"]):
        np.testing.assert_array_equal(plots._layers(frame, 2)[-1][0], candidate)
    # Actual commands and actual pusher positions are separate; neither overwrites plans.
    assert actions[0, 0] == 512 and states[1, 0] == 84
    assert trace["old_plan_raw"][0, 0] == 600
    assert "2 NFE/step" in plots._caption(revision[-1], config)
    assert "Midpoint accepted steps" in plots._caption(revision[-1], config)
    assert "robot time" not in plots._caption(frames[0], config)


def test_alignment_shifts_indices_and_appends_only_actual_tail():
    episode, config = _example()
    _, _, traces = plots._prepare(episode, config, 1, 1)
    trace = traces[0]
    frames = list(plots._phase_frames(trace, 4))
    aligned = next(frame for frame in frames if frame.phase == "ALIGN")
    layers = plots._layers(aligned, 2)
    assert layers[0][1] == -2
    np.testing.assert_array_equal(layers[1][0], trace["old_plan_raw"][2:])
    completed = [frame for frame in frames if frame.phase == "COMPLETE"]
    assert [frame.count for frame in completed] == [3, 4, 4]
    np.testing.assert_array_equal(plots._layers(completed[0], 2)[-1][0], trace["completed_raw"][2:3])
    noisy = next(frame for frame in frames if frame.phase == "SOURCE_NOISE")
    np.testing.assert_array_equal(plots._layers(noisy, 2)[-1][0], trace["perturbed_source_raw"])
    assert not np.array_equal(trace["completed_raw"], trace["perturbed_source_raw"])


def test_full_horizon_execution_is_command_anchor_startup_not_tail_completion():
    episode, config = _example(execute=4)
    _, _, traces = plots._prepare(episode, config, 1, 1)
    trace = traces[0]
    assert trace["startup"] and trace["old_executed"] == 4
    assert not np.array_equal(trace["startup_anchor_raw"], trace["state_raw"][:2])
    frames = list(plots._phase_frames(trace, 4))
    startup = next(frame for frame in frames if frame.phase == "STARTUP")
    assert "no old-tail completion" in plots._caption(startup, config)
    np.testing.assert_array_equal(plots._layers(startup, 4)[0][0][0], episode["actions_raw"][3])
    assert [frame.count for frame in frames if frame.phase == "COMPLETE"] == [1, 2, 3, 4, 4]


def test_initial_startup_and_missing_trace_boundary():
    episode, config = _example()
    _, _, traces = plots._prepare(episode, config, 0, 1)
    assert next(plots._phase_frames(traces[0], 4)).phase == "STARTUP"
    episode["plan_traces"] = episode["plan_traces"][::2]
    _, _, traces = plots._prepare(episode, config, 0, 5)
    assert [trace["replan"] for trace in traces] == [0]  # Never jump the missing replan.
    with pytest.raises(ValueError, match="start_replan=9.*no recorded trace"):
        plots._prepare(episode, config, 9, 1)


@pytest.mark.parametrize("field", ["completed_raw", "perturbed_source_raw", "revision_times",
                                    "revision_candidate_plans_raw"])
def test_missing_full_trace_is_an_explicit_error(field, tmp_path):
    episode, config = _example()
    del episode["plan_traces"][1][field]
    with pytest.raises(ValueError, match="full traces|revision_times|revision_candidate"):
        plots.render_revision_process(episode, config, tmp_path)
    assert not list(tmp_path.iterdir())


def test_scene_is_frozen_and_outside_targets_are_not_clipped():
    episode, config = _example()
    states, actions, traces = plots._prepare(episode, config, 1, 1)
    plt = plots._import_pyplot()
    fig, axes = plt.subplots(1, 3)
    try:
        bounds = plots._bounds(traces, states)
        assert bounds[1][0] > 600
        frame = plots._Frame(traces[0], "REVISE", traces[0]["step"], revision=3)
        plots._draw_panel(axes, frame, states, actions, config, bounds)
        pusher = next(patch for patch in axes[0].patches if patch.get_label() == "Recorded pusher")
        np.testing.assert_array_equal(pusher.center, states[2, :2])
        revised = next(line for line in axes[0].lines if line.get_label() == "Revised targets")
        assert revised.get_xdata()[0] == 601  # Not clipped to arena or actual commands.
        assert axes[0].get_xlim()[1] > 601
    finally:
        plt.close(fig)


def test_real_mp4_storyboard_and_optional_gif(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    episode, config = _example()
    result = plots.render_revision_process(episode, config, tmp_path / "process.mp4",
                                           start_replan=2, replans=3, fps=2, save_gif=True)
    json.dumps(result)
    assert result["replans"] == [2]
    assert result["last_physical_step"] == 5  # Early end: one command, not K=2.
    assert result["phases"][-1] == "EXECUTE_FINAL"
    assert len(result["storyboards"]) == 1 and result["illustrative_timing"]
    with Image.open(result["storyboards"][0]) as storyboard:
        assert storyboard.format == "PNG" and storyboard.width > 1500
    with imageio.get_reader(result["video"], format="FFMPEG") as reader:
        assert reader.get_meta_data()["fps"] == 2
        assert reader.count_frames() == result["frame_count"]
        assert reader.get_data(0).shape == (720, 1280, 3)
    with Image.open(result["gif"]) as gif:
        assert gif.format == "GIF"
        assert gif.info["duration"] >= 500


def test_mp4_streams_frames_default_without_gif_and_preserves_episode(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    episode, config = _example()
    before = episode["plan_traces"][1]["final_raw"].copy()
    observed = []

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, pixels):
            assert pixels.shape == (720, 1280, 3)
            observed.append(1)

    def writer(path, **kwargs):
        assert kwargs["ffmpeg_params"] == ["-threads", "1"]
        assert kwargs["fps"] == 10
        return Writer()

    monkeypatch.setattr(imageio, "get_writer", writer)
    monkeypatch.setattr(imageio, "mimsave", lambda *args, **kwargs: pytest.fail("GIF must be opt-in"))
    monkeypatch.setattr(plots, "_storyboard", lambda *args: None)
    monkeypatch.setattr(plots, "_draw_panel", lambda *args, **kwargs: None)
    result = plots.render_revision_process(episode, config, tmp_path, replans=1)
    assert result["gif"] is None and result["frame_count"] == len(observed)
    assert result["replans"] == [1] and len(result["storyboards"]) == 1
    np.testing.assert_array_equal(episode["plan_traces"][1]["final_raw"], before)


def test_writer_failure_closes_matplotlib_figure(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    episode, config = _example()
    plt = plots._import_pyplot()
    previous = plt.get_fignums()
    monkeypatch.setattr(plots, "_storyboard", lambda *args: None)

    def fail(*args, **kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(imageio, "get_writer", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        plots.render_revision_process(episode, config, tmp_path)
    assert plt.get_fignums() == previous


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": float("nan")}, {"replans": 0}, {"start_replan": -1}])
def test_bad_render_arguments_fail_before_output(tmp_path, kwargs):
    episode, config = _example()
    with pytest.raises(ValueError):
        plots.render_revision_process(episode, config, tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())
