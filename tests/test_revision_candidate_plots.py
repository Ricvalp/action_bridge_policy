"""A candidate plot holds the scene fixed and never invents executed paths."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from action_bridge.eval import revision_pusht_candidate_plots as plots


def example(count=5, *, startup=False, execute=2):
    completed = np.array([[180., 205.], [185., 210.], [192., 215.], [198., 220.]])
    candidates = completed[None] + np.arange(count)[:, None, None] * np.array([3., -2.])
    return {"state_raw": np.array([170., 200., 250., 245., .4]),
            "obs_history_raw": np.zeros((2, 5)), "action_history_raw": np.zeros((2, 2)),
            "old_plan_raw": None if startup else completed - 10,
            "completed_raw": completed, "sources_raw": np.repeat(completed[None], count, axis=0),
            "candidates_raw": candidates, "executed": 0 if startup else execute, "execute": execute,
            "has_previous_plan": not startup,
            "replan": 0 if startup else 3, "robot_step": 0 if startup else 6}


@pytest.mark.parametrize("count,startup,execute", [(20, False, 2), (1, True, 2), (1, True, 4), (1, False, 4)])
def test_render_outputs_and_preserves_input(tmp_path, count, startup, execute):
    data = example(count, startup=startup, execute=execute)
    before = {key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
    paths = plots.render_candidate_batch(data, tmp_path, title="OU-SB smoke")
    assert set(paths) == {"overlay_png", "overlay_svg", "grid_png"}
    for path in paths.values():
        assert (tmp_path / path).is_file()
        assert (tmp_path / path).stat().st_size > 1000
        if path.endswith(".png"):
            with Image.open(tmp_path / path) as picture:
                assert picture.width > 500
                assert picture.height > 500
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
    svg = (tmp_path / paths["overlay_svg"]).read_text()
    assert "NOT simulated pusher paths" in svg
    assert "GT commands" not in svg
    assert "targets outside [0, 512]" in svg
    assert "Startup anchor" in svg if startup else "Current pusher" in svg
    assert not plots._import_pyplot().get_fignums()


def test_limits_include_out_of_workspace_commands_and_retained_suffix():
    data = example()
    data["candidates_raw"][0, 0] = [-400., 1000.]
    np.testing.assert_array_equal(plots._retained(data), data["old_plan_raw"][2:])
    xlim, ylim = plots._limits([data["candidates_raw"], data["state_raw"][:2]], workspace=True)
    assert xlim[0] < -400 and xlim[1] > 512
    assert ylim[0] > 1000 and ylim[1] < 0


def test_rejects_nonfinite_targets(tmp_path):
    data = example()
    data["candidates_raw"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        plots.render_candidate_batch(data, tmp_path)


def test_synthetic_scene_labels_axis_distribution_and_unchanged_input(tmp_path):
    data = example()
    data.update(scene_kind="synthetic_symmetric", replan=None, robot_step=None,
                symmetry_axis_raw=np.array([0., 1.]), lateral_axis_raw=np.array([1., 0.]),
                goal_pose_raw=np.array([250., 256., 0.]))
    before = {key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
    paths = plots.render_candidate_batch(data, tmp_path, title="OU-SB symmetric probe")
    assert set(paths) == {"overlay_png", "overlay_svg", "grid_png", "lateral_png"}
    svg = (tmp_path / paths["overlay_svg"]).read_text()
    assert "synthetic symmetric scene" in svg
    assert "manually specified, not a simulator rollout" in svg
    assert "Symmetry axis" in svg
    assert "learner-reached" not in svg
    assert "robot step None" not in svg
    with Image.open(tmp_path / paths["lateral_png"]) as picture:
        assert picture.width > 1000
    np.testing.assert_allclose(plots._lateral_offsets(data, 1), data["candidates_raw"][:, 1, 0] - 250.)
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
    assert not plots._import_pyplot().get_fignums()


def test_lateral_offset_respects_rotated_axis():
    data = example(1)
    data.update(goal_pose_raw=np.array([250., 256., .5]),
                lateral_axis_raw=np.array([1., -1.]) / np.sqrt(2.))
    expected = ((data["candidates_raw"][0, -1, 0] - 250.) -
                (data["candidates_raw"][0, -1, 1] - 256.)) / np.sqrt(2.)
    np.testing.assert_allclose(plots._lateral_offsets(data, -1), [expected])


def test_large_batch_uses_translucent_commands():
    plt = plots._import_pyplot()
    fig, ax = plt.subplots()
    try:
        plots._commands(ax, example(1000)["candidates_raw"], executed=2)
        assert ax.lines[0].get_alpha() == pytest.approx(.008)
        assert ax.collections[0].get_alpha() == pytest.approx(.03)
    finally:
        plt.close(fig)


def test_source_comparison_uses_shared_axes_and_does_not_change_data(tmp_path, monkeypatch):
    conditions = {}
    for column, label in enumerate(("Axial source", "15 deg right tilt", "90 deg right turn", "Right-side route")):
        data = example(8)
        data.update(scene_kind="synthetic_symmetric", lateral_axis_raw=np.array([1., 0.]),
                    symmetry_axis_raw=np.array([0., 1.]), goal_pose_raw=np.array([250., 256., 0.]))
        data["sources_raw"][:, :, 0] += column * 70
        data["candidates_raw"][:, :, 0] += column * 15
        conditions[label] = data
    before = [{key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
              for data in conditions.values()]
    plt = plots._import_pyplot()
    close = plt.close
    limits = []

    def remember_limits(figure):
        if hasattr(figure, "axes"):
            limits.extend((ax.get_xlim(), ax.get_ylim()) for ax in figure.axes)
        close(figure)

    monkeypatch.setattr(plt, "close", remember_limits)
    paths = plots.render_source_comparison(conditions, tmp_path, title="OU source probe")
    assert set(paths) == {"comparison_png", "comparison_svg"}
    svg = (tmp_path / paths["comparison_svg"]).read_text()
    for label in conditions:
        assert label in svg
    assert "Fixed input source" in svg
    assert "NOT simulated physical paths" in svg
    assert "Input source endpoint" in svg
    for group in (limits[:4], limits[4:]):
        assert len(group) == 4
        for bounds in group:
            np.testing.assert_allclose(bounds, group[0])
    with Image.open(tmp_path / paths["comparison_png"]) as picture:
        assert picture.width > 2000 and picture.height > 1000
    for data, original in zip(conditions.values(), before):
        for key, value in original.items():
            np.testing.assert_array_equal(data[key], value)
    assert not plt.get_fignums()


def test_source_intervention_context_label_is_used(tmp_path):
    data = example(1)
    data.update(scene_kind="synthetic_symmetric", context_label="synthetic symmetric scene",
                source_label="15 deg right tilt", lateral_axis_raw=np.array([1., 0.]))
    plots.render_candidate_batch(data, tmp_path)
    svg = (tmp_path / "candidate_overlay.svg").read_text()
    assert "synthetic symmetric scene" in svg
    assert "15 deg right tilt" in svg
    assert "symmetric context" not in svg
