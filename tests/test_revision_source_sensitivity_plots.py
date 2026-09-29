"""Source sensitivity plots keep matched draws, scene coordinates, and time grids."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from action_bridge.eval import revision_pusht_source_sensitivity_plots as plots


def example(samples=3, families=("translate_x", "translate_y", "bow_x", "bow_y")):
    source = np.array([[180., 205.], [185., 210.], [192., 215.], [198., 220.]])
    sources, labels, names, amplitudes = [source], ["baseline"], ["baseline"], [0.]
    for family in families:
        profile = np.ones(4) if family.startswith("translate") else np.array([0., 1., 1., 0.])
        coordinate = 0 if family.endswith("x") else 1
        for amplitude in (-20., 20.):
            change = np.zeros_like(source)
            change[:, coordinate] = amplitude * profile
            sources.append(source + change)
            labels.append(f"{family} {amplitude:+g} px")
            names.append(family)
            amplitudes.append(amplitude)
    sources = np.stack(sources)
    noise = np.arange(samples)[:, None, None] * np.array([3., -2.])
    chunks = source[None, None] + .25 * (sources[:, None] - source[None, None]) + noise[None]
    times = np.array([0., .1, 1.])
    traces = sources[:, None, None] * (1 - times[None, None, :, None, None])
    traces = traces + chunks[:, :, None] * times[None, None, :, None, None]
    return {"state_raw": np.array([170., 200., 250., 245., .4]),
            "completed_raw": source, "fixed_source_raw": source, "old_plan_raw": source - 10,
            "sources_raw": sources, "candidates_raw": chunks, "revision_states_raw": traces,
            "revision_times": times, "repeat_baseline_raw": chunks[0].copy(),
            "independent_baseline_raw": chunks[0] + np.array([4., 3.]),
            "variant_labels": labels, "families": names, "amplitudes_px": np.array(amplitudes),
            "execute": 2, "executed": 2, "has_previous_plan": True, "replan": 3, "robot_step": 6,
            "intervention_region": "whole"}


@pytest.mark.parametrize("samples,families", [(3, ("translate_x", "translate_y", "bow_x", "bow_y")),
                                             (1, ("translate_x",))])
def test_outputs_are_valid_labeled_and_input_is_unchanged(tmp_path, samples, families):
    data = example(samples, families)
    before = {key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
    paths = plots.render_source_sensitivity(data, tmp_path, title="OU-SB source probe")
    assert set(paths) == {f"{name}_{extension}" for name in (
        "source_response", "paired_paths", "sensitivity_curves", "revision_sensitivity")
                          for extension in ("png", "svg")}
    for relative in paths.values():
        path = tmp_path / relative
        assert path.is_file() and path.stat().st_size > 1000
        if path.suffix == ".png":
            with Image.open(path) as image:
                assert image.width > 700 and image.height > 500
        else:
            ET.parse(path)
    svg = (tmp_path / paths["source_response_svg"]).read_text()
    assert "Command K=2" in svg and "Last command H=4" in svg
    assert "Mean chunks can lie between modes" in svg
    assert "NOT physical trajectories" in svg
    curves = (tmp_path / paths["sensitivity_curves_svg"]).read_text()
    assert "Same source, independent noise" in curves
    assert "Same source, repeated seed" in curves
    assert "zero-input gain is undefined" in curves
    assert "global independence" in curves
    assert "same sampler noise" in (tmp_path / paths["paired_paths_svg"]).read_text()
    for key, expected in before.items():
        np.testing.assert_array_equal(data[key], expected)
    assert not plots._import_pyplot().get_fignums()


def test_axes_preserve_outside_commands_shared_bounds_and_true_time_grid(tmp_path, monkeypatch):
    data = example(families=("translate_x", "translate_y"))
    data["sources_raw"][1, 0] = [-400., 1000.]
    data["candidates_raw"][2, 0, 2] = [1200., -500.]
    plt = plots._import_pyplot()
    close = plt.close
    captured = []

    def remember(figure):
        if hasattr(figure, "axes"):
            axes = figure.axes
            first_collection = axes[0].collections[0] if axes[0].collections else None
            captured.append({"limits": [(ax.get_xlim(), ax.get_ylim()) for ax in axes],
                             "mesh": first_collection.get_coordinates().copy()
                             if hasattr(first_collection, "get_coordinates") else None})
        close(figure)

    monkeypatch.setattr(plt, "close", remember)
    plots.render_source_sensitivity(data, tmp_path)
    for limits in captured[0]["limits"][:4]:
        np.testing.assert_allclose(limits, captured[0]["limits"][0])
        xlim, ylim = limits
        assert xlim[0] < -400 and xlim[1] > 1200
        assert ylim[0] > 1000 and ylim[1] < -500
    np.testing.assert_allclose(captured[-1]["mesh"][0, :, 0], [0., .05, .55, 1.])
    assert not plt.get_fignums()


def test_rms_means_euclidean_pixels_and_known_gain():
    data = example()
    assert plots._rms(np.array([[3., 4.], [3., 4.]])) == pytest.approx(5.)
    for index in range(1, len(data["sources_raw"])):
        input_rms = plots._rms(data["sources_raw"][index] - data["sources_raw"][0])
        output_rms = plots._rms(data["candidates_raw"][index] - data["candidates_raw"][0])
        assert output_rms / input_rms == pytest.approx(.25)


@pytest.mark.parametrize("name", ["sources_raw", "candidates_raw", "revision_states_raw",
                                  "repeat_baseline_raw", "independent_baseline_raw", "state_raw"])
def test_rejects_nonfinite_data_before_creating_figures(tmp_path, name):
    data = example()
    data[name].flat[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        plots.render_source_sensitivity(data, tmp_path)
    assert not list(tmp_path.iterdir())
    assert not plots._import_pyplot().get_fignums()


def test_rejects_bad_trace_shape_and_time_grid(tmp_path):
    data = example()
    data["revision_times"] = np.array([0., .1, .1])
    with pytest.raises(ValueError, match="strictly increasing"):
        plots.render_source_sensitivity(data, tmp_path)
    data["revision_times"] = np.array([0., .1, 1.])
    data["revision_states_raw"] = data["revision_states_raw"][:, :, :-1]
    with pytest.raises(ValueError, match="revision_states_raw"):
        plots.render_source_sensitivity(data, tmp_path)
