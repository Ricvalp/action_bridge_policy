"""New completion/reference settings survive real inference and diagnostic plots."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.eval import revision_pusht_parallel as parallel
from action_bridge.eval import revision_pusht_plots as plots
from action_bridge.eval import revision_pusht_visualization as visualization
from action_bridge.plan_revision.completion import DirectTailPredictor
from action_bridge.plan_revision.models import build_policy
from action_bridge.plan_revision.training import direct_tail_config
from test_revision_generation_trace import scenario
from test_revision_plots import capture_plots, replay_examples
from test_revision_process_plots import _example


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def direct_scenario(method, monkeypatch):
    _, config, metadata, dependencies = scenario(method, monkeypatch)
    config.update(completion_id=3, training_completion_modes=[0, 1, 3],
                  direct_tail_hidden_dim=8)
    direct_config = direct_tail_config(config)
    direct = DirectTailPredictor(**direct_config)
    dependencies.update(direct_tail_config=direct_config, direct_tail_state=direct.state_dict())
    return build_policy(config), config, metadata, dependencies


@pytest.mark.parametrize("method", ["fm_paired", "sb_ou", "sb_kinetic"])
def test_direct_completion_closed_loop_preserves_overlap_and_full_trace(monkeypatch, tmp_path, method):
    policy, config, metadata, dependencies = direct_scenario(method, monkeypatch)
    result = evaluator.evaluate(policy, config, metadata, dependencies, "cpu", output=tmp_path,
                                completion_id=3, seeds=[71], progress=False, trace_generation=True)
    assert result["completion"] == "direct_mlp" and result["completion_id"] == 3
    episode = json.loads((tmp_path / "episode-seed71.json").read_text())
    traces = episode["plan_traces"]
    assert len(traces) == 5 and traces[0]["startup"]
    for previous, trace in zip(traces, traces[1:]):
        assert trace["completion_id"] == 3 and not trace["startup"]
        assert trace["aligned_old_raw"] == previous["final_raw"][2:]
        assert trace["completed_raw"][:2] == trace["aligned_old_raw"]
        np.testing.assert_array_equal(trace["revision_candidate_plans_raw"][0], trace["perturbed_source_raw"])
        np.testing.assert_array_equal(trace["revision_candidate_plans_raw"][-1], trace["final_raw"])


@pytest.mark.parametrize("failure", ["untrained", "missing", "execute"])
def test_direct_eval_rejects_invalid_setup_before_starting_simulation(monkeypatch, failure):
    policy, config, metadata, dependencies = direct_scenario("fm_paired", monkeypatch)
    if failure == "untrained":
        config["training_completion_modes"] = [0, 1, 2]
        message = "not a trained completion mode"
    elif failure == "missing":
        del dependencies["direct_tail_config"], dependencies["direct_tail_state"]
        message = "requires a fitted DirectTailPredictor"
    else:
        config["execute"] = 1
        message = "execute must match"

    def should_not_open(**kwargs):
        pytest.fail("invalid direct-tail evaluation opened a simulator")

    monkeypatch.setattr(evaluator, "_make_pusht_env", should_not_open)
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(policy, config, metadata, dependencies, "cpu", completion_id=3,
                           seeds=[1], progress=False)


def test_parallel_evaluation_validates_trained_modes_before_spawning(tmp_path):
    config = {"method": "fm_paired", "training_completion_modes": [0, 1, 2]}
    with pytest.raises(ValueError, match="not a trained completion mode"):
        parallel.evaluate_parallel({}, config, "cpu", output=tmp_path, completion_id=3,
                                    seeds=[1], workers=2)


@pytest.mark.parametrize("kind", ["brownian", "isotropic_ou"])
def test_evaluation_reports_applied_reference_rates_not_original_precision(monkeypatch, kind):
    policy, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    config["reference_kind"] = kind
    references = []
    original = evaluator.reference_for

    def capture(batch, cfg):
        reference = original(batch, cfg)
        references.append(reference)
        return reference

    monkeypatch.setattr(evaluator, "reference_for", capture)
    result = evaluator.evaluate(policy, config, metadata, dependencies, "cpu", seeds=[71], progress=False)
    assert result["reference_kind"] == kind
    eigenvalues = [torch.linalg.eigvalsh(reference.precision) for reference in references]
    expected_min = np.mean([float(values.amin(-1).mean()) for values in eigenvalues])
    expected_max = np.mean([float(values.amax(-1).mean()) for values in eigenvalues])
    assert result["reference_rate_min"] == expected_min
    assert result["reference_rate_max"] == expected_max
    if kind == "brownian":
        assert result["reference_rate_min"] == result["reference_rate_max"] == 0
        assert result["reference_stabilized_fraction"] == 0
    else:
        assert result["reference_rate_min"] > 0
        assert result["reference_rate_min"] == result["reference_rate_max"]


def direct_examples():
    config, records, metadata = replay_examples()
    config.update(obs_dim=5, action_dim=2, obs_history=2, action_history=2,
                  direct_tail_hidden_dim=8)
    # Distinct targets make accidental replacement of the expert old suffix visible.
    records["future_actions"] = torch.arange(8 * 4 * 2).reshape(8, 4, 2).float() / 100
    model = DirectTailPredictor(**direct_tail_config(config))
    return config, records, metadata, model


def test_direct_training_preview_keeps_old_expert_suffix_and_labels_it(tmp_path, monkeypatch):
    config, records, metadata, model = direct_examples()
    model.encoder.eval()  # Preserve mixed module training states.
    seen = capture_plots(monkeypatch)
    plot = plots.make_direct_tail_plotter(records, config, metadata, tmp_path, "cpu", count=2)
    before = torch.get_rng_state().clone()
    paths = plot(model, 3)
    assert len(paths) == 2 and paths[0].name == "step_00000003_example_00.png"
    assert torch.equal(before, torch.get_rng_state())
    assert model.training and not model.encoder.training
    for image in seen:
        np.testing.assert_array_equal(image["predicted"][:2], image["retained"])
        assert image["retained"].shape == (2, 2)
        assert "expert chunk (input)" in image["retained_label"]
        assert "expert old-plan input" in image["title"]
        assert "MLP tail" in image["prediction_label"]
        assert image.get("source") is None


def test_direct_preview_current_labels_do_not_enter_predictions(tmp_path, monkeypatch):
    config, records, metadata, model = direct_examples()
    # Only the final pair is selected, so changing its current GT cannot alter
    # either its old-plan inputs or a later preview's old expert plan.
    records = {key: value[:2] for key, value in records.items()}
    seen = capture_plots(monkeypatch)
    first = plots.make_direct_tail_plotter(records, config, metadata, tmp_path, "cpu", count=1)
    first(model, 1)
    records["future_actions"][1] += 100
    second = plots.make_direct_tail_plotter(records, config, metadata, tmp_path, "cpu", count=1)
    second(model, 2)
    np.testing.assert_array_equal(seen[0]["predicted"], seen[1]["predicted"])
    np.testing.assert_array_equal(seen[0]["retained"], seen[1]["retained"])
    assert not np.array_equal(seen[0]["expert"], seen[1]["expert"])


def test_direct_preview_renders_png_and_handles_disabled_logging(tmp_path):
    config, records, metadata, model = direct_examples()
    plot = plots.make_direct_tail_plotter(records, config, metadata, tmp_path, "cpu", count=1)
    image = plot(model, 7)[0]
    with Image.open(image) as opened:
        assert opened.format == "PNG" and opened.size == (720, 720)
    assert plots.make_direct_tail_plotter({}, {}, {}, tmp_path, "cpu", count=0)(model, 8) == []
    with pytest.raises(ValueError, match="negative"):
        plots.make_direct_tail_plotter(records, config, metadata, tmp_path, "cpu", count=-1)


def test_direct_completion_phase_caption_names_the_actual_predictor():
    episode, config = _example()
    episode["plan_traces"][1]["completion_id"] = 3
    _, _, traces = visualization._prepare(episode, config, 1, 1)
    complete = next(frame for frame in visualization._phase_frames(traces[0], 4)
                    if frame.phase == "COMPLETE")
    assert "direct MLP completion" in visualization._caption(complete, config)
