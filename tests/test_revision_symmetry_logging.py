"""Connect the fixed symmetric probe to the existing image hook, not optimization."""
from pathlib import Path

import pytest
import torch

from action_bridge.plan_revision.tracking import TrackingOptions
from action_bridge.plan_revision.training import track_images
from action_bridge.scripts import sb_pusht


@pytest.mark.parametrize("stage", ["ddim", "sb_ou", "sb_kinetic"])
def test_policy_stage_logs_both_panels_with_same_model_and_step(tmp_path, monkeypatch, stage):
    from action_bridge.eval import revision_pusht_plots, revision_pusht_symmetry

    calls = []
    factories = []

    def plotter(name):
        def plot(model, step):
            assert not model.training and not torch.is_grad_enabled()
            calls.append((name, model, step))
            return [Path(f"{name}-{step}.png")]
        return plot

    monkeypatch.setattr(revision_pusht_plots, "make_action_chunk_plotter",
                        lambda *args, **kwargs: plotter("ordinary"))

    def symmetry(config, metadata, output, device, **kwargs):
        factories.append((config, output, device, kwargs))
        return plotter("symmetric")

    monkeypatch.setattr(revision_pusht_symmetry, "make_symmetry_plotter", symmetry)
    config = {"method": stage, "execute": 8}
    dependencies = {"frozen": "reference"}
    tracker, visualize = sb_pusht.stage_tracking(
        stage, tmp_path, {}, config, {}, "cpu",
        TrackingOptions(enabled=True, images_every=10, symmetry_candidates=17), dependencies)
    assert factories == [(config, tmp_path / stage, "cpu",
                          {"dependencies": dependencies, "candidates": 17})]
    captured = []
    monkeypatch.setattr(tracker, "images", lambda step, panels: captured.append((step, panels)))
    model = torch.nn.Linear(2, 2).train()
    track_images(tracker, visualize, model, 9)
    assert calls == []
    track_images(tracker, visualize, model, 10)
    assert calls == [("ordinary", model, 10), ("symmetric", model, 10)]
    assert captured == [(10, {"examples/action_chunks": [Path("ordinary-10.png")],
                              "examples/symmetric_probe": [Path("symmetric-10.png")]})]
    assert model.training


@pytest.mark.parametrize("stage,options", [
    ("sb_ou", TrackingOptions()),
    ("sb_ou", TrackingOptions(enabled=True, image_count=0)),
    ("sb_ou", TrackingOptions(enabled=True, symmetry_candidates=0)),
    ("reference", TrackingOptions(enabled=True)),
    ("direct_tail", TrackingOptions(enabled=True)),
    ("fm_paired", TrackingOptions(enabled=True)),
])
def test_disabled_or_nonpolicy_stages_do_not_construct_probe(tmp_path, monkeypatch, stage, options):
    from action_bridge.eval import revision_pusht_plots, revision_pusht_symmetry

    def unexpected(*args, **kwargs):
        pytest.fail("This stage must not construct the symmetric probe")

    monkeypatch.setattr(revision_pusht_symmetry, "make_symmetry_plotter", unexpected)
    monkeypatch.setattr(revision_pusht_plots, "make_action_chunk_plotter",
                        lambda *args, **kwargs: lambda model, step: ["ordinary.png"])
    monkeypatch.setattr(revision_pusht_plots, "make_direct_tail_plotter",
                        lambda *args, **kwargs: lambda model, step: ["ordinary.png"])
    _, visualize = sb_pusht.stage_tracking(stage, tmp_path, {}, {}, {}, "cpu", options)
    if options.enabled and options.image_count:
        assert visualize(None, 1) == ["ordinary.png"]
    else:
        assert visualize is None


def test_unsupported_probe_does_not_disable_ordinary_images(tmp_path, monkeypatch):
    from action_bridge.eval import revision_pusht_plots, revision_pusht_symmetry

    monkeypatch.setattr(revision_pusht_symmetry, "make_symmetry_plotter", lambda *args, **kwargs: None)
    monkeypatch.setattr(revision_pusht_plots, "make_action_chunk_plotter",
                        lambda *args, **kwargs: lambda model, step: ["ordinary.png"])
    _, visualize = sb_pusht.stage_tracking("sb_ou", tmp_path, {}, {}, {}, "cpu", TrackingOptions(enabled=True))
    assert visualize(None, 1) == ["ordinary.png"]
