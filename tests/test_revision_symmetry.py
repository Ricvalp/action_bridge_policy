"""Synthetic training probes preserve training state and never run physics."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.eval import revision_pusht_symmetry as symmetry
from action_bridge.plan_revision.models import build_policy
from test_revision_generation_trace import scenario


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic", "ddim"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_real_probe_is_repeatable_fixed_context_and_preserves_training(
        tmp_path, monkeypatch, method, conditioned):
    model, config, metadata, dependencies = scenario(method, monkeypatch)
    if conditioned:
        config.update(protocol="self_source_k_curriculum_v1", condition_on_k=True,
                      model_schema="k_conditioned_v1", active_k=1)
        model = build_policy(config)
    model.train()
    # Mixed train/eval flags and gradients should also remain untouched.
    next(iter(model.children())).eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    flags = [module.training for module in model.modules()]
    weights = copy.deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    config_before, metadata_before = copy.deepcopy(config), copy.deepcopy(metadata)
    dependency_before = copy.deepcopy(dependencies)
    calls, rendered = [], []
    original = type(model).sample

    def sample(self, obs, actions, *args, **kwargs):
        calls.append((obs.clone(), actions.clone(), kwargs.copy()))
        return original(self, obs, actions, *args, **kwargs)

    monkeypatch.setattr(type(model), "sample", sample)
    monkeypatch.setattr(evaluator, "_make_pusht_env", lambda **kwargs: pytest.fail("No simulator should be opened"))
    monkeypatch.setattr(symmetry, "_render", lambda data, output, **kwargs: rendered.append(data) or [
        output / "symmetric_overlay.png", output / "symmetric_lateral.png"])
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    plot = symmetry.make_symmetry_plotter(config, metadata, tmp_path, "cpu", dependencies=dependencies,
                                         candidates=5, batch_size=3, seed=71)
    assert config == config_before and metadata == metadata_before
    # Even if the caller reuses and edits its stage config, probe deployment K
    # and normalizers were captured when the plotter was created.
    config["execute"] = 1
    metadata["codec"]["mean"][0] = -100
    for step in (10, 20):
        paths = plot(model, step)
        assert len(paths) == 2 and all(path.parent.name == f"step_{step:06d}" for path in paths)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and numpy_rng[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert flags == [module.training for module in model.modules()]
    for key in weights:
        torch.testing.assert_close(model.state_dict()[key], weights[key], rtol=0, atol=0)
    for before, parameter in zip(gradients, model.parameters()):
        torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
    for key in dependency_before["completion_state"]:
        torch.testing.assert_close(dependencies["completion_state"][key], dependency_before["completion_state"][key])
    first, second = rendered
    np.testing.assert_array_equal(first["candidates_raw"], second["candidates_raw"])
    assert first["candidates_raw"].shape == (5, 4, 2)
    assert not np.array_equal(first["candidates_raw"][0], first["candidates_raw"][1])
    report = json.loads((tmp_path / "symmetric_probe/step_000010/probe.json").read_text())
    assert report["deployment_k"] == 2 and report["sampling_seed"] == 71
    assert report["batch_size"] == 3 and report["candidates"] == 5
    assert not report["simulated_candidates"] and report["action_units"] == "pixels"
    assert report["condition_on_k"] == (conditioned and method != "ddim")
    assert [len(obs) for obs, _, _ in calls] == [3, 2, 3, 2]
    for obs, actions, kwargs in calls:
        torch.testing.assert_close(obs, obs[:1].expand_as(obs))
        torch.testing.assert_close(actions, actions[:1].expand_as(actions))
        if method == "ddim":
            assert set(kwargs) == {"generator"}
        else:
            torch.testing.assert_close(kwargs["source_actions"], kwargs["source_actions"][:1].expand_as(kwargs["source_actions"]))
            assert kwargs["has_previous_plan"].all()
            if conditioned:
                assert (kwargs["execution_k"] == 2).all()
    with np.load(tmp_path / "symmetric_probe/step_000010/samples.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["candidates_raw"], first["candidates_raw"])
        assert arrays["lateral_offsets_px"].shape == (5, 2)
        assert ("fixed_source_raw" in arrays) == (method != "ddim")
    if method != "ddim":
        source = first["fixed_source_raw"]
        lateral = (source - first["goal_pose_raw"][:2]) @ first["lateral_axis_raw"]
        np.testing.assert_allclose(lateral, 0., atol=3e-5)
        np.testing.assert_allclose(first["sources_raw"], np.repeat(source[None], 5, axis=0), atol=3e-5)
        assert report["completion_id"] == 1 and report["source_noise"] == "none"
    else:
        assert report["completion_id"] is None and "no source plan" in report["note"]
        assert not any(key in first for key in ("sources_raw", "fixed_source_raw", "completed_raw"))


@pytest.mark.parametrize("method", ["ddim", "sb_ou"])
def test_real_figures_exist_and_close(tmp_path, monkeypatch, method):
    model, config, metadata, dependencies = scenario(method, monkeypatch)
    plot = symmetry.make_symmetry_plotter(config, metadata, tmp_path, "cpu", dependencies=dependencies,
                                         candidates=4, batch_size=3)
    from action_bridge.eval.visualization import _import_pyplot
    plt = _import_pyplot()
    before = plt.get_fignums()
    paths = plot(model, 1)
    assert plt.get_fignums() == before
    assert len(paths) == 2 and all(path.stat().st_size > 1000 for path in paths)


@pytest.mark.parametrize("override,metadata_override", [
    ({"method": "fm_paired"}, None), ({"method": "reference"}, None),
    ({"obs_dim": 6}, None), ({"action_dim": 3}, None),
    ({}, {"units": "meters"}), ({}, {"semantics": "delta_target"}),
])
def test_inapplicable_profiles_skip_without_sampling(tmp_path, monkeypatch, override, metadata_override):
    _, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    config.update(override)
    if metadata_override:
        metadata["codec"].update(metadata_override)
    assert symmetry.make_symmetry_plotter(config, metadata, tmp_path, "cpu", dependencies=dependencies) is None
    assert not list(tmp_path.iterdir())


def test_disabled_and_untrained_mode_skip_with_clear_warning(tmp_path, monkeypatch):
    _, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    assert symmetry.make_symmetry_plotter(config, metadata, tmp_path, "cpu", candidates=0) is None
    with pytest.warns(UserWarning, match="fixed_damped"):
        assert symmetry.make_symmetry_plotter(config | {"training_completion_modes": [2]}, metadata,
                                              tmp_path, "cpu", dependencies=dependencies) is None
    with pytest.warns(UserWarning, match="deployment K"):
        assert symmetry.make_symmetry_plotter(config | {"execute": 4}, metadata, tmp_path,
                                              "cpu", dependencies=dependencies) is None
    with pytest.warns(UserWarning, match="reference"):
        assert symmetry.make_symmetry_plotter(config, metadata, tmp_path, "cpu") is None


@pytest.mark.parametrize("options", [{"candidates": -1}, {"batch_size": 0}, {"seed": -1}, {"seed": 2**63}])
def test_invalid_options(options, tmp_path, monkeypatch):
    _, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    with pytest.raises(ValueError, match="candidates"):
        symmetry.make_symmetry_plotter(config, metadata, tmp_path, "cpu", dependencies=dependencies, **options)
