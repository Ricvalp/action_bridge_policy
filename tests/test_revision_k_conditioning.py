"""K is a fixed-width condition; early checkpoints still deploy at configured K."""
import copy
import json

import pytest
import torch

from action_bridge.eval import revision_pusht as evaluator
from action_bridge.eval import revision_pusht_cli as cli
from action_bridge.plan_revision import checkpoints
from action_bridge.plan_revision.gaussian import GaussianReference
from action_bridge.plan_revision.models import build_policy
from test_revision_eval_cli import evaluation
from test_revision_generation_trace import scenario
from test_revision_models import batch, config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def conditioned_config(method, horizon=9, action_dim=3):
    return config(method, horizon, action_dim) | {
        "protocol": "self_source_k_curriculum_v1", "condition_on_k": True,
        "model_schema": "k_conditioned_v1", "execute": 8,
    }


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_all_revisers_keep_shapes_parameters_and_ema_compatible_across_k(method):
    cfg = conditioned_config(method)
    model = build_policy(cfg)
    data = batch(cfg)
    n = cfg["horizon"] * cfg["action_dim"]
    reference = (GaussianReference(torch.eye(n).expand(2, n, n), torch.zeros(2, n),
                                  kind="kinetic" if method == "sb_kinetic" else "ou")
                 if method.startswith("sb_") else None)
    parameter_ids = [id(value) for value in model.parameters()]
    shapes = {name: value.shape for name, value in model.state_dict().items()}
    for k in (1, 2, 4, 8, 9):
        data["execution_k"] = torch.full((2,), k)
        kwargs = {"reference": reference} if reference is not None else {}
        directions = ("forward", "reverse") if reference is not None else (None,)
        for direction in directions:
            loss = model.loss(data, **kwargs, **({"direction": direction} if direction else {}))["loss"]
            assert torch.isfinite(loss)
            loss.backward()
        sampled, metrics = model.sample(
            data["obs_hist"], data["act_hist"], data["completion_id"],
            source_actions=data["source_actions"], reference=reference, execution_k=data["execution_k"],
            generator=torch.Generator().manual_seed(77),
        )
        assert sampled.shape == (2, 9, 3) and torch.isfinite(sampled).all()
        assert metrics["nfe"] == cfg["num_inference_steps"]
        assert parameter_ids == [id(value) for value in model.parameters()]
        assert shapes == {name: value.shape for name, value in model.state_dict().items()}
    restored = checkpoints.restore_policy({"config": cfg, "ema": copy.deepcopy(model.state_dict())})
    assert shapes == {name: value.shape for name, value in restored.state_dict().items()}
    assert all(value.grad is not None for value in model.parameters())


def test_k_is_explicit_integral_bounded_and_not_inferred_from_active_training_stage():
    cfg = conditioned_config("fm_paired") | {"active_k": 1}
    field = build_policy(cfg).field
    data = batch(cfg)
    args = (data["obs_hist"], data["act_hist"], data["completion_id"])
    for value in (None, 0, 10, 1.5, True, torch.ones(2, 1), torch.ones(3)):
        with pytest.raises(ValueError, match="execution_k"):
            field.conditioning(*args, execution_k=value)
    one = field.conditioning(*args, execution_k=1)
    eight = field.conditioning(*args, execution_k=8)
    assert one.shape == eight.shape and not torch.equal(one, eight)
    # Startup still knows the NEXT execution interval, while the separate bit
    # says there was no previous plan and hence no elapsed alignment shift.
    startup = field.conditioning(*args, has_previous_plan=False, execution_k=8)
    assert not torch.equal(eight, startup)


def test_legacy_architecture_and_ddim_are_unchanged_and_migration_not_silent():
    legacy_config = config("fm_paired")
    legacy = build_policy(legacy_config)
    assert not any("k_embedding" in name for name in legacy.state_dict())
    restored = checkpoints.restore_policy({"config": legacy_config, "ema": legacy.state_dict()})
    assert set(restored.state_dict()) == set(legacy.state_dict())
    for extra in ({"condition_on_k": True}, {"model_schema": "k_conditioned_v1"},
                  {"model_schema": "unknown"}):
        with pytest.raises(ValueError, match="schema"):
            build_policy(legacy_config | extra)
    new = build_policy(legacy_config | {"condition_on_k": True, "model_schema": "k_conditioned_v1"})
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(legacy.state_dict())
    ordinary = build_policy(config("ddim"))
    conditioned = build_policy(config("ddim") | {"condition_on_k": True, "model_schema": "k_conditioned_v1"})
    assert {name: item.shape for name, item in ordinary.state_dict().items()} == {
        name: item.shape for name, item in conditioned.state_dict().items()}


@pytest.mark.parametrize("override", [None, 2])
def test_early_curriculum_checkpoint_evaluates_deployment_k_or_explicit_override(evaluation, override):
    run = evaluation("sb_ou")
    run.payload["config"].update(protocol="self_source_k_curriculum_v1", condition_on_k=True,
                                 model_schema="k_conditioned_v1", execute=8)
    run.payload["curriculum"] = {"active_k": 1}
    arguments = ["--checkpoint", str(run.checkpoint), "--output-dir", str(run.output)]
    if override is not None:
        arguments.extend(["--execute", str(override)])
    assert cli.main("sb_ou", arguments) == 0
    actual = 8 if override is None else override
    assert run.evaluate.call_args.args[1]["execute"] == actual
    identity = json.loads((run.output / "evaluation.json").read_text())
    assert identity["training_active_k"] == 1
    assert identity["deployment_k"] == 8 and identity["execution_k"] == actual
    assert identity["evaluation_panel"] == ("deployment_k_8" if override is None else "diagnostic_k_2")


@pytest.mark.parametrize("execute", [1, 2, 4])
def test_simulator_passes_actual_k_and_full_completion_does_not_restart(tmp_path, monkeypatch, execute):
    _, cfg, metadata, dependencies = scenario("fm_paired", monkeypatch, execute=execute)
    cfg.update(protocol="self_source_k_curriculum_v1", condition_on_k=True,
               model_schema="k_conditioned_v1", active_k=1, max_episode_steps=9)
    policy = build_policy(cfg).eval()
    original = policy.sample
    intervals = []

    def sample(*args, **kwargs):
        intervals.append(kwargs["execution_k"].item())
        return original(*args, **kwargs)

    monkeypatch.setattr(policy, "sample", sample)
    metrics = evaluator.evaluate(policy, cfg, metadata, dependencies, "cpu", output=tmp_path,
                                 seeds=[71], progress=False, trace_generation=True)
    assert set(intervals) == {execute} and metrics["execution_k"] == execute
    assert metrics["startups"] == 1
    episode = json.loads((tmp_path / "episode-seed71.json").read_text())
    from action_bridge.eval.revision_pusht_visualization import _prepare
    _prepare(episode, cfg, 0, len(episode["plan_traces"]))
    for index, trace in enumerate(episode["plan_traces"]):
        assert trace["elapsed_count"] == (execute if index else 0)
        assert trace["execution_k"] == execute
        assert trace["startup"] == (index == 0)
        assert trace["overlap_length"] == (4 - execute if index else 0)
        if index:
            assert trace["aligned_old_raw"] == episode["plan_traces"][index - 1]["final_raw"][execute:]
            assert len(trace["completed_raw"]) == 4
    assert metrics["gap_samples"] == 0 if execute == 4 else metrics["gap_samples"] > 0


def dense_case():
    from action_bridge.configs.sb_pusht import get_k_curriculum_config
    from action_bridge.plan_revision.completion import LearnedCompletion
    from action_bridge.plan_revision.training import completion_config
    from test_revision_plots import replay_examples
    _, records, metadata = replay_examples()
    records = {key: value[:1].expand(17, *value.shape[1:]).clone() for key, value in records.items()}
    records.update(time_index=torch.arange(17), episode_id=torch.zeros(17, dtype=torch.long),
                   future_actions=torch.zeros(17, 16, 2), valid_mask=torch.ones(17, 16, dtype=torch.bool),
                   startup_actions=torch.ones(17, 16, 2))
    cfg = get_k_curriculum_config("fm_paired") | dict(
        channels=[8, 16], history_dim=16, hidden_dim=16, time_dim=8,
        num_inference_steps=2, reference_hidden_dim=8, updates=8, phase_updates=1)
    reference_config = completion_config(cfg)
    reference = LearnedCompletion(**reference_config)
    dependencies = {"completion_config": reference_config, "completion_state": reference.state_dict(),
                    "innovation_variance": torch.ones(2)}
    return cfg, records, metadata, dependencies


def test_training_previews_replay_dense_windows_on_each_stage_grid(tmp_path, monkeypatch):
    from action_bridge.eval import revision_pusht_plots as plots
    from test_revision_plots import capture_plots
    cfg, records, metadata, dependencies = dense_case()
    model = build_policy(cfg)
    seen = capture_plots(monkeypatch)
    plot = plots.make_action_chunk_plotter(records, cfg, metadata, tmp_path, "cpu",
                                           count=1, dependencies=dependencies)
    for step, k in zip((1, 3, 5, 7), (1, 2, 4, 8)):
        plot(model, step)
        assert len(seen[-1]["retained"]) == 16 - k
        assert len(seen[-1]["completed"]) == 16
        assert f"K={k}" in seen[-1]["title"]
        assert f"t={2*k}" in seen[-1]["title"]


def test_offline_gap_diagnostic_uses_deployment_k_not_early_training_k():
    from action_bridge.eval.revision_pusht_gaps import diagnose_offline_sources
    cfg, records, metadata, dependencies = dense_case()
    cfg["active_k"] = 1
    payload = {"metadata": metadata, "records": {"val": records},
               "data_spec": {key: cfg[key] for key in ("horizon", "execute", "obs_history", "action_history")}}
    diagnostic = diagnose_offline_sources(build_policy(cfg), cfg, metadata, dependencies, payload,
                                         "cpu", episodes=1, replans=3)
    assert [row["robot_step"] for row in diagnostic["records"]["self"]] == [8, 16]
    assert diagnostic["selection"]["execute"] == 8
    assert cfg["active_k"] == 1 and cfg["k_values_by_block"] == [1, 2, 4, 8]


def test_common_source_probe_uses_actual_deployment_k_for_all_conditioned_revisers(tmp_path):
    from action_bridge.plan_revision import reporting
    cfg, records, metadata, dependencies = dense_case()
    configs = {method: cfg | {"method": method, "active_k": 1} for method in reporting.REVISERS}
    policies = {method: build_policy(config) for method, config in configs.items()}
    result = reporting.common_source_probe(policies, {"validation": records, "test": records},
                                           configs, dependencies, "cpu", tmp_path, count=2)
    saved = torch.load(tmp_path / "common_sources.pt", weights_only=False)
    assert saved["pools"]["test"]["time_index"].tolist() == [8, 16] * 4
    assert saved["pools"]["test"]["execution_k"].tolist() == [8] * 8
    assert result["protocol"] == "self_source_k_curriculum_v1"
