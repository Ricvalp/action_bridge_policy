from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from action_bridge.jax.eval.rlbench_online import (
    CANONICAL_RLBENCH_ACTION_LAYOUT,
    CANONICAL_RLBENCH_STATE_LAYOUT,
    OnlineEvaluationMetadata,
)
from action_bridge.jax.eval.rlbench_online import eval_rlbench_online as online_cli


def _metadata(**overrides: object) -> OnlineEvaluationMetadata:
    values: dict[str, object] = {
        "task_to_id": {"reach_target": 0},
        "task_variation_to_id": {"reach_target:0": 0},
        "point_count": 4,
        "observation_history": 2,
        "observation_stride": 1,
        "action_history": 2,
        "action_stride": 1,
        "action_offset": 1,
        "action_horizon": 3,
        "action_representation": "absolute",
        "state_layout": list(CANONICAL_RLBENCH_STATE_LAYOUT),
        "action_layout": list(CANONICAL_RLBENCH_ACTION_LAYOUT),
        "training_cache_identity": {
            "available": False,
            "manifest_sha256": None,
            "schema_name": None,
            "schema_version": None,
        },
        "include_rgb": True,
        "include_mask_id": False,
        "policy_type": "direct_chunk_bc",
    }
    values.update(overrides)
    return OnlineEvaluationMetadata.from_mapping(values)


def _adapter(metadata: OnlineEvaluationMetadata | None = None) -> object:
    return SimpleNamespace(
        metadata=_metadata() if metadata is None else metadata,
        checkpoint_identifier="sha256:" + "a" * 64,
    )


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "checkpoint.pkl"),
        "--trusted-checkpoint",
        "--task",
        "reach_target",
        "--variation",
        "0",
        "--episodes",
        "2",
        "--actions-per-plan",
        "1",
        "--reject-out-of-bounds-actions",
        "--headless",
        "--record-video",
        "--run-dir",
        str(tmp_path / "run"),
    ]


def _install_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
) -> None:
    class FakeRuntime:
        def __init__(self, profile: object) -> None:
            state["runtime"] = self
            state["runtime_profile"] = profile
            self.launch_calls = 0
            self.close_calls = 0

        def launch(self) -> None:
            self.launch_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    class FakeArtifacts:
        @classmethod
        def create(
            cls,
            run_directory: Path,
            *,
            resolved_config: dict[str, object],
            provenance: dict[str, object],
        ) -> object:
            state["run_directory"] = run_directory
            state["resolved_config"] = resolved_config
            state["provenance"] = provenance
            return cls()

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            state["runner_kwargs"] = kwargs

        def run(self, *, artifacts: object) -> object:
            del artifacts
            state["runtime"].close()
            return SimpleNamespace(
                summary=SimpleNamespace(
                    attempted_episodes=2,
                    successful_episodes=1,
                    exception_records=0,
                    runtime_cleanup_succeeded=True,
                ),
                episodes=(SimpleNamespace(video_path="videos/episode-000000.mp4"),),
            )

    monkeypatch.setattr(online_cli, "RLBenchRuntime", FakeRuntime)
    monkeypatch.setattr(online_cli, "EvaluationArtifacts", FakeArtifacts)
    monkeypatch.setattr(online_cli, "EvaluationRunner", FakeRunner)
    monkeypatch.setattr(
        online_cli,
        "collect_provenance",
        lambda **kwargs: {"schema_name": "phi.robotics.provenance"},
    )
    monkeypatch.setattr(
        online_cli,
        "collect_policy_source_identity",
        lambda: {"schema_name": "phi.action_bridge_policy.source"},
    )


def test_online_cli_composes_policy_runtime_and_default_latent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state: dict[str, Any] = {}
    _install_success_fakes(monkeypatch, state)

    def fake_loader(*args: object, **kwargs: object) -> object:
        del args
        state["loader_kwargs"] = kwargs
        return _adapter()

    monkeypatch.setattr(online_cli, "load_jax_policy_adapter", fake_loader)

    assert online_cli.main(_arguments(tmp_path)) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["recorded_videos"] == 1
    assert state["loader_kwargs"]["deterministic_latent"] is None
    config = state["resolved_config"]
    assert config["latent_sampling"] == {
        "deterministic": True,
        "source": "checkpoint_or_metadata_default",
    }
    assert config["online_evaluation"] == _metadata().to_json_dict()
    assert config["preprocessing"]["num_points"] == 4
    assert config["runtime"]["headless"] is True
    assert state["runtime"].launch_calls == 1
    assert state["runtime"].close_calls == 1


def test_online_cli_stochastic_latent_override_is_passed_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state: dict[str, Any] = {}
    _install_success_fakes(monkeypatch, state)
    stochastic = _metadata(
        policy_type="action_bridge",
        deterministic_latent=False,
    )

    def fake_loader(*args: object, **kwargs: object) -> object:
        del args
        state["loader_kwargs"] = kwargs
        return _adapter(stochastic)

    monkeypatch.setattr(online_cli, "load_jax_policy_adapter", fake_loader)

    assert online_cli.main([*_arguments(tmp_path), "--stochastic-latent"]) == 0
    assert state["loader_kwargs"]["deterministic_latent"] is False
    assert state["resolved_config"]["latent_sampling"] == {
        "deterministic": False,
        "source": "command_line_override",
    }


def test_hardened_preprocessing_is_an_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state: dict[str, Any] = {}
    _install_success_fakes(monkeypatch, state)
    monkeypatch.setattr(
        online_cli,
        "load_jax_policy_adapter",
        lambda *args, **kwargs: _adapter(),
    )

    assert (
        online_cli.main([*_arguments(tmp_path), "--preprocessing-profile", "hardened"])
        == 0
    )
    assert (
        state["resolved_config"]["preprocessing"]["profile_name"]
        == "rlbench_hardened_evaluation_v1"
    )
    assert (
        state["resolved_config"]["preprocessing_rng_recipe"]
        == "phi_episode_seed_sequence_v1"
    )


def test_legacy_preprocessing_stream_reuses_seed_plus_eleven_rng() -> None:
    values: list[float] = []

    class FakeDelegate:
        def process(self, observation: object, *, rng: np.random.Generator) -> object:
            del observation
            values.append(float(rng.random()))
            return object()

    stream = online_cli._LegacyContinuousPreprocessingStream(  # type: ignore[arg-type]
        FakeDelegate(),
        base_seed=7,
    )
    ignored = np.random.default_rng(999)
    stream.process(object(), rng=ignored)
    stream.process(object(), rng=ignored)

    np.testing.assert_array_equal(values, np.random.default_rng(18).random(2))


def test_fail_fast_reraises_after_runner_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state: dict[str, Any] = {}

    class FakeRuntime:
        def __init__(self, profile: object) -> None:
            del profile
            self.close_calls = 0
            state["runtime"] = self

        def launch(self) -> None:
            pass

        def close(self) -> None:
            self.close_calls += 1

    class FakeArtifacts:
        @classmethod
        def create(cls, run_directory: Path, **kwargs: object) -> object:
            del run_directory, kwargs
            return cls()

    class FailingRunner:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self, *, artifacts: object) -> object:
            del artifacts
            state["runtime"].close()
            raise RuntimeError("fail-fast online episode")

    monkeypatch.setattr(
        online_cli,
        "load_jax_policy_adapter",
        lambda *args, **kwargs: _adapter(),
    )
    monkeypatch.setattr(online_cli, "RLBenchRuntime", FakeRuntime)
    monkeypatch.setattr(online_cli, "EvaluationArtifacts", FakeArtifacts)
    monkeypatch.setattr(online_cli, "EvaluationRunner", FailingRunner)
    monkeypatch.setattr(online_cli, "collect_provenance", lambda **kwargs: {})
    monkeypatch.setattr(online_cli, "collect_policy_source_identity", dict)

    with pytest.raises(RuntimeError, match="fail-fast online episode"):
        online_cli.main([*_arguments(tmp_path), "--fail-fast"])
    assert state["runtime"].close_calls == 1


def test_launch_failure_retries_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instances: list[object] = []

    class FailingRuntime:
        def __init__(self, profile: object) -> None:
            del profile
            self.close_calls = 0
            instances.append(self)

        def launch(self) -> None:
            raise RuntimeError("native launch unavailable")

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("transient cleanup failure")

    monkeypatch.setattr(
        online_cli,
        "load_jax_policy_adapter",
        lambda *args, **kwargs: _adapter(),
    )
    monkeypatch.setattr(online_cli, "RLBenchRuntime", FailingRuntime)

    assert online_cli.main(_arguments(tmp_path)) == 1
    assert instances[0].close_calls == 2
    errors = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [item["stage"] for item in errors] == [
        "launch",
        "launch_cleanup_attempt_1",
    ]


def test_available_cache_identity_is_preserved_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(
        training_cache_identity={
            "available": True,
            "manifest_sha256": "a" * 64,
            "schema_name": "action_bridge.rlbench_dense",
            "schema_version": 1,
            "cache_bundle_sha256": "b" * 64,
            "preprocessing_sidecar_sha256": "c" * 64,
        }
    )
    monkeypatch.setattr(online_cli, "collect_provenance", lambda **kwargs: {})

    record = online_cli._collect_run_provenance(metadata, preprocessing_config={})

    assert record["cache"]["manifest"] == {
        "available": True,
        "sha256": "a" * 64,
        "schema_name": "action_bridge.rlbench_dense",
        "schema_version": 1,
        "bundle_sha256": "b" * 64,
        "preprocessing_sidecar_sha256": "c" * 64,
    }


def test_online_cli_requires_explicit_pickle_trust(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.remove("--trusted-checkpoint")
    with pytest.raises(SystemExit, match="2"):
        online_cli.main(arguments)


def test_online_cli_requires_explicit_action_bounds_policy(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.remove("--reject-out-of-bounds-actions")
    with pytest.raises(SystemExit, match="2"):
        online_cli.main(arguments)


def test_online_cli_reports_checkpoint_failure_without_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        online_cli,
        "load_jax_policy_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("JAX unavailable")),
    )

    assert online_cli.main(_arguments(tmp_path)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["stage"] == "checkpoint_load"
    assert error["message"] == "JAX unavailable"
