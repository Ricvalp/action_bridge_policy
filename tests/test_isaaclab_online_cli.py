from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from action_bridge.eval.isaaclab_online import eval_isaaclab_online

_MANIFEST_VALUE = {
    "schema_name": "phi.isaaclab.episode_hdf5",
    "schema_version": 2,
    "task_name": "franka_cube_lift",
    "variation_id": 0,
    "observation_profile": "phi.isaaclab.franka_cube_lift.state.v2",
    "action_profile": "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v2",
    "episode_count": 3,
}
_MANIFEST_PAYLOAD = (
    json.dumps(_MANIFEST_VALUE, allow_nan=False, indent=2, sort_keys=True) + "\n"
).encode()
_MANIFEST_SHA256 = hashlib.sha256(_MANIFEST_PAYLOAD).hexdigest()


class _Metadata:
    def to_json_dict(self):
        return {
            "schema_name": "action_bridge.isaaclab_online",
            "schema_version": 2,
            "collection_identity": {
                "schema_name": "phi.isaaclab.episode_hdf5",
                "schema_version": 2,
                "manifest_sha256": _MANIFEST_SHA256,
            },
        }


def _manifest_path(tmp_path):
    path = tmp_path / "manifest.json"
    if not path.exists():
        path.write_bytes(_MANIFEST_PAYLOAD)
    return path


def _shared_identities(tmp_path):
    return (
        eval_isaaclab_online._cache_manifest_identity(
            _manifest_path(tmp_path), expected_sha256=_MANIFEST_SHA256
        ),
        eval_isaaclab_online._preprocessing_identity(_Metadata().to_json_dict()),
    )


def _arguments(tmp_path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--collection-manifest",
        str(_manifest_path(tmp_path)),
        "--trusted-checkpoint",
        "--device",
        "cpu",
        "--isaaclab-source-root",
        str(tmp_path),
        "--writable-root",
        str(tmp_path),
        "--run-dir",
        str(tmp_path / "run"),
    ]


def test_checkpoint_failure_never_launches_isaac_runtime(
    tmp_path, monkeypatch, capsys
) -> None:
    launched = False

    def fail_load(*args, **kwargs):
        del args, kwargs
        raise ValueError("invalid metadata")

    def record_launch(*args, **kwargs):
        del args, kwargs
        nonlocal launched
        launched = True
        return {}

    monkeypatch.setattr(eval_isaaclab_online, "load_torch_policy_adapter", fail_load)
    monkeypatch.setattr(eval_isaaclab_online, "_run_native", record_launch)

    assert eval_isaaclab_online.main(_arguments(tmp_path)) == 2
    assert launched is False
    error = json.loads(capsys.readouterr().err)
    assert error["stage"] == "checkpoint_load"
    assert error["message"] == "invalid metadata"


def test_collection_manifest_mismatch_never_launches_native_runtime(
    tmp_path, monkeypatch, capsys
) -> None:
    launched = False

    class Metadata:
        def to_json_dict(self):
            value = _Metadata().to_json_dict()
            value["collection_identity"]["manifest_sha256"] = "f" * 64
            return value

    adapter = SimpleNamespace(
        checkpoint_identifier="sha256:" + "a" * 64,
        metadata=Metadata(),
    )
    monkeypatch.setattr(
        eval_isaaclab_online,
        "load_torch_policy_adapter",
        lambda *_args, **_kwargs: adapter,
    )

    def launch(*_args, **_kwargs):
        nonlocal launched
        launched = True
        return {}

    monkeypatch.setattr(eval_isaaclab_online, "_run_native", launch)
    assert eval_isaaclab_online.main(_arguments(tmp_path)) == 2
    assert launched is False
    error = json.loads(capsys.readouterr().err)
    assert error["stage"] == "checkpoint_load"
    assert "manifest SHA-256 disagrees" in error["message"]


@pytest.mark.parametrize(
    ("profile_key", "replacement"),
    [
        ("observation_profile", "phi.isaaclab.franka_cube_lift.state.v1"),
        ("action_profile", "phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v1"),
    ],
)
def test_collection_manifest_profile_drift_is_rejected_before_native_launch(
    tmp_path, monkeypatch, profile_key, replacement
) -> None:
    manifest = dict(_MANIFEST_VALUE)
    manifest[profile_key] = replacement
    payload = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = tmp_path / "drifted-manifest.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=profile_key):
        eval_isaaclab_online._cache_manifest_identity(
            path, expected_sha256=hashlib.sha256(payload).hexdigest()
        )


def test_preflight_identifier_is_forwarded_to_native_runner(
    tmp_path, monkeypatch, capsys
) -> None:
    identifier = "sha256:" + "a" * 64

    class _Adapter:
        checkpoint_identifier = identifier
        metadata = _Metadata()

    monkeypatch.setattr(
        eval_isaaclab_online,
        "load_torch_policy_adapter",
        lambda *args, **kwargs: _Adapter(),
    )

    def run(
        _args,
        *,
        checkpoint_identifier,
        cache_manifest_identity,
        preprocessing_identity,
    ):
        assert checkpoint_identifier == identifier
        assert cache_manifest_identity["sha256"] == _MANIFEST_SHA256
        assert preprocessing_identity["algorithm"] == "sha256"
        return {
            "attempted_episodes": 100,
            "successful_episodes": 100,
            "checkpoint_identifier": identifier,
        }

    monkeypatch.setattr(eval_isaaclab_online, "_run_native", run)
    arguments = [*_arguments(tmp_path), "--json"]
    assert eval_isaaclab_online.main(arguments) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["checkpoint_identifier"] == identifier


def test_native_entrypoint_flushes_then_uses_forced_exit_after_runtime(
    tmp_path, monkeypatch, capsys
) -> None:
    from phi_isaaclab.sim import runtime

    identifier = "sha256:" + "d" * 64
    forced_statuses: list[int] = []

    class _Adapter:
        checkpoint_identifier = identifier
        metadata = _Metadata()

    class _ForcedExit(Exception):
        pass

    monkeypatch.setattr(
        eval_isaaclab_online,
        "load_torch_policy_adapter",
        lambda *args, **kwargs: _Adapter(),
    )
    monkeypatch.setattr(
        eval_isaaclab_online,
        "_run_native",
        lambda *_args, **_kwargs: {
            "attempted_episodes": 1,
            "successful_episodes": 1,
            "checkpoint_identifier": identifier,
        },
    )

    def force_exit(status: int) -> None:
        forced_statuses.append(status)
        raise _ForcedExit

    monkeypatch.setattr(runtime, "native_cli_process_exit_required", lambda: True)
    monkeypatch.setattr(runtime, "finalize_native_cli_process", force_exit)
    with pytest.raises(_ForcedExit):
        eval_isaaclab_online.native_main(
            [*_arguments(tmp_path), "--episodes", "1", "--json"]
        )

    assert forced_statuses == [0]
    assert json.loads(capsys.readouterr().out)["checkpoint_identifier"] == identifier


def test_native_entrypoint_forces_nonzero_exit_after_native_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    from phi_isaaclab.sim import runtime

    identifier = "sha256:" + "e" * 64
    forced_statuses: list[int] = []

    class _Adapter:
        checkpoint_identifier = identifier
        metadata = _Metadata()

    class _ForcedExit(Exception):
        pass

    monkeypatch.setattr(
        eval_isaaclab_online,
        "load_torch_policy_adapter",
        lambda *args, **kwargs: _Adapter(),
    )
    monkeypatch.setattr(
        eval_isaaclab_online,
        "_run_native",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native failed")),
    )

    def force_exit(status: int) -> None:
        forced_statuses.append(status)
        raise _ForcedExit

    monkeypatch.setattr(runtime, "native_cli_process_exit_required", lambda: True)
    monkeypatch.setattr(runtime, "finalize_native_cli_process", force_exit)
    with pytest.raises(_ForcedExit):
        eval_isaaclab_online.native_main(_arguments(tmp_path))

    assert forced_statuses == [4]
    error = json.loads(capsys.readouterr().err)
    assert error["stage"] == "native_evaluation"


def test_native_device_validation_requires_an_explicit_visible_cuda_index(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="explicit device index"):
        eval_isaaclab_online._validate_native_device("cuda")
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        eval_isaaclab_online._validate_native_device("cuda:1")
    eval_isaaclab_online._validate_native_device("cuda:0")


def test_preload_rejects_paths_outside_explicit_writable_root(tmp_path, capsys) -> None:
    writable = tmp_path / "writable"
    writable.mkdir()
    arguments = _arguments(writable)
    run_index = arguments.index("--run-dir")
    arguments[run_index + 1] = str(tmp_path / "outside" / "run")
    assert eval_isaaclab_online.main(arguments) == 2
    assert "--run-dir must be a child" in capsys.readouterr().err

    arguments = _arguments(writable)
    arguments.extend(["--portable-root", str(tmp_path / "outside-kit")])
    assert eval_isaaclab_online.main(arguments) == 2
    assert "--portable-root must be a child" in capsys.readouterr().err


def test_native_runner_persists_policy_identity_and_owns_cleanup(
    tmp_path, monkeypatch
) -> None:
    import torch
    from phi_isaaclab import evaluation
    from phi_isaaclab.sim import runtime, task

    identifier = "sha256:" + "b" * 64
    closed = {"environment": 0, "application": 0}
    captured = {}

    class App:
        def __init__(self):
            self.runtime_provenance = {
                "device": "cpu",
                "isaac_lab_git_head": "f" * 40,
                "torch_runtime": {"torch_version": "test"},
            }

        def close(self):
            closed["application"] += 1

    class Environment:
        unwrapped = SimpleNamespace(device="cpu")

        def close(self):
            closed["environment"] += 1

    class Adapter:
        checkpoint_identifier = identifier
        backend = SimpleNamespace(device=torch.device("cpu"))
        metadata = _Metadata()

    class Result:
        def as_dict(self):
            return {
                "attempted_episodes": 1,
                "successful_episodes": 1,
                "num_envs": 2,
                "checkpoint_identifier": identifier,
                "runtime_cleanup_succeeded": True,
            }

    class Runner:
        def __init__(self, environment, adapter, config, *, runtime_cleanup):
            captured["environment"] = environment
            captured["adapter"] = adapter
            captured["config"] = config
            captured["cleanup"] = runtime_cleanup

        def run(self):
            captured["cleanup"]()
            return Result()

    monkeypatch.setattr(runtime, "launch_native_app", lambda _config: App())
    monkeypatch.setattr(task, "build_lift_task_config", lambda **_kwargs: object())
    monkeypatch.setattr(
        task, "create_lift_environment", lambda *_args, **_kwargs: Environment()
    )
    monkeypatch.setattr(evaluation, "BatchedEvaluationRunner", Runner)
    monkeypatch.setattr(
        eval_isaaclab_online,
        "load_torch_policy_adapter",
        lambda *_args, **_kwargs: Adapter(),
    )

    cache_identity, preprocessing_identity = _shared_identities(tmp_path)
    result = eval_isaaclab_online._run_native(
        Namespace(
            portable_root=tmp_path / "portable",
            run_dir=tmp_path / "run",
            writable_root=tmp_path,
            isaaclab_source_root=tmp_path,
            device="cpu",
            headless=True,
            record_video=False,
            num_envs=2,
            max_steps=25,
            checkpoint=tmp_path / "policy.pt",
            collection_manifest=_manifest_path(tmp_path),
            online_metadata=None,
            episodes=1,
            seed=7,
            video_fps=50.0,
        ),
        checkpoint_identifier=identifier,
        cache_manifest_identity=cache_identity,
        preprocessing_identity=preprocessing_identity,
    )

    config = captured["config"]
    assert config.checkpoint_identifier == identifier
    assert config.policy_provenance["checkpoint_identifier"] == identifier
    assert config.policy_provenance["online_evaluation"]["schema_version"] == 2
    assert config.runtime_provenance["isaac_lab_git_head"] == "f" * 40
    assert config.cache_manifest_identity["sha256"] == _MANIFEST_SHA256
    assert config.preprocessing_identity["algorithm"] == "sha256"
    assert result["checkpoint_identifier"] == identifier
    assert result["num_envs"] == 2
    assert closed == {"environment": 1, "application": 1}


def test_native_setup_failure_still_closes_application(tmp_path, monkeypatch) -> None:
    from phi_isaaclab.sim import runtime, task

    closed = 0

    class App:
        def __init__(self):
            self.runtime_provenance = {
                "device": "cpu",
                "isaac_lab_git_head": "f" * 40,
                "torch_runtime": {"torch_version": "test"},
            }

        def close(self):
            nonlocal closed
            closed += 1

    monkeypatch.setattr(runtime, "launch_native_app", lambda _config: App())
    monkeypatch.setattr(task, "build_lift_task_config", lambda **_kwargs: object())
    monkeypatch.setattr(
        task,
        "create_lift_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("create failed")),
    )

    cache_identity, preprocessing_identity = _shared_identities(tmp_path)
    with pytest.raises(RuntimeError, match="create failed"):
        eval_isaaclab_online._run_native(
            Namespace(
                portable_root=tmp_path / "portable",
                run_dir=tmp_path / "run",
                writable_root=tmp_path,
                isaaclab_source_root=tmp_path,
                device="cpu",
                headless=True,
                record_video=False,
                num_envs=1,
                max_steps=25,
                checkpoint=tmp_path / "policy.pt",
                collection_manifest=_manifest_path(tmp_path),
                online_metadata=None,
                episodes=1,
                seed=0,
                video_fps=50.0,
            ),
            checkpoint_identifier="sha256:" + "c" * 64,
            cache_manifest_identity=cache_identity,
            preprocessing_identity=preprocessing_identity,
        )
    assert closed == 1
