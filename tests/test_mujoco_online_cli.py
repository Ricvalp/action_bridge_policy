from __future__ import annotations

import json

from action_bridge.eval.mujoco_online import eval_mujoco_online


def _arguments(tmp_path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--trusted-checkpoint",
        "--run-dir",
        str(tmp_path / "run"),
    ]


def test_checkpoint_validation_failure_never_launches_native_runtime(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    launched = False

    def fail_load(*args, **kwargs):
        del args, kwargs
        raise ValueError("invalid metadata")

    def record_launch(*args, **kwargs):
        del args, kwargs
        nonlocal launched
        launched = True
        return 0

    monkeypatch.setattr(eval_mujoco_online, "load_torch_policy_adapter", fail_load)
    monkeypatch.setattr(eval_mujoco_online, "_run_native", record_launch)

    assert eval_mujoco_online.main(_arguments(tmp_path)) == 2
    assert launched is False
    error = json.loads(capsys.readouterr().err)
    assert error["stage"] == "checkpoint_load"
    assert error["message"] == "invalid metadata"
