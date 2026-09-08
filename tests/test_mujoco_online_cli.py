from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from test_mujoco_online_adapter import RecordingBackend
from test_mujoco_online_metadata import make_metadata

from action_bridge.eval.mujoco_online import eval_mujoco_online
from action_bridge.eval.mujoco_online.adapter import ActionBridgeMujocoPolicyAdapter


def arguments(tmp_path):
    return [
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--trusted-checkpoint",
        "--run-dir",
        str(tmp_path / "run"),
    ]


def test_checkpoint_failure_never_launches_simulator(tmp_path, monkeypatch, capsys):
    def fail_load(*args, **kwargs):
        raise ValueError("invalid metadata")

    def fail_launch(*args, **kwargs):
        raise AssertionError("simulator must not launch")

    monkeypatch.setattr(eval_mujoco_online, "load_torch_policy_adapter", fail_load)
    monkeypatch.setattr(eval_mujoco_online, "EvaluationRunner", fail_launch)
    assert eval_mujoco_online.main(arguments(tmp_path)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["stage"] == "checkpoint_load"


def test_cli_uses_public_runner_and_writes_policy_metadata(
    tmp_path, monkeypatch, capsys
):
    metadata = make_metadata()
    backend = RecordingBackend()
    adapter = ActionBridgeMujocoPolicyAdapter(
        metadata=metadata, backend=backend, checkpoint_identifier="sha256:example"
    )
    closed = []
    commands = []

    class Environment:
        def reset(self):
            self.step_index = 0
            print("simulator startup")
            return np.zeros(23, dtype=np.float32), {}

        def step(self, action):
            commands.append(action.copy())
            self.step_index += 1
            done = self.step_index == 3
            return (
                np.full(23, self.step_index, dtype=np.float32),
                float(done),
                done,
                False,
                {"success": done},
            )

        def close(self):
            closed.append(True)

    original = adapter.integration
    adapter.integration = SimpleNamespace(
        spec=original.spec,
        capabilities=original.capabilities,
        action_history_padding=original.action_history_padding,
        validate_action=original.validate_action,
        project_action=original.project_action,
        create_environment=lambda **kwargs: Environment(),
        extract_observation=lambda env, state: {"state": state},
        success=lambda info: info["success"],
    )
    monkeypatch.setattr(
        eval_mujoco_online, "load_torch_policy_adapter", lambda *a, **kw: adapter
    )
    status = eval_mujoco_online.main(
        arguments(tmp_path) + ["--episodes", "2", "--quiet", "--json"]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["successful_episodes"] == 2
    assert len(closed) == 2
    assert len(commands) == 6
    assert len(backend.batches) == 4
    assert "simulator startup" in (tmp_path / "run.log").read_text()
    record = json.loads((tmp_path / "run" / "policy_metadata.json").read_text())
    assert record["actions_per_plan"] == 2
    assert record["checkpoint_identifier"] == "sha256:example"
