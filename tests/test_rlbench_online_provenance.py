from __future__ import annotations

import hashlib
from collections.abc import Sequence

from phi_rlbench.provenance import CommandResult

from action_bridge.jax.eval.rlbench_online.policy_provenance import (
    collect_policy_source_identity,
)


def test_policy_source_identity_records_commit_version_and_lock_without_path(
    tmp_path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "action-bridge-policy"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    lock_bytes = b"version = 1\n"
    (tmp_path / "uv.lock").write_bytes(lock_bytes)

    def runner(command: Sequence[str], timeout_s: float) -> CommandResult:
        assert timeout_s == 5.0
        if "rev-parse" in command:
            return CommandResult(0, "A" * 40 + "\n")
        assert "status" in command
        return CommandResult(0, " M action_bridge/example.py\n")

    identity = collect_policy_source_identity(tmp_path, command_runner=runner)

    assert identity["schema_name"] == "phi.action_bridge_policy.source"
    assert identity["package_name"] == "action-bridge-policy"
    assert identity["package_version"] == "0.1.0"
    assert identity["git"] == {
        "available": True,
        "commit": "a" * 40,
        "dirty": True,
    }
    assert identity["uv_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert str(tmp_path) not in repr(identity)


def test_policy_source_identity_refuses_symlink_lock(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "action-bridge-policy"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    target = tmp_path / "actual.lock"
    target.write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "uv.lock").symlink_to(target)

    identity = collect_policy_source_identity(tmp_path)

    assert identity["git"] == {
        "available": False,
        "commit": None,
        "dirty": None,
    }
    assert identity["uv_lock_sha256"] is None
