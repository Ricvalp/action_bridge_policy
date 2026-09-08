"""Small source record for a MuJoCo training run."""

from __future__ import annotations

import hashlib
import subprocess
from importlib import metadata
from pathlib import Path


def _git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_record(directory: Path) -> dict[str, object]:
    git_root = _git(directory, "rev-parse", "--show-toplevel")
    root = Path(git_root) if git_root is not None else directory
    status = _git(root, "status", "--porcelain")
    lock = root / "uv.lock"
    return {
        "root": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty": status != "" if status is not None else None,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.is_file()
        else None,
    }


def training_provenance() -> dict[str, object]:
    """Record the policy checkout and the backend Python actually imports."""

    import phi_mujoco

    policy_root = Path(__file__).resolve().parents[2]
    backend_directory = Path(phi_mujoco.__file__).resolve().parent
    return {
        "action_bridge": _source_record(policy_root),
        "phi_mujoco": {
            **_source_record(backend_directory),
            "version": metadata.version("phi-mujoco"),
        },
    }
