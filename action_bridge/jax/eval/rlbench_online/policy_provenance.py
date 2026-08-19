"""Privacy-bounded source identity for policy-owned RLBench evaluation."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import tomllib
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from phi_rlbench.provenance import CommandResult, run_command

POLICY_SOURCE_SCHEMA = "phi.action_bridge_policy.source"
POLICY_SOURCE_SCHEMA_VERSION = 1
_PROJECT_NAME = "action-bridge-policy"
_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _project_name_and_version(root: Path) -> tuple[str | None, str | None]:
    path = root / "pyproject.toml"
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return None, None
    project = value.get("project")
    if not isinstance(project, dict):
        return None, None
    raw_name = project.get("name")
    raw_version = project.get("version")
    name = raw_name if isinstance(raw_name, str) and raw_name else None
    version = raw_version if isinstance(raw_version, str) and raw_version else None
    return name, version


def _find_policy_root() -> Path | None:
    """Find an installed/source Action Bridge checkout without importing it."""

    try:
        spec = importlib.util.find_spec("action_bridge")
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    locations: list[Path] = []
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        locations.append(Path(spec.origin).resolve(strict=False).parent)
    if spec.submodule_search_locations is not None:
        locations.extend(
            Path(item).resolve(strict=False) for item in spec.submodule_search_locations
        )
    for location in locations:
        for candidate in (location, *location.parents):
            project_name, _ = _project_name_and_version(candidate)
            if project_name == _PROJECT_NAME:
                return candidate
    return None


def _sha256_regular_file(path: Path) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git_identity(
    root: Path | None,
    *,
    command_runner: Callable[[Sequence[str], float], CommandResult],
) -> dict[str, bool | str | None]:
    unavailable: dict[str, bool | str | None] = {
        "available": False,
        "commit": None,
        "dirty": None,
    }
    if root is None or not (root / ".git").exists():
        return unavailable
    revision = command_runner(
        ("git", "-C", str(root), "rev-parse", "--verify", "HEAD"),
        5.0,
    )
    commit = revision.stdout.strip()
    if revision.returncode != 0 or _GIT_COMMIT.fullmatch(commit) is None:
        return unavailable
    status = command_runner(
        ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"),
        5.0,
    )
    return {
        "available": True,
        "commit": commit.lower(),
        "dirty": None if status.returncode != 0 else bool(status.stdout.strip()),
    }


def collect_policy_source_identity(
    source_root: str | Path | None = None,
    *,
    command_runner: Callable[[Sequence[str], float], CommandResult] = run_command,
) -> dict[str, Any]:
    """Collect Action Bridge version, revision, and lock digest without paths."""

    root = (
        _find_policy_root()
        if source_root is None
        else Path(source_root).expanduser().resolve(strict=False)
    )
    project_name: str | None = None
    project_version: str | None = None
    if root is not None:
        project_name, project_version = _project_name_and_version(root)
    if project_version is None:
        try:
            project_version = metadata.version(_PROJECT_NAME)
        except metadata.PackageNotFoundError:
            project_version = None
    return {
        "schema_name": POLICY_SOURCE_SCHEMA,
        "schema_version": POLICY_SOURCE_SCHEMA_VERSION,
        "package_name": project_name or _PROJECT_NAME,
        "package_version": project_version,
        "source_root_detected": root is not None,
        "git": _git_identity(root, command_runner=command_runner),
        "uv_lock_sha256": None
        if root is None
        else _sha256_regular_file(root / "uv.lock"),
    }


__all__ = [
    "POLICY_SOURCE_SCHEMA",
    "POLICY_SOURCE_SCHEMA_VERSION",
    "collect_policy_source_identity",
]
