"""Small, local experiment artifacts; no simulator installation is involved."""

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run(method):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "runs" / f"{method}-{stamp}"


def provenance(dataset):
    repo = ROOT.parent.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    # Hash the actual working sources too: an uncommitted experiment is not
    # faithfully identified by its parent repository's commit alone.
    sources = list(ROOT.glob("*.py")) + list((ROOT / "configs").glob("*.json"))
    sources += [repo / name for name in (
        "action_bridge/models/action_bridge_policy.py",
        "action_bridge/models/references.py",
        "action_bridge/models/diffusion_policy.py",
        "action_bridge/models/encoders.py",
        "action_bridge/data/action_coordinates.py",
        "action_bridge/training/losses.py",
    )]
    return {
        "policy_commit": commit,
        "sources_sha256": {str(p.relative_to(repo)): sha256(p) for p in sources},
        "lock_sha256": sha256(repo / "uv.lock"),
        "dataset_manifest_sha256": sha256(Path(dataset) / "manifest.json"),
        "dataset_episodes_sha256": sha256(Path(dataset) / "episodes.npz"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "diffusers")},
        "observation_profile": "surface-contact-state-v1",
        "action_profile": "absolute-cartesian-target-2d-v1",
        "simulator": "local deterministic compliant-contact integrator",
    }
