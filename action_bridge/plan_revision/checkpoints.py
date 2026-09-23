"""Self-contained experiment checkpoints, including exact phase/RNG resume."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def same_training_config(left, right):
    """Evaluation cadence can change on resume without changing the learner."""
    return ({key: value for key, value in left.items() if key != "validation_every"}
            == {key: value for key, value in right.items() if key != "validation_every"})


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_identity():
    """Include new/untracked source too: `git diff` alone would miss it."""
    checkout = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    source_files = {str(path.relative_to(checkout)): digest(path)
                    for path in sorted((checkout / "action_bridge").rglob("*.py"))}
    lock = checkout / "uv.lock"
    return {"policy_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_files_sha256": source_files,
            "lock_sha256": digest(lock) if lock.exists() else None}


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def frozen_copy(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


@torch.no_grad()
def update_ema(ema, model, decay=0.999):
    for averaged, current in zip(ema.parameters(), model.parameters(), strict=True):
        averaged.lerp_(current, 1 - decay)
    for averaged, current in zip(ema.buffers(), model.buffers(), strict=True):
        averaged.copy_(current)


def save(path, payload):
    """Atomic replacement of this experiment's own resumable checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": "plan_revision_v1", **payload, "rng": rng_state(),
                "runtime": runtime_identity()}, temporary)
    temporary.replace(path)


def load(path, device="cpu"):
    # Only load locally produced / explicitly trusted checkpoints. Optimizer and
    # Python/NumPy RNG states intentionally need more than weights_only=True.
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "plan_revision_v1":
        raise ValueError("Not a plan-revision checkpoint")
    return payload


def restore_policy(payload, device="cpu", *, encoder=None):
    """Rebuild inference without any dataset, simulator, or original file path."""
    from action_bridge.plan_revision.models import build_policy
    policy = build_policy(payload["config"], encoder=encoder).to(device)
    policy.load_state_dict(payload["ema"])
    return policy.eval()
