"""Atomic pickle checkpoints for JAX training state."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import jax

from action_bridge.config import to_plain_dict


def save_checkpoint(
    path: str | Path,
    *,
    state: Any,
    config: Any,
    best_val_loss: float,
    wandb_run_id: str | None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(jax.device_get(state.step)),
        "params": jax.device_get(state.params),
        "opt_state": jax.device_get(state.opt_state),
        "rng": jax.device_get(state.rng),
        "config": to_plain_dict(config),
        "best_val_loss": float(best_val_loss),
        "wandb_run_id": wandb_run_id,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)

