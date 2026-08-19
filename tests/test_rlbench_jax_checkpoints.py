from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from action_bridge.jax.training.checkpoints import load_checkpoint, save_checkpoint


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        step=np.asarray(7, dtype=np.int32),
        params={"weight": np.asarray([1.0, 2.0], dtype=np.float32)},
        opt_state={"count": np.asarray(3, dtype=np.int32)},
        rng=np.asarray([0, 1], dtype=np.uint32),
    )


def test_checkpoint_copies_online_evaluation_metadata_to_top_level(tmp_path):
    online_evaluation = {
        "schema_name": "action_bridge.rlbench_online",
        "schema_version": 1,
        "task_to_id": {"reach_target": 0},
    }

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        state=_state(),
        config={"seed": 11, "online_evaluation": online_evaluation},
        best_val_loss=0.25,
        wandb_run_id="run-1",
    )

    payload = load_checkpoint(path)
    assert payload["online_evaluation"] == online_evaluation
    assert payload["online_evaluation"] == payload["config"]["online_evaluation"]


def test_checkpoint_without_online_evaluation_keeps_legacy_payload_shape(tmp_path):
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        state=_state(),
        config={"seed": 11},
        best_val_loss=0.25,
        wandb_run_id=None,
    )

    payload = load_checkpoint(path)
    assert "online_evaluation" not in payload
    assert payload["config"] == {"seed": 11}


def test_checkpoint_rejects_non_mapping_online_evaluation(tmp_path):
    path = tmp_path / "checkpoint.pt"

    with pytest.raises(
        TypeError,
        match="config.online_evaluation must serialize to a mapping",
    ):
        save_checkpoint(
            path,
            state=_state(),
            config={"online_evaluation": ["not", "a", "mapping"]},
            best_val_loss=0.25,
            wandb_run_id=None,
        )

    assert not path.exists()
