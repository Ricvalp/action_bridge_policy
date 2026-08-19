import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

from action_bridge.config import load_config
from action_bridge.jax.training.train_rlbench import (
    _merge_device_batch,
    _shard_batch,
    _training_devices,
)


def test_shard_and_merge_batch_preserve_examples():
    batch = {
        "features": np.arange(48, dtype=np.float32).reshape(8, 2, 3),
        "ids": np.arange(8, dtype=np.int32),
    }
    sharded = _shard_batch(batch, (object(), object()))

    assert sharded["features"].shape == (2, 4, 2, 3)
    assert sharded["ids"].shape == (2, 4)
    merged = _merge_device_batch(sharded)
    np.testing.assert_array_equal(merged["features"], batch["features"])
    np.testing.assert_array_equal(merged["ids"], batch["ids"])


def test_shard_batch_rejects_non_divisible_global_batch():
    with pytest.raises(ValueError, match="Cannot shard batch leaf"):
        _shard_batch({"features": np.zeros((5, 3))}, (object(), object()))


def test_training_devices_validate_requested_count_and_global_batch(monkeypatch):
    fake_devices = (object(), object(), object(), object())
    monkeypatch.setattr(jax, "local_devices", lambda: list(fake_devices))
    config = load_config("rlbench_jax_contact_bridge")
    config.distributed.num_devices = 2
    config.optim.batch_size = 8
    assert _training_devices(config) == fake_devices[:2]

    config.optim.batch_size = 7
    with pytest.raises(ValueError, match="must be divisible"):
        _training_devices(config)
