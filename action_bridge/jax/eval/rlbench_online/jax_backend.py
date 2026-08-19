"""Lazy JAX/checkpoint backend owned by the Action Bridge integration."""

from __future__ import annotations

import hashlib
import os
import pickle
import stat
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from action_bridge.jax.eval.rlbench_online.adapter import (
    ActionBridgeRLBenchPolicyAdapter,
)
from action_bridge.jax.eval.rlbench_online.metadata import (
    OnlineEvaluationMetadata,
    OnlineMetadataError,
    resolve_online_metadata,
)


def _load_checkpoint_snapshot(path: Path) -> tuple[Mapping[str, object], str]:
    """Unpickle and identify one immutable in-memory snapshot of a trusted file."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError("Action Bridge checkpoint must be a regular file.")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    # Existing checkpoints are pickle payloads. Hashing and unpickling the same
    # byte snapshot prevents path replacement from corrupting recorded identity.
    checkpoint = pickle.loads(payload)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Action Bridge checkpoint must contain a mapping.")
    identifier = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return checkpoint, identifier


def _read_config_value(config: Any, *names: str) -> Any:
    current = config
    for name in names:
        if isinstance(current, Mapping):
            if name not in current:
                raise OnlineMetadataError(
                    f"Checkpoint config is missing {'.'.join(names)}."
                )
            current = current[name]
        else:
            try:
                current = getattr(current, name)
            except AttributeError as exc:
                raise OnlineMetadataError(
                    f"Checkpoint config is missing {'.'.join(names)}."
                ) from exc
    return current


def validate_checkpoint_config(
    config: Any,
    metadata: OnlineEvaluationMetadata,
) -> None:
    """Reject online metadata that disagrees with trained model geometry."""

    expected = {
        ("policy_type",): metadata.policy_type,
        ("data", "point_count"): metadata.point_count,
        ("data", "obs_history"): metadata.observation_history,
        ("data", "obs_stride"): metadata.observation_stride,
        ("data", "action_history"): metadata.action_history,
        ("data", "action_stride"): metadata.action_stride,
        ("data", "action_offset"): metadata.action_offset,
        ("data", "chunk_horizon"): metadata.action_horizon,
        ("data", "action_representation"): metadata.action_representation,
    }
    for path, required in expected.items():
        actual = _read_config_value(config, *path)
        if actual != required:
            raise OnlineMetadataError(
                f"Online metadata {'.'.join(path)}={required!r} disagrees with "
                f"checkpoint config value {actual!r}."
            )

    # New configs may carry layouts directly. Older checkpoints can supply
    # them through the strict explicit metadata fallback.
    for field, required in (
        ("state_layout", metadata.state_layout),
        ("action_layout", metadata.action_layout),
    ):
        try:
            actual = _read_config_value(config, "data", field)
        except OnlineMetadataError:
            continue
        if (
            isinstance(actual, (str, bytes))
            or not isinstance(actual, Sequence)
            or tuple(actual) != required
        ):
            raise OnlineMetadataError(
                f"Online metadata data.{field}={list(required)!r} disagrees with "
                f"checkpoint config value {actual!r}."
            )

    # A modality affected parameter shapes only when the dataset emitted it
    # and the encoder consumed it.
    for name, required in (
        ("rgb", metadata.include_rgb),
        ("mask_id", metadata.include_mask_id),
    ):
        data_value = _read_config_value(config, "data", f"include_{name}")
        encoder_value = _read_config_value(config, "encoder", f"use_{name}")
        if not isinstance(data_value, bool) or not isinstance(encoder_value, bool):
            raise OnlineMetadataError(
                f"Checkpoint modality flags for {name} must be boolean."
            )
        effective_value = data_value and encoder_value
        if effective_value != required:
            raise OnlineMetadataError(
                f"Online metadata include_{name}={required!r} disagrees with "
                "the effective checkpoint input modality "
                f"(data.include_{name}={data_value!r}, "
                f"encoder.use_{name}={encoder_value!r})."
            )

    for path, required in (
        (("encoder", "max_obs_history"), metadata.observation_history),
        (("encoder", "max_action_history"), metadata.action_history),
    ):
        actual = _read_config_value(config, *path)
        if isinstance(actual, bool) or not isinstance(actual, Integral):
            raise OnlineMetadataError(
                f"Checkpoint config {'.'.join(path)} must be an integer."
            )
        if int(actual) < required:
            raise OnlineMetadataError(
                f"Online history length {required} exceeds checkpoint capacity "
                f"{'.'.join(path)}={actual}."
            )


class JaxCheckpointBackend:
    """Compiled prior/direct inference with explicit per-episode PRNG state."""

    def __init__(
        self,
        *,
        jax_module: Any,
        params: Any,
        apply_inference: Any,
    ) -> None:
        self._jax = jax_module
        self._params = params
        self._apply_inference = apply_inference
        self._rng: Any | None = None

    def reset(self, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("Action Bridge policy seed must be an integer.")
        normalized = int(seed)
        if not 0 <= normalized <= 2**32 - 1:
            raise ValueError("Action Bridge policy seed must fit uint32.")
        self._rng = self._jax.random.PRNGKey(normalized)

    def predict(
        self,
        batch: Mapping[str, NDArray[np.generic]],
    ) -> tuple[NDArray[np.float32], Mapping[str, object]]:
        if self._rng is None:
            raise RuntimeError("JAX inference backend must be reset before prediction.")
        self._rng, inference_rng = self._jax.random.split(self._rng)
        device_batch = self._jax.tree_util.tree_map(
            lambda value: self._jax.device_put(np.asarray(value)),
            dict(batch),
        )
        actions, diagnostics = self._apply_inference(
            self._params,
            device_batch,
            inference_rng,
        )
        host_actions = np.asarray(self._jax.device_get(actions), dtype=np.float32)
        host_diagnostics = self._jax.device_get(diagnostics)
        return host_actions, {
            str(key): float(np.asarray(value))
            for key, value in host_diagnostics.items()
        }


def load_jax_policy_adapter(
    checkpoint_path: str | Path,
    *,
    online_metadata_path: str | Path | None = None,
    trusted_checkpoint: bool = False,
    deterministic_latent: bool | None = None,
) -> ActionBridgeRLBenchPolicyAdapter:
    """Load a trusted pickle checkpoint without importing JAX at module import time."""

    if not trusted_checkpoint:
        raise ValueError(
            "Action Bridge checkpoints are pickle files and may execute code while loading; "
            "set trusted_checkpoint=True only for a trusted local checkpoint."
        )
    if deterministic_latent is not None and not isinstance(deterministic_latent, bool):
        raise TypeError("deterministic_latent override must be boolean or None.")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is not a regular file: {path}")

    try:
        import jax
        import jax.numpy as jnp
        from ml_collections import ConfigDict

        from action_bridge.jax.models.config import policy_config_from_config
        from action_bridge.jax.models.rlbench_policy import (
            DirectChunkBCPolicy,
            RLBenchActionBridgePolicy,
        )
    except ImportError as exc:
        raise RuntimeError(
            "JAX online evaluation requires Action Bridge with its jax-cpu or "
            "jax-cu13 extra plus phi-rlbench."
        ) from exc

    checkpoint, checkpoint_identifier = _load_checkpoint_snapshot(path)
    if "params" not in checkpoint or "config" not in checkpoint:
        raise KeyError("Action Bridge checkpoint requires params and config.")
    metadata = resolve_online_metadata(
        checkpoint,
        explicit_path=online_metadata_path,
    )
    if deterministic_latent is not None:
        metadata = replace(metadata, deterministic_latent=deterministic_latent)
    config = ConfigDict(checkpoint["config"])
    validate_checkpoint_config(config, metadata)
    policy_config = policy_config_from_config(config)
    common = {
        "cfg": policy_config,
        "state_dim": len(metadata.state_layout),
        "action_dim": len(metadata.action_layout),
        "num_tasks": len(metadata.task_to_id),
        "num_task_variations": len(metadata.task_variation_to_id),
    }
    is_bridge = metadata.policy_type == "action_bridge"
    model = (
        RLBenchActionBridgePolicy(**common)
        if is_bridge
        else DirectChunkBCPolicy(**common)
    )
    resolved_deterministic_latent = metadata.deterministic_latent

    def inference(params: Any, batch: Any, latent_rng: Any) -> tuple[Any, Any]:
        if is_bridge:
            output = model.apply(
                {"params": params},
                batch,
                train=False,
                use_posterior=False,
                deterministic_latent=resolved_deterministic_latent,
                rngs={"latent": latent_rng},
            )
            latent_l2 = jnp.mean(jnp.square(output["latent"]))
        else:
            output = model.apply({"params": params}, batch, train=False)
            latent_l2 = jnp.asarray(0.0, dtype=jnp.float32)
        actions = output["actions"]
        diagnostics = {
            "action_min": jnp.min(actions),
            "action_max": jnp.max(actions),
            "mean_gripper": jnp.mean(actions[..., 7]),
            "latent_l2": latent_l2,
        }
        return actions, diagnostics

    backend = JaxCheckpointBackend(
        jax_module=jax,
        params=jax.device_put(checkpoint["params"]),
        apply_inference=jax.jit(inference),
    )
    return ActionBridgeRLBenchPolicyAdapter(
        metadata=metadata,
        backend=backend,
        checkpoint_identifier=checkpoint_identifier,
    )


__all__ = [
    "JaxCheckpointBackend",
    "load_jax_policy_adapter",
    "validate_checkpoint_config",
]
