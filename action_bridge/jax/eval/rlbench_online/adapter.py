"""Action Bridge policy adapter over the public phi-rlbench NumPy contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from phi_rlbench.evaluation import PolicyInput, PolicyOutput

from action_bridge.jax.eval.rlbench_online.metadata import OnlineEvaluationMetadata


class InferenceBackend(Protocol):
    """Framework-owned compiled inference boundary used by the thin adapter."""

    def reset(self, *, seed: int) -> None:
        """Reset episode-local framework RNG state."""

    def predict(
        self,
        batch: Mapping[str, NDArray[np.generic]],
    ) -> tuple[NDArray[np.float32], Mapping[str, object]]:
        """Return canonical absolute actions ``[1, H, 8]`` and diagnostics."""


def _batched(value: NDArray[np.generic]) -> NDArray[np.generic]:
    return np.ascontiguousarray(value[None])


def build_online_batch(
    observation: PolicyInput,
    metadata: OnlineEvaluationMetadata,
) -> dict[str, NDArray[np.generic]]:
    """Construct the exact inference-only batch accepted by current JAX policies."""

    history, points = observation.point_cloud_history.shape[:2]
    if history != metadata.observation_history:
        raise ValueError(
            f"PolicyInput history is {history}, checkpoint requires "
            f"{metadata.observation_history}."
        )
    if points != metadata.point_count:
        raise ValueError(
            f"PolicyInput point count is {points}, checkpoint requires {metadata.point_count}."
        )
    if observation.state_history.shape[-1] != len(metadata.state_layout):
        raise ValueError(
            "PolicyInput state width disagrees with the exact checkpoint state_layout."
        )
    try:
        expected_task_id = metadata.task_to_id[observation.task_name]
    except KeyError as exc:
        raise KeyError(
            f"Task {observation.task_name!r} is absent from checkpoint metadata."
        ) from exc
    variation_key = f"{observation.task_name}:{observation.variation_id}"
    try:
        task_variation_id = metadata.task_variation_to_id[variation_key]
    except KeyError as exc:
        raise KeyError(
            f"Task variation {variation_key!r} is absent from checkpoint metadata."
        ) from exc
    if observation.task_id is not None and observation.task_id != expected_task_id:
        raise ValueError(
            f"PolicyInput task_id={observation.task_id} disagrees with checkpoint "
            f"task_id={expected_task_id}."
        )
    if metadata.include_rgb and observation.rgb_history is None:
        raise ValueError("Checkpoint encoder requires rgb_history, but it is absent.")
    if metadata.include_mask_id and observation.mask_id_history is None:
        raise ValueError(
            "Checkpoint encoder requires mask_id_history, but it is absent."
        )

    # For the explicitly validated v1 geometry, cached `action[t]` and
    # `state[t]` are the same observed gripper pose/open vector.
    batch: dict[str, NDArray[np.generic]] = {
        "obs_hist": _batched(observation.state_history.astype(np.float32, copy=False)),
        "point_cloud_hist": _batched(
            observation.point_cloud_history.astype(np.float32, copy=False)
        ),
        "point_valid_hist": _batched(
            observation.point_valid_history.astype(np.bool_, copy=False)
        ),
        "act_hist": _batched(observation.state_history.astype(np.float32, copy=False)),
        "obs_history_mask": _batched(
            observation.observation_history_mask.astype(np.bool_, copy=False)
        ),
        "action_history_mask": _batched(
            observation.observation_history_mask.astype(np.bool_, copy=False)
        ),
        "action_is_absolute": np.asarray([True], dtype=np.bool_),
        "task_id": np.asarray([expected_task_id], dtype=np.int32),
        "task_variation_id": np.asarray([task_variation_id], dtype=np.int32),
        "variation_id": np.asarray([observation.variation_id], dtype=np.int32),
    }
    if metadata.include_rgb:
        assert observation.rgb_history is not None
        batch["rgb_hist"] = _batched(
            observation.rgb_history.astype(np.float32, copy=False)
        )
    if metadata.include_mask_id:
        assert observation.mask_id_history is not None
        batch["mask_id_hist"] = _batched(
            observation.mask_id_history.astype(np.int32, copy=False)
        )
    return batch


class ActionBridgeRLBenchPolicyAdapter:
    """Stateful policy adapter; it neither launches RLBench nor writes artifacts."""

    def __init__(
        self,
        *,
        metadata: OnlineEvaluationMetadata,
        backend: InferenceBackend,
        checkpoint_identifier: str,
    ) -> None:
        if (
            not isinstance(checkpoint_identifier, str)
            or not checkpoint_identifier.strip()
        ):
            raise ValueError("checkpoint_identifier must be non-empty.")
        self.metadata = metadata
        self.backend = backend
        self.checkpoint_identifier = checkpoint_identifier.strip()
        self._episode: tuple[str, int] | None = None
        self._policy_calls = 0

    def reset(self, *, task_name: str, variation_id: int, seed: int) -> None:
        variation_key = f"{task_name}:{variation_id}"
        if task_name not in self.metadata.task_to_id:
            raise KeyError(f"Task {task_name!r} is absent from checkpoint metadata.")
        if variation_key not in self.metadata.task_variation_to_id:
            raise KeyError(
                f"Task variation {variation_key!r} is absent from checkpoint metadata."
            )
        self.backend.reset(seed=seed)
        self._episode = (task_name, variation_id)
        self._policy_calls = 0

    def predict(self, observation: PolicyInput) -> PolicyOutput:
        if self._episode is None:
            raise RuntimeError("Action Bridge adapter must be reset before prediction.")
        if (observation.task_name, observation.variation_id) != self._episode:
            raise ValueError(
                "PolicyInput task/variation changed without an adapter reset."
            )
        batch = build_online_batch(observation, self.metadata)
        actions, diagnostics = self.backend.predict(batch)
        source = np.asarray(actions)
        expected = (
            1,
            self.metadata.action_horizon,
            len(self.metadata.action_layout),
        )
        if source.shape != expected:
            raise ValueError(
                f"Inference backend returned actions {source.shape}, expected {expected}."
            )
        if source.dtype.kind not in "iuf" or not np.isfinite(source).all():
            raise ValueError(
                "Inference backend returned non-finite/non-numeric actions."
            )
        self._policy_calls += 1
        output_diagnostics = {
            **dict(diagnostics),
            "adapter": "action_bridge_rlbench",
            "policy_type": self.metadata.policy_type,
            "deterministic_latent": self.metadata.deterministic_latent,
            "policy_call": self._policy_calls,
        }
        return PolicyOutput(
            actions=source[0].astype(np.float32, copy=True),
            diagnostics=output_diagnostics,
        )


__all__ = [
    "ActionBridgeRLBenchPolicyAdapter",
    "InferenceBackend",
    "build_online_batch",
]
