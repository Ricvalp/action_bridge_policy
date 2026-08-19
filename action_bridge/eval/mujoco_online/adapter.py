"""NumPy policy adapter for Action Bridge on the phi-mujoco v1 contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from phi_mujoco.evaluation import PolicyInput, PolicyOutput

from action_bridge.eval.mujoco_online.metadata import OnlineEvaluationMetadata


class InferenceBackend(Protocol):
    """Framework-owned normalized inference boundary."""

    def reset(self, *, seed: int) -> None:
        """Reset episode-local framework RNG state."""

    def predict(
        self, batch: Mapping[str, NDArray[np.float32]]
    ) -> tuple[NDArray[np.float32], Mapping[str, object]]:
        """Return normalized action chunks with shape ``[1, H, 2]``."""


def _normalized_observations(
    observations: NDArray[np.float32], metadata: OnlineEvaluationMetadata
) -> NDArray[np.float32]:
    stats = metadata.normalization
    mean = np.asarray(stats.obs_mean, dtype=np.float32)
    std = np.asarray(stats.obs_std, dtype=np.float32)
    output = (observations - mean) / std
    return np.ascontiguousarray(output, dtype=np.float32)


def _normalized_actions(
    actions: NDArray[np.float32], metadata: OnlineEvaluationMetadata
) -> NDArray[np.float32]:
    stats = metadata.normalization
    mean = np.asarray(stats.action_mean, dtype=np.float32)
    std = np.asarray(stats.action_std, dtype=np.float32)
    output = (actions - mean) / std
    return np.ascontiguousarray(output, dtype=np.float32)


def _raw_actions(
    actions: NDArray[np.float32], metadata: OnlineEvaluationMetadata
) -> NDArray[np.float32]:
    stats = metadata.normalization
    mean = np.asarray(stats.action_mean, dtype=np.float32)
    std = np.asarray(stats.action_std, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        output = actions * std + mean
    return np.ascontiguousarray(output, dtype=np.float32)


class ActionBridgeMujocoPolicyAdapter:
    """Maintain exact loader-compatible histories around a framework backend.

    The v1 evaluator executes exactly one action before returning the next
    observation.  This lets the adapter commit one previously returned torque
    and construct histories identical to the offline loader without simulator
    access or guessed intermediate observations.
    """

    def __init__(
        self,
        *,
        metadata: OnlineEvaluationMetadata,
        backend: InferenceBackend,
        checkpoint_identifier: str,
    ) -> None:
        if not isinstance(checkpoint_identifier, str) or not checkpoint_identifier.strip():
            raise ValueError("checkpoint_identifier must be non-empty.")
        self.metadata = metadata
        self.backend = backend
        self.checkpoint_identifier = checkpoint_identifier.strip()
        self._episode: tuple[str, int] | None = None
        self._last_episode_step: int | None = None
        self._observation_history: NDArray[np.float32] | None = None
        self._action_history: NDArray[np.float32] | None = None
        self._pending_action: NDArray[np.float32] | None = None
        self._policy_calls = 0
        self._clipped_action_calls = 0
        self._clipped_value_count = 0
        self._max_clip_correction_nm = 0.0

    def reset(self, *, task_name: str, variation_id: int, seed: int) -> None:
        if task_name != self.metadata.task_name or variation_id != self.metadata.variation_id:
            raise ValueError(
                "task/variation does not match the checkpoint's exact MuJoCo metadata."
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("policy seed must be an integer in the uint32 range.")
        self.backend.reset(seed=seed)
        self._episode = (task_name, variation_id)
        self._last_episode_step = None
        self._observation_history = None
        self._action_history = None
        self._pending_action = None
        self._policy_calls = 0
        self._clipped_action_calls = 0
        self._clipped_value_count = 0
        self._max_clip_correction_nm = 0.0

    def _update_histories(self, observation: PolicyInput) -> None:
        raw_observation = np.asarray(observation.observation, dtype=np.float32)
        if self._last_episode_step is None:
            if observation.episode_step != 0:
                raise ValueError("the first policy observation after reset must be episode_step=0.")
            self._observation_history = np.repeat(
                raw_observation[None], self.metadata.observation_history, axis=0
            )
            self._action_history = np.zeros(
                (self.metadata.action_history, self.metadata.action_dim),
                dtype=np.float32,
            )
        else:
            expected_step = self._last_episode_step + self.metadata.actions_per_plan
            if observation.episode_step != expected_step:
                raise ValueError(
                    f"expected consecutive episode_step={expected_step}, got "
                    f"{observation.episode_step}; v1 requires actions_per_plan=1."
                )
            if self._pending_action is None:
                raise RuntimeError("no previously returned action is available to commit.")
            assert self._observation_history is not None
            assert self._action_history is not None
            self._observation_history = np.concatenate(
                [self._observation_history[1:], raw_observation[None]], axis=0
            ).astype(np.float32, copy=False)
            self._action_history = np.concatenate(
                [self._action_history[1:], self._pending_action[None]], axis=0
            ).astype(np.float32, copy=False)
        self._last_episode_step = observation.episode_step

    def predict(self, observation: PolicyInput) -> PolicyOutput:
        if self._episode is None:
            raise RuntimeError("Action Bridge MuJoCo adapter must be reset before prediction.")
        if (observation.task_name, observation.variation_id) != self._episode:
            raise ValueError("PolicyInput task/variation changed without an adapter reset.")
        self._update_histories(observation)
        assert self._observation_history is not None
        assert self._action_history is not None
        batch = {
            "obs_hist": _normalized_observations(self._observation_history, self.metadata)[None],
            "act_hist": _normalized_actions(self._action_history, self.metadata)[None],
        }
        normalized, backend_diagnostics = self.backend.predict(batch)
        source = np.asarray(normalized)
        expected_shape = (1, self.metadata.action_horizon, self.metadata.action_dim)
        if source.shape != expected_shape:
            raise ValueError(
                f"inference backend returned actions {source.shape}, expected {expected_shape}."
            )
        if source.dtype.kind not in "iuf" or not np.isfinite(source).all():
            raise ValueError("inference backend returned non-finite or non-numeric actions.")
        with np.errstate(over="ignore", invalid="ignore"):
            normalized_actions = source.astype(np.float32, copy=False)
        actions = _raw_actions(normalized_actions, self.metadata)[0]
        if not np.isfinite(actions).all():
            raise ValueError(
                "inference backend actions become non-finite after float32 "
                "conversion or denormalization."
            )
        lower = np.asarray(self.metadata.action_lower, dtype=np.float32)
        upper = np.asarray(self.metadata.action_upper, dtype=np.float32)
        violations = (actions < lower) | (actions > upper)
        clipped = bool(np.any(violations))
        max_correction = 0.0
        if clipped:
            clipped_actions = np.clip(actions, lower, upper).astype(np.float32, copy=False)
            max_correction = float(np.max(np.abs(clipped_actions - actions)))
            if not self.metadata.clip_actions:
                raise ValueError(
                    "denormalized policy action violates the exact torque bounds; "
                    "checkpoint metadata clip_actions=false."
                )
            actions = np.ascontiguousarray(clipped_actions)
            self._clipped_action_calls += 1
            self._clipped_value_count += int(np.count_nonzero(violations))
            self._max_clip_correction_nm = max(
                self._max_clip_correction_nm,
                max_correction,
            )

        self._pending_action = actions[0].copy()
        self._policy_calls += 1
        diagnostics = {
            **dict(backend_diagnostics),
            "adapter": "action_bridge_mujoco",
            "policy_type": self.metadata.policy_type,
            "policy_call": self._policy_calls,
            "deterministic_latent": self.metadata.deterministic_latent,
            "clip_actions": self.metadata.clip_actions,
            "action_clipped": clipped,
            "clipped_value_count": int(np.count_nonzero(violations)),
            "max_clip_correction_nm": max_correction,
            "clipped_action_calls_total": self._clipped_action_calls,
            "clipped_value_count_total": self._clipped_value_count,
            "max_clip_correction_nm_episode": self._max_clip_correction_nm,
        }
        return PolicyOutput(actions=actions, diagnostics=diagnostics)


__all__ = ["ActionBridgeMujocoPolicyAdapter", "InferenceBackend"]
