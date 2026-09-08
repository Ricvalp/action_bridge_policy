"""Maintain training-compatible histories while executing action chunks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from phi_mujoco.evaluation import PolicyInput
from phi_mujoco.integrations import IntegrationSpec, get_integration

from .metadata import OnlineEvaluationMetadata


class InferenceBackend(Protocol):
    def reset(self, *, seed: int) -> None: ...

    def predict(
        self, batch: Mapping[str, NDArray[np.float32]]
    ) -> tuple[NDArray[np.float32], Mapping[str, object]]:
        """Return normalized actions with shape (1, horizon, action_dim)."""
        ...


class ActionBridgeMujocoPolicyAdapter:
    """Return one action per callback, replanning every actions_per_plan steps.

    Keeping the chunk here lets every intervening observation and executed
    action enter the histories, even when the model is not called each step.
    """

    def __init__(
        self,
        *,
        metadata: OnlineEvaluationMetadata,
        backend: InferenceBackend,
        checkpoint_identifier: str,
        actions_per_plan: int | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        self.metadata = metadata
        self.backend = backend
        self.integration = get_integration(metadata.integration.name)
        self.checkpoint_identifier = checkpoint_identifier
        self.provenance = dict(provenance or {})
        self.actions_per_plan = (
            metadata.actions_per_plan if actions_per_plan is None else actions_per_plan
        )
        if not 1 <= self.actions_per_plan <= metadata.action_horizon:
            raise ValueError(
                "actions_per_plan must lie between 1 and the trained action horizon"
            )
        self._ready = False

    def reset(self, *, integration: IntegrationSpec, seed: int) -> None:
        if integration != self.metadata.integration:
            raise ValueError("evaluation integration disagrees with checkpoint profile")
        self.backend.reset(seed=seed)
        self._seed = seed
        self._last_step = -1
        self._observations: NDArray[np.float32] | None = None
        self._actions: NDArray[np.float32] | None = None
        self._pending_action: NDArray[np.float32] | None = None
        self._plan: list[NDArray[np.float32]] = []
        self._ready = True

    def predict(self, inputs: PolicyInput) -> NDArray[np.float32]:
        if not self._ready:
            raise RuntimeError("reset the policy before prediction")
        if inputs.integration != self.metadata.integration or inputs.seed != self._seed:
            raise ValueError("evaluation identity changed without a policy reset")
        if inputs.step_index != self._last_step + 1:
            raise ValueError(
                "the evaluator must call the adapter at every consecutive step"
            )
        state = np.asarray(inputs.observations["state"], dtype=np.float32)
        if (
            state.shape != (self.metadata.observation_dim,)
            or not np.isfinite(state).all()
        ):
            raise ValueError(
                "policy state has the wrong shape or contains non-finite values"
            )
        if self._last_step == -1:
            self._observations = np.repeat(
                state[None], self.metadata.observation_history, axis=0
            )
            padding = self.integration.action_history_padding(inputs.observations)
            self._actions = np.repeat(
                padding[None], self.metadata.action_history, axis=0
            )
        else:
            self._observations = np.concatenate([self._observations[1:], state[None]])
            self._actions = np.concatenate(
                [self._actions[1:], self._pending_action[None]]
            )
        self._last_step = inputs.step_index

        stats = self.metadata.normalization
        if not self._plan:
            normalized, _ = self.backend.predict(
                {
                    "obs_hist": stats.normalize_observations(self._observations)[None],
                    "act_hist": stats.normalize_actions(self._actions)[None],
                }
            )
            normalized = np.asarray(normalized)
            expected = (1, self.metadata.action_horizon, self.metadata.action_dim)
            if normalized.shape != expected or not np.isfinite(normalized).all():
                raise ValueError(f"backend actions must have finite shape {expected}")
            raw = stats.denormalize_actions(normalized)[0]
            self._plan = [action.copy() for action in raw[: self.actions_per_plan]]
        action = np.asarray(self._plan.pop(0), dtype=np.float32)
        if self.metadata.clip_actions:
            action = self.integration.project_action(action, inputs.observations)
        self.integration.validate_action(action, strict=True)
        # Commit the projected command, which is what the simulator will execute.
        self._pending_action = np.asarray(action, dtype=np.float32).copy()
        return self._pending_action.copy()
