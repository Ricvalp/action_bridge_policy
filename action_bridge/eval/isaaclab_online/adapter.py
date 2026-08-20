"""Device-native batched Action Bridge policy adapter for PHI Isaac Lab."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch

from action_bridge.eval.isaaclab_online.contracts import TCP_POSE_SLICE
from action_bridge.eval.isaaclab_online.metadata import OnlineEvaluationMetadata


class BatchedPolicyInputLike(Protocol):
    """Structural view of ``phi_isaaclab.evaluation.BatchedPolicyInput``."""

    observation: torch.Tensor
    episode_ids: torch.Tensor
    step_indices: torch.Tensor
    task_id: str
    variation_id: int


class BatchedInferenceBackend(Protocol):
    """Framework inference boundary that never leaves the Torch device."""

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all or selected episode-local latent state."""

    def predict(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        """Return normalized chunks with shape ``[B,H,8]``."""


class ActionBridgeIsaacLabPolicyAdapter:
    """Maintain exact per-environment histories around a batched Torch policy.

    The adapter consumes and returns tensors on the simulator device.  It does
    not convert through NumPy or loop over environments.  Isaac Lab v1 executes
    one action from each predicted chunk before providing the next observation.
    """

    def __init__(
        self,
        *,
        metadata: OnlineEvaluationMetadata,
        backend: BatchedInferenceBackend,
        checkpoint_identifier: str,
    ) -> None:
        if not isinstance(checkpoint_identifier, str) or not checkpoint_identifier.strip():
            raise ValueError("checkpoint_identifier must be non-empty")
        self.metadata = metadata
        self.backend = backend
        self.checkpoint_identifier = checkpoint_identifier.strip()
        self._initialized = False
        self._reset_before_initialization = False
        self._episode_ids: torch.Tensor | None = None
        self._last_steps: torch.Tensor | None = None
        self._reset_pending: torch.Tensor | None = None
        self._observation_history: torch.Tensor | None = None
        self._action_history: torch.Tensor | None = None
        self._pending_action: torch.Tensor | None = None
        self._stats_cache: tuple[
            torch.device,
            torch.dtype,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Mark all or selected vector environments for episode initialization."""

        if not self._initialized:
            if env_ids is not None:
                raise RuntimeError("the first adapter reset must use env_ids=None")
            self._reset_before_initialization = True
            self.backend.reset(None)
            return
        assert self._episode_ids is not None
        assert self._last_steps is not None
        assert self._reset_pending is not None
        if env_ids is None:
            self._episode_ids.fill_(-1)
            self._last_steps.fill_(-1)
            self._reset_pending.fill_(True)
        else:
            env_ids = self._validated_env_ids(env_ids)
            self._episode_ids[env_ids] = -1
            self._last_steps[env_ids] = -1
            self._reset_pending[env_ids] = True
        self.backend.reset(env_ids)

    def _validated_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        assert self._episode_ids is not None
        if not torch.is_tensor(env_ids) or env_ids.ndim != 1:
            raise TypeError("env_ids must be a rank-one Torch tensor")
        if env_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("env_ids must have an integer dtype")
        if env_ids.device != self._episode_ids.device:
            raise ValueError("env_ids must remain on the policy tensor device")
        if env_ids.numel() and bool(
            ((env_ids < 0) | (env_ids >= self._episode_ids.shape[0])).any()
        ):
            raise IndexError("env_ids contain an out-of-range vector environment index")
        if env_ids.numel() != torch.unique(env_ids).numel():
            raise ValueError("env_ids must contain no duplicates")
        return env_ids.to(dtype=torch.int64)

    def _initialize(self, observation: torch.Tensor, episode_ids: torch.Tensor) -> None:
        if not self._reset_before_initialization:
            raise RuntimeError("Action Bridge Isaac Lab adapter must be reset before prediction")
        batch_size = observation.shape[0]
        self._episode_ids = torch.full(
            (batch_size,), -1, dtype=episode_ids.dtype, device=episode_ids.device
        )
        self._last_steps = torch.full_like(self._episode_ids, -1)
        self._reset_pending = torch.ones(
            (batch_size,), dtype=torch.bool, device=observation.device
        )
        self._observation_history = torch.empty(
            (batch_size, self.metadata.observation_history, self.metadata.observation_dim),
            dtype=observation.dtype,
            device=observation.device,
        )
        self._action_history = torch.empty(
            (batch_size, self.metadata.action_history, self.metadata.action_dim),
            dtype=observation.dtype,
            device=observation.device,
        )
        self._pending_action = torch.empty(
            (batch_size, self.metadata.action_dim),
            dtype=observation.dtype,
            device=observation.device,
        )
        self._initialized = True

    def _validate_inputs(
        self, inputs: BatchedPolicyInputLike
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            inputs.task_id != self.metadata.task_name
            or inputs.variation_id != self.metadata.variation_id
        ):
            raise ValueError("task/variation does not match the checkpoint's Isaac Lab metadata")
        observation = inputs.observation
        episode_ids = inputs.episode_ids
        steps = inputs.step_indices
        if not torch.is_tensor(observation) or observation.ndim != 2:
            raise TypeError("observation must be a rank-two Torch tensor")
        if observation.shape[1] != self.metadata.observation_dim:
            raise ValueError(
                f"observation must have shape [B,{self.metadata.observation_dim}]"
            )
        if observation.dtype != torch.float32:
            raise TypeError("Isaac Lab v1 observations must use torch.float32")
        if not torch.isfinite(observation).all():
            raise ValueError("observation contains non-finite values")
        for value, name in ((episode_ids, "episode_ids"), (steps, "step_indices")):
            if not torch.is_tensor(value) or value.ndim != 1:
                raise TypeError(f"{name} must be a rank-one Torch tensor")
            if value.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must have an integer dtype")
            if value.shape[0] != observation.shape[0]:
                raise ValueError(f"{name} batch dimension disagrees with observation")
            if value.device != observation.device:
                raise ValueError(f"{name} must remain on the observation device")
            if bool((value < 0).any()):
                raise ValueError(f"{name} must be non-negative")
        if self._initialized and self._episode_ids is not None:
            if observation.shape[0] != self._episode_ids.shape[0]:
                raise ValueError("the vector environment batch size changed without reconstruction")
            if observation.device != self._episode_ids.device:
                raise ValueError("the vector environment device changed without reconstruction")
        return observation, episode_ids, steps

    def _stats(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._stats_cache
        if cached is not None and cached[0] == reference.device and cached[1] == reference.dtype:
            return cached[2], cached[3], cached[4], cached[5]
        normalization = self.metadata.normalization
        obs_mean = torch.tensor(
            normalization.obs_mean, dtype=reference.dtype, device=reference.device
        )
        obs_std = torch.tensor(
            normalization.obs_std, dtype=reference.dtype, device=reference.device
        )
        action_mean = torch.tensor(
            normalization.action_mean, dtype=reference.dtype, device=reference.device
        )
        action_std = torch.tensor(
            normalization.action_std, dtype=reference.dtype, device=reference.device
        )
        self._stats_cache = (
            reference.device,
            reference.dtype,
            obs_mean,
            obs_std,
            action_mean,
            action_std,
        )
        return obs_mean, obs_std, action_mean, action_std

    def _update_histories(
        self,
        observation: torch.Tensor,
        episode_ids: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        assert self._episode_ids is not None
        assert self._last_steps is not None
        assert self._reset_pending is not None
        assert self._observation_history is not None
        assert self._action_history is not None
        assert self._pending_action is not None

        reset_mask = self._reset_pending
        changed_episode = episode_ids != self._episode_ids
        if not torch.equal(changed_episode, reset_mask):
            raise ValueError(
                "episode_ids changed without reset(env_ids), or reset environments retained "
                "their prior episode id"
            )
        if bool((steps[reset_mask] != 0).any()):
            raise ValueError("the first observation after reset must have step_indices=0")
        continuing = ~reset_mask
        if bool((steps[continuing] != self._last_steps[continuing] + 1).any()):
            raise ValueError(
                "Isaac Lab v1 requires consecutive step_indices because actions_per_plan=1"
            )

        if bool(continuing.any()):
            self._observation_history[continuing, :-1] = self._observation_history[
                continuing, 1:
            ].clone()
            self._observation_history[continuing, -1] = observation[continuing]
            self._action_history[continuing, :-1] = self._action_history[
                continuing, 1:
            ].clone()
            self._action_history[continuing, -1] = self._pending_action[continuing]
        if bool(reset_mask.any()):
            reset_observation = observation[reset_mask]
            self._observation_history[reset_mask] = reset_observation.unsqueeze(1).expand(
                -1, self.metadata.observation_history, -1
            )
            hold_open = torch.cat(
                [
                    reset_observation[:, slice(*TCP_POSE_SLICE)],
                    torch.ones(
                        (reset_observation.shape[0], 1),
                        dtype=reset_observation.dtype,
                        device=reset_observation.device,
                    ),
                ],
                dim=-1,
            )
            self._action_history[reset_mask] = hold_open.unsqueeze(1).expand(
                -1, self.metadata.action_history, -1
            )
        self._episode_ids.copy_(episode_ids)
        self._last_steps.copy_(steps)
        self._reset_pending.fill_(False)

    def _project_actions(self, actions: torch.Tensor) -> torch.Tensor:
        projection = self.metadata.action_projection
        if not torch.isfinite(actions).all():
            raise ValueError("denormalized policy action contains non-finite values")
        lower = actions.new_tensor(projection.position_lower_m)
        upper = actions.new_tensor(projection.position_upper_m)
        position = torch.maximum(torch.minimum(actions[..., :3], upper), lower)
        quaternion = actions[..., 3:7]
        quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if bool((quaternion_norm < projection.quaternion_epsilon).any()):
            raise ValueError("policy produced a near-zero quaternion that cannot be normalized")
        quaternion = quaternion / quaternion_norm
        sign = torch.where(
            quaternion[..., 3:4] < 0.0,
            quaternion.new_tensor(-1.0),
            quaternion.new_tensor(1.0),
        )
        quaternion = quaternion * sign
        gripper = torch.where(
            actions[..., 7:8] >= projection.gripper_threshold,
            actions.new_tensor(projection.gripper_open_action),
            actions.new_tensor(projection.gripper_close_action),
        )
        return torch.cat([position, quaternion, gripper], dim=-1).contiguous()

    def predict(self, inputs: BatchedPolicyInputLike) -> torch.Tensor:
        """Return one strict physical action for every active vector environment."""

        observation, episode_ids, steps = self._validate_inputs(inputs)
        if not self._initialized:
            self._initialize(observation, episode_ids)
        self._update_histories(observation, episode_ids, steps)
        assert self._observation_history is not None
        assert self._action_history is not None
        assert self._pending_action is not None
        obs_mean, obs_std, action_mean, action_std = self._stats(observation)
        batch = {
            "obs_hist": (self._observation_history - obs_mean) / obs_std,
            "act_hist": (self._action_history - action_mean) / action_std,
        }
        normalized_actions, _ = self.backend.predict(batch)
        expected_shape = (
            observation.shape[0],
            self.metadata.action_horizon,
            self.metadata.action_dim,
        )
        if not torch.is_tensor(normalized_actions) or normalized_actions.shape != expected_shape:
            shape = getattr(normalized_actions, "shape", None)
            raise ValueError(
                f"inference backend returned actions {shape}, expected {expected_shape}"
            )
        if normalized_actions.device != observation.device:
            raise ValueError("inference backend moved actions off the simulator device")
        if normalized_actions.dtype != observation.dtype:
            raise TypeError("inference backend action dtype disagrees with observations")
        projected = self._project_actions(normalized_actions * action_std + action_mean)
        self._pending_action.copy_(projected[:, 0])
        return projected[:, 0]


__all__ = [
    "ActionBridgeIsaacLabPolicyAdapter",
    "BatchedInferenceBackend",
    "BatchedPolicyInputLike",
]
