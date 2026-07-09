"""Coordinate adapters for reference processes.

The dataset raw action is not always the coordinate where a reference process
should live. This adapter keeps that conversion explicit while preserving the
raw action format expected by training and evaluation code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class ActionCoordinateAdapter:
    """Convert raw actions to reference coordinates and back.

    Supported coordinate modes:
    - raw_action: q is the dataset action.
    - absolute_action: q is an absolute target action; decoding returns q.
    - absolute_from_delta: raw actions are deltas, q is their integrated
      absolute position path.
    """

    def __init__(self, coordinate_mode: str = "raw_action", dt: float = 1.0, action_dim: int = 2):
        if coordinate_mode not in {"raw_action", "absolute_action", "absolute_from_delta"}:
            raise ValueError(f"Unknown coordinate_mode {coordinate_mode!r}.")
        self.coordinate_mode = coordinate_mode
        self.dt = float(dt)
        self.action_dim = int(action_dim)

    def _future_actions(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "actions" in batch:
            return batch["actions"]
        return batch["future_actions"]

    def _prev_actions(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "prev_actions" in batch:
            return batch["prev_actions"]
        return batch["act_hist"]

    def _current_position(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "current_position" in batch:
            return batch["current_position"]
        if "future_positions" in batch:
            return batch["future_positions"][:, 0]
        if "obs_hist" in batch:
            return batch["obs_hist"][:, -1, : self.action_dim]
        raise KeyError("absolute_from_delta requires current_position, future_positions, or obs_hist.")

    def _previous_position(self, batch: Dict[str, Any], q0: torch.Tensor) -> torch.Tensor:
        if "previous_position" in batch:
            return batch["previous_position"]
        prev_actions = self._prev_actions(batch)
        return q0 - prev_actions[:, -1]

    def build_q_sequence(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return q_seq with shape [B, H+1, A]."""

        actions = self._future_actions(batch)
        if self.coordinate_mode in {"raw_action", "absolute_action"}:
            prev_actions = self._prev_actions(batch)
            q0 = prev_actions[:, -1:]
            return torch.cat([q0, actions], dim=1)

        q0 = self._current_position(batch)
        q_future = q0[:, None] + torch.cumsum(actions, dim=1)
        return torch.cat([q0[:, None], q_future], dim=1)

    def q_minus_one(self, batch: Dict[str, Any], q_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q_seq is None:
            q_seq = self.build_q_sequence(batch)
        if self.coordinate_mode in {"raw_action", "absolute_action"}:
            prev_actions = self._prev_actions(batch)
            if prev_actions.shape[1] >= 2:
                return prev_actions[:, -2]
            return q_seq[:, 0]
        return self._previous_position(batch, q_seq[:, 0])

    def build_p_sequence(self, q_seq: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Return p_seq with shape [B, H+1, A]."""

        q_prev0 = self.q_minus_one(batch, q_seq)
        q_prev = torch.cat([q_prev0[:, None], q_seq[:, :-1]], dim=1)
        return (q_seq - q_prev) / self.dt

    def init_qp_from_history(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return q0, p0 for inference."""

        prev_actions = self._prev_actions(batch)
        if self.coordinate_mode in {"raw_action", "absolute_action"}:
            q0 = prev_actions[:, -1]
            q_minus_1 = prev_actions[:, -2] if prev_actions.shape[1] >= 2 else q0
        else:
            q0 = self._current_position(batch)
            q_minus_1 = self._previous_position(batch, q0)
        p0 = (q0 - q_minus_1) / self.dt
        return q0, p0

    def decode_step(self, q: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        if self.coordinate_mode == "absolute_from_delta":
            return q_next - q
        return q_next

    def decode_raw_actions(self, q_seq: torch.Tensor) -> torch.Tensor:
        """Return raw action sequence with shape [B, H, A]."""

        if self.coordinate_mode == "absolute_from_delta":
            return q_seq[:, 1:] - q_seq[:, :-1]
        return q_seq[:, 1:]
