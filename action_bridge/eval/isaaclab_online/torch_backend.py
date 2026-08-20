"""Trusted PyTorch checkpoint loading and batched Isaac Lab inference."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from action_bridge.eval.isaaclab_online.adapter import ActionBridgeIsaacLabPolicyAdapter
from action_bridge.eval.isaaclab_online.metadata import (
    OnlineEvaluationMetadata,
    resolve_online_metadata,
    validate_checkpoint_config,
)
from action_bridge.eval.rollout import generate_chunk, predict_actions
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.training.common import build_model, resolve_device


def _checkpoint_snapshot(path: str | Path) -> tuple[bytes, str]:
    """Read one stable, regular-file checkpoint snapshot and identify its bytes."""

    source = Path(path).expanduser().absolute()
    try:
        visible = os.lstat(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"checkpoint does not exist: {source}") from exc
    if stat.S_ISLNK(visible.st_mode) or not stat.S_ISREG(visible.st_mode):
        raise ValueError(f"checkpoint must be a regular non-symlink file: {source}")
    descriptor = os.open(
        source,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            visible.st_dev,
            visible.st_ino,
        ):
            raise ValueError("checkpoint changed while it was opened")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        if (
            final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ValueError("checkpoint changed while it was read")
    finally:
        os.close(descriptor)
    return b"".join(chunks), f"sha256:{digest.hexdigest()}"


class TorchInferenceBackend:
    """Run a reconstructed Action Bridge model on device-native tensor batches."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        metadata: OnlineEvaluationMetadata,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.metadata = metadata
        self.device = device
        self._reset = False
        self._episode_latent_embedding: torch.Tensor | None = None
        self._episode_latent_valid: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._episode_latent_embedding = None
            self._episode_latent_valid = None
        elif self._episode_latent_valid is not None:
            if env_ids.device != self.device:
                raise ValueError("env_ids must remain on the inference device")
            self._episode_latent_valid[env_ids.to(torch.int64)] = False
        self._reset = True

    def _latent_embedding(
        self, model: ActionBridgePolicy, history_embedding: torch.Tensor
    ) -> torch.Tensor:
        if model.latent_type == "continuous":
            mean, _ = model.latent.prior_params(history_embedding)
            return model.latent.embed(mean)
        if model.latent_type == "categorical":
            logits = model.latent.prior_logits(history_embedding)
            return model.latent.embed_ids(logits.argmax(dim=-1))
        return model.zero_z_embedding(
            history_embedding.shape[0],
            history_embedding.device,
            history_embedding.dtype,
        )

    def _committed_latent_embedding(
        self, model: ActionBridgePolicy, history_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        candidate = self._latent_embedding(model, history_embedding)
        if self.metadata.latent_commitment == "chunk" or model.latent_type == "none":
            return candidate, None
        batch_size = history_embedding.shape[0]
        if (
            self._episode_latent_embedding is None
            or self._episode_latent_embedding.shape != candidate.shape
        ):
            self._episode_latent_embedding = candidate.detach().clone()
            self._episode_latent_valid = torch.ones(
                (batch_size,), dtype=torch.bool, device=self.device
            )
            reused = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
            return candidate, reused
        assert self._episode_latent_valid is not None
        reused = self._episode_latent_valid.clone()
        selected = torch.where(
            reused.unsqueeze(-1), self._episode_latent_embedding, candidate
        )
        self._episode_latent_embedding.copy_(selected.detach())
        self._episode_latent_valid.fill_(True)
        return selected, reused

    @torch.inference_mode()
    def predict(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        if not self._reset:
            raise RuntimeError("Torch inference backend must be reset before prediction")
        obs_hist = batch["obs_hist"]
        act_hist = batch["act_hist"]
        if not torch.is_tensor(obs_hist) or not torch.is_tensor(act_hist):
            raise TypeError("Torch inference backend requires tensor histories")
        if obs_hist.device != self.device or act_hist.device != self.device:
            raise ValueError("history tensors must remain on the inference device")
        diagnostics: dict[str, object] = {
            "framework": "torch",
            "device": str(self.device),
            "latent_commitment": self.metadata.latent_commitment,
        }
        if isinstance(self.model, ActionBridgePolicy):
            history_embedding = self.model.encode_history(obs_hist, act_hist)
            latent_embedding, reused = self._committed_latent_embedding(
                self.model, history_embedding
            )
            output = generate_chunk(
                self.model,
                obs_hist,
                act_hist,
                deterministic=True,
                z_emb=latent_embedding,
            )
            if reused is not None:
                diagnostics["episode_latent_reused"] = reused
        else:
            output = predict_actions(
                self.model,
                {"obs_hist": obs_hist, "act_hist": act_hist},
                deterministic=True,
            )
        actions = output["actions"]
        if not torch.is_tensor(actions):
            raise TypeError("model returned non-tensor actions")
        return actions, diagnostics


def load_torch_policy_adapter(
    checkpoint_path: str | Path,
    *,
    online_metadata_path: str | Path | None = None,
    trusted_checkpoint: bool = False,
    device: str = "cuda",
    strict_device: bool = False,
) -> ActionBridgeIsaacLabPolicyAdapter:
    """Load one explicitly trusted checkpoint without importing Isaac Lab."""

    if not trusted_checkpoint:
        raise ValueError(
            "Action Bridge PyTorch checkpoints use pickle and may execute code while loading; "
            "set trusted_checkpoint=True only for a trusted local checkpoint."
        )
    payload, identifier = _checkpoint_snapshot(checkpoint_path)
    requested_device = torch.device(device)
    if strict_device and requested_device.type == "cuda" and requested_device.index is None:
        raise ValueError("strict CUDA inference requires an explicit device index")
    resolved_device = resolve_device(device)
    if strict_device and resolved_device.type != requested_device.type:
        raise RuntimeError(
            f"requested inference device {requested_device} is unavailable; refusing "
            f"silent fallback to {resolved_device}"
        )
    if (
        strict_device
        and requested_device.index is not None
        and resolved_device.index != requested_device.index
    ):
        raise RuntimeError(
            f"requested inference device {requested_device}, resolved {resolved_device}"
        )
    checkpoint: Any = torch.load(
        io.BytesIO(payload),
        map_location=resolved_device,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Action Bridge checkpoint must contain a mapping")
    if "config" not in checkpoint or "model_state" not in checkpoint:
        raise KeyError("Action Bridge checkpoint requires config and model_state")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise TypeError("Action Bridge checkpoint config must be a mapping")
    metadata = resolve_online_metadata(checkpoint, explicit_path=online_metadata_path)
    validate_checkpoint_config(config, metadata)
    model = build_model(config).to(resolved_device)
    state = checkpoint["model_state"]
    if not isinstance(state, Mapping):
        raise TypeError("Action Bridge checkpoint model_state must be a mapping")
    model.load_state_dict(state, strict=True)
    backend = TorchInferenceBackend(model=model, metadata=metadata, device=resolved_device)
    return ActionBridgeIsaacLabPolicyAdapter(
        metadata=metadata,
        backend=backend,
        checkpoint_identifier=identifier,
    )


__all__ = ["TorchInferenceBackend", "load_torch_policy_adapter"]
