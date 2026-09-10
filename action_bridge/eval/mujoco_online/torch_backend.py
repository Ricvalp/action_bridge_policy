"""Trusted PyTorch checkpoint loading and normalized MuJoCo inference."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from action_bridge.eval.mujoco_online.adapter import ActionBridgeMujocoPolicyAdapter
from action_bridge.eval.mujoco_online.metadata import (
    OnlineEvaluationMetadata,
    resolve_online_metadata,
    validate_checkpoint_config,
)
from action_bridge.eval.rollout import generate_chunk, predict_actions
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.models.diffusion_policy import DiffusionPolicy
from action_bridge.training.common import build_model, resolve_device


def _checkpoint_snapshot(path: str | Path) -> tuple[bytes, str]:
    """Hash the same checkpoint bytes that will be deserialized."""
    payload = Path(path).expanduser().read_bytes()
    return payload, f"sha256:{hashlib.sha256(payload).hexdigest()}"


class TorchInferenceBackend:
    """Run one reconstructed Action Bridge model on normalized NumPy batches."""

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
        self._generator = torch.Generator(device=device)
        self._reset = False
        self._episode_latent: torch.Tensor | None = None
        self._episode_latent_embedding: torch.Tensor | None = None
        self._episode_latent_diagnostics: dict[str, object] = {}

    def reset(self, *, seed: int) -> None:
        self._generator.manual_seed(int(seed))
        self._episode_latent = None
        self._episode_latent_embedding = None
        self._episode_latent_diagnostics = {}
        self._reset = True

    def _latent(
        self, model: ActionBridgePolicy, h_emb: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, object]]:
        if model.latent_type == "continuous":
            mean, log_variance = model.latent.prior_params(h_emb)
            if self.metadata.deterministic_latent:
                latent = mean
            else:
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=mean.device,
                    generator=self._generator,
                )
                latent = mean + torch.exp(0.5 * log_variance) * noise
            return (
                latent,
                model.latent.embed(latent),
                {
                    "latent_l2": float(
                        torch.linalg.vector_norm(latent).detach().cpu().item()
                    )
                },
            )
        if model.latent_type == "categorical":
            logits = model.latent.prior_logits(h_emb)
            if self.metadata.deterministic_latent:
                latent = logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                latent = torch.multinomial(
                    probabilities, num_samples=1, generator=self._generator
                ).squeeze(-1)
            return (
                latent,
                model.latent.embed_ids(latent),
                {"latent_id": int(latent.detach().cpu().reshape(-1)[0].item())},
            )
        embedding = model.zero_z_embedding(h_emb.shape[0], h_emb.device, h_emb.dtype)
        return None, embedding, {}

    @torch.inference_mode()
    def predict(
        self, batch: Mapping[str, NDArray[np.float32]]
    ) -> tuple[NDArray[np.float32], Mapping[str, object]]:
        if not self._reset:
            raise RuntimeError(
                "Torch inference backend must be reset before prediction."
            )
        obs_hist = torch.as_tensor(
            batch["obs_hist"], dtype=torch.float32, device=self.device
        )
        act_hist = torch.as_tensor(
            batch["act_hist"], dtype=torch.float32, device=self.device
        )
        tensor_batch = {"obs_hist": obs_hist, "act_hist": act_hist}
        diagnostics: dict[str, object] = {
            "framework": "torch",
            "device": str(self.device),
        }
        if isinstance(self.model, ActionBridgePolicy):
            diagnostics["latent_commitment"] = self.metadata.latent_commitment
            h_emb = self.model.encode_history(obs_hist, act_hist)
            reuse_episode_latent = (
                self.metadata.latent_commitment == "episode"
                and self._episode_latent_embedding is not None
            )
            if reuse_episode_latent:
                latent = self._episode_latent
                latent_embedding = self._episode_latent_embedding
                latent_diagnostics = self._episode_latent_diagnostics
            else:
                latent, latent_embedding, latent_diagnostics = self._latent(
                    self.model, h_emb
                )
                if self.metadata.latent_commitment == "episode":
                    self._episode_latent = (
                        None if latent is None else latent.detach().clone()
                    )
                    self._episode_latent_embedding = latent_embedding.detach().clone()
                    self._episode_latent_diagnostics = dict(latent_diagnostics)
            output = generate_chunk(
                self.model,
                obs_hist,
                act_hist,
                deterministic=True,
                z=latent,
                z_emb=latent_embedding,
            )
            diagnostics.update(latent_diagnostics)
            diagnostics["episode_latent_reused"] = reuse_episode_latent
        else:
            output = predict_actions(
                self.model,
                tensor_batch,
                deterministic=True,
                generator=self._generator,
            )
            if isinstance(self.model, DiffusionPolicy):
                diagnostics["sampler"] = "ddim"
                diagnostics["sampling_seed"] = self._generator.initial_seed()
        path_kl = output.get("path_kl_energy")
        if path_kl is not None:
            diagnostics["normalized_path_kl_energy"] = float(
                path_kl.detach().mean().cpu().item()
            )
        actions = output["actions"].detach().cpu().numpy().astype(np.float32, copy=True)
        return np.ascontiguousarray(actions), diagnostics


def load_torch_policy_adapter(
    checkpoint_path: str | Path,
    *,
    trusted_checkpoint: bool = False,
    device: str = "cpu",
    actions_per_plan: int | None = None,
) -> ActionBridgeMujocoPolicyAdapter:
    """Load one explicitly trusted checkpoint without launching MuJoCo."""

    if not trusted_checkpoint:
        raise ValueError(
            "Action Bridge PyTorch checkpoints use pickle and may execute code while loading; "
            "set trusted_checkpoint=True only for a trusted local checkpoint."
        )
    payload, identifier = _checkpoint_snapshot(checkpoint_path)
    resolved_device = resolve_device(device)
    checkpoint: Any = torch.load(
        io.BytesIO(payload),
        map_location=resolved_device,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Action Bridge checkpoint must contain a mapping.")
    if "config" not in checkpoint or "model_state" not in checkpoint:
        raise KeyError("Action Bridge checkpoint requires config and model_state.")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise TypeError("Action Bridge checkpoint config must be a mapping.")
    metadata = resolve_online_metadata(checkpoint)
    validate_checkpoint_config(config, metadata)
    model = build_model(config).to(resolved_device)
    state = checkpoint["model_state"]
    if not isinstance(state, Mapping):
        raise TypeError("Action Bridge checkpoint model_state must be a mapping.")
    model.load_state_dict(state, strict=True)
    backend = TorchInferenceBackend(
        model=model, metadata=metadata, device=resolved_device
    )
    return ActionBridgeMujocoPolicyAdapter(
        metadata=metadata,
        backend=backend,
        checkpoint_identifier=identifier,
        actions_per_plan=actions_per_plan,
        provenance=config.get("provenance", {}),
    )


__all__ = ["TorchInferenceBackend", "load_torch_policy_adapter"]
