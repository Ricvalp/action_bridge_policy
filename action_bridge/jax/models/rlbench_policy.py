"""JAX RLBench policies with a Euclidean XYZ contact reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from action_bridge.jax.models.rlbench_encoder import (
    ActionQueryDecoder,
    EncoderConfig,
    RLBenchHistoryEncoder,
)


@dataclass(frozen=True)
class DecoderConfig:
    num_layers: int = 4


@dataclass(frozen=True)
class BridgeConfig:
    z_dim: int = 4
    z_embed_dim: int = 64
    hidden_dim: int = 512
    control_depth: int = 3
    reference_depth: int = 2
    auxiliary_depth: int = 2
    dt: float = 1.0
    sigma: float = 0.1
    k_min: float = 0.0
    k_max: float = 2.0
    gamma_min: float = 0.0
    gamma_max: float = 0.95
    xyz_center: Tuple[float, float, float] = (0.0, 0.0, 1.25)
    xyz_scale: Tuple[float, float, float] = (1.0, 1.0, 1.25)


@dataclass(frozen=True)
class RLBenchPolicyConfig:
    horizon: int = 16
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)


def normalize_xyz(xyz: jnp.ndarray, cfg: BridgeConfig) -> jnp.ndarray:
    center = jnp.asarray(cfg.xyz_center, dtype=jnp.float32)
    scale = jnp.asarray(cfg.xyz_scale, dtype=jnp.float32)
    return (xyz.astype(jnp.float32) - center) / scale


def denormalize_xyz(xyz: jnp.ndarray, cfg: BridgeConfig) -> jnp.ndarray:
    center = jnp.asarray(cfg.xyz_center, dtype=jnp.float32)
    scale = jnp.asarray(cfg.xyz_scale, dtype=jnp.float32)
    return xyz.astype(jnp.float32) * scale + center


def normalize_quaternion(quaternion: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(quaternion.astype(jnp.float32), axis=-1, keepdims=True)
    fallback = jnp.zeros_like(quaternion).at[..., 3].set(1.0)
    return jnp.where(norm > 1e-6, quaternion / jnp.maximum(norm, 1e-6), fallback)


def canonicalize_quaternion(
    quaternion: jnp.ndarray,
    reference: jnp.ndarray,
) -> jnp.ndarray:
    quaternion = normalize_quaternion(quaternion)
    reference = normalize_quaternion(reference)
    sign = jnp.where(
        jnp.sum(quaternion * reference, axis=-1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    return quaternion * sign


class MLPHead(nn.Module):
    hidden_dim: int
    depth: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for index in range(int(self.depth)):
            x = nn.Dense(int(self.hidden_dim), name=f"hidden_{index}")(x)
            x = nn.silu(x)
        return nn.Dense(int(self.output_dim), name="output")(x)


class GaussianHead(nn.Module):
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(int(self.hidden_dim), name="hidden")(x)
        x = nn.silu(x)
        parameters = nn.Dense(2 * int(self.z_dim), name="parameters")(x)
        mean, log_variance = jnp.split(parameters, 2, axis=-1)
        return mean, jnp.clip(log_variance, -10.0, 4.0)


class ContactParameterHead(nn.Module):
    cfg: BridgeConfig

    @nn.compact
    def __call__(self, context: jnp.ndarray):
        output = MLPHead(
            int(self.cfg.hidden_dim),
            int(self.cfg.reference_depth),
            7,
            name="network",
        )(context)
        attractor_raw, stiffness_raw, damping_raw = jnp.split(output, [3, 6], axis=-1)
        attractor = jnp.tanh(attractor_raw)
        stiffness = float(self.cfg.k_min) + (
            float(self.cfg.k_max) - float(self.cfg.k_min)
        ) * jax.nn.sigmoid(stiffness_raw)
        damping = float(self.cfg.gamma_min) + (
            float(self.cfg.gamma_max) - float(self.cfg.gamma_min)
        ) * jax.nn.sigmoid(damping_raw)
        return attractor, stiffness, damping


class ResidualControlHead(nn.Module):
    cfg: BridgeConfig

    @nn.compact
    def __call__(
        self,
        step_context: jnp.ndarray,
        position: jnp.ndarray,
        momentum: jnp.ndarray,
        latent_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        features = jnp.concatenate(
            [step_context, position, momentum, latent_embedding], axis=-1
        )
        return MLPHead(
            int(self.cfg.hidden_dim),
            int(self.cfg.control_depth),
            3,
            name="network",
        )(features)


class AuxiliaryActionHead(nn.Module):
    cfg: BridgeConfig

    @nn.compact
    def __call__(
        self,
        step_context: jnp.ndarray,
        position: jnp.ndarray,
        momentum: jnp.ndarray,
        latent_embedding: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        features = jnp.concatenate(
            [step_context, position, momentum, latent_embedding], axis=-1
        )
        output = MLPHead(
            int(self.cfg.hidden_dim),
            int(self.cfg.auxiliary_depth),
            5,
            name="network",
        )(features)
        return output[..., :4], output[..., 4]


class RLBenchActionBridgePolicy(nn.Module):
    cfg: RLBenchPolicyConfig
    state_dim: int
    action_dim: int
    num_tasks: int
    num_task_variations: int

    def setup(self):
        if int(self.action_dim) != 8:
            raise ValueError("RLBenchActionBridgePolicy expects 8D pose-plus-gripper actions.")
        self.encoder = RLBenchHistoryEncoder(
            self.cfg.encoder,
            self.num_tasks,
            self.num_task_variations,
            name="encoder",
        )
        self.decoder = ActionQueryDecoder(
            d_model=int(self.cfg.encoder.d_model),
            n_heads=int(self.cfg.encoder.n_heads),
            mlp_mult=int(self.cfg.encoder.mlp_mult),
            dropout=float(self.cfg.encoder.dropout),
            num_layers=int(self.cfg.decoder.num_layers),
            horizon=int(self.cfg.horizon),
            name="action_query_decoder",
        )
        self.prior = GaussianHead(
            int(self.cfg.bridge.hidden_dim), int(self.cfg.bridge.z_dim), name="prior"
        )
        self.posterior = GaussianHead(
            int(self.cfg.bridge.hidden_dim), int(self.cfg.bridge.z_dim), name="posterior"
        )
        self.latent_embedding = MLPHead(
            int(self.cfg.bridge.hidden_dim),
            1,
            int(self.cfg.bridge.z_embed_dim),
            name="latent_embedding",
        )
        self.contact_parameters = ContactParameterHead(
            self.cfg.bridge, name="contact_parameters"
        )
        self.control = ResidualControlHead(self.cfg.bridge, name="control")
        self.auxiliary = AuxiliaryActionHead(self.cfg.bridge, name="auxiliary")

    def _initial_state(self, batch: Dict[str, jnp.ndarray]):
        actions = batch["act_hist"].astype(jnp.float32)
        q0 = normalize_xyz(actions[:, -1, :3], self.cfg.bridge)
        if int(actions.shape[1]) > 1:
            previous = normalize_xyz(actions[:, -2, :3], self.cfg.bridge)
            valid = (
                batch["action_history_mask"][:, -1]
                & batch["action_history_mask"][:, -2]
            ).astype(jnp.float32)[:, None]
            p0 = (q0 - previous) / float(self.cfg.bridge.dt) * valid
        else:
            p0 = jnp.zeros_like(q0)
        return q0, p0

    def _sample_latent(
        self,
        mean: jnp.ndarray,
        log_variance: jnp.ndarray,
        deterministic: bool,
    ) -> jnp.ndarray:
        if deterministic:
            return mean
        noise = jax.random.normal(self.make_rng("latent"), mean.shape)
        return mean + jnp.exp(0.5 * log_variance) * noise

    def _auxiliary_actions(
        self,
        step_context: jnp.ndarray,
        position: jnp.ndarray,
        momentum: jnp.ndarray,
        latent_embedding: jnp.ndarray,
        current_quaternion: jnp.ndarray,
    ):
        quaternion_raw, gripper_logits = self.auxiliary(
            step_context,
            position,
            momentum,
            latent_embedding,
        )
        reference = current_quaternion[:, None, :]
        quaternion = canonicalize_quaternion(quaternion_raw, reference)
        return quaternion, gripper_logits

    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        *,
        train: bool = False,
        use_posterior: bool = True,
        deterministic_latent: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        context, context_mask, global_embedding = self.encoder(batch, train=train)
        step_context = self.decoder(context, context_mask, train=train)
        prior_mean, prior_log_variance = self.prior(global_embedding)

        future = batch.get("future_actions")
        if future is not None:
            future_normalized = future.astype(jnp.float32).at[..., :3].set(
                normalize_xyz(future[..., :3], self.cfg.bridge)
            )
            posterior_features = jnp.concatenate(
                [global_embedding, future_normalized.reshape(future.shape[0], -1)], axis=-1
            )
            posterior_mean, posterior_log_variance = self.posterior(posterior_features)
        else:
            posterior_mean, posterior_log_variance = prior_mean, prior_log_variance
        latent_mean = posterior_mean if use_posterior and future is not None else prior_mean
        latent_log_variance = (
            posterior_log_variance if use_posterior and future is not None else prior_log_variance
        )
        latent = self._sample_latent(
            latent_mean, latent_log_variance, bool(deterministic_latent)
        )
        latent_embedding = self.latent_embedding(latent)
        latent_steps = jnp.broadcast_to(
            latent_embedding[:, None],
            (
                latent_embedding.shape[0],
                int(self.cfg.horizon),
                latent_embedding.shape[-1],
            ),
        )
        attractor, stiffness, damping = self.contact_parameters(step_context)
        q0, p0 = self._initial_state(batch)

        free_positions = []
        free_momenta = []
        free_controls = []
        free_reference_forces = []
        q, p = q0, p0
        for step in range(int(self.cfg.horizon)):
            control = self.control(
                step_context[:, step], q, p, latent_steps[:, step]
            )
            reference_force = -stiffness[:, step] * (q - attractor[:, step])
            reference_force = reference_force - damping[:, step] * p
            p = p + float(self.cfg.bridge.dt) * (
                reference_force + float(self.cfg.bridge.sigma) * control
            )
            q = q + float(self.cfg.bridge.dt) * p
            free_positions.append(q)
            free_momenta.append(p)
            free_controls.append(control)
            free_reference_forces.append(reference_force)
        free_position = jnp.stack(free_positions, axis=1)
        free_momentum = jnp.stack(free_momenta, axis=1)
        free_control = jnp.stack(free_controls, axis=1)
        free_reference_force = jnp.stack(free_reference_forces, axis=1)

        current_quaternion = batch["obs_hist"][:, -1, 3:7]
        quaternion, gripper_logits = self._auxiliary_actions(
            step_context,
            free_position,
            free_momentum,
            latent_steps,
            current_quaternion,
        )
        actions = jnp.concatenate(
            [
                denormalize_xyz(free_position, self.cfg.bridge),
                quaternion,
                jax.nn.sigmoid(gripper_logits)[..., None],
            ],
            axis=-1,
        )
        output = {
            "actions": actions,
            "free_position": free_position,
            "free_momentum": free_momentum,
            "free_control": free_control,
            "free_reference_force": free_reference_force,
            "quaternion": quaternion,
            "gripper_logits": gripper_logits,
            "attractor": attractor,
            "stiffness": stiffness,
            "damping": damping,
            "step_context": step_context,
            "latent": latent,
            "prior_mean": prior_mean,
            "prior_log_variance": prior_log_variance,
            "posterior_mean": posterior_mean,
            "posterior_log_variance": posterior_log_variance,
        }

        if future is not None:
            target_position = normalize_xyz(future[..., :3], self.cfg.bridge)
            current_position = jnp.concatenate([q0[:, None], target_position[:, :-1]], axis=1)
            target_momentum = (
                target_position - current_position
            ) / float(self.cfg.bridge.dt)
            current_momentum = jnp.concatenate([p0[:, None], target_momentum[:, :-1]], axis=1)
            teacher_control = self.control(
                step_context,
                current_position,
                current_momentum,
                latent_steps,
            )
            teacher_reference_force = -stiffness * (current_position - attractor)
            teacher_reference_force = teacher_reference_force - damping * current_momentum
            teacher_momentum = current_momentum + float(self.cfg.bridge.dt) * (
                teacher_reference_force + float(self.cfg.bridge.sigma) * teacher_control
            )
            teacher_position = current_position + float(self.cfg.bridge.dt) * teacher_momentum
            teacher_quaternion, teacher_gripper_logits = self._auxiliary_actions(
                step_context,
                teacher_position,
                teacher_momentum,
                latent_steps,
                current_quaternion,
            )
            output.update(
                {
                    "target_position": target_position,
                    "target_momentum": target_momentum,
                    "teacher_position": teacher_position,
                    "teacher_momentum": teacher_momentum,
                    "teacher_control": teacher_control,
                    "teacher_reference_force": teacher_reference_force,
                    "teacher_quaternion": teacher_quaternion,
                    "teacher_gripper_logits": teacher_gripper_logits,
                }
            )
        return output


class DirectChunkBCPolicy(nn.Module):
    cfg: RLBenchPolicyConfig
    state_dim: int
    action_dim: int
    num_tasks: int
    num_task_variations: int

    @nn.compact
    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        *,
        train: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        context, context_mask, _ = RLBenchHistoryEncoder(
            self.cfg.encoder,
            self.num_tasks,
            self.num_task_variations,
            name="encoder",
        )(batch, train=train)
        step_context = ActionQueryDecoder(
            d_model=int(self.cfg.encoder.d_model),
            n_heads=int(self.cfg.encoder.n_heads),
            mlp_mult=int(self.cfg.encoder.mlp_mult),
            dropout=float(self.cfg.encoder.dropout),
            num_layers=int(self.cfg.decoder.num_layers),
            horizon=int(self.cfg.horizon),
            name="action_query_decoder",
        )(context, context_mask, train=train)
        raw = MLPHead(
            int(self.cfg.bridge.hidden_dim), 2, int(self.action_dim), name="action_head"
        )(step_context)
        xyz_normalized = jnp.tanh(raw[..., :3])
        current_quaternion = batch["obs_hist"][:, -1, 3:7][:, None]
        quaternion = canonicalize_quaternion(raw[..., 3:7], current_quaternion)
        gripper_logits = raw[..., 7]
        actions = jnp.concatenate(
            [
                denormalize_xyz(xyz_normalized, self.cfg.bridge),
                quaternion,
                jax.nn.sigmoid(gripper_logits)[..., None],
            ],
            axis=-1,
        )
        return {
            "actions": actions,
            "position": xyz_normalized,
            "quaternion": quaternion,
            "gripper_logits": gripper_logits,
            "step_context": step_context,
        }

