"""Losses for JAX RLBench bridge and direct chunk policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import optax

from action_bridge.jax.models.rlbench_policy import BridgeConfig, normalize_quaternion


@dataclass(frozen=True)
class LossConfig:
    xyz_weight: float = 1.0
    momentum_weight: float = 0.25
    unroll_weight: float = 0.1
    quaternion_weight: float = 1.0
    gripper_weight: float = 0.1
    beta_R: float = 0.001
    beta_z_start: float = 0.0
    beta_z_end: float = 0.001
    beta_z_warmup_steps: int = 20000
    free_nats: float = 0.05


def _masked_mean(value: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mask = mask.astype(jnp.float32)
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(jnp.ones_like(value) * mask), 1.0)


def _quaternion_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    prediction = normalize_quaternion(prediction)
    target = normalize_quaternion(target)
    dot = jnp.sum(prediction * target, axis=-1)
    return 1.0 - jnp.square(dot)


def _gaussian_kl(
    posterior_mean: jnp.ndarray,
    posterior_log_variance: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_log_variance: jnp.ndarray,
) -> jnp.ndarray:
    variance_ratio = jnp.exp(posterior_log_variance - prior_log_variance)
    mean_term = jnp.square(posterior_mean - prior_mean) / jnp.exp(prior_log_variance)
    return 0.5 * jnp.sum(
        prior_log_variance
        - posterior_log_variance
        + variance_ratio
        + mean_term
        - 1.0,
        axis=-1,
    )


def _gaussian_entropy(log_variance: jnp.ndarray) -> jnp.ndarray:
    constant = jnp.log(jnp.asarray(2.0 * jnp.pi * jnp.e, dtype=jnp.float32))
    return 0.5 * jnp.sum(constant + log_variance, axis=-1)


def beta_z_at_step(config: LossConfig, step: jnp.ndarray) -> jnp.ndarray:
    if int(config.beta_z_warmup_steps) <= 0:
        return jnp.asarray(config.beta_z_end, dtype=jnp.float32)
    fraction = jnp.clip(
        step.astype(jnp.float32) / float(config.beta_z_warmup_steps), 0.0, 1.0
    )
    return float(config.beta_z_start) + fraction * (
        float(config.beta_z_end) - float(config.beta_z_start)
    )


def bridge_loss(
    output: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    config: LossConfig,
    bridge_config: BridgeConfig,
    step: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    mask = batch["future_action_mask"].astype(jnp.float32)
    target = batch["future_actions"].astype(jnp.float32)
    xyz_loss = _masked_mean(
        jnp.square(output["teacher_position"] - output["target_position"]), mask
    )
    momentum_loss = _masked_mean(
        jnp.square(output["teacher_momentum"] - output["target_momentum"]), mask
    )
    unroll_loss = _masked_mean(
        jnp.square(output["free_position"] - output["target_position"]), mask
    )
    quaternion_loss = _masked_mean(
        _quaternion_loss(output["teacher_quaternion"], target[..., 3:7]), mask
    )
    gripper_loss = _masked_mean(
        optax.sigmoid_binary_cross_entropy(
            output["teacher_gripper_logits"], target[..., 7]
        ),
        mask,
    )
    path_kl = _masked_mean(
        0.5 * jnp.sum(jnp.square(output["teacher_control"]), axis=-1), mask
    )
    latent_kl_raw = _gaussian_kl(
        output["posterior_mean"],
        output["posterior_log_variance"],
        output["prior_mean"],
        output["prior_log_variance"],
    )
    latent_kl = jnp.mean(latent_kl_raw)
    latent_kl_loss = jnp.mean(
        jnp.maximum(latent_kl_raw, jnp.asarray(config.free_nats, dtype=jnp.float32))
    )
    beta_z = beta_z_at_step(config, step)
    loss = (
        float(config.xyz_weight) * xyz_loss
        + float(config.momentum_weight) * momentum_loss
        + float(config.unroll_weight) * unroll_loss
        + float(config.quaternion_weight) * quaternion_loss
        + float(config.gripper_weight) * gripper_loss
        + float(config.beta_R) * path_kl
        + beta_z * latent_kl_loss
    )
    raw_action_mse = _masked_mean(jnp.square(output["actions"] - target), mask)
    reference_norm = _masked_mean(
        jnp.linalg.norm(output["teacher_reference_force"], axis=-1), mask
    )
    control_acceleration_norm = _masked_mean(
        jnp.linalg.norm(
            float(bridge_config.sigma) * output["teacher_control"], axis=-1
        ),
        mask,
    )
    return {
        "loss": loss,
        "xyz_loss": xyz_loss,
        "momentum_loss": momentum_loss,
        "unroll_loss": unroll_loss,
        "quaternion_loss": quaternion_loss,
        "gripper_loss": gripper_loss,
        "path_kl": path_kl,
        "latent_kl": latent_kl,
        "latent_kl_loss": latent_kl_loss,
        "beta_z": beta_z,
        "action_mse": raw_action_mse,
        "prior_entropy": jnp.mean(_gaussian_entropy(output["prior_log_variance"])),
        "posterior_entropy": jnp.mean(
            _gaussian_entropy(output["posterior_log_variance"])
        ),
        "reference_force_norm": reference_norm,
        "control_acceleration_norm": control_acceleration_norm,
        "stiffness_mean": _masked_mean(output["stiffness"], mask),
        "damping_mean": _masked_mean(output["damping"], mask),
        "attractor_norm": _masked_mean(
            jnp.linalg.norm(output["attractor"], axis=-1), mask
        ),
    }


def direct_bc_loss(
    output: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    config: LossConfig,
    bridge_config: BridgeConfig = BridgeConfig(),
) -> Dict[str, jnp.ndarray]:
    mask = batch["future_action_mask"].astype(jnp.float32)
    target = batch["future_actions"].astype(jnp.float32)
    target_xyz = (
        target[..., :3] - jnp.asarray(bridge_config.xyz_center, dtype=jnp.float32)
    ) / jnp.asarray(bridge_config.xyz_scale, dtype=jnp.float32)
    xyz_loss = _masked_mean(jnp.square(output["position"] - target_xyz), mask)
    quaternion_loss = _masked_mean(
        _quaternion_loss(output["quaternion"], target[..., 3:7]), mask
    )
    gripper_loss = _masked_mean(
        optax.sigmoid_binary_cross_entropy(output["gripper_logits"], target[..., 7]),
        mask,
    )
    loss = (
        float(config.xyz_weight) * xyz_loss
        + float(config.quaternion_weight) * quaternion_loss
        + float(config.gripper_weight) * gripper_loss
    )
    return {
        "loss": loss,
        "xyz_loss": xyz_loss,
        "quaternion_loss": quaternion_loss,
        "gripper_loss": gripper_loss,
        "action_mse": _masked_mean(jnp.square(output["actions"] - target), mask),
    }
