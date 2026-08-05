"""Convert project configs into immutable JAX model/loss configs."""

from __future__ import annotations

from action_bridge.jax.models.rlbench_encoder import EncoderConfig
from action_bridge.jax.models.rlbench_policy import (
    BridgeConfig,
    DecoderConfig,
    RLBenchPolicyConfig,
)
from action_bridge.jax.training.losses import LossConfig


def policy_config_from_config(config) -> RLBenchPolicyConfig:
    encoder = config.encoder
    bridge = config.bridge
    return RLBenchPolicyConfig(
        horizon=int(config.data.chunk_horizon),
        encoder=EncoderConfig(
            encoder_type=str(encoder.type),
            d_model=int(encoder.d_model),
            n_heads=int(encoder.n_heads),
            mlp_mult=int(encoder.mlp_mult),
            dropout=float(encoder.dropout),
            frame_num_latents=int(encoder.frame_num_latents),
            frame_layers=int(encoder.frame_layers),
            supernodes=int(encoder.supernodes),
            supernode_temperature=float(encoder.supernode_temperature),
            supernode_center_sampling=str(encoder.supernode_center_sampling),
            supernode_layers=int(encoder.supernode_layers),
            history_layers=int(encoder.history_layers),
            query_num_latents=int(encoder.query_num_latents),
            query_layers=int(encoder.query_layers),
            max_obs_history=int(encoder.max_obs_history),
            max_action_history=int(encoder.max_action_history),
            mask_id_vocab=int(encoder.mask_id_vocab),
            use_rgb=bool(encoder.use_rgb),
            use_mask_id=bool(encoder.use_mask_id),
            use_task_tokens=bool(encoder.use_task_tokens),
        ),
        decoder=DecoderConfig(num_layers=int(config.decoder.num_layers)),
        bridge=BridgeConfig(
            z_dim=int(bridge.z_dim),
            z_embed_dim=int(bridge.z_embed_dim),
            hidden_dim=int(bridge.hidden_dim),
            control_depth=int(bridge.control_depth),
            reference_depth=int(bridge.reference_depth),
            auxiliary_depth=int(bridge.auxiliary_depth),
            dt=float(bridge.dt),
            sigma=float(bridge.sigma),
            k_min=float(bridge.k_min),
            k_max=float(bridge.k_max),
            gamma_min=float(bridge.gamma_min),
            gamma_max=float(bridge.gamma_max),
            xyz_center=tuple(float(value) for value in bridge.xyz_center),
            xyz_scale=tuple(float(value) for value in bridge.xyz_scale),
        ),
    )


def loss_config_from_config(config) -> LossConfig:
    loss = config.loss
    return LossConfig(
        xyz_weight=float(loss.xyz_weight),
        momentum_weight=float(loss.momentum_weight),
        unroll_weight=float(loss.unroll_weight),
        quaternion_weight=float(loss.quaternion_weight),
        gripper_weight=float(loss.gripper_weight),
        beta_R=float(loss.beta_R),
        beta_z_start=float(loss.beta_z_start),
        beta_z_end=float(loss.beta_z_end),
        beta_z_warmup_steps=int(loss.beta_z_warmup_steps),
        free_nats=float(loss.free_nats),
    )

