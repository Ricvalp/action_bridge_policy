import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
import numpy as np

from action_bridge.jax.models.rlbench_encoder import EncoderConfig
from action_bridge.jax.models.rlbench_policy import (
    BridgeConfig,
    DecoderConfig,
    DirectChunkBCPolicy,
    RLBenchActionBridgePolicy,
    RLBenchPolicyConfig,
)
from action_bridge.jax.training.losses import LossConfig, bridge_loss, direct_bc_loss


def _config():
    return RLBenchPolicyConfig(
        horizon=4,
        encoder=EncoderConfig(
            encoder_type="supernode",
            d_model=32,
            n_heads=4,
            mlp_mult=2,
            frame_num_latents=4,
            frame_layers=1,
            supernodes=8,
            supernode_layers=1,
            history_layers=1,
            query_num_latents=8,
            query_layers=1,
            max_obs_history=4,
            max_action_history=4,
            use_rgb=True,
            use_task_tokens=True,
        ),
        decoder=DecoderConfig(num_layers=2),
        bridge=BridgeConfig(
            z_dim=4,
            z_embed_dim=16,
            hidden_dim=64,
            control_depth=2,
            reference_depth=1,
            auxiliary_depth=1,
            sigma=0.1,
        ),
    )


def _batch(batch_size=2):
    rng = np.random.default_rng(3)
    future = np.zeros((batch_size, 4, 8), dtype=np.float32)
    future[..., :3] = rng.uniform(
        [-0.4, -0.4, 0.75], [0.4, 0.4, 1.5], size=(batch_size, 4, 3)
    )
    future[..., 3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    future[..., 7] = rng.integers(0, 2, size=(batch_size, 4))
    state = np.zeros((batch_size, 2, 8), dtype=np.float32)
    state[..., :3] = future[:, :1, :3]
    state[..., 3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    actions = state.copy()
    return {
        "obs_hist": jnp.asarray(state),
        "point_cloud_hist": jnp.asarray(
            rng.uniform(-1, 1, size=(batch_size, 2, 16, 3)).astype(np.float32)
        ),
        "point_valid_hist": jnp.ones((batch_size, 2, 16), dtype=jnp.bool_),
        "rgb_hist": jnp.asarray(
            rng.uniform(0, 1, size=(batch_size, 2, 16, 3)).astype(np.float32)
        ),
        "act_hist": jnp.asarray(actions),
        "future_actions": jnp.asarray(future),
        "obs_history_mask": jnp.ones((batch_size, 2), dtype=jnp.bool_),
        "action_history_mask": jnp.ones((batch_size, 2), dtype=jnp.bool_),
        "future_action_mask": jnp.ones((batch_size, 4), dtype=jnp.bool_),
        "task_id": jnp.arange(batch_size, dtype=jnp.int32) % 2,
        "task_variation_id": jnp.arange(batch_size, dtype=jnp.int32) % 3,
        "variation_id": jnp.zeros((batch_size,), dtype=jnp.int32),
        "episode_id": jnp.zeros((batch_size,), dtype=jnp.int32),
        "time_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        "action_is_absolute": jnp.ones((batch_size,), dtype=jnp.bool_),
    }


def test_contact_bridge_forward_loss_and_gradients_are_finite():
    config = _config()
    batch = _batch()
    model = RLBenchActionBridgePolicy(config, 8, 8, 2, 3)
    rng = jax.random.PRNGKey(4)
    variables = model.init(
        {"params": rng, "latent": rng, "dropout": rng},
        batch,
        train=True,
        use_posterior=True,
        deterministic_latent=False,
    )

    def objective(params):
        output = model.apply(
            {"params": params},
            batch,
            train=True,
            use_posterior=True,
            deterministic_latent=False,
            rngs={"latent": rng, "dropout": rng},
        )
        metrics = bridge_loss(
            output, batch, LossConfig(), config.bridge, jnp.asarray(10)
        )
        return metrics["loss"], (output, metrics)

    (loss, (output, metrics)), gradients = jax.value_and_grad(
        objective, has_aux=True
    )(variables["params"])
    assert output["actions"].shape == (2, 4, 8)
    assert output["attractor"].shape == (2, 4, 3)
    assert output["teacher_control"].shape == (2, 4, 3)
    assert np.isfinite(np.asarray(loss))
    assert np.isfinite(np.asarray(metrics["path_kl"]))
    assert all(
        np.isfinite(np.asarray(value)).all()
        for value in jax.tree_util.tree_leaves(gradients)
    )
    quaternion_norm = np.linalg.norm(np.asarray(output["quaternion"]), axis=-1)
    assert np.allclose(quaternion_norm, 1.0, atol=1e-5)


def test_direct_chunk_bc_forward_and_loss_are_finite():
    config = _config()
    batch = _batch()
    model = DirectChunkBCPolicy(config, 8, 8, 2, 3)
    rng = jax.random.PRNGKey(7)
    variables = model.init({"params": rng, "dropout": rng}, batch, train=True)
    output = model.apply({"params": variables["params"]}, batch, train=False)
    metrics = direct_bc_loss(output, batch, LossConfig())
    assert output["actions"].shape == (2, 4, 8)
    assert np.isfinite(np.asarray(metrics["loss"]))
