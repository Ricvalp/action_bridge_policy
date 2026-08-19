from __future__ import annotations

import pickle

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

from phi_rlbench.evaluation import PolicyInput

from action_bridge.config import to_plain_dict
from action_bridge.configs.rlbench_jax_direct_chunk_bc import get_config
from action_bridge.jax.eval.rlbench_online import (
    CANONICAL_RLBENCH_ACTION_LAYOUT,
    CANONICAL_RLBENCH_STATE_LAYOUT,
    OnlineEvaluationMetadata,
    build_online_batch,
)
from action_bridge.jax.eval.rlbench_online.jax_backend import (
    load_jax_policy_adapter,
)
from action_bridge.jax.models.config import policy_config_from_config
from action_bridge.jax.models.rlbench_policy import DirectChunkBCPolicy


def _small_config():
    config = get_config()
    config.data.point_count = 4
    config.data.obs_history = 2
    config.data.obs_stride = 1
    config.data.action_history = 2
    config.data.action_stride = 1
    config.data.action_offset = 1
    config.data.chunk_horizon = 3
    config.data.include_rgb = False
    config.data.include_mask_id = False
    config.encoder.d_model = 16
    config.encoder.n_heads = 2
    config.encoder.mlp_mult = 2
    config.encoder.frame_num_latents = 2
    config.encoder.frame_layers = 1
    config.encoder.supernodes = 2
    config.encoder.supernode_layers = 1
    config.encoder.history_layers = 1
    config.encoder.query_num_latents = 2
    config.encoder.query_layers = 1
    config.encoder.max_obs_history = 2
    config.encoder.max_action_history = 2
    config.encoder.use_rgb = False
    config.encoder.use_mask_id = False
    config.encoder.use_task_tokens = True
    config.decoder.num_layers = 1
    config.bridge.hidden_dim = 16
    return config


def _metadata() -> OnlineEvaluationMetadata:
    return OnlineEvaluationMetadata(
        task_to_id={"reach_target": 0},
        task_variation_to_id={"reach_target:0": 0},
        point_count=4,
        observation_history=2,
        observation_stride=1,
        action_history=2,
        action_stride=1,
        action_offset=1,
        action_horizon=3,
        action_representation="absolute",
        state_layout=CANONICAL_RLBENCH_STATE_LAYOUT,
        action_layout=CANONICAL_RLBENCH_ACTION_LAYOUT,
        training_cache_identity={
            "available": False,
            "manifest_sha256": None,
            "schema_name": None,
            "schema_version": None,
        },
        include_rgb=False,
        include_mask_id=False,
        policy_type="direct_chunk_bc",
    )


def _observation() -> PolicyInput:
    return PolicyInput(
        point_cloud_history=np.asarray(
            [
                [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5], [0.1, 0.1, 0.5]],
                [[0.0, 0.0, 0.6], [0.1, 0.0, 0.6], [0.0, 0.1, 0.6], [0.1, 0.1, 0.6]],
            ],
            dtype=np.float32,
        ),
        point_valid_history=np.ones((2, 4), dtype=np.bool_),
        state_history=np.asarray(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        observation_history_mask=np.ones(2, dtype=np.bool_),
        task_name="reach_target",
        task_id=0,
        variation_id=0,
        episode_step=1,
    )


def test_trusted_checkpoint_loader_matches_direct_model_inference(tmp_path) -> None:
    config = _small_config()
    metadata = _metadata()
    host_batch = build_online_batch(_observation(), metadata)
    device_batch = jax.tree_util.tree_map(jax.device_put, host_batch)
    model = DirectChunkBCPolicy(
        cfg=policy_config_from_config(config),
        state_dim=8,
        action_dim=8,
        num_tasks=1,
        num_task_variations=1,
    )
    variables = model.init(jax.random.PRNGKey(5), device_batch, train=False)
    expected = np.asarray(
        model.apply(
            {"params": variables["params"]},
            device_batch,
            train=False,
        )["actions"]
    )
    checkpoint = {
        "params": jax.device_get(variables["params"]),
        "config": to_plain_dict(config),
        "online_evaluation": metadata.to_json_dict(),
    }
    checkpoint_path = tmp_path / "checkpoint.pkl"
    checkpoint_path.write_bytes(
        pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
    )

    adapter = load_jax_policy_adapter(
        checkpoint_path,
        trusted_checkpoint=True,
    )
    adapter.reset(task_name="reach_target", variation_id=0, seed=17)
    actual = adapter.predict(_observation())

    np.testing.assert_allclose(actual.actions, expected[0], rtol=1e-6, atol=1e-6)
    assert actual.actions.shape == (3, 8)
    assert adapter.checkpoint_identifier.startswith("sha256:")
    assert actual.diagnostics["latent_l2"] == 0.0
