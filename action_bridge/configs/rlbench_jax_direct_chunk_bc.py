"""JAX query-only direct action-chunk BC baseline for RLBench."""

from action_bridge.configs.rlbench_jax_contact_bridge import get_config as bridge_config


def get_config():
    config = bridge_config()
    config.run_id = "rlbench_jax_direct_chunk_bc"
    config.policy_type = "direct_chunk_bc"
    return config

