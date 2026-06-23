"""Long shared-prefix path-Sinkhorn delayed modes.

This keeps the delayed top/bottom task ambiguous for more prediction times.
The shared prefix uses a slower horizontal approach, so increasing
`shared_prefix_steps` creates more shared moving states instead of mostly
waiting at the prefix target.
"""

from __future__ import annotations

from action_bridge.configs.latent_path_sinkhorn_delayed_modes import get_config as get_base_config


def get_config():
    cfg = get_base_config()

    cfg.data.shared_prefix_steps = 12
    cfg.data.shared_prefix_speed = 0.28

    return cfg
