"""Push-T entrypoint placeholder for the toy-first pilot."""

from __future__ import annotations

import argparse

from action_bridge.config import apply_overrides, load_config
from action_bridge.data.pusht_adapter import PushTLowDimDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="pusht_lowdim_continuous")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config_name), args.overrides)
    data_cfg = config.get("data", {})
    PushTLowDimDataset(
        dataset_path=data_cfg.get("dataset_path"),
        backend=data_cfg.get("backend", "auto"),
        obs_history=int(config.get("obs_history", 2)),
        action_history=int(config.get("action_history", 2)),
        chunk_horizon=int(config.get("chunk_horizon", 16)),
    )


if __name__ == "__main__":
    main()
