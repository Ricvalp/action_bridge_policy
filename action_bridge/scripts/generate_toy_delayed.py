"""Generate a delayed-branch toy dataset artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from action_bridge.data.toy_obstacle import save_delayed_branch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-contexts", type=int, default=32)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-len", type=int, default=64)
    parser.add_argument("--chunk-horizon", type=int, default=16)
    args = parser.parse_args()
    path = save_delayed_branch(
        Path(args.out),
        num_contexts=args.num_contexts,
        seed=args.seed,
        trajectory_len=args.trajectory_len,
        chunk_horizon=args.chunk_horizon,
    )
    print(path)


if __name__ == "__main__":
    main()
