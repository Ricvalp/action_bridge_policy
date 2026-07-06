"""Generate an annular toy dataset artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from action_bridge.data.toy_annular import generate_annular_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-contexts", type=int, default=32)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-len", type=int, default=64)
    args = parser.parse_args()
    arrays = generate_annular_arrays(num_contexts=args.num_contexts, seed=args.seed, trajectory_len=args.trajectory_len)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(arrays, path)
    print(path)


if __name__ == "__main__":
    main()
