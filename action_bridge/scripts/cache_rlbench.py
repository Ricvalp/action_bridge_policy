"""Command-line entrypoint for building the RLBench HDF5 cache."""

from __future__ import annotations

import argparse

from action_bridge.data.rlbench_cache_builder import (
    DEFAULT_LOW_DIM_FIELDS,
    DEFAULT_WORKSPACE_BOUNDS,
    convert_rlbench_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw RLBench point-cloud demonstrations to flexible HDF5 episodes."
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tasks", nargs="*", default=())
    parser.add_argument("--start-variation", type=int, default=0)
    parser.add_argument("--num-variations", type=int, default=-1)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes-per-variation", type=int)
    parser.add_argument("--include-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-mask-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--workspace-bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=None,
        help="Defaults to -1 1 -1 1 0 2.5. Ignored with --no-workspace-filter.",
    )
    parser.add_argument("--no-workspace-filter", action="store_true")
    parser.add_argument("--low-dim-fields", nargs="+", default=DEFAULT_LOW_DIM_FIELDS)
    parser.add_argument("--allow-length-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_workspace_filter:
        bounds = None
    elif args.workspace_bounds is None:
        bounds = DEFAULT_WORKSPACE_BOUNDS
    else:
        values = args.workspace_bounds
        bounds = ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))
    manifest = convert_rlbench_dataset(
        args.raw_root,
        args.cache_root,
        tasks=args.tasks,
        start_variation=args.start_variation,
        num_variations=args.num_variations,
        num_points=args.num_points,
        compression=args.compression,
        seed=args.seed,
        include_rgb=args.include_rgb,
        include_mask_id=args.include_mask_id,
        ignore_background=args.ignore_background,
        workspace_bounds=bounds,
        low_dim_fields=args.low_dim_fields,
        strict_lengths=not args.allow_length_mismatch,
        max_episodes_per_variation=args.max_episodes_per_variation,
        overwrite=args.overwrite,
    )
    print(f"RLBench cache manifest: {manifest}")


if __name__ == "__main__":
    main()
