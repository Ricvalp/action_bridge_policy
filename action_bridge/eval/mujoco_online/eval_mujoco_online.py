"""Evaluate a trusted Action Bridge checkpoint through ``phi-mujoco``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path

from action_bridge.eval.mujoco_online.adapter import ActionBridgeMujocoPolicyAdapter
from action_bridge.eval.mujoco_online.torch_backend import load_torch_policy_adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m action_bridge.eval.mujoco_online",
        description="Evaluate one trusted Action Bridge checkpoint through phi-mujoco.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--online-metadata", type=Path)
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="acknowledge that the local PyTorch pickle checkpoint is trusted",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--actions-per-plan", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="return non-zero unless every attempted episode succeeds",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--discard-failed-video", action="store_true")
    parser.add_argument("--video-frame-stride", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_preload_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.trusted_checkpoint:
        parser.error(
            "--trusted-checkpoint is required because PyTorch checkpoints use pickle and "
            "may execute code while loading"
        )
    if args.episodes < 1 or args.max_steps < 1 or args.actions_per_plan < 1:
        parser.error("--episodes, --max-steps, and --actions-per-plan must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must fit the uint32 range")
    if args.seed + args.episodes - 1 > 2**32 - 1:
        parser.error("--seed + --episodes - 1 must fit the uint32 range")
    if args.video_frame_stride < 1 or not math.isfinite(args.video_fps) or args.video_fps <= 0.0:
        parser.error("video frame stride and FPS must be positive")
    if args.render_width < 1 or args.render_height < 1:
        parser.error("render dimensions must be positive")
    if not args.run_dir.parent.is_dir():
        parser.error(f"--run-dir parent must already exist: {args.run_dir.parent}")


def _emit_error(stage: str, error: BaseException) -> None:
    record = {
        "stage": stage,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }
    print(json.dumps(record, sort_keys=True, allow_nan=False), file=sys.stderr)


def _action_bridge_version() -> str | None:
    try:
        return importlib_metadata.version("action-bridge-policy")
    except importlib_metadata.PackageNotFoundError:
        return None


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_native(
    args: argparse.Namespace,
    adapter: ActionBridgeMujocoPolicyAdapter,
) -> int:
    """Import and launch the native runtime only after checkpoint validation."""

    from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner, VideoConfig
    from phi_mujoco.provenance import (
        EXPECTED_MODEL_SHA256,
        collect_model_identity,
        collect_provenance,
        selected_gl_backend,
    )
    from phi_mujoco.sim import PlanarReachConfig, PlanarReachRuntime
    from phi_mujoco.tasks.planar_reach import DEFAULT_MAX_EPISODE_STEPS

    if args.max_steps > DEFAULT_MAX_EPISODE_STEPS:
        raise ValueError(
            f"--max-steps must not exceed the v1 task limit {DEFAULT_MAX_EPISODE_STEPS}."
        )
    render_backend = selected_gl_backend(os.environ)
    if args.record_video and render_backend not in {"egl", "osmesa", "glfw"}:
        raise ValueError(
            "--record-video requires an explicit MuJoCo rendering backend; set "
            "MUJOCO_GL=egl|osmesa|glfw before starting Python."
        )

    runtime_config = PlanarReachConfig(
        render_width=args.render_width,
        render_height=args.render_height,
    )
    evaluation = EvaluationConfig(
        task_name=adapter.metadata.task_name,
        variation_id=adapter.metadata.variation_id,
        episodes=args.episodes,
        max_simulator_steps=args.max_steps,
        actions_per_plan=args.actions_per_plan,
        base_seed=args.seed,
        observation_profile=adapter.metadata.observation_profile,
        action_profile=adapter.metadata.action_profile,
        checkpoint_identifier=adapter.checkpoint_identifier,
    )
    video = VideoConfig(
        enabled=bool(args.record_video),
        frame_stride=args.video_frame_stride,
        frames_per_second=args.video_fps,
        record_failed_episodes=not bool(args.discard_failed_video),
    )
    model_identity = collect_model_identity(expected_sha256=EXPECTED_MODEL_SHA256)
    if not model_identity.get("matches_expected"):
        raise ValueError("the packaged planar_reach model does not match its expected SHA-256")
    preprocessing = {
        "normalization": adapter.metadata.normalization.to_json_dict(),
        "observation_history": adapter.metadata.observation_history,
        "action_history": adapter.metadata.action_history,
        "action_horizon": adapter.metadata.action_horizon,
        "actions_per_plan": adapter.metadata.actions_per_plan,
    }
    provenance = collect_provenance()
    provenance["cache"] = {
        "manifest": {
            "available": True,
            "sha256": adapter.metadata.collection_identity.manifest_sha256,
            "schema_name": adapter.metadata.collection_identity.schema_name,
            "schema_version": adapter.metadata.collection_identity.schema_version,
            "valid_json": True,
        }
    }
    provenance["preprocessing"] = {
        "algorithm": "sha256",
        "sha256": _canonical_sha256(preprocessing),
    }
    resolved_config = {
        "evaluation": evaluation.to_json_dict(),
        "video": video.to_json_dict(),
        "task": runtime_config.to_dict(),
        "model": model_identity,
        "render_backend": render_backend,
        "policy": {
            "adapter": "action_bridge_mujoco",
            "package_version": _action_bridge_version(),
            "checkpoint_identifier": adapter.checkpoint_identifier,
            "online_evaluation": adapter.metadata.to_json_dict(),
            "preprocessing": preprocessing,
        },
    }
    runtime = PlanarReachRuntime(runtime_config)
    result = EvaluationRunner(
        runtime=runtime,
        policy=adapter,
        config=evaluation,
        video_config=video,
        frame_capture=(lambda _observation: runtime.render()) if args.record_video else None,
    ).run(
        run_directory=args.run_dir,
        provenance=provenance,
        resolved_config=resolved_config,
    )

    summary = result.summary.to_json_dict()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"run directory: {result.run_directory}")
        print(
            f"episodes: {summary['successful_episodes']}/{summary['attempted_episodes']} successful"
        )
        print(f"simulator steps: {summary['total_simulator_steps']}")
        print(f"runtime cleanup succeeded: {summary['runtime_cleanup_succeeded']}")

    all_succeeded = summary["successful_episodes"] == summary["attempted_episodes"]
    videos_complete = not args.record_video or all(
        episode.video_path is not None or (args.discard_failed_video and not episode.success)
        for episode in result.episodes
    )
    accepted = bool(
        summary["runtime_cleanup_succeeded"]
        and not result.exceptions
        and videos_complete
        and (all_succeeded or not args.require_success)
    )
    return 0 if accepted else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_preload_args(parser, args)
    try:
        adapter = load_torch_policy_adapter(
            args.checkpoint,
            online_metadata_path=args.online_metadata,
            trusted_checkpoint=True,
            device=args.device,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        _emit_error("checkpoint_load", error)
        return 2
    if args.actions_per_plan != adapter.metadata.actions_per_plan:
        parser.error(
            f"--actions-per-plan={args.actions_per_plan} disagrees with checkpoint "
            f"metadata value {adapter.metadata.actions_per_plan}"
        )
    try:
        return _run_native(args, adapter)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        _emit_error("native_evaluation", error)
        return 3


__all__ = ["main"]
