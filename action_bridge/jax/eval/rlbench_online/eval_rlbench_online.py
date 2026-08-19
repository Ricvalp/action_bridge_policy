"""Evaluate an Action Bridge checkpoint through phi-rlbench closed loop."""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from phi_rlbench.evaluation import (
    EvaluationArtifacts,
    EvaluationConfig,
    EvaluationRunner,
    VideoConfig,
)
from phi_rlbench.observations import (
    HARDENED_EVALUATION_PROFILE,
    HARDENED_LIVE_ADAPTER_PROFILE,
    LEGACY_LIVE_ADAPTER_PROFILE,
    LEGACY_PREPROCESSING_PROFILE,
    LiveObservationPreprocessor,
    ProcessedFrame,
)
from phi_rlbench.provenance import collect_provenance
from phi_rlbench.sim import (
    ActionProfile,
    ObservationProfile,
    RLBenchActionValidator,
    RLBenchRuntime,
    RuntimeProfile,
)

from action_bridge.jax.eval.rlbench_online.jax_backend import (
    load_jax_policy_adapter,
)
from action_bridge.jax.eval.rlbench_online.metadata import OnlineEvaluationMetadata
from action_bridge.jax.eval.rlbench_online.policy_provenance import (
    collect_policy_source_identity,
)


class _LegacyContinuousPreprocessingStream:
    """Reproduce the audited online evaluator's one continuous sampling RNG."""

    def __init__(
        self, delegate: LiveObservationPreprocessor, *, base_seed: int
    ) -> None:
        self._delegate = delegate
        self._rng = np.random.default_rng(int(base_seed) + 11)

    def process(
        self,
        observation: object,
        *,
        rng: np.random.Generator,
    ) -> ProcessedFrame:
        del rng
        return self._delegate.process(observation, rng=self._rng)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m action_bridge.jax.eval.rlbench_online",
        description="Evaluate one trusted Action Bridge checkpoint through phi-rlbench.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--online-metadata",
        type=Path,
        help="Explicit metadata for an old checkpoint without task vocabularies.",
    )
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="Acknowledge that the local pickle checkpoint is trusted code/data.",
    )
    latent = parser.add_mutually_exclusive_group()
    latent.add_argument(
        "--deterministic-latent",
        dest="deterministic_latent",
        action="store_true",
        help="Use the prior mean for Action Bridge inference.",
    )
    latent.add_argument(
        "--stochastic-latent",
        dest="deterministic_latent",
        action="store_false",
        help="Sample the Action Bridge prior using the episode policy RNG.",
    )
    parser.set_defaults(deterministic_latent=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--actions-per-plan", type=int, default=1)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Re-raise the first episode failure after recording it and cleaning up.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--renderer", choices=("opengl", "opengl3"), default="opengl")
    parser.add_argument(
        "--preprocessing-profile",
        choices=("legacy", "hardened"),
        default="legacy",
        help="Use legacy checkpoint-parity preprocessing or explicit hardened validation.",
    )
    parser.add_argument("--collision-checking", action="store_true")
    action_bounds = parser.add_mutually_exclusive_group(required=True)
    action_bounds.add_argument(
        "--reject-out-of-bounds-actions",
        dest="reject_out_of_bounds_actions",
        action="store_true",
        help="Reject xyz outside --action-position-bounds.",
    )
    action_bounds.add_argument(
        "--allow-out-of-bounds-actions",
        dest="reject_out_of_bounds_actions",
        action="store_false",
        help="Disable the optional xyz bounds check.",
    )
    parser.add_argument(
        "--action-position-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=(-1.0, 1.0, -1.0, 1.0, 0.0, 2.5),
    )
    parser.add_argument(
        "--max-position-delta",
        type=float,
        help="Optional rejection threshold in metres; actions are never clipped.",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="Launch a visible window.")
    display.add_argument("--headless", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-view", default="front_rgb")
    parser.add_argument("--video-frame-stride", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument(
        "--discard-failed-video",
        action="store_true",
        help="Retain videos only for successful episodes.",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--run-dir", type=Path)
    output.add_argument("--output-root", type=Path)
    return parser


def _new_run_directory(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve(strict=False)
    assert args.output_root is not None
    root = Path(args.output_root).expanduser().resolve(strict=False)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"action-bridge-{stamp}-{uuid.uuid4().hex[:8]}"


def _error_payload(stage: str, error: BaseException) -> dict[str, object]:
    error_type = type(error)
    return {
        "status": "failed",
        "stage": stage,
        "error_type": f"{error_type.__module__}.{error_type.__qualname__}",
        "message": str(error),
    }


def _emit_error(stage: str, error: BaseException) -> None:
    sys.stderr.write(json.dumps(_error_payload(stage, error), sort_keys=True) + "\n")


def _close_after_setup_failure(runtime: RLBenchRuntime, *, stage: str) -> bool:
    """Try cleanup twice before declaring an early setup failure unclean."""

    for attempt in range(1, 3):
        try:
            runtime.close()
        except KeyboardInterrupt:
            if attempt == 2:
                raise
        except Exception as error:  # noqa: BLE001
            _emit_error(f"{stage}_cleanup_attempt_{attempt}", error)
        else:
            return True
    return False


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.trusted_checkpoint:
        parser.error(
            "--trusted-checkpoint is required because existing Action Bridge checkpoints "
            "use pickle and may execute code while loading"
        )
    if args.variation < 0:
        parser.error("--variation must be non-negative")
    if args.episodes < 1 or args.max_steps < 1 or args.actions_per_plan < 1:
        parser.error("--episodes, --max-steps, and --actions-per-plan must be positive")
    if args.image_size < 1:
        parser.error("--image-size must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must fit the uint32 range")
    if args.seed + args.episodes - 1 > 2**32 - 1:
        parser.error("--seed + --episodes - 1 must fit the uint32 range")
    if args.max_position_delta is not None and (
        not math.isfinite(args.max_position_delta) or args.max_position_delta <= 0.0
    ):
        parser.error("--max-position-delta must be finite and positive")
    bounds = tuple(float(value) for value in args.action_position_bounds)
    if not all(math.isfinite(value) for value in bounds):
        parser.error("--action-position-bounds must be finite")
    if any(bounds[index] > bounds[index + 1] for index in (0, 2, 4)):
        parser.error("--action-position-bounds must contain ordered min/max pairs")


def _action_position_bounds(
    args: argparse.Namespace,
) -> tuple[tuple[float, float], ...]:
    values = tuple(float(value) for value in args.action_position_bounds)
    return (
        (values[0], values[1]),
        (values[2], values[3]),
        (values[4], values[5]),
    )


def _collect_run_provenance(
    metadata: OnlineEvaluationMetadata,
    *,
    preprocessing_config: dict[str, object],
) -> dict[str, Any]:
    """Attach the checkpoint's explicit training-cache identity to provenance."""

    record = collect_provenance(preprocessing_config=preprocessing_config)
    identity = metadata.training_cache_identity
    if bool(identity["available"]):
        manifest: dict[str, object] = {
            "available": True,
            "sha256": identity["manifest_sha256"],
            "schema_name": identity["schema_name"],
            "schema_version": identity["schema_version"],
            "bundle_sha256": identity["cache_bundle_sha256"],
            "preprocessing_sidecar_sha256": identity["preprocessing_sidecar_sha256"],
        }
    else:
        manifest = {"available": False, "sha256": None}
    record["cache"] = {"manifest": manifest}
    return record


def main(argv: list[str] | None = None) -> int:
    """Load a trusted checkpoint and run one explicit task/variation evaluation."""

    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        video_config = VideoConfig(
            enabled=bool(args.record_video),
            source_view=args.video_view,
            frame_stride=args.video_frame_stride,
            frames_per_second=args.video_fps,
            record_failed_episodes=not bool(args.discard_failed_video),
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    try:
        adapter = load_jax_policy_adapter(
            args.checkpoint,
            online_metadata_path=args.online_metadata,
            trusted_checkpoint=True,
            deterministic_latent=args.deterministic_latent,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001
        _emit_error("checkpoint_load", error)
        return 2

    metadata = adapter.metadata
    preprocessing_base = (
        LEGACY_PREPROCESSING_PROFILE
        if args.preprocessing_profile == "legacy"
        else HARDENED_EVALUATION_PROFILE
    )
    live_adapter_base = (
        LEGACY_LIVE_ADAPTER_PROFILE
        if args.preprocessing_profile == "legacy"
        else HARDENED_LIVE_ADAPTER_PROFILE
    )
    preprocessing_profile = replace(
        preprocessing_base,
        num_points=metadata.point_count,
        include_rgb=metadata.include_rgb,
        include_mask_id=metadata.include_mask_id,
    )
    live_adapter_profile = replace(
        live_adapter_base,
        include_rgb=metadata.include_rgb,
    )
    observation_profile = ObservationProfile(
        image_size=(args.image_size, args.image_size),
        renderer=args.renderer,
        rgb=metadata.include_rgb or video_config.enabled,
    )
    action_profile = ActionProfile(
        collision_checking=bool(args.collision_checking),
        workspace_bounds=(
            _action_position_bounds(args) if args.reject_out_of_bounds_actions else None
        ),
        max_position_delta=args.max_position_delta,
    )
    runtime_profile = RuntimeProfile(
        observation=observation_profile,
        action=action_profile,
        headless=bool(args.headless),
    )
    if args.task not in metadata.task_to_id:
        parser.error(f"task {args.task!r} is absent from checkpoint metadata")
    variation_key = f"{args.task}:{args.variation}"
    if variation_key not in metadata.task_variation_to_id:
        parser.error(
            f"task variation {variation_key!r} is absent from checkpoint metadata"
        )
    try:
        evaluation_config = EvaluationConfig(
            task_name=args.task,
            variation_id=args.variation,
            episodes=args.episodes,
            max_simulator_steps=args.max_steps,
            observation_history=metadata.observation_history,
            observation_stride=metadata.observation_stride,
            actions_per_plan=args.actions_per_plan,
            expected_action_horizon=metadata.action_horizon,
            base_seed=args.seed,
            task_id=metadata.task_to_id[args.task],
            observation_profile=observation_profile.name,
            action_profile=action_profile.name,
            checkpoint_identifier=adapter.checkpoint_identifier,
            fail_fast=bool(args.fail_fast),
            close_runtime=True,
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    live_preprocessor = LiveObservationPreprocessor(
        preprocessing_profile,
        adapter_config=live_adapter_profile,
    )
    preprocessing_rng_recipe = "phi_episode_seed_sequence_v1"
    runner_preprocessor: Any = live_preprocessor
    if args.preprocessing_profile == "legacy":
        preprocessing_rng_recipe = "action_bridge_continuous_seed_plus_11_v1"
        runner_preprocessor = _LegacyContinuousPreprocessingStream(
            live_preprocessor,
            base_seed=args.seed,
        )
    try:
        run_directory = _new_run_directory(args)
        if run_directory.exists() or run_directory.is_symlink():
            raise FileExistsError(
                "the selected run directory already exists; artifacts never overwrite"
            )
        resolved_config: dict[str, Any] = {
            "command": "action_bridge_rlbench_online",
            "task_name": args.task,
            "variation_id": args.variation,
            "episodes": args.episodes,
            "max_simulator_steps": args.max_steps,
            "actions_per_plan": args.actions_per_plan,
            "fail_fast": bool(args.fail_fast),
            "base_seed": args.seed,
            "checkpoint_identifier": adapter.checkpoint_identifier,
            "policy_source": collect_policy_source_identity(),
            "online_evaluation": metadata.to_json_dict(),
            "latent_sampling": {
                "deterministic": metadata.deterministic_latent,
                "source": (
                    "checkpoint_or_metadata_default"
                    if args.deterministic_latent is None
                    else "command_line_override"
                ),
            },
            "runtime": asdict(runtime_profile),
            "action_validation": {
                "reject_out_of_bounds_actions": bool(args.reject_out_of_bounds_actions),
                "action_position_bounds": [
                    list(axis) for axis in _action_position_bounds(args)
                ],
                "max_position_delta": args.max_position_delta,
                "normalize_quaternion": True,
                "discretize_gripper": True,
            },
            "preprocessing": preprocessing_profile.to_dict(),
            "live_adapter": asdict(live_adapter_profile),
            "preprocessing_rng_recipe": preprocessing_rng_recipe,
            "video": video_config.to_json_dict(),
        }
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001
        _emit_error("configuration", error)
        return 2

    runtime = RLBenchRuntime(runtime_profile)
    runner = EvaluationRunner(
        runtime=runtime,
        preprocessor=runner_preprocessor,
        policy=adapter,
        action_validator=RLBenchActionValidator(action_profile),
        video_config=video_config,
        config=evaluation_config,
    )
    try:
        runtime.launch()
    except KeyboardInterrupt:
        _close_after_setup_failure(runtime, stage="launch")
        return 130
    except Exception as error:  # noqa: BLE001
        _emit_error("launch", error)
        _close_after_setup_failure(runtime, stage="launch")
        return 1

    try:
        artifacts = EvaluationArtifacts.create(
            run_directory,
            resolved_config=resolved_config,
            provenance=_collect_run_provenance(
                metadata,
                preprocessing_config=preprocessing_profile.to_dict(),
            ),
        )
    except KeyboardInterrupt:
        _close_after_setup_failure(runtime, stage="artifact_setup")
        return 130
    except Exception as error:  # noqa: BLE001
        _emit_error("artifact_setup", error)
        _close_after_setup_failure(runtime, stage="artifact_setup")
        return 1

    try:
        result = runner.run(artifacts=artifacts)
    except KeyboardInterrupt as error:
        _emit_error("evaluation", error)
        return 130
    except Exception as error:
        if args.fail_fast:
            raise
        _emit_error("evaluation", error)
        return 1

    output = {
        "status": "completed",
        "run_directory": str(run_directory),
        "checkpoint_identifier": adapter.checkpoint_identifier,
        "attempted_episodes": result.summary.attempted_episodes,
        "successful_episodes": result.summary.successful_episodes,
        "exception_records": result.summary.exception_records,
        "runtime_cleanup_succeeded": result.summary.runtime_cleanup_succeeded,
        "recorded_videos": sum(
            episode.video_path is not None for episode in result.episodes
        ),
    }
    sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
    return 0 if result.summary.exception_records == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
