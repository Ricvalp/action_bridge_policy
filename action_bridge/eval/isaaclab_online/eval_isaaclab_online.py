"""Evaluate a trusted Action Bridge checkpoint through native PHI Isaac Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
import traceback
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Never

from action_bridge.eval.isaaclab_online.contracts import (
    ACTION_PROFILE,
    COLLECTION_SCHEMA_NAME,
    COLLECTION_SCHEMA_VERSION,
    CONTROL_TIMESTEP_S,
    OBSERVATION_PROFILE,
    TASK_ID,
    VARIATION_ID,
)
from action_bridge.eval.isaaclab_online.torch_backend import load_torch_policy_adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m action_bridge.eval.isaaclab_online",
        description=(
            "Evaluate one trusted Action Bridge checkpoint through phi-isaaclab."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--online-metadata", type=Path)
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        required=True,
        help="exact processed-cache manifest bound into the trusted checkpoint",
    )
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="acknowledge that the local PyTorch pickle checkpoint is trusted",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isaaclab-source-root", type=Path, required=True)
    parser.add_argument(
        "--writable-root",
        type=Path,
        required=True,
        help="existing authorized root containing run and Kit/cache directories",
    )
    parser.add_argument(
        "--portable-root",
        type=Path,
        help="writable Kit/cache root; defaults below --writable-root",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument(
        "--require-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="return non-zero unless every requested episode succeeds",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_preload_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not args.trusted_checkpoint:
        parser.error(
            "--trusted-checkpoint is required because PyTorch checkpoints use "
            "pickle and "
            "may execute code while loading"
        )
    if args.episodes < 1 or args.num_envs < 1 or args.max_steps < 1:
        parser.error("--episodes, --num-envs, and --max-steps must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must fit the uint32 range")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0.0:
        parser.error("--video-fps must be finite and positive")
    writable_root = args.writable_root.expanduser().resolve()
    if not writable_root.is_dir():
        parser.error(f"--writable-root must be an existing directory: {writable_root}")
    run_directory = args.run_dir.expanduser().resolve()
    if writable_root not in run_directory.parents:
        parser.error("--run-dir must be a child of --writable-root")
    if not run_directory.parent.is_dir():
        parser.error(f"--run-dir parent must already exist: {run_directory.parent}")
    if run_directory.exists():
        parser.error(f"--run-dir must not already exist: {run_directory}")
    if args.portable_root is not None:
        portable_root = args.portable_root.expanduser().resolve()
        if writable_root not in portable_root.parents:
            parser.error("--portable-root must be a child of --writable-root")


def _emit_error(stage: str, error: BaseException) -> None:
    record = {
        "stage": stage,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }
    print(json.dumps(record, sort_keys=True, allow_nan=False), file=sys.stderr)


def _validate_native_device(device: str) -> None:
    import torch

    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("native Isaac inference device must be cpu or cuda:N")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested native inference device {requested} is unavailable; "
            "CPU fallback "
            "would violate the Isaac Lab batched tensor contract"
        )
    if requested.type == "cuda" and requested.index is None:
        raise ValueError(
            "native CUDA inference requires an explicit device index, "
            "for example cuda:0"
        )
    if requested.type == "cuda" and requested.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested native inference device {requested} is unavailable; "
            f"only {torch.cuda.device_count()} CUDA device(s) are visible"
        )


def _action_bridge_version() -> str | None:
    try:
        return importlib_metadata.version("action-bridge-policy")
    except importlib_metadata.PackageNotFoundError:
        return None


def _reject_nonstandard_constant(value: str) -> Never:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def _cache_manifest_identity(
    path_value: Path, *, expected_sha256: str
) -> dict[str, object]:
    path = path_value.expanduser()
    try:
        file_info = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"collection manifest does not exist: {path}") from exc
    if not stat.S_ISREG(file_info.st_mode):
        raise ValueError(
            f"collection manifest must be a regular file, not a link: {path}"
        )
    if file_info.st_size > 64 * 1024 * 1024:
        raise ValueError("collection manifest exceeds the 64 MiB safety bound")
    payload = path.read_bytes()
    if len(payload) != file_info.st_size:
        raise ValueError("collection manifest changed while it was read")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "collection manifest SHA-256 disagrees with checkpoint metadata: "
            f"expected {expected_sha256}, computed {digest}"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonstandard_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("collection manifest must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("collection manifest must contain one JSON object")
    if value.get("schema_name") != COLLECTION_SCHEMA_NAME:
        raise ValueError("collection manifest has the wrong schema_name")
    if value.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise ValueError("collection manifest has the wrong schema_version")
    if value.get("task_name") != TASK_ID or value.get("variation_id") != VARIATION_ID:
        raise ValueError(
            "collection manifest task identity disagrees with the Isaac Lab contract"
        )
    if value.get("observation_profile") != OBSERVATION_PROFILE:
        raise ValueError(
            "collection manifest observation_profile disagrees with the Isaac Lab contract"
        )
    if value.get("action_profile") != ACTION_PROFILE:
        raise ValueError(
            "collection manifest action_profile disagrees with the Isaac Lab contract"
        )
    episodes = value.get("episode_count")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0:
        raise ValueError(
            "collection manifest episode_count must be a non-negative integer"
        )
    return {
        "available": True,
        "sha256": digest,
        "size_bytes": len(payload),
        "valid_json": True,
        "schema_name": COLLECTION_SCHEMA_NAME,
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "num_tasks": 1,
        "num_variations": 1,
        "num_episodes": episodes,
    }


def _preprocessing_identity(online_metadata: Mapping[str, object]) -> dict[str, object]:
    payload = json.dumps(
        {
            "schema_name": "phi.isaaclab.online_preprocessing",
            "schema_version": 1,
            "online_evaluation": dict(online_metadata),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {"algorithm": "sha256", "sha256": hashlib.sha256(payload).hexdigest()}


def _shared_provenance_identities(
    adapter: Any, manifest_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    metadata_value = adapter.metadata.to_json_dict()
    if not isinstance(metadata_value, Mapping):
        raise TypeError("online checkpoint metadata must serialize to an object")
    collection = metadata_value.get("collection_identity")
    if not isinstance(collection, Mapping):
        raise TypeError("online checkpoint metadata lacks collection_identity")
    expected = collection.get("manifest_sha256")
    if not isinstance(expected, str):
        raise TypeError("checkpoint collection manifest identity is invalid")
    return (
        _cache_manifest_identity(manifest_path, expected_sha256=expected),
        _preprocessing_identity(metadata_value),
    )


def _run_native(
    args: argparse.Namespace,
    *,
    checkpoint_identifier: str,
    cache_manifest_identity: Mapping[str, object],
    preprocessing_identity: Mapping[str, object],
) -> dict[str, object]:
    """Launch Isaac only after CPU checkpoint validation has completed."""

    from phi_isaaclab.evaluation import BatchedEvaluationRunner, EvaluationConfig
    from phi_isaaclab.sim.runtime import (
        NativeRuntimeConfig,
        launch_native_app,
    )

    native_config = NativeRuntimeConfig(
        portable_root=(
            args.portable_root
            if args.portable_root is not None
            else args.writable_root / ".phi-isaaclab-kit" / args.run_dir.name
        ),
        writable_root=args.writable_root,
        isaaclab_source_root=args.isaaclab_source_root,
        device=args.device,
        headless=bool(args.headless),
        enable_cameras=bool(args.record_video),
    )
    native_app = launch_native_app(native_config)
    runtime_provenance = native_app.runtime_provenance
    environment = None
    environment_closed = False
    application_closed = False

    def close_runtime() -> None:
        nonlocal environment_closed, application_closed
        failures: list[tuple[str, BaseException]] = []
        if environment is not None and not environment_closed:
            try:
                environment.close()
            except BaseException as exc:
                failures.append(("environment", exc))
            else:
                environment_closed = True
        if not application_closed:
            try:
                native_app.close()
            except BaseException as exc:
                failures.append(("application", exc))
            else:
                application_closed = True
        if failures:
            details = "; ".join(f"{name}: {error}" for name, error in failures)
            raise RuntimeError(
                f"native Isaac cleanup failed ({details})"
            ) from failures[0][1]

    try:
        # Isaac Lab modules may only be imported after AppLauncher starts.
        from phi_isaaclab.sim.task import (
            build_lift_task_config,
            create_lift_environment,
        )

        task_config = build_lift_task_config(
            device=args.device,
            num_envs=args.num_envs,
            episode_length_s=args.max_steps * CONTROL_TIMESTEP_S,
        )
        environment = create_lift_environment(
            task_config,
            render_mode="rgb_array" if args.record_video else None,
        )
        adapter = load_torch_policy_adapter(
            args.checkpoint,
            online_metadata_path=args.online_metadata,
            trusted_checkpoint=True,
            device=args.device,
            strict_device=True,
        )
        if adapter.checkpoint_identifier != checkpoint_identifier:
            raise RuntimeError("checkpoint bytes changed after preflight validation")
        native_cache_identity, native_preprocessing_identity = (
            _shared_provenance_identities(adapter, args.collection_manifest)
        )
        if native_cache_identity != dict(
            cache_manifest_identity
        ) or native_preprocessing_identity != dict(preprocessing_identity):
            raise RuntimeError(
                "checkpoint metadata or collection manifest changed after "
                "preflight validation"
            )
        import torch

        environment_device = torch.device(str(environment.unwrapped.device))
        policy_device = adapter.backend.device
        if environment_device.type != policy_device.type or (
            environment_device.index is not None
            and policy_device.index is not None
            and environment_device.index != policy_device.index
        ):
            raise RuntimeError(
                f"policy device {policy_device} disagrees with environment device "
                f"{environment_device}"
            )
        policy_provenance = {
            "adapter": "action_bridge.eval.isaaclab_online",
            "action_bridge_version": _action_bridge_version(),
            "checkpoint_identifier": adapter.checkpoint_identifier,
            "online_evaluation": adapter.metadata.to_json_dict(),
        }
        result = BatchedEvaluationRunner(
            environment,
            adapter,
            EvaluationConfig(
                episodes=args.episodes,
                max_episode_steps=args.max_steps,
                seed=args.seed,
                run_directory=args.run_dir,
                record_video=args.record_video,
                video_fps=args.video_fps,
                checkpoint_identifier=adapter.checkpoint_identifier,
                policy_provenance=policy_provenance,
                runtime_provenance=runtime_provenance,
                cache_manifest_identity=cache_manifest_identity,
                preprocessing_identity=preprocessing_identity,
            ),
            runtime_cleanup=close_runtime,
        ).run()
    finally:
        if not environment_closed or not application_closed:
            active_error = sys.exc_info()[1]
            try:
                close_runtime()
            except BaseException:
                if active_error is None:
                    raise
    value = result.as_dict()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        _validate_preload_args(parser, args)
    except SystemExit as exc:
        return int(exc.code)

    # CPU reconstruction validates metadata, model geometry, and the complete
    # state dict before incurring Isaac Sim startup or creating run artifacts.
    try:
        preflight = load_torch_policy_adapter(
            args.checkpoint,
            online_metadata_path=args.online_metadata,
            trusted_checkpoint=True,
            device="cpu",
        )
    except Exception as exc:
        _emit_error("checkpoint_load", exc)
        return 2
    try:
        cache_manifest_identity, preprocessing_identity = _shared_provenance_identities(
            preflight,
            args.collection_manifest,
        )
    except Exception as exc:
        _emit_error("checkpoint_load", exc)
        return 2
    try:
        _validate_native_device(args.device)
    except Exception as exc:
        _emit_error("native_precondition", exc)
        return 3
    try:
        result = _run_native(
            args,
            checkpoint_identifier=preflight.checkpoint_identifier,
            cache_manifest_identity=cache_manifest_identity,
            preprocessing_identity=preprocessing_identity,
        )
    except Exception as exc:
        _emit_error("native_evaluation", exc)
        return 4
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(result)
    return (
        5
        if args.require_success and int(result["successful_episodes"]) != args.episodes
        else 0
    )


def native_main(argv: Sequence[str] | None = None) -> int:
    """Run the executable boundary with Isaac's teardown workaround enabled."""

    from phi_isaaclab.sim.runtime import (
        finalize_native_cli_process,
        native_cli_process_exit_required,
    )

    try:
        status = main(argv)
    except BaseException:
        if native_cli_process_exit_required():
            traceback.print_exc()
            finalize_native_cli_process(1)
        raise
    if native_cli_process_exit_required():
        finalize_native_cli_process(status)
    return status


__all__ = ["main", "native_main"]
