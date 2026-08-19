"""Build checkpoint-safe RLBench online metadata from instantiated datasets."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, MutableMapping
from numbers import Integral
from typing import Any

from phi_rlbench.data.schema import ACTION_COMPONENTS, STATE_COMPONENTS
from phi_rlbench.provenance import cache_manifest_identity

from action_bridge.jax.eval.rlbench_online.metadata import (
    OnlineEvaluationMetadata,
    OnlineMetadataError,
    resolve_online_metadata,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOW_FIELDS = (
    "obs_history",
    "obs_stride",
    "action_history",
    "action_stride",
    "action_offset",
    "chunk_horizon",
)


class RLBenchOnlineMetadataWarning(UserWarning):
    """Training can continue, but online checkpoint semantics need attention."""


def _config_value(config: Any, *path: str, default: Any = ...) -> Any:
    value = config
    for name in path:
        if isinstance(value, Mapping) and name in value:
            value = value[name]
            continue
        try:
            value = getattr(value, name)
        except AttributeError as exc:
            if default is not ...:
                return default
            raise ValueError(f"Training config is missing {'.'.join(path)}.") from exc
    return value


def _remove_config_metadata(config: Any) -> None:
    if isinstance(config, MutableMapping):
        config.pop("online_evaluation", None)
        return
    if hasattr(config, "online_evaluation"):
        delattr(config, "online_evaluation")


def _set_config_metadata(config: Any, value: Mapping[str, object]) -> None:
    if isinstance(config, MutableMapping):
        config["online_evaluation"] = dict(value)
        return
    config.online_evaluation = dict(value)


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {normalized}.")
    return normalized


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _dataset_mapping(dataset: object, name: str, *, split: str) -> dict[str, int]:
    value = getattr(dataset, name, None)
    if not isinstance(value, Mapping):
        raise TypeError(f"{split} dataset {name} must be a mapping.")
    return dict(value)


def _dataset_integer(dataset: object, name: str, *, split: str) -> int:
    try:
        value = getattr(dataset, name)
    except AttributeError as exc:
        raise ValueError(f"{split} dataset is missing {name}.") from exc
    return _integer(value, name=f"{split} dataset {name}")


def _dataset_boolean(dataset: object, name: str, *, split: str) -> bool:
    try:
        value = getattr(dataset, name)
    except AttributeError as exc:
        raise ValueError(f"{split} dataset is missing {name}.") from exc
    return _boolean(value, name=f"{split} dataset {name}")


def _validate_dataset_pair(
    train_dataset: object, validation_dataset: object
) -> dict[str, object]:
    task_to_id = _dataset_mapping(train_dataset, "task_to_id", split="train")
    validation_tasks = _dataset_mapping(
        validation_dataset, "task_to_id", split="validation"
    )
    if task_to_id != validation_tasks:
        raise ValueError(
            "Train/validation task_to_id mappings differ; checkpoint task embeddings "
            "would be ambiguous."
        )

    task_variation_to_id = _dataset_mapping(
        train_dataset,
        "task_variation_to_id",
        split="train",
    )
    validation_variations = _dataset_mapping(
        validation_dataset,
        "task_variation_to_id",
        split="validation",
    )
    if task_variation_to_id != validation_variations:
        raise ValueError(
            "Train/validation task_variation_to_id mappings differ; checkpoint variation "
            "embeddings would be ambiguous."
        )

    output: dict[str, object] = {
        "task_to_id": task_to_id,
        "task_variation_to_id": task_variation_to_id,
    }
    for name in ("state_dim", "action_dim", "point_count", *_WINDOW_FIELDS):
        train_value = _dataset_integer(train_dataset, name, split="train")
        validation_value = _dataset_integer(
            validation_dataset, name, split="validation"
        )
        if train_value != validation_value:
            raise ValueError(
                f"Train/validation {name} differ: {train_value} != {validation_value}."
            )
        output[name] = train_value

    for name in ("include_rgb", "include_mask_id"):
        train_value = _dataset_boolean(train_dataset, name, split="train")
        validation_value = _dataset_boolean(
            validation_dataset, name, split="validation"
        )
        if train_value != validation_value:
            raise ValueError(
                f"Train/validation {name} differ: {train_value} != {validation_value}."
            )
        output[name] = train_value

    train_representation = getattr(train_dataset, "action_representation", None)
    validation_representation = getattr(
        validation_dataset, "action_representation", None
    )
    if not isinstance(train_representation, str) or not isinstance(
        validation_representation, str
    ):
        raise TypeError("Train/validation action_representation must be strings.")
    if train_representation != validation_representation:
        raise ValueError(
            "Train/validation action_representation differ: "
            f"{train_representation!r} != {validation_representation!r}."
        )
    output["action_representation"] = train_representation

    if output["state_dim"] != len(STATE_COMPONENTS):
        raise ValueError(
            f"RLBench state_dim must be {len(STATE_COMPONENTS)}, "
            f"got {output['state_dim']}."
        )
    if output["action_dim"] != len(ACTION_COMPONENTS):
        raise ValueError(
            f"RLBench action_dim must be {len(ACTION_COMPONENTS)}, "
            f"got {output['action_dim']}."
        )
    return output


def _unavailable_cache_identity() -> dict[str, object]:
    return {
        "available": False,
        "manifest_sha256": None,
        "schema_name": None,
        "schema_version": None,
        "cache_bundle_sha256": None,
        "preprocessing_sidecar_sha256": None,
    }


def _sha256(value: object) -> str | None:
    if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None:
        return value
    return None


def _cache_identity(cache_root: object) -> dict[str, object]:
    try:
        source = cache_manifest_identity(str(cache_root))
    except OSError:
        return _unavailable_cache_identity()
    if not isinstance(source, Mapping) or source.get("available") is not True:
        return _unavailable_cache_identity()

    manifest_digest = _sha256(source.get("sha256"))
    bundle_digest = _sha256(source.get("bundle_sha256"))
    schema_name = source.get("schema_name")
    schema_version = source.get("schema_version")
    if (
        source.get("valid_json") is not True
        or manifest_digest is None
        or bundle_digest is None
        or not isinstance(schema_name, str)
        or not schema_name.strip()
        or schema_name != schema_name.strip()
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, Integral)
        or int(schema_version) < 1
    ):
        return _unavailable_cache_identity()

    sidecar_digest: str | None = None
    sidecar = source.get("preprocessing_sidecar")
    if not isinstance(sidecar, Mapping):
        return _unavailable_cache_identity()
    if sidecar.get("available") is True:
        sidecar_digest = _sha256(sidecar.get("sha256"))
        if sidecar_digest is None:
            return _unavailable_cache_identity()

    return {
        "available": True,
        "manifest_sha256": manifest_digest,
        "schema_name": schema_name,
        "schema_version": int(schema_version),
        "cache_bundle_sha256": bundle_digest,
        "preprocessing_sidecar_sha256": sidecar_digest,
    }


def _resume_contains_metadata(payload: Mapping[str, object]) -> bool:
    if payload.get("online_evaluation") is not None:
        return True
    saved_config = payload.get("config")
    return (
        isinstance(saved_config, Mapping)
        and saved_config.get("online_evaluation") is not None
    )


def _unsupported_online_geometry(semantics: Mapping[str, object]) -> tuple[str, ...]:
    reasons: list[str] = []
    if semantics["action_history"] != semantics["obs_history"]:
        reasons.append("action_history must equal obs_history")
    if (
        semantics["obs_stride"],
        semantics["action_stride"],
        semantics["action_offset"],
    ) != (1, 1, 1):
        reasons.append("obs_stride, action_stride, and action_offset must all equal 1")
    if semantics["action_representation"] != "absolute":
        reasons.append("action_representation must be absolute")
    return tuple(reasons)


def _compare_resume_metadata(
    resume_metadata: OnlineEvaluationMetadata,
    current_metadata: OnlineEvaluationMetadata,
) -> None:
    resume_value = resume_metadata.to_json_dict()
    current_value = current_metadata.to_json_dict()
    resume_value.pop("deterministic_latent", None)
    current_value.pop("deterministic_latent", None)
    mismatched = sorted(
        key
        for key in set(resume_value) | set(current_value)
        if resume_value.get(key) != current_value.get(key)
    )
    if mismatched:
        raise OnlineMetadataError(
            "Resume checkpoint online_evaluation metadata disagrees with the current "
            f"training semantics in fields {mismatched}; only deterministic_latent may differ."
        )


def configure_rlbench_online_metadata(
    config: Any,
    train_dataset: object,
    validation_dataset: object,
    *,
    resume_payload: Mapping[str, object] | None = None,
) -> OnlineEvaluationMetadata | None:
    """Validate dataset semantics and attach checkpoint-ready online metadata.

    Offline geometries that the current evaluator cannot execute remain valid
    training jobs, but they deliberately receive no ``online_evaluation``
    metadata. Resuming a checkpoint that already claimed online compatibility
    under such a geometry is rejected rather than silently weakening its
    semantic contract.
    """

    if resume_payload is not None and not isinstance(resume_payload, Mapping):
        raise TypeError("resume_payload must be a mapping when provided.")
    _remove_config_metadata(config)
    semantics = _validate_dataset_pair(train_dataset, validation_dataset)

    selected_points = int(semantics["point_count"])
    configured_points = _integer(
        _config_value(config, "data", "point_count"),
        name="config.data.point_count",
    )
    if configured_points != selected_points:
        raise ValueError(
            "The instantiated dataset point_count disagrees with config.data.point_count: "
            f"{selected_points} != {configured_points}."
        )

    for name in ("include_rgb", "include_mask_id"):
        configured = _boolean(
            _config_value(config, "data", name),
            name=f"config.data.{name}",
        )
        if configured != semantics[name]:
            raise ValueError(
                f"The instantiated dataset {name}={semantics[name]!r} disagrees with "
                f"config.data.{name}={configured!r}."
            )

    resume_has_metadata = resume_payload is not None and _resume_contains_metadata(
        resume_payload
    )
    unsupported = _unsupported_online_geometry(semantics)
    if unsupported:
        details = "; ".join(unsupported)
        if resume_has_metadata:
            raise OnlineMetadataError(
                "Cannot resume a checkpoint with online_evaluation metadata under a "
                f"currently unsupported online geometry: {details}."
            )
        warnings.warn(
            "RLBench offline training geometry is unsupported by the current online "
            f"evaluator ({details}); no online_evaluation metadata will be attached.",
            RLBenchOnlineMetadataWarning,
            stacklevel=2,
        )
        return None

    dataset_rgb = bool(semantics["include_rgb"])
    dataset_mask_id = bool(semantics["include_mask_id"])
    encoder_rgb = _boolean(
        _config_value(config, "encoder", "use_rgb"),
        name="config.encoder.use_rgb",
    )
    encoder_mask_id = _boolean(
        _config_value(config, "encoder", "use_mask_id"),
        name="config.encoder.use_mask_id",
    )
    deterministic_latent = _boolean(
        _config_value(
            config,
            "checkpoint",
            "online_deterministic_latent_default",
            default=True,
        ),
        name="config.checkpoint.online_deterministic_latent_default",
    )

    metadata = OnlineEvaluationMetadata(
        task_to_id=semantics["task_to_id"],  # type: ignore[arg-type]
        task_variation_to_id=semantics["task_variation_to_id"],  # type: ignore[arg-type]
        point_count=selected_points,
        observation_history=int(semantics["obs_history"]),
        observation_stride=int(semantics["obs_stride"]),
        action_history=int(semantics["action_history"]),
        action_stride=int(semantics["action_stride"]),
        action_offset=int(semantics["action_offset"]),
        action_horizon=int(semantics["chunk_horizon"]),
        action_representation=str(semantics["action_representation"]),
        state_layout=tuple(STATE_COMPONENTS),
        action_layout=tuple(ACTION_COMPONENTS),
        training_cache_identity=_cache_identity(
            _config_value(config, "data", "cache_root")
        ),
        include_rgb=dataset_rgb and encoder_rgb,
        include_mask_id=dataset_mask_id and encoder_mask_id,
        policy_type=str(_config_value(config, "policy_type")),
        deterministic_latent=deterministic_latent,
    )

    if resume_has_metadata:
        assert resume_payload is not None
        resume_metadata = resolve_online_metadata(resume_payload)
        _compare_resume_metadata(resume_metadata, metadata)
    elif resume_payload is not None:
        warnings.warn(
            "Resuming a legacy checkpoint without online_evaluation metadata. Current "
            "dataset semantics will be attached to future checkpoints, but the original "
            "checkpoint task/cache identity could not be verified.",
            RLBenchOnlineMetadataWarning,
            stacklevel=2,
        )

    _set_config_metadata(config, metadata.to_json_dict())
    return metadata


__all__ = [
    "RLBenchOnlineMetadataWarning",
    "configure_rlbench_online_metadata",
]
