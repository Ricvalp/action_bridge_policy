"""Strict metadata required to reconstruct an Action Bridge online input."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

from phi_rlbench.data.schema import ACTION_COMPONENTS, STATE_COMPONENTS


class OnlineMetadataError(ValueError):
    """Checkpoint/run metadata is insufficient or ambiguous for online use."""


# Keep the policy-owned validator tied to the simulator package's authoritative
# v1 component vocabulary rather than maintaining a second set of aliases.
CANONICAL_RLBENCH_STATE_LAYOUT = tuple(STATE_COMPONENTS)
CANONICAL_RLBENCH_ACTION_LAYOUT = tuple(ACTION_COMPONENTS)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise OnlineMetadataError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < 1:
        raise OnlineMetadataError(f"{name} must be positive.")
    return normalized


def _dense_identifier_map(value: object, *, name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise OnlineMetadataError(f"{name} must be a non-empty mapping.")
    output: dict[str, int] = {}
    for raw_key, raw_identifier in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise OnlineMetadataError(f"{name} keys must be non-empty strings.")
        if raw_key != raw_key.strip():
            raise OnlineMetadataError(
                f"{name} keys must be canonical and contain no surrounding whitespace."
            )
        if isinstance(raw_identifier, bool) or not isinstance(raw_identifier, Integral):
            raise OnlineMetadataError(f"{name} values must be integers.")
        output[raw_key] = int(raw_identifier)
    if sorted(output.values()) != list(range(len(output))):
        raise OnlineMetadataError(f"{name} values must be unique and dense from zero.")
    return MappingProxyType(output)


def _exact_component_layout(
    value: object,
    *,
    name: str,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OnlineMetadataError(f"{name} must be an ordered component sequence.")
    normalized = tuple(value)
    if any(not isinstance(component, str) for component in normalized):
        raise OnlineMetadataError(f"{name} components must be strings.")
    if normalized != expected:
        raise OnlineMetadataError(
            f"{name} must exactly match the canonical RLBench layout {list(expected)!r}; "
            f"received {list(normalized)!r}."
        )
    return normalized  # type: ignore[return-value]


def _training_cache_identity(value: object) -> Mapping[str, object]:
    name = "training_cache_identity"
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OnlineMetadataError(f"{name} must be an object with string keys.")
    required = {"available", "manifest_sha256", "schema_name", "schema_version"}
    missing = sorted(required - value.keys())
    optional = {"cache_bundle_sha256", "preprocessing_sidecar_sha256"}
    unexpected = sorted(value.keys() - required - optional)
    if missing:
        raise OnlineMetadataError(f"{name} is missing fields: {missing}.")
    if unexpected:
        raise OnlineMetadataError(f"{name} contains unsupported fields: {unexpected}.")

    available = value["available"]
    if not isinstance(available, bool):
        raise OnlineMetadataError(f"{name}.available must be boolean.")
    digest = value["manifest_sha256"]
    schema_name = value["schema_name"]
    schema_version = value["schema_version"]
    bundle_digest = value.get("cache_bundle_sha256")
    sidecar_digest = value.get("preprocessing_sidecar_sha256")
    if not available:
        if any(
            item is not None
            for item in (
                digest,
                schema_name,
                schema_version,
                bundle_digest,
                sidecar_digest,
            )
        ):
            raise OnlineMetadataError(
                f"Unavailable {name} must use null digest and schema fields."
            )
    else:
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise OnlineMetadataError(
                f"{name}.manifest_sha256 must be one lowercase SHA-256 digest."
            )
        if (
            not isinstance(schema_name, str)
            or not schema_name
            or schema_name != schema_name.strip()
        ):
            raise OnlineMetadataError(f"{name}.schema_name must be a canonical string.")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, Integral)
            or int(schema_version) < 1
        ):
            raise OnlineMetadataError(
                f"{name}.schema_version must be a positive integer."
            )
        schema_version = int(schema_version)
        for field_name, field_value in (
            ("cache_bundle_sha256", bundle_digest),
            ("preprocessing_sidecar_sha256", sidecar_digest),
        ):
            if field_value is not None and (
                not isinstance(field_value, str)
                or _SHA256_PATTERN.fullmatch(field_value) is None
            ):
                raise OnlineMetadataError(
                    f"{name}.{field_name} must be null or one lowercase SHA-256 digest."
                )
    return MappingProxyType(
        {
            "available": available,
            "manifest_sha256": digest,
            "schema_name": schema_name,
            "schema_version": schema_version,
            "cache_bundle_sha256": bundle_digest,
            "preprocessing_sidecar_sha256": sidecar_digest,
        }
    )


@dataclass(frozen=True)
class OnlineEvaluationMetadata:
    """Checkpoint-adjacent semantics that old Action Bridge checkpoints omit."""

    task_to_id: Mapping[str, int]
    task_variation_to_id: Mapping[str, int]
    point_count: int
    observation_history: int
    observation_stride: int
    action_history: int
    action_stride: int
    action_offset: int
    action_horizon: int
    action_representation: str
    state_layout: tuple[str, ...]
    action_layout: tuple[str, ...]
    training_cache_identity: Mapping[str, object]
    include_rgb: bool
    include_mask_id: bool
    policy_type: str
    deterministic_latent: bool = True
    schema_name: str = "action_bridge.rlbench_online"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_name, str)
            or self.schema_name != "action_bridge.rlbench_online"
        ):
            raise OnlineMetadataError("Unsupported online metadata schema_name.")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or self.schema_version != 1
        ):
            raise OnlineMetadataError("Unsupported online metadata schema_version.")
        task_to_id = _dense_identifier_map(self.task_to_id, name="task_to_id")
        variation_to_id = _dense_identifier_map(
            self.task_variation_to_id,
            name="task_variation_to_id",
        )
        variation_tasks: set[str] = set()
        for key in variation_to_id:
            try:
                task_name, variation_text = key.rsplit(":", maxsplit=1)
                variation_id = int(variation_text)
            except (TypeError, ValueError) as exc:
                raise OnlineMetadataError(
                    "task_variation_to_id keys must use '<task>:<nonnegative-id>'."
                ) from exc
            if (
                task_name not in task_to_id
                or variation_id < 0
                or variation_text != str(variation_id)
            ):
                raise OnlineMetadataError(
                    f"Unknown or negative task variation mapping key: {key!r}."
                )
            variation_tasks.add(task_name)
        missing_task_variations = sorted(set(task_to_id) - variation_tasks)
        if missing_task_variations:
            raise OnlineMetadataError(
                "Every task must have at least one task variation mapping; missing "
                f"{missing_task_variations}."
            )
        point_count = _positive_integer(self.point_count, name="point_count")
        observation_history = _positive_integer(
            self.observation_history,
            name="observation_history",
        )
        observation_stride = _positive_integer(
            self.observation_stride,
            name="observation_stride",
        )
        action_history = _positive_integer(
            self.action_history,
            name="action_history",
        )
        action_stride = _positive_integer(self.action_stride, name="action_stride")
        action_offset = _positive_integer(self.action_offset, name="action_offset")
        action_horizon = _positive_integer(self.action_horizon, name="action_horizon")
        if action_history != observation_history:
            raise OnlineMetadataError(
                "The current PolicyInput can reproduce Action Bridge action history only "
                "when action_history == observation_history."
            )
        if (observation_stride, action_stride, action_offset) != (1, 1, 1):
            raise OnlineMetadataError(
                "The current online runner executes actions on consecutive simulator steps "
                "and therefore requires observation_stride=action_stride=action_offset=1."
            )
        if (
            not isinstance(self.action_representation, str)
            or self.action_representation != "absolute"
        ):
            raise OnlineMetadataError(
                "Closed-loop Action Bridge evaluation requires absolute actions; "
                "delta_xyz would otherwise be decoded twice or against ambiguous state."
            )
        state_layout = _exact_component_layout(
            self.state_layout,
            name="state_layout",
            expected=CANONICAL_RLBENCH_STATE_LAYOUT,
        )
        action_layout = _exact_component_layout(
            self.action_layout,
            name="action_layout",
            expected=CANONICAL_RLBENCH_ACTION_LAYOUT,
        )
        training_cache_identity = _training_cache_identity(self.training_cache_identity)
        if not isinstance(self.policy_type, str) or self.policy_type not in {
            "action_bridge",
            "direct_chunk_bc",
        }:
            raise OnlineMetadataError(
                "policy_type must be action_bridge or direct_chunk_bc."
            )
        for name in ("include_rgb", "include_mask_id", "deterministic_latent"):
            if not isinstance(getattr(self, name), bool):
                raise OnlineMetadataError(f"{name} must be boolean.")
        if self.policy_type == "direct_chunk_bc" and not self.deterministic_latent:
            raise OnlineMetadataError(
                "direct_chunk_bc has no stochastic latent and requires "
                "deterministic_latent=true."
            )
        object.__setattr__(self, "task_to_id", task_to_id)
        object.__setattr__(self, "task_variation_to_id", variation_to_id)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "observation_history", observation_history)
        object.__setattr__(self, "observation_stride", observation_stride)
        object.__setattr__(self, "action_history", action_history)
        object.__setattr__(self, "action_stride", action_stride)
        object.__setattr__(self, "action_offset", action_offset)
        object.__setattr__(self, "action_horizon", action_horizon)
        object.__setattr__(self, "state_layout", state_layout)
        object.__setattr__(self, "action_layout", action_layout)
        object.__setattr__(self, "training_cache_identity", training_cache_identity)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OnlineEvaluationMetadata:
        if any(not isinstance(key, str) for key in value):
            raise OnlineMetadataError("Online metadata keys must be strings.")
        required = {
            "task_to_id",
            "task_variation_to_id",
            "point_count",
            "observation_history",
            "observation_stride",
            "action_history",
            "action_stride",
            "action_offset",
            "action_horizon",
            "action_representation",
            "state_layout",
            "action_layout",
            "training_cache_identity",
            "include_rgb",
            "include_mask_id",
            "policy_type",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise OnlineMetadataError(f"Online metadata is missing fields: {missing}.")
        allowed = required | {
            "deterministic_latent",
            "schema_name",
            "schema_version",
        }
        unexpected = sorted(value.keys() - allowed)
        if unexpected:
            raise OnlineMetadataError(
                f"Online metadata contains unsupported fields: {unexpected}."
            )
        return cls(**dict(value))  # type: ignore[arg-type]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "task_to_id": dict(self.task_to_id),
            "task_variation_to_id": dict(self.task_variation_to_id),
            "point_count": self.point_count,
            "observation_history": self.observation_history,
            "observation_stride": self.observation_stride,
            "action_history": self.action_history,
            "action_stride": self.action_stride,
            "action_offset": self.action_offset,
            "action_horizon": self.action_horizon,
            "action_representation": self.action_representation,
            "state_layout": list(self.state_layout),
            "action_layout": list(self.action_layout),
            "training_cache_identity": dict(self.training_cache_identity),
            "include_rgb": self.include_rgb,
            "include_mask_id": self.include_mask_id,
            "policy_type": self.policy_type,
            "deterministic_latent": self.deterministic_latent,
        }


def _strict_json_object(path: Path) -> Mapping[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OnlineMetadataError(f"Cannot read online metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise OnlineMetadataError("Online metadata JSON must contain one object.")
    return value


def resolve_online_metadata(
    checkpoint: Mapping[str, object],
    *,
    explicit_path: str | Path | None = None,
) -> OnlineEvaluationMetadata:
    """Apply the non-guessing old-checkpoint metadata hierarchy."""

    embedded = checkpoint.get("online_evaluation")
    if embedded is not None:
        if not isinstance(embedded, Mapping):
            raise OnlineMetadataError("checkpoint online_evaluation must be a mapping.")
        return OnlineEvaluationMetadata.from_mapping(embedded)
    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        saved = config.get("online_evaluation")
        if saved is not None:
            if not isinstance(saved, Mapping):
                raise OnlineMetadataError("config online_evaluation must be a mapping.")
            return OnlineEvaluationMetadata.from_mapping(saved)
    if explicit_path is not None:
        return OnlineEvaluationMetadata.from_mapping(
            _strict_json_object(Path(explicit_path).expanduser())
        )
    raise OnlineMetadataError(
        "This checkpoint has no online_evaluation metadata. Supply an explicit "
        "--online-metadata JSON file with task vocabularies, exact component layouts, "
        "training-cache identity, and training window semantics; task IDs and layouts "
        "must not be inferred from dimensions or filesystem ordering."
    )


__all__ = [
    "CANONICAL_RLBENCH_ACTION_LAYOUT",
    "CANONICAL_RLBENCH_STATE_LAYOUT",
    "OnlineEvaluationMetadata",
    "OnlineMetadataError",
    "resolve_online_metadata",
]
