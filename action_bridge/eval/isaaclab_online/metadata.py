"""Strict checkpoint metadata for batched Action Bridge evaluation in Isaac Lab."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import NoReturn

from action_bridge.eval.isaaclab_online.contracts import (
    ACTION_DIM,
    ACTION_PROFILE,
    BENCHMARK_NAME,
    COLLECTION_SCHEMA_NAME,
    COLLECTION_SCHEMA_VERSION,
    CONTROL_TIMESTEP_S,
    GRIPPER_CLOSE_ACTION,
    GRIPPER_OPEN_ACTION,
    GRIPPER_THRESHOLD,
    OBSERVATION_DIM,
    OBSERVATION_PROFILE,
    ONLINE_SCHEMA_NAME,
    ONLINE_SCHEMA_VERSION,
    POSITION_LOWER_M,
    POSITION_PROJECTION,
    POSITION_UPPER_M,
    QUATERNION_EPSILON,
    QUATERNION_ORDER,
    QUATERNION_PROJECTION,
    SUPPORTED_LATENT_COMMITMENTS,
    SUPPORTED_POLICY_TYPES,
    TASK_ID,
    VARIATION_ID,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OnlineMetadataError(ValueError):
    """Checkpoint metadata is absent, malformed, or incompatible."""


def _fail(message: str) -> NoReturn:
    raise OnlineMetadataError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"{name} must be an object with string keys.")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail(f"{name} must be an integer.")
    output = int(value)
    if output < minimum:
        _fail(f"{name} must be at least {minimum}.")
    return output


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be a boolean.")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{name} must be a non-empty canonical string.")
    return value


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(f"{name} must be a real number.")
    output = float(value)
    if not math.isfinite(output) or (positive and output <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        _fail(f"{name} must be {qualifier}.")
    return output


def _float_vector(
    value: object,
    *,
    name: str,
    size: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{name} must be an array of length {size}.")
    output = tuple(
        _finite_float(item, name=f"{name}[{index}]", positive=positive)
        for index, item in enumerate(value)
    )
    if len(output) != size:
        _fail(f"{name} must have length {size}, got {len(output)}.")
    return output


@dataclass(frozen=True, slots=True)
class StandardNormalization:
    obs_mean: tuple[float, ...]
    obs_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    eps: float
    type: str = "standard"

    def __post_init__(self) -> None:
        if self.type != "standard":
            _fail("normalization.type must equal 'standard'.")
        obs_mean = _float_vector(self.obs_mean, name="normalization.obs_mean", size=OBSERVATION_DIM)
        obs_std = _float_vector(
            self.obs_std,
            name="normalization.obs_std",
            size=OBSERVATION_DIM,
            positive=True,
        )
        action_mean = _float_vector(
            self.action_mean, name="normalization.action_mean", size=ACTION_DIM
        )
        action_std = _float_vector(
            self.action_std,
            name="normalization.action_std",
            size=ACTION_DIM,
            positive=True,
        )
        eps = _finite_float(self.eps, name="normalization.eps", positive=True)
        if any(value < eps for value in (*obs_std, *action_std)):
            _fail("normalization standard deviations must be at least eps.")
        object.__setattr__(self, "obs_mean", obs_mean)
        object.__setattr__(self, "obs_std", obs_std)
        object.__setattr__(self, "action_mean", action_mean)
        object.__setattr__(self, "action_std", action_std)
        object.__setattr__(self, "eps", eps)

    @classmethod
    def from_mapping(cls, value: object) -> StandardNormalization:
        data = _mapping(value, name="normalization")
        _exact_keys(
            data,
            {"type", "eps", "obs_mean", "obs_std", "action_mean", "action_std"},
            name="normalization",
        )
        return cls(
            type=_string(data["type"], name="normalization.type"),
            eps=_finite_float(data["eps"], name="normalization.eps", positive=True),
            obs_mean=_float_vector(
                data["obs_mean"], name="normalization.obs_mean", size=OBSERVATION_DIM
            ),
            obs_std=_float_vector(
                data["obs_std"],
                name="normalization.obs_std",
                size=OBSERVATION_DIM,
                positive=True,
            ),
            action_mean=_float_vector(
                data["action_mean"], name="normalization.action_mean", size=ACTION_DIM
            ),
            action_std=_float_vector(
                data["action_std"],
                name="normalization.action_std",
                size=ACTION_DIM,
                positive=True,
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "eps": self.eps,
            "obs_mean": list(self.obs_mean),
            "obs_std": list(self.obs_std),
            "action_mean": list(self.action_mean),
            "action_std": list(self.action_std),
        }


@dataclass(frozen=True, slots=True)
class CollectionIdentity:
    manifest_sha256: str
    schema_name: str = COLLECTION_SCHEMA_NAME
    schema_version: int = COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != COLLECTION_SCHEMA_NAME:
            _fail(f"collection_identity.schema_name must equal {COLLECTION_SCHEMA_NAME!r}.")
        if self.schema_version != COLLECTION_SCHEMA_VERSION:
            _fail(
                "collection_identity.schema_version must equal "
                f"{COLLECTION_SCHEMA_VERSION}."
            )
        if not isinstance(self.manifest_sha256, str) or _SHA256.fullmatch(
            self.manifest_sha256
        ) is None:
            _fail("collection_identity.manifest_sha256 must be lowercase SHA-256 hex.")

    @classmethod
    def from_mapping(cls, value: object) -> CollectionIdentity:
        data = _mapping(value, name="collection_identity")
        _exact_keys(
            data,
            {"schema_name", "schema_version", "manifest_sha256"},
            name="collection_identity",
        )
        return cls(
            schema_name=_string(data["schema_name"], name="collection_identity.schema_name"),
            schema_version=_integer(
                data["schema_version"], name="collection_identity.schema_version", minimum=1
            ),
            manifest_sha256=_string(
                data["manifest_sha256"], name="collection_identity.manifest_sha256"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class ActionProjection:
    position_lower_m: tuple[float, ...] = POSITION_LOWER_M
    position_upper_m: tuple[float, ...] = POSITION_UPPER_M
    position_projection: str = POSITION_PROJECTION
    quaternion_order: str = QUATERNION_ORDER
    quaternion_projection: str = QUATERNION_PROJECTION
    quaternion_epsilon: float = QUATERNION_EPSILON
    gripper_threshold: float = GRIPPER_THRESHOLD
    gripper_open_action: float = GRIPPER_OPEN_ACTION
    gripper_close_action: float = GRIPPER_CLOSE_ACTION

    def __post_init__(self) -> None:
        constants = {
            "position_lower_m": (tuple(self.position_lower_m), POSITION_LOWER_M),
            "position_upper_m": (tuple(self.position_upper_m), POSITION_UPPER_M),
            "position_projection": (self.position_projection, POSITION_PROJECTION),
            "quaternion_order": (self.quaternion_order, QUATERNION_ORDER),
            "quaternion_projection": (self.quaternion_projection, QUATERNION_PROJECTION),
            "quaternion_epsilon": (self.quaternion_epsilon, QUATERNION_EPSILON),
            "gripper_threshold": (self.gripper_threshold, GRIPPER_THRESHOLD),
            "gripper_open_action": (self.gripper_open_action, GRIPPER_OPEN_ACTION),
            "gripper_close_action": (self.gripper_close_action, GRIPPER_CLOSE_ACTION),
        }
        for name, (actual, expected) in constants.items():
            if actual != expected:
                _fail(f"action_projection.{name} must equal {expected!r}.")

    @classmethod
    def from_mapping(cls, value: object) -> ActionProjection:
        data = _mapping(value, name="action_projection")
        expected = {
            "position_lower_m",
            "position_upper_m",
            "position_projection",
            "quaternion_order",
            "quaternion_projection",
            "quaternion_epsilon",
            "gripper_threshold",
            "gripper_open_action",
            "gripper_close_action",
        }
        _exact_keys(data, expected, name="action_projection")
        return cls(
            position_lower_m=_float_vector(
                data["position_lower_m"], name="action_projection.position_lower_m", size=3
            ),
            position_upper_m=_float_vector(
                data["position_upper_m"], name="action_projection.position_upper_m", size=3
            ),
            position_projection=_string(
                data["position_projection"], name="action_projection.position_projection"
            ),
            quaternion_order=_string(
                data["quaternion_order"], name="action_projection.quaternion_order"
            ),
            quaternion_projection=_string(
                data["quaternion_projection"], name="action_projection.quaternion_projection"
            ),
            quaternion_epsilon=_finite_float(
                data["quaternion_epsilon"],
                name="action_projection.quaternion_epsilon",
                positive=True,
            ),
            gripper_threshold=_finite_float(
                data["gripper_threshold"], name="action_projection.gripper_threshold"
            ),
            gripper_open_action=_finite_float(
                data["gripper_open_action"], name="action_projection.gripper_open_action"
            ),
            gripper_close_action=_finite_float(
                data["gripper_close_action"], name="action_projection.gripper_close_action"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "position_lower_m": list(self.position_lower_m),
            "position_upper_m": list(self.position_upper_m),
            "position_projection": self.position_projection,
            "quaternion_order": self.quaternion_order,
            "quaternion_projection": self.quaternion_projection,
            "quaternion_epsilon": self.quaternion_epsilon,
            "gripper_threshold": self.gripper_threshold,
            "gripper_open_action": self.gripper_open_action,
            "gripper_close_action": self.gripper_close_action,
        }


@dataclass(frozen=True, slots=True)
class OnlineEvaluationMetadata:
    observation_history: int
    action_history: int
    action_horizon: int
    normalization: StandardNormalization
    collection_identity: CollectionIdentity
    action_projection: ActionProjection
    policy_type: str
    latent_commitment: str
    deterministic_latent: bool = True
    schema_name: str = ONLINE_SCHEMA_NAME
    schema_version: int = ONLINE_SCHEMA_VERSION
    task_name: str = TASK_ID
    variation_id: int = VARIATION_ID
    observation_profile: str = OBSERVATION_PROFILE
    action_profile: str = ACTION_PROFILE
    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM
    actions_per_plan: int = 1
    control_timestep_s: float = CONTROL_TIMESTEP_S

    def __post_init__(self) -> None:
        constants = {
            "schema_name": (self.schema_name, ONLINE_SCHEMA_NAME),
            "schema_version": (self.schema_version, ONLINE_SCHEMA_VERSION),
            "task_name": (self.task_name, TASK_ID),
            "variation_id": (self.variation_id, VARIATION_ID),
            "observation_profile": (self.observation_profile, OBSERVATION_PROFILE),
            "action_profile": (self.action_profile, ACTION_PROFILE),
            "observation_dim": (self.observation_dim, OBSERVATION_DIM),
            "action_dim": (self.action_dim, ACTION_DIM),
            "actions_per_plan": (self.actions_per_plan, 1),
        }
        for name, (actual, expected) in constants.items():
            if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
                _fail(f"{name} must equal {expected!r}.")
        for name, value, minimum in (
            ("observation_history", self.observation_history, 1),
            ("action_history", self.action_history, 2),
            ("action_horizon", self.action_horizon, 1),
        ):
            _integer(value, name=name, minimum=minimum)
        if not isinstance(self.normalization, StandardNormalization):
            _fail("normalization must be StandardNormalization.")
        if not isinstance(self.collection_identity, CollectionIdentity):
            _fail("collection_identity must be CollectionIdentity.")
        if not isinstance(self.action_projection, ActionProjection):
            _fail("action_projection must be ActionProjection.")
        if self.policy_type not in SUPPORTED_POLICY_TYPES:
            _fail(f"policy_type must be one of {sorted(SUPPORTED_POLICY_TYPES)}.")
        if self.latent_commitment not in SUPPORTED_LATENT_COMMITMENTS:
            _fail(f"latent_commitment must be one of {sorted(SUPPORTED_LATENT_COMMITMENTS)}.")
        if self.deterministic_latent is not True:
            _fail("Isaac Lab requires deterministic_latent=true.")
        if not math.isclose(float(self.control_timestep_s), CONTROL_TIMESTEP_S, abs_tol=1e-12):
            _fail(f"control_timestep_s must equal {CONTROL_TIMESTEP_S!r}.")

    @classmethod
    def from_mapping(cls, value: object) -> OnlineEvaluationMetadata:
        data = _mapping(value, name="online_evaluation")
        expected = {
            "schema_name",
            "schema_version",
            "task_name",
            "variation_id",
            "observation_profile",
            "action_profile",
            "observation_dim",
            "action_dim",
            "observation_history",
            "action_history",
            "action_horizon",
            "actions_per_plan",
            "control_timestep_s",
            "normalization",
            "collection_identity",
            "action_projection",
            "policy_type",
            "latent_commitment",
            "deterministic_latent",
        }
        _exact_keys(data, expected, name="online_evaluation")
        return cls(
            schema_name=_string(data["schema_name"], name="schema_name"),
            schema_version=_integer(data["schema_version"], name="schema_version", minimum=1),
            task_name=_string(data["task_name"], name="task_name"),
            variation_id=_integer(data["variation_id"], name="variation_id"),
            observation_profile=_string(data["observation_profile"], name="observation_profile"),
            action_profile=_string(data["action_profile"], name="action_profile"),
            observation_dim=_integer(data["observation_dim"], name="observation_dim", minimum=1),
            action_dim=_integer(data["action_dim"], name="action_dim", minimum=1),
            observation_history=_integer(
                data["observation_history"], name="observation_history", minimum=1
            ),
            action_history=_integer(data["action_history"], name="action_history", minimum=2),
            action_horizon=_integer(data["action_horizon"], name="action_horizon", minimum=1),
            actions_per_plan=_integer(data["actions_per_plan"], name="actions_per_plan", minimum=1),
            control_timestep_s=_finite_float(
                data["control_timestep_s"], name="control_timestep_s", positive=True
            ),
            normalization=StandardNormalization.from_mapping(data["normalization"]),
            collection_identity=CollectionIdentity.from_mapping(data["collection_identity"]),
            action_projection=ActionProjection.from_mapping(data["action_projection"]),
            policy_type=_string(data["policy_type"], name="policy_type"),
            latent_commitment=_string(data["latent_commitment"], name="latent_commitment"),
            deterministic_latent=_boolean(
                data["deterministic_latent"], name="deterministic_latent"
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "variation_id": self.variation_id,
            "observation_profile": self.observation_profile,
            "action_profile": self.action_profile,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observation_history": self.observation_history,
            "action_history": self.action_history,
            "action_horizon": self.action_horizon,
            "actions_per_plan": self.actions_per_plan,
            "control_timestep_s": self.control_timestep_s,
            "normalization": self.normalization.to_json_dict(),
            "collection_identity": self.collection_identity.to_json_dict(),
            "action_projection": self.action_projection.to_json_dict(),
            "policy_type": self.policy_type,
            "latent_commitment": self.latent_commitment,
            "deterministic_latent": self.deterministic_latent,
        }


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            _fail(f"duplicate JSON key is forbidden: {key!r}")
        output[key] = value
    return output


def load_online_metadata(path: str | Path) -> OnlineEvaluationMetadata:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        _fail(f"online metadata must be a regular non-symlink file: {source}")
    try:
        text = source.read_text(encoding="utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OnlineMetadataError(f"cannot read strict online metadata: {source}: {exc}") from exc
    return OnlineEvaluationMetadata.from_mapping(value)


def resolve_online_metadata(
    checkpoint: Mapping[str, object],
    *,
    explicit_path: str | Path | None = None,
) -> OnlineEvaluationMetadata:
    embedded_value = checkpoint.get("online_evaluation")
    embedded = (
        None if embedded_value is None else OnlineEvaluationMetadata.from_mapping(embedded_value)
    )
    explicit = None if explicit_path is None else load_online_metadata(explicit_path)
    if embedded is not None and explicit is not None and embedded != explicit:
        _fail("explicit online metadata disagrees with checkpoint online_evaluation metadata.")
    if embedded is not None:
        return embedded
    if explicit is not None:
        return explicit
    _fail(
        "checkpoint has no online_evaluation metadata; provide explicit metadata JSON "
        "only when its training semantics are known."
    )


def _config_value(config: Mapping[str, object], key: str, *, name: str) -> object:
    if key not in config:
        _fail(f"checkpoint config is missing {name}.")
    return config[key]


def validate_checkpoint_config(
    config: Mapping[str, object], metadata: OnlineEvaluationMetadata
) -> None:
    if config.get("benchmark") != BENCHMARK_NAME:
        _fail(f"checkpoint config benchmark must equal {BENCHMARK_NAME!r}.")
    for key, expected in {
        "obs_dim": metadata.observation_dim,
        "action_dim": metadata.action_dim,
        "obs_history": metadata.observation_history,
        "action_history": metadata.action_history,
        "chunk_horizon": metadata.action_horizon,
    }.items():
        actual = _integer(_config_value(config, key, name=key), name=f"config.{key}", minimum=1)
        if actual != expected:
            _fail(f"checkpoint config {key}={actual} disagrees with metadata value {expected}.")

    model = _mapping(_config_value(config, "model", name="model"), name="config.model")
    if model.get("policy_type") != metadata.policy_type:
        _fail("checkpoint model.policy_type disagrees with online metadata.")
    inference = _mapping(
        _config_value(config, "inference", name="inference"), name="config.inference"
    )
    if inference.get("latent_commitment") != metadata.latent_commitment:
        _fail("checkpoint inference.latent_commitment disagrees with online metadata.")
    if inference.get("deterministic") != metadata.deterministic_latent:
        _fail("checkpoint inference.deterministic disagrees with online metadata.")

    data = _mapping(_config_value(config, "data", name="data"), name="config.data")
    if data.get("observation_profile") != metadata.observation_profile:
        _fail("checkpoint data.observation_profile disagrees with online metadata.")
    if data.get("action_profile") != metadata.action_profile:
        _fail("checkpoint data.action_profile disagrees with online metadata.")
    if data.get("normalize") is not True:
        _fail("Isaac Lab online checkpoints require data.normalize=true.")
    if StandardNormalization.from_mapping(
        _config_value(data, "normalization_stats", name="data.normalization_stats")
    ) != metadata.normalization:
        _fail("checkpoint normalization_stats disagree with online metadata.")
    if CollectionIdentity.from_mapping(
        _config_value(data, "collection_identity", name="data.collection_identity")
    ) != metadata.collection_identity:
        _fail("checkpoint collection_identity disagrees with online metadata.")


__all__ = [
    "ActionProjection",
    "CollectionIdentity",
    "OnlineEvaluationMetadata",
    "OnlineMetadataError",
    "StandardNormalization",
    "load_online_metadata",
    "resolve_online_metadata",
    "validate_checkpoint_config",
]
