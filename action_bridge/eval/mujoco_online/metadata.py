"""Strict metadata for Action Bridge evaluation on ``phi-mujoco``.

The online adapter deliberately does not infer policy semantics from tensor
shapes.  A checkpoint must describe the exact task/profile contract, history
geometry, normalization, and training collection identity used by the model.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import NoReturn

from phi_mujoco.dataset import SCHEMA_NAME as COLLECTION_SCHEMA_NAME
from phi_mujoco.dataset import SCHEMA_VERSION as COLLECTION_SCHEMA_VERSION
from phi_mujoco.evaluation.protocol import ACTION_DIM, OBSERVATION_DIM
from phi_mujoco.tasks.planar_reach import (
    ACTION_PROFILE,
    DEFAULT_ACTION_REPEAT,
    MODEL_TIMESTEP_S,
    OBSERVATION_PROFILE,
    TASK_ID,
    VARIATION_ID,
)

ONLINE_SCHEMA_NAME = "action_bridge.mujoco_online"
ONLINE_SCHEMA_VERSION = 1
BENCHMARK_NAME = "mujoco_planar_reach"
CONTROL_TIMESTEP_S = MODEL_TIMESTEP_S * DEFAULT_ACTION_REPEAT
SUPPORTED_POLICY_TYPES = {
    "action_bridge",
    "ar_bc",
    "autoregressive_bc",
    "bc_smooth",
    "direct_bc",
}
SUPPORTED_LATENT_COMMITMENTS = {"chunk", "episode"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OnlineMetadataError(ValueError):
    """Checkpoint metadata is absent, malformed, or semantically incompatible."""


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
    """Standard-score parameters embedded in the checkpoint contract."""

    obs_mean: tuple[float, ...]
    obs_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    eps: float
    type: str = "standard"

    def __post_init__(self) -> None:
        if self.type != "standard":
            _fail("normalization.type must equal 'standard'.")
        obs_mean = _float_vector(
            self.obs_mean,
            name="normalization.obs_mean",
            size=OBSERVATION_DIM,
        )
        obs_std = _float_vector(
            self.obs_std,
            name="normalization.obs_std",
            size=OBSERVATION_DIM,
            positive=True,
        )
        action_mean = _float_vector(
            self.action_mean,
            name="normalization.action_mean",
            size=ACTION_DIM,
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
        if data["type"] != "standard":
            _fail("normalization.type must equal 'standard'.")
        return cls(
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
            eps=_finite_float(data["eps"], name="normalization.eps", positive=True),
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
    """Content identity of the validated collection used for training."""

    manifest_sha256: str
    schema_name: str = COLLECTION_SCHEMA_NAME
    schema_version: int = COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_name = _string(
            self.schema_name,
            name="collection_identity.schema_name",
        )
        schema_version = _integer(
            self.schema_version,
            name="collection_identity.schema_version",
            minimum=1,
        )
        digest = _string(
            self.manifest_sha256,
            name="collection_identity.manifest_sha256",
        )
        if schema_name != COLLECTION_SCHEMA_NAME or schema_version != COLLECTION_SCHEMA_VERSION:
            _fail(
                "collection_identity must identify the phi-mujoco collection schema "
                f"{COLLECTION_SCHEMA_NAME!r} version {COLLECTION_SCHEMA_VERSION}."
            )
        if _SHA256.fullmatch(digest) is None:
            _fail("collection_identity.manifest_sha256 must be lowercase SHA-256 hex.")
        object.__setattr__(self, "schema_name", schema_name)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "manifest_sha256", digest)

    @classmethod
    def from_mapping(cls, value: object) -> CollectionIdentity:
        data = _mapping(value, name="collection_identity")
        _exact_keys(
            data,
            {"schema_name", "schema_version", "manifest_sha256"},
            name="collection_identity",
        )
        schema_name = _string(data["schema_name"], name="collection_identity.schema_name")
        schema_version = _integer(
            data["schema_version"], name="collection_identity.schema_version", minimum=1
        )
        digest = _string(data["manifest_sha256"], name="collection_identity.manifest_sha256")
        if schema_name != COLLECTION_SCHEMA_NAME or schema_version != COLLECTION_SCHEMA_VERSION:
            _fail(
                "collection_identity must identify the phi-mujoco collection schema "
                f"{COLLECTION_SCHEMA_NAME!r} version {COLLECTION_SCHEMA_VERSION}."
            )
        if _SHA256.fullmatch(digest) is None:
            _fail("collection_identity.manifest_sha256 must be lowercase SHA-256 hex.")
        return cls(manifest_sha256=digest)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class OnlineEvaluationMetadata:
    """Exact policy/runtime boundary required for closed-loop reconstruction."""

    observation_history: int
    action_history: int
    action_horizon: int
    normalization: StandardNormalization
    collection_identity: CollectionIdentity
    policy_type: str
    latent_commitment: str
    deterministic_latent: bool = True
    clip_actions: bool = False
    schema_name: str = ONLINE_SCHEMA_NAME
    schema_version: int = ONLINE_SCHEMA_VERSION
    task_name: str = TASK_ID
    variation_id: int = VARIATION_ID
    observation_profile: str = OBSERVATION_PROFILE.name
    action_profile: str = ACTION_PROFILE.name
    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM
    actions_per_plan: int = 1
    action_lower: tuple[float, ...] = ACTION_PROFILE.lower
    action_upper: tuple[float, ...] = ACTION_PROFILE.upper
    control_timestep_s: float = CONTROL_TIMESTEP_S

    def __post_init__(self) -> None:
        if not isinstance(self.normalization, StandardNormalization):
            _fail("normalization must be StandardNormalization.")
        if not isinstance(self.collection_identity, CollectionIdentity):
            _fail("collection_identity must be CollectionIdentity.")
        constants = {
            "schema_name": (self.schema_name, ONLINE_SCHEMA_NAME),
            "schema_version": (self.schema_version, ONLINE_SCHEMA_VERSION),
            "task_name": (self.task_name, TASK_ID),
            "variation_id": (self.variation_id, VARIATION_ID),
            "observation_profile": (self.observation_profile, OBSERVATION_PROFILE.name),
            "action_profile": (self.action_profile, ACTION_PROFILE.name),
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
        if self.policy_type not in SUPPORTED_POLICY_TYPES:
            _fail(f"policy_type must be one of {sorted(SUPPORTED_POLICY_TYPES)}.")
        if self.latent_commitment not in SUPPORTED_LATENT_COMMITMENTS:
            _fail(f"latent_commitment must be one of {sorted(SUPPORTED_LATENT_COMMITMENTS)}.")
        if not isinstance(self.deterministic_latent, bool) or not isinstance(
            self.clip_actions, bool
        ):
            _fail("deterministic_latent and clip_actions must be booleans.")
        action_lower = _float_vector(
            self.action_lower,
            name="action_lower",
            size=ACTION_DIM,
        )
        action_upper = _float_vector(
            self.action_upper,
            name="action_upper",
            size=ACTION_DIM,
        )
        if action_lower != tuple(ACTION_PROFILE.lower):
            _fail(f"action_lower must equal {list(ACTION_PROFILE.lower)!r}.")
        if action_upper != tuple(ACTION_PROFILE.upper):
            _fail(f"action_upper must equal {list(ACTION_PROFILE.upper)!r}.")
        control_timestep_s = _finite_float(
            self.control_timestep_s,
            name="control_timestep_s",
            positive=True,
        )
        if not math.isclose(
            control_timestep_s,
            CONTROL_TIMESTEP_S,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"control_timestep_s must equal {CONTROL_TIMESTEP_S!r}.")
        object.__setattr__(self, "action_lower", action_lower)
        object.__setattr__(self, "action_upper", action_upper)
        object.__setattr__(self, "control_timestep_s", control_timestep_s)

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
            "action_lower",
            "action_upper",
            "control_timestep_s",
            "normalization",
            "collection_identity",
            "policy_type",
            "latent_commitment",
            "deterministic_latent",
            "clip_actions",
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
            action_lower=_float_vector(data["action_lower"], name="action_lower", size=ACTION_DIM),
            action_upper=_float_vector(data["action_upper"], name="action_upper", size=ACTION_DIM),
            control_timestep_s=_finite_float(
                data["control_timestep_s"], name="control_timestep_s", positive=True
            ),
            normalization=StandardNormalization.from_mapping(data["normalization"]),
            collection_identity=CollectionIdentity.from_mapping(data["collection_identity"]),
            policy_type=_string(data["policy_type"], name="policy_type"),
            latent_commitment=_string(data["latent_commitment"], name="latent_commitment"),
            deterministic_latent=_boolean(
                data["deterministic_latent"], name="deterministic_latent"
            ),
            clip_actions=_boolean(data["clip_actions"], name="clip_actions"),
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
            "action_lower": list(self.action_lower),
            "action_upper": list(self.action_upper),
            "control_timestep_s": self.control_timestep_s,
            "normalization": self.normalization.to_json_dict(),
            "collection_identity": self.collection_identity.to_json_dict(),
            "policy_type": self.policy_type,
            "latent_commitment": self.latent_commitment,
            "deterministic_latent": self.deterministic_latent,
            "clip_actions": self.clip_actions,
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
    """Read strict JSON metadata from one explicit regular file."""

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
    """Resolve embedded metadata, with a non-guessing explicit legacy fallback."""

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
        "checkpoint has no online_evaluation metadata; provide an explicit metadata JSON "
        "only when its training semantics are known."
    )


def _config_value(config: Mapping[str, object], key: str, *, name: str) -> object:
    if key not in config:
        _fail(f"checkpoint config is missing {name}.")
    return config[key]


def validate_checkpoint_config(
    config: Mapping[str, object], metadata: OnlineEvaluationMetadata
) -> None:
    """Require model/data config fields to agree with the embedded contract."""

    if config.get("benchmark") != BENCHMARK_NAME:
        _fail(f"checkpoint config benchmark must equal {BENCHMARK_NAME!r}.")
    expected_dimensions = {
        "obs_dim": metadata.observation_dim,
        "action_dim": metadata.action_dim,
        "obs_history": metadata.observation_history,
        "action_history": metadata.action_history,
        "chunk_horizon": metadata.action_horizon,
    }
    for key, expected in expected_dimensions.items():
        actual = _integer(_config_value(config, key, name=key), name=f"config.{key}", minimum=1)
        if actual != expected:
            _fail(f"checkpoint config {key}={actual} disagrees with metadata value {expected}.")

    model = _mapping(_config_value(config, "model", name="model"), name="config.model")
    policy_type = _string(
        _config_value(model, "policy_type", name="model.policy_type"),
        name="config.model.policy_type",
    )
    if policy_type != metadata.policy_type:
        _fail("checkpoint model.policy_type disagrees with online metadata.")

    inference = _mapping(
        _config_value(config, "inference", name="inference"),
        name="config.inference",
    )
    commitment = _string(
        _config_value(
            inference,
            "latent_commitment",
            name="inference.latent_commitment",
        ),
        name="config.inference.latent_commitment",
    )
    if commitment != metadata.latent_commitment:
        _fail("checkpoint inference.latent_commitment disagrees with online metadata.")
    deterministic = _boolean(
        _config_value(inference, "deterministic", name="inference.deterministic"),
        name="config.inference.deterministic",
    )
    if deterministic != metadata.deterministic_latent:
        _fail("checkpoint inference.deterministic disagrees with online metadata.")

    data = _mapping(_config_value(config, "data", name="data"), name="config.data")
    if data.get("normalize") is not True:
        _fail("MuJoCo online checkpoints require data.normalize=true.")
    normalization = StandardNormalization.from_mapping(
        _config_value(data, "normalization_stats", name="data.normalization_stats")
    )
    if normalization != metadata.normalization:
        _fail("checkpoint normalization_stats disagree with online metadata.")
    identity = CollectionIdentity.from_mapping(
        _config_value(data, "collection_identity", name="data.collection_identity")
    )
    if identity != metadata.collection_identity:
        _fail("checkpoint collection_identity disagrees with online metadata.")


__all__ = [
    "BENCHMARK_NAME",
    "CONTROL_TIMESTEP_S",
    "ONLINE_SCHEMA_NAME",
    "ONLINE_SCHEMA_VERSION",
    "SUPPORTED_LATENT_COMMITMENTS",
    "CollectionIdentity",
    "OnlineEvaluationMetadata",
    "OnlineMetadataError",
    "StandardNormalization",
    "load_online_metadata",
    "resolve_online_metadata",
    "validate_checkpoint_config",
]
