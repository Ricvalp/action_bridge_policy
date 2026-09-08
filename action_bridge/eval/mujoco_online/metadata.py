"""Checkpoint contract for a policy trained on a named MuJoCo integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from phi_mujoco.integrations import IntegrationSpec, get_integration
from phi_mujoco.offline import StandardNormalization

ONLINE_SCHEMA_NAME = "action_bridge.mujoco_online"
ONLINE_SCHEMA_VERSION = 2
BENCHMARK_NAME = "mujoco"


class OnlineMetadataError(ValueError):
    """Checkpoint metadata disagrees with the training or simulator contract."""


@dataclass(frozen=True)
class OnlineEvaluationMetadata:
    integration: IntegrationSpec
    normalization: StandardNormalization
    collection_identity: dict[str, object]
    splits: dict[str, list[int]]
    observation_history: int
    action_history: int
    action_horizon: int
    actions_per_plan: int
    policy_type: str
    latent_commitment: str
    deterministic_latent: bool
    clip_actions: bool

    def __post_init__(self) -> None:
        if self.integration != get_integration(self.integration.name).spec:
            raise OnlineMetadataError(
                "checkpoint integration/profile disagrees with phi-mujoco"
            )
        if set(self.integration.observations) != {"state"}:
            raise OnlineMetadataError(
                "this adapter requires the state observation modality"
            )
        for name, minimum in (
            ("observation_history", 1),
            ("action_history", 2),
            ("action_horizon", 1),
            ("actions_per_plan", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise OnlineMetadataError(f"{name} must be an integer >= {minimum}")
        if self.actions_per_plan > self.action_horizon:
            raise OnlineMetadataError("actions_per_plan must not exceed action_horizon")
        if self.latent_commitment not in {"chunk", "episode"}:
            raise OnlineMetadataError("latent_commitment must be 'chunk' or 'episode'")
        if (
            type(self.deterministic_latent) is not bool
            or type(self.clip_actions) is not bool
        ):
            raise OnlineMetadataError(
                "deterministic_latent and clip_actions must be boolean"
            )
        if (
            self.normalization.observation_modality != "state"
            or len(self.normalization.obs_mean) != self.observation_dim
            or len(self.normalization.action_mean) != self.action_dim
        ):
            raise OnlineMetadataError(
                "normalization dimensions disagree with integration"
            )
        if set(self.splits) != {"train", "val", "test"}:
            raise OnlineMetadataError("splits must contain train, val and test")
        indices = [index for split in self.splits.values() for index in split]
        if any(type(index) is not int or index < 0 for index in indices) or len(
            set(indices)
        ) != len(indices):
            raise OnlineMetadataError(
                "split episode indices must be non-negative and disjoint"
            )
        if tuple(self.splits["train"]) != self.normalization.source_episode_indices:
            raise OnlineMetadataError(
                "normalization must be fitted on exactly the train split"
            )
        for name in ("manifest_sha256", "data_sha256"):
            digest = self.collection_identity.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise OnlineMetadataError(
                    f"collection_identity.{name} must be a SHA256 digest"
                )

    @property
    def observation_dim(self) -> int:
        return self.integration.observations["state"].shape[0]

    @property
    def action_dim(self) -> int:
        return self.integration.action.shape[0]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_name": ONLINE_SCHEMA_NAME,
            "schema_version": ONLINE_SCHEMA_VERSION,
            "integration": self.integration.to_dict(),
            "normalization": self.normalization.to_dict(),
            "collection_identity": self.collection_identity,
            "splits": self.splits,
            "observation_history": self.observation_history,
            "action_history": self.action_history,
            "action_horizon": self.action_horizon,
            "actions_per_plan": self.actions_per_plan,
            "policy_type": self.policy_type,
            "latent_commitment": self.latent_commitment,
            "deterministic_latent": self.deterministic_latent,
            "clip_actions": self.clip_actions,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OnlineEvaluationMetadata:
        if not isinstance(value, Mapping):
            raise OnlineMetadataError(
                "checkpoint requires embedded online_evaluation metadata"
            )
        data = dict(value)
        if (
            data.pop("schema_name", None) != ONLINE_SCHEMA_NAME
            or data.pop("schema_version", None) != ONLINE_SCHEMA_VERSION
        ):
            raise OnlineMetadataError(
                "unsupported online_evaluation schema; retrain this checkpoint"
            )
        try:
            data["integration"] = IntegrationSpec.from_dict(data["integration"])
            data["normalization"] = StandardNormalization.from_dict(
                data["normalization"]
            )
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise OnlineMetadataError(
                f"invalid online_evaluation metadata: {exc}"
            ) from exc


def resolve_online_metadata(
    checkpoint: Mapping[str, object],
) -> OnlineEvaluationMetadata:
    return OnlineEvaluationMetadata.from_mapping(checkpoint.get("online_evaluation"))


def validate_checkpoint_config(
    config: Mapping[str, object], metadata: OnlineEvaluationMetadata
) -> None:
    """Reject stale profiles, normalization, histories and inference settings."""
    expected = {
        "benchmark": BENCHMARK_NAME,
        "obs_dim": metadata.observation_dim,
        "action_dim": metadata.action_dim,
        "obs_history": metadata.observation_history,
        "action_history": metadata.action_history,
        "chunk_horizon": metadata.action_horizon,
        "online_evaluation": metadata.to_json_dict(),
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise OnlineMetadataError(
                f"checkpoint config {name} disagrees with online metadata"
            )
    sections = {
        "model": {"policy_type": metadata.policy_type},
        "inference": {
            "deterministic": metadata.deterministic_latent,
            "latent_commitment": metadata.latent_commitment,
        },
        "eval": {
            "actions_per_plan": metadata.actions_per_plan,
            "clip_actions": metadata.clip_actions,
        },
        "data": {
            "integration": metadata.integration.name,
            "observation_profile": metadata.integration.observation_profile,
            "action_profile": metadata.integration.action_profile,
            "normalization": metadata.normalization.to_dict(),
            "collection_identity": metadata.collection_identity,
        },
    }
    for name, fields in sections.items():
        section = config.get(name, {})
        if not isinstance(section, Mapping):
            raise OnlineMetadataError(f"checkpoint config {name} must be a mapping")
        for field, value in fields.items():
            if section.get(field) != value:
                raise OnlineMetadataError(
                    f"checkpoint config {name}.{field} disagrees with metadata"
                )
