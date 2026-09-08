"""Record the processed dataset and integration used to train a checkpoint."""

from __future__ import annotations

import hashlib
from typing import Any

from phi_mujoco.offline import MANIFEST_FILENAME, SCHEMA_NAME, SCHEMA_VERSION

from action_bridge.config import to_plain_dict
from action_bridge.eval.mujoco_online.metadata import OnlineEvaluationMetadata


def configure_mujoco_online_metadata(
    config: Any,
    train_dataset: Any,
    validation_dataset: Any,
    test_dataset: Any = None,
) -> dict[str, object]:
    """Attach the exact dataset/profile contract; reject inconsistent resumes."""
    for name, other in (("validation", validation_dataset), ("test", test_dataset)):
        if other is None:
            continue
        if (
            other.spec != train_dataset.spec
            or other.bundle.data_sha256 != train_dataset.bundle.data_sha256
            or other.split_plan != train_dataset.split_plan
            or other.normalization != train_dataset.normalization
            or other.window_config != train_dataset.window_config
        ):
            raise ValueError(
                f"MuJoCo {name} dataset disagrees with the training dataset"
            )
    windows = train_dataset.window_config
    if (config.obs_history, config.action_history, config.chunk_horizon) != (
        windows.observation_history,
        windows.action_history,
        windows.prediction_horizon,
    ):
        raise ValueError("configured histories/horizon disagree with dataset windows")
    if config.data.integration != train_dataset.spec.name:
        raise ValueError("configured integration disagrees with training dataset")
    identity = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(
            (train_dataset.bundle.root / MANIFEST_FILENAME).read_bytes()
        ).hexdigest(),
        "data_sha256": train_dataset.bundle.data_sha256,
    }
    metadata = OnlineEvaluationMetadata(
        integration=train_dataset.spec,
        normalization=train_dataset.normalization,
        collection_identity=identity,
        splits={
            name: list(train_dataset.split_plan.episode_indices(name))
            for name in ("train", "val", "test")
        },
        observation_history=int(config.obs_history),
        action_history=int(config.action_history),
        action_horizon=int(config.chunk_horizon),
        actions_per_plan=int(config.eval.actions_per_plan),
        policy_type=str(config.model.policy_type),
        latent_commitment=str(config.inference.latent_commitment),
        deterministic_latent=config.inference.deterministic,
        clip_actions=config.eval.clip_actions,
    ).to_json_dict()
    previous = config.get("online_evaluation")
    if previous is not None and to_plain_dict(previous) != metadata:
        raise ValueError(
            "checkpoint/config online metadata disagrees with the training dataset"
        )
    config.obs_dim = train_dataset.obs_dim
    config.action_dim = train_dataset.action_dim
    config.data.normalization = train_dataset.normalization.to_dict()
    config.data.normalization_stats = train_dataset.normalization_stats
    config.data.collection_identity = identity
    config.data.observation_profile = train_dataset.spec.observation_profile
    config.data.action_profile = train_dataset.spec.action_profile
    config.online_evaluation = metadata
    return metadata
