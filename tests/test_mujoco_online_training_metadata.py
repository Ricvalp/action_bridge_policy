from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
from ml_collections import ConfigDict
from phi_mujoco.offline import MANIFEST_FILENAME, EpisodeSplit, WindowConfig
from test_mujoco_online_metadata import make_config, make_metadata

from action_bridge.eval.mujoco_online.metadata import (
    OnlineEvaluationMetadata,
    validate_checkpoint_config,
)
from action_bridge.training.mujoco_online_metadata import (
    configure_mujoco_online_metadata,
)


def dataset_and_config(tmp_path):
    metadata = make_metadata()
    config = ConfigDict(make_config(metadata))
    del config.online_evaluation
    (tmp_path / MANIFEST_FILENAME).write_text('{"example":true}\n')
    dataset = SimpleNamespace(
        spec=metadata.integration,
        obs_dim=23,
        action_dim=7,
        normalization=metadata.normalization,
        normalization_stats={"type": "standard", **metadata.normalization.to_dict()},
        bundle=SimpleNamespace(root=tmp_path, data_sha256="b" * 64),
        window_config=WindowConfig(
            observation_history=2, action_history=2, prediction_horizon=3
        ),
        split_plan=EpisodeSplit(None, True, (0, 1, 2), (0, 1), (2,), ()),
    )
    return dataset, config


def test_training_metadata_records_exact_cache_and_optional_test_split(tmp_path):
    dataset, config = dataset_and_config(tmp_path)
    value = configure_mujoco_online_metadata(config, dataset, deepcopy(dataset))
    assert value["splits"] == {"train": [0, 1], "val": [2], "test": []}
    assert value["collection_identity"]["data_sha256"] == "b" * 64
    assert (
        value["collection_identity"]["manifest_sha256"]
        == hashlib.sha256((tmp_path / MANIFEST_FILENAME).read_bytes()).hexdigest()
    )
    validate_checkpoint_config(
        config.to_dict(), OnlineEvaluationMetadata.from_mapping(value)
    )


def test_training_metadata_rejects_split_contract_drift(tmp_path):
    dataset, config = dataset_and_config(tmp_path)
    validation = deepcopy(dataset)
    validation.bundle.data_sha256 = "c" * 64
    with pytest.raises(ValueError, match="validation dataset disagrees"):
        configure_mujoco_online_metadata(config, dataset, validation)


def test_training_metadata_rejects_changed_cache_on_resume(tmp_path):
    dataset, config = dataset_and_config(tmp_path)
    configure_mujoco_online_metadata(config, dataset, dataset)
    (tmp_path / MANIFEST_FILENAME).write_text('{"different":true}\n')
    with pytest.raises(ValueError, match="metadata disagrees"):
        configure_mujoco_online_metadata(config, dataset, dataset)
