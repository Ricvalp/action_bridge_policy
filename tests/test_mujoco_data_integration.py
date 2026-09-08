from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from phi_mujoco.offline import EpisodeData, get_integration, write_processed_bundle
from torch.utils.data import DataLoader

from action_bridge.config import load_config
from action_bridge.training.common import (
    build_dataset,
    build_model,
    writable_numpy_collate,
)
from action_bridge.training.losses import model_loss


def make_cache(tmp_path: Path, integration_name: str) -> Path:
    """Three synthetic episodes with an explicit held-out validation episode."""

    integration = get_integration(integration_name)
    obs_dim = integration.spec.observations["state"].shape[0]
    action_dim = integration.spec.action.shape[0]
    steps = 12
    episodes = []
    for index in range(3):
        state = (
            np.arange((steps + 1) * obs_dim, dtype=np.float32).reshape(
                steps + 1, obs_dim
            )
            / 100
            + index * 100
        )
        actions = (
            np.linspace(-0.9, 0.3, steps * action_dim, dtype=np.float32).reshape(
                steps, action_dim
            )
            + index * 0.3
        )
        success = np.zeros(steps, dtype=np.bool_)
        success[-1] = True
        episodes.append(
            EpisodeData(
                episode_index=index,
                seed=index,
                observations={"state": state},
                actions=actions,
                action_history_padding=np.zeros(action_dim, dtype=np.float32),
                rewards=success.astype(np.float64),
                terminated=success.copy(),
                truncated=np.zeros(steps, dtype=np.bool_),
                success=success,
                termination_reason="success",
                source_episode_id=f"demo_{index}",
            )
        )
    root = tmp_path / integration_name
    write_processed_bundle(
        root,
        integration=integration,
        episodes=episodes,
        splits={"train": [0, 1], "val": [2], "test": []},
    )
    return root


@pytest.mark.parametrize("task,obs_dim", [("square", 23), ("tool_hang", 53)])
def test_official_splits_train_only_stats_and_aligned_windows(
    tmp_path: Path, task: str, obs_dim: int
) -> None:
    config = load_config(f"mujoco_robomimic_{task}")
    config.data.cache_root = str(make_cache(tmp_path, f"robomimic_{task}"))
    # Manifest partitions take precedence over these fallback fractions.
    config.data.train_fraction = 0.5
    config.data.val_fraction = 0.25
    config.data.split_seed = 123
    train = build_dataset(config, split="train")
    config.data.normalization = train.normalization.to_dict()
    validation = build_dataset(config, split="val")

    assert train.split_plan.config is None
    assert train.episode_indices == (0, 1)
    assert validation.episode_indices == (2,)
    assert train.split_plan.test_episode_indices == ()
    assert validation.normalization == train.normalization
    assert train.normalization.source_episode_indices == (0, 1)
    assert train.normalization_stats["type"] == "standard"
    assert (train.obs_dim, train.action_dim) == (obs_dim, 7)
    episodes = [train.bundle.load_episode(i) for i in train.episode_indices]
    expected_states = np.concatenate(
        [episode.observations["state"] for episode in episodes]
    )
    expected_actions = np.concatenate([episode.actions for episode in episodes])
    np.testing.assert_allclose(
        train.normalization.obs_mean, expected_states.mean(0), rtol=1e-6
    )
    np.testing.assert_allclose(
        train.normalization.action_mean, expected_actions.mean(0), atol=1e-7
    )

    start = train.item_from_episode_time(0, 0)
    assert start["obs_hist"].shape == (2, obs_dim)
    assert start["future_actions"].shape == (8, 7)
    assert start["obs_history_mask"].tolist() == [False, True]
    assert start["action_history_mask"].tolist() == [False, False]
    np.testing.assert_array_equal(start["obs_hist"][0], start["obs_hist"][1])
    np.testing.assert_allclose(
        start["act_hist"],
        train.normalization.normalize_actions(np.zeros((2, 7), dtype=np.float32)),
    )
    item = train.item_from_episode_time(0, 3)
    np.testing.assert_allclose(
        item["obs_hist"],
        train.normalization.normalize_observations(
            episodes[0].observations["state"][2:4]
        ),
    )
    np.testing.assert_allclose(
        item["act_hist"],
        train.normalization.normalize_actions(episodes[0].actions[1:3]),
    )
    np.testing.assert_allclose(
        item["future_actions"],
        train.normalization.normalize_actions(episodes[0].actions[3:11]),
    )
    assert item["future_action_mask"].all()
    assert len(train) == 2 * (12 - 8 + 1)
    assert train.sample_batch(3, np.random.default_rng(0))["obs_hist"].shape == (
        3,
        2,
        obs_dim,
    )
    with pytest.raises(ValueError, match="no eligible episodes"):
        build_dataset(config, split="test")


@pytest.mark.parametrize("task", ["square", "tool_hang"])
@pytest.mark.parametrize("latent_type", ["none", "continuous"])
def test_robomimic_state_windows_train_existing_bridge_model(
    tmp_path: Path, task: str, latent_type: str
) -> None:
    config = load_config(f"mujoco_robomimic_{task}")
    config.data.cache_root = str(make_cache(tmp_path, f"robomimic_{task}"))
    config.model.latent_type = latent_type
    config.model.hidden_dim = 8
    config.model.h_emb_dim = 8
    config.reference.hidden_dim = 8
    train = build_dataset(config, split="train")
    config.data.normalization_stats = train.normalization_stats
    batch = next(
        iter(DataLoader(train, batch_size=2, collate_fn=writable_numpy_collate))
    )
    model = build_model(config)
    loss = model_loss(model, batch, config.loss, global_step=1)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_offline_mujoco_adapter_does_not_import_simulator() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from action_bridge.config import load_config; "
            "load_config('mujoco_robomimic_square'); "
            "import action_bridge.data.mujoco_adapter; "
            "assert not {'mujoco', 'robosuite', 'gymnasium'} & sys.modules.keys()",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
