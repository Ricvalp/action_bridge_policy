from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from phi_isaaclab.dataset import EpisodeData, write_processed_bundle
from torch.utils.data import DataLoader

from action_bridge.config import load_config
from action_bridge.eval.isaaclab_online.torch_backend import load_torch_policy_adapter
from action_bridge.training.common import (
    build_dataset,
    build_model,
    move_to_device,
    writable_numpy_collate,
)
from action_bridge.training.isaaclab_online_metadata import (
    configure_isaaclab_online_metadata,
)
from action_bridge.training.losses import model_loss
from action_bridge.training.train_isaaclab import train
from action_bridge.training.train_toy import save_checkpoint


def _episode(index: int, *, steps: int = 6) -> EpisodeData:
    observations = np.zeros((steps + 1, 35), dtype=np.float32)
    observations[:, 18:25] = np.asarray(
        [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    observations[:, 25:32] = np.asarray(
        [0.5, 0.0, 0.055, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    observations[:, 32:35] = np.asarray([0.5, 0.0, 0.35], dtype=np.float32)
    actions = np.repeat(
        np.asarray(
            [[0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
        steps,
        axis=0,
    )
    terminated = np.zeros(steps, dtype=np.bool_)
    terminated[-1] = True
    success = np.zeros(steps, dtype=np.bool_)
    success[-1] = True
    return EpisodeData(
        episode_index=index,
        seed=100 + index,
        observations=observations,
        actions=actions,
        rewards=np.zeros(steps, dtype=np.float64),
        terminated=terminated,
        truncated=np.zeros(steps, dtype=np.bool_),
        success=success,
        termination_reason="success",
        source_episode_id=f"fixture-{index}",
    )


def _collection(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "collection"
    write_processed_bundle(
        root,
        [_episode(index) for index in range(6)],
        resolved_config={"fixture": "action-bridge-isaaclab-integration"},
        provenance={"fixture": True},
    )
    return root


@dataclass
class _Input:
    observation: torch.Tensor
    episode_ids: torch.Tensor
    step_indices: torch.Tensor
    task_id: str = "franka_cube_lift"
    variation_id: int = 0


def test_validated_hdf5_reaches_loss_checkpoint_and_batched_adapter(
    tmp_path: Path,
) -> None:
    config = load_config("isaaclab_franka_cube_lift_direct_chunk_bc")
    config.device = "cpu"
    config.data.collection_root = str(_collection(tmp_path))
    config.model.hidden_dim = 8
    config.model.h_emb_dim = 8
    config.model.depth = 1

    train = build_dataset(config, split="train")
    config.data.normalization_stats = train.normalization_stats
    config.data.normalization = train.normalization.to_json_dict()
    validation = build_dataset(config, split="val")
    test = build_dataset(config, split="test")
    metadata = configure_isaaclab_online_metadata(
        config, train, validation, test
    )
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0
    assert metadata["collection_identity"] == train.collection_identity
    first = train.item_from_episode_time(train.episode_indices[0], 0)
    assert first["action_history_mask"].tolist() == [False, False]
    expected_hold = np.asarray([0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0])
    denormalized_history = train.normalization.denormalize_actions(first["act_hist"])
    np.testing.assert_allclose(denormalized_history, np.repeat(expected_hold[None], 2, axis=0))

    loader = DataLoader(
        train,
        batch_size=2,
        shuffle=False,
        collate_fn=writable_numpy_collate,
    )
    batch = move_to_device(next(iter(loader)), torch.device("cpu"))
    model = build_model(config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    output = model_loss(model, batch, config.loss, global_step=1)
    assert torch.isfinite(output["loss"])

    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        config,
        1,
        float(output["loss"].detach()),
    )
    adapter = load_torch_policy_adapter(
        checkpoint, trusted_checkpoint=True, device="cpu"
    )
    adapter.reset()
    reset_observation = torch.from_numpy(_episode(0).observations[0]).unsqueeze(0)
    action = adapter.predict(
        _Input(
            reset_observation,
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        )
    )
    assert action.shape == (1, 8)
    assert torch.isfinite(action).all()
    torch.testing.assert_close(action[0], torch.tensor(expected_hold, dtype=torch.float32))


@torch.no_grad()
def _checkpoint_prediction(checkpoint: Path) -> torch.Tensor:
    adapter = load_torch_policy_adapter(
        checkpoint, trusted_checkpoint=True, device="cpu"
    )
    adapter.reset()
    observation = torch.from_numpy(_episode(0).observations[0]).unsqueeze(0)
    return adapter.predict(
        _Input(
            observation,
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        )
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "isaaclab_franka_cube_lift_direct_chunk_bc",
        "isaaclab_franka_cube_lift_no_latent",
    ],
)
def test_one_step_train_writes_reloadable_checkpoint(
    tmp_path: Path, config_name: str
) -> None:
    config = load_config(config_name)
    config.device = "cpu"
    config.data.collection_root = str(_collection(tmp_path / config_name))
    config.output_dir = str(tmp_path / "runs")
    config.run_id = config_name
    config.optim.max_steps = 1
    config.optim.batch_size = 2
    config.logging.log_every_steps = 1
    config.logging.eval_every_steps = 1
    config.logging.checkpoint_every_steps = 0
    config.logging.full_eval_every_steps = 0
    config.eval.offline_max_batches = 1
    config.model.hidden_dim = 8
    config.model.h_emb_dim = 8
    if config.model.policy_type == "direct_bc":
        config.model.depth = 1
    else:
        config.model.time_emb_dim = 4
        config.reference.hidden_dim = 8
        config.reference.time_emb_dim = 4

    run_directory = train(config)
    best = run_directory / "checkpoints" / "best.pt"
    latest = run_directory / "checkpoints" / "latest.pt"
    assert best.is_file() and latest.is_file()
    best_prediction = _checkpoint_prediction(best)
    latest_prediction = _checkpoint_prediction(latest)
    torch.testing.assert_close(best_prediction, latest_prediction)
    assert best_prediction.shape == (1, 8)
