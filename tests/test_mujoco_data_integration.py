from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from phi_mujoco.collection import CollectionConfig, CollectionRunner
from phi_mujoco.evaluation import PolicyInput, PolicyOutput
from torch.utils.data import DataLoader

from action_bridge.config import load_config
from action_bridge.eval.mujoco_online.torch_backend import load_torch_policy_adapter
from action_bridge.training.common import (
    build_dataset,
    build_model,
    move_to_device,
    writable_numpy_collate,
)
from action_bridge.training.losses import model_loss
from action_bridge.training.mujoco_online_metadata import (
    configure_mujoco_online_metadata,
)
from action_bridge.training.train_toy import save_checkpoint


@dataclass
class _Observation:
    values: np.ndarray


@dataclass
class _Reset:
    observation: _Observation
    info: dict[str, object]
    seed_report: dict[str, object]


@dataclass
class _Step:
    observation: _Observation
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, object]


class _Runtime:
    def __init__(self) -> None:
        self.seed = 0
        self.step_index = 0

    def _observation(self) -> _Observation:
        seed = float(self.seed)
        step = float(self.step_index)
        return _Observation(
            np.asarray(
                [seed, step, seed + step, seed - step, step, -step, 0.2, -0.1],
                dtype=np.float32,
            )
        )

    def reset(self, *, seed: int | None = None) -> _Reset:
        self.seed = 0 if seed is None else seed
        self.step_index = 0
        return _Reset(
            observation=self._observation(),
            info={},
            seed_report={
                "generator": "test.sequence",
                "requested_seed": self.seed,
                "resolved_seed": self.seed,
            },
        )

    def step(self, action: np.ndarray) -> _Step:
        assert action.shape == (2,)
        self.step_index += 1
        success = self.step_index == 6
        return _Step(
            observation=self._observation(),
            reward=-float(self.step_index),
            terminated=success,
            truncated=False,
            success=success,
            info={},
        )

    def close(self) -> None:
        return None


class _Policy:
    def reset(self, *, task_name: str, variation_id: int, seed: int) -> None:
        del task_name, variation_id, seed

    def predict(self, observation: PolicyInput) -> PolicyOutput:
        value = 0.1 * (observation.episode_step + 1)
        return PolicyOutput(np.asarray([value, -value], dtype=np.float32))


def _provenance() -> dict[str, object]:
    return {
        "backend": {},
        "cache": {},
        "environment": {},
        "generated_at_utc": "2026-08-19T12:00:00+00:00",
        "host": {},
        "preprocessing": None,
        "schema_name": "phi.robotics.provenance",
        "schema_version": 1,
        "simulator": {},
        "software": {},
    }


def _collection(tmp_path: Path) -> Path:
    root = tmp_path / "collection"
    CollectionRunner(
        runtime=_Runtime(),
        policy=_Policy(),
        config=CollectionConfig(episodes=6, max_steps=6, base_seed=100),
    ).run(
        output_directory=root,
        provenance=_provenance(),
        resolved_config={"fixture": "action-bridge-mujoco-integration"},
    )
    return root


def test_validated_bundle_reaches_loss_checkpoint_and_online_adapter(
    tmp_path: Path,
) -> None:
    config = load_config("mujoco_planar_reach_direct_chunk_bc")
    config.device = "cpu"
    config.data.collection_root = str(_collection(tmp_path))
    config.eval.clip_actions = True
    config.model.hidden_dim = 8
    config.model.h_emb_dim = 8
    config.model.depth = 1

    train = build_dataset(config, split="train")
    config.data.normalization_stats = train.normalization_stats
    config.data.normalization = train.normalization.to_json_dict()
    validation = build_dataset(config, split="val")
    test = build_dataset(config, split="test")
    metadata = configure_mujoco_online_metadata(config, train, validation, test)

    assert len(train) > 0 and len(validation) > 0 and len(test) > 0
    assert metadata["collection_identity"] == train.collection_identity
    first = train.item_from_episode_time(train.episode_indices[0], 0)
    assert first["action_history_mask"].tolist() == [False, False]

    loader = DataLoader(
        train,
        batch_size=2,
        shuffle=False,
        collate_fn=writable_numpy_collate,
    )
    batch = move_to_device(next(iter(loader)), torch.device("cpu"))
    model = build_model(config)
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
        checkpoint,
        trusted_checkpoint=True,
        device="cpu",
    )
    adapter.reset(task_name="planar_reach", variation_id=0, seed=5)
    policy_output = adapter.predict(
        PolicyInput(
            observation=np.zeros(8, dtype=np.float32),
            task_name="planar_reach",
            variation_id=0,
            episode_step=0,
        )
    )
    assert policy_output.actions.shape == (config.chunk_horizon, 2)
    assert np.isfinite(policy_output.actions).all()
