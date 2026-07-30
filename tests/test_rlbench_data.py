from pathlib import Path
import json
import pickle
import types

import h5py
import numpy as np
import torch

from action_bridge.data.rlbench_cache import (
    CACHE_SCHEMA_NAME,
    RLBenchCacheStore,
    build_cache_keys,
)
from action_bridge.data.rlbench_cache_builder import convert_rlbench_dataset
from action_bridge.data.rlbench_dataset import (
    RLBenchDataset,
    decode_action_chunk,
)


def _write_fake_raw_rlbench(root: Path, *, episodes: int = 5, length: int = 6) -> None:
    variation = root / "reach_target" / "variation0"
    (variation / "episodes").mkdir(parents=True)
    (variation / "mask_to_label.json").write_text(
        json.dumps({"1": "Floor", "2": "target", "3": "panda_gripper"}),
        encoding="utf-8",
    )
    with (variation / "variation_descriptions.pkl").open("wb") as handle:
        pickle.dump(["reach the target"], handle)

    for episode_id in range(episodes):
        episode = variation / "episodes" / f"episode{episode_id}"
        point_dir = episode / "merged_point_cloud"
        point_dir.mkdir(parents=True)
        observations = []
        for time_index in range(length):
            x = float(episode_id + 0.1 * time_index)
            observations.append(
                types.SimpleNamespace(
                    gripper_pose=np.asarray(
                        [x, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0],
                        dtype=np.float32,
                    ),
                    gripper_open=float(time_index % 2),
                    joint_positions=np.arange(7, dtype=np.float32) + time_index,
                    joint_velocities=np.full(7, time_index, dtype=np.float32),
                    joint_forces=np.full(7, 2 * time_index, dtype=np.float32),
                    gripper_joint_positions=np.asarray([0.01, 0.01], dtype=np.float32),
                    gripper_touch_forces=None,
                    task_low_dim_state=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
                )
            )
            points = np.asarray(
                [
                    [0.0, 0.0, 0.1],
                    [0.1, 0.0, 0.8],
                    [0.2, 0.1, 0.9],
                    [0.3, 0.2, 1.0],
                    [0.4, 0.3, 1.1],
                    [3.0, 0.0, 0.8],
                ],
                dtype=np.float32,
            )
            colors = np.linspace(0.0, 1.0, len(points) * 3, dtype=np.float32).reshape(-1, 3)
            masks = np.asarray([1, 2, 2, 3, 3, 2], dtype=np.int32)
            np.savez(point_dir / f"{time_index}.npz", points=points, colors=colors, masks=masks)
        with (episode / "low_dim_obs.pkl").open("wb") as handle:
            pickle.dump(observations, handle)


def _build_fake_cache(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    _write_fake_raw_rlbench(raw_root)
    manifest = convert_rlbench_dataset(
        raw_root,
        cache_root,
        num_points=4,
        compression="none",
    )
    assert manifest == cache_root / "manifest.json"
    return cache_root


def test_rlbench_cache_builder_preserves_episode_data(tmp_path):
    cache_root = _build_fake_cache(tmp_path)
    cache_path = cache_root / "reach_target" / "variation0.h5"
    with h5py.File(cache_path, "r") as handle:
        assert handle.attrs["schema_name"] == CACHE_SCHEMA_NAME
        assert handle.attrs["action_semantics"] == "absolute_gripper_pose_plus_open"
        assert handle["episode_ids"][:].tolist() == [0, 1, 2, 3, 4]
        episode = handle["episodes"]["0"]
        assert episode["xyz"].shape == (6, 4, 3)
        assert episode["state"].shape == (6, 8)
        assert episode["action"].shape == (6, 8)
        assert episode["joint_positions"].shape == (6, 7)
        assert episode["task_low_dim_state"].shape == (6, 3)
        assert np.all(episode["mask_id"][:] != 1)
        assert np.max(episode["xyz"][:, :, 0]) <= 1.0

    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_tasks"] == 1
    assert manifest["num_episodes"] == 5
    assert manifest["num_frames"] == 30


def test_rlbench_cache_store_loads_repeated_rows_and_extra_fields(tmp_path):
    cache_root = _build_fake_cache(tmp_path)
    keys, tasks = build_cache_keys(cache_root)
    assert tasks == ["reach_target"]
    store = RLBenchCacheStore(keys, keep_open=True)
    rows = store.load_episode_slices(
        0,
        0,
        [2, 0, 2],
        fields=("state", "joint_positions"),
    )
    assert rows["state"].shape == (3, 8)
    assert np.allclose(rows["state"][0], rows["state"][2])
    assert np.isclose(rows["state"][1, 0], 0.0)
    assert store.infer_dims() == (4, 8, 8)
    store.close()


def test_rlbench_dataset_builds_flexible_absolute_windows(tmp_path):
    cache_root = _build_fake_cache(tmp_path)
    dataset = RLBenchDataset(
        str(cache_root),
        split="train",
        obs_history=2,
        action_history=2,
        chunk_horizon=2,
        action_offset=1,
        action_representation="absolute",
        include_rgb=True,
        include_mask_id=True,
        extra_observation_fields=("joint_positions",),
        point_count=3,
        pad_episode_starts=True,
    )
    assert len(dataset.episodes) == 3
    assert len(dataset) == 12
    item = dataset[0]
    assert item["obs_hist"].shape == (2, 8)
    assert item["point_cloud_hist"].shape == (2, 3, 3)
    assert item["point_valid_hist"].shape == (2, 3)
    assert item["rgb_hist"].shape == (2, 3, 3)
    assert item["mask_id_hist"].shape == (2, 3)
    assert item["act_hist"].shape == (2, 8)
    assert item["future_actions"].shape == (2, 8)
    assert item["low_dim"]["joint_positions"].shape == (2, 7)
    assert item["obs_history_mask"].tolist() == [False, True]
    assert item["action_history_mask"].tolist() == [False, True]
    assert item["future_action_mask"].all()
    assert bool(item["action_is_absolute"])
    episode_weights = dataset.sampling_weights("episode_uniform")
    assert episode_weights.shape == (len(dataset),)
    assert torch.isclose(episode_weights.sum(), torch.tensor(3.0, dtype=torch.double))


def test_rlbench_delta_xyz_is_loader_time_and_round_trips(tmp_path):
    cache_root = _build_fake_cache(tmp_path)
    dataset = RLBenchDataset(
        str(cache_root),
        split="all",
        obs_history=2,
        action_history=2,
        chunk_horizon=2,
        action_offset=1,
        action_representation="delta_xyz",
        include_rgb=False,
        include_mask_id=False,
        return_absolute_actions=True,
    )
    item = dataset.item_from_episode_time(0, 0, 2)
    assert torch.allclose(item["act_hist"][:, 0], torch.tensor([0.1, 0.1]), atol=1e-5)
    assert torch.allclose(item["future_actions"][:, 0], torch.tensor([0.1, 0.1]), atol=1e-5)
    decoded = decode_action_chunk(
        item["future_actions"].numpy(),
        observation_state=item["obs_hist"].numpy(),
        representation="delta_xyz",
    )
    assert np.allclose(decoded, item["future_actions_absolute"].numpy(), atol=1e-5)


def test_rlbench_dataset_can_be_serialized_for_dataloader_workers(tmp_path):
    cache_root = _build_fake_cache(tmp_path)
    dataset = RLBenchDataset(
        str(cache_root),
        split="all",
        chunk_horizon=2,
        include_rgb=False,
        include_mask_id=False,
    )
    _ = dataset[0]
    import pickle as pickle_module

    restored = pickle_module.loads(pickle_module.dumps(dataset))
    restored_item = restored[0]
    assert restored_item["future_actions"].shape == (2, 8)
