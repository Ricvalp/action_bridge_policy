from action_bridge.data.toy_obstacle import DelayedBranchObstacleDataset
from action_bridge.eval.selection import history_index_metadata, representative_history_indices, select_history_index


def _dataset():
    return DelayedBranchObstacleDataset(
        {
            "num_contexts": 16,
            "trajectory_len": 64,
            "chunk_horizon": 16,
            "obs_history": 2,
            "action_history": 2,
            "shared_prefix_steps": 8,
        },
        split="all",
    )


def test_select_history_index_uses_middle_time_not_first_chunk():
    dataset = _dataset()
    idx = select_history_index(dataset, trajectory_fraction=0.5, time_fraction=0.5)
    traj_id, time_index = dataset.indices[idx]
    first_time = min(t for candidate_traj, t in dataset.indices if candidate_traj == traj_id)
    last_time = max(t for candidate_traj, t in dataset.indices if candidate_traj == traj_id)

    assert time_index > first_time
    assert abs(time_index - (first_time + last_time) / 2) <= 1


def test_representative_history_indices_spread_trajectory_and_time():
    dataset = _dataset()
    indices = representative_history_indices(dataset, count=6, time_fractions=(0.0, 0.5, 1.0))
    metadata = history_index_metadata(dataset, indices)
    trajectory_ids = {item["trajectory_id"] for item in metadata}
    time_indices = {item["time_index"] for item in metadata}

    assert len(indices) == 6
    assert len(trajectory_ids) >= 2
    assert len(time_indices) >= 3
