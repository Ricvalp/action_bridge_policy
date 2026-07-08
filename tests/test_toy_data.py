import torch

from action_bridge.data.toy_obstacle import DelayedBranchObstacleDataset, generate_delayed_branch_arrays


def test_delayed_dataset_schema_shapes():
    ds = DelayedBranchObstacleDataset(
        {
            "num_contexts": 8,
            "trajectory_len": 24,
            "chunk_horizon": 6,
            "obs_history": 2,
            "action_history": 2,
            "shared_prefix_steps": 4,
        },
        split="train",
    )
    item = ds[0]
    assert item["obs_hist"].shape == (2, 4)
    assert item["act_hist"].shape == (2, 2)
    assert item["future_actions"].shape == (6, 2)
    assert item["future_positions"].shape == (7, 2)
    assert item["current_position"].shape == (2,)
    assert item["previous_position"].shape == (2,)
    assert item["action_is_absolute"].shape == ()
    assert item["true_mode_probs"].shape == (2,)
    assert "goal" in item["context"]


def test_delayed_dataset_absolute_actions_are_future_targets():
    ds = DelayedBranchObstacleDataset(
        {
            "num_contexts": 8,
            "trajectory_len": 24,
            "chunk_horizon": 6,
            "obs_history": 2,
            "action_history": 2,
            "shared_prefix_steps": 4,
            "train_absolute_actions": True,
        },
        split="train",
    )
    item = ds[0]
    assert bool(item["action_is_absolute"])
    assert torch.allclose(item["act_hist"][-1], item["current_position"])
    assert torch.allclose(item["future_actions"], item["future_positions"][1:])


def test_paired_delayed_prefix_actions_are_identical():
    arrays = generate_delayed_branch_arrays(
        num_contexts=4,
        paired_fraction=1.0,
        trajectory_len=24,
        shared_prefix_steps=5,
        action_noise_std=0.0,
        seed=3,
    )
    first = arrays["actions"][0, :5]
    second = arrays["actions"][1, :5]
    assert torch.allclose(first, second)
    assert int(arrays["modes"][0]) == -int(arrays["modes"][1])


def test_delayed_trajectories_clear_obstacle():
    arrays = generate_delayed_branch_arrays(
        num_contexts=64,
        paired_fraction=0.5,
        trajectory_len=64,
        shared_prefix_steps=8,
        seed=11,
    )
    positions = arrays["positions"]
    centers = arrays["obstacle_centers"][:, None]
    radii = arrays["obstacle_radii"][:, None]
    clearance = torch.linalg.norm(positions - centers, dim=-1) - radii
    assert float(clearance.min()) > 0.0
