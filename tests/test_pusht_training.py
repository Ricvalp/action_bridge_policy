from pathlib import Path

import numpy as np

from action_bridge.config import apply_overrides, load_config
from action_bridge.data.pusht_adapter import PushTLowDimDataset
from action_bridge.eval.eval_pusht import representative_plot_indices
from action_bridge.training.train_pusht import train


def write_tiny_pusht_npz(path: Path) -> None:
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(5, 14, 6)).astype(np.float32)
    actions = rng.normal(scale=0.05, size=(5, 14, 2)).astype(np.float32)
    np.savez(path, observations=observations, actions=actions)


def test_pusht_npz_dataset_shapes(tmp_path):
    dataset_path = tmp_path / "pusht_tiny.npz"
    write_tiny_pusht_npz(dataset_path)
    dataset = PushTLowDimDataset(
        dataset_path=str(dataset_path),
        backend="npz",
        split="train",
        obs_history=2,
        action_history=2,
        chunk_horizon=4,
    )
    item = dataset[0]
    assert dataset.obs_dim == 6
    assert dataset.action_dim == 2
    assert item["obs_hist"].shape == (2, 6)
    assert item["act_hist"].shape == (2, 2)
    assert item["future_actions"].shape == (4, 2)


def test_representative_plot_indices_spread_across_episodes(tmp_path):
    dataset_path = tmp_path / "pusht_tiny.npz"
    write_tiny_pusht_npz(dataset_path)
    dataset = PushTLowDimDataset(
        dataset_path=str(dataset_path),
        backend="npz",
        split="all",
        obs_history=2,
        action_history=2,
        chunk_horizon=4,
    )
    indices = representative_plot_indices(dataset, max_items=4)
    episodes = {dataset.indices[idx][0] for idx in indices}
    assert len(indices) == 4
    assert len(episodes) == 4


def test_train_pusht_smoke(tmp_path):
    dataset_path = tmp_path / "pusht_tiny.npz"
    write_tiny_pusht_npz(dataset_path)
    config = apply_overrides(
        load_config("pusht_lowdim_continuous"),
        [
            "device=cpu",
            f"output_dir={tmp_path}",
            "run_id=pusht_smoke",
            f"data.dataset_path={dataset_path}",
            "data.backend=npz",
            "chunk_horizon=4",
            "optim.max_steps=1",
            "optim.batch_size=4",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.z_embed_dim=4",
            "model.z_dim=2",
            "logging.full_eval_every_steps=0",
            "eval.offline_rollout_episodes=1",
            "inference.num_samples=2",
        ],
    )
    run_dir = train(config)
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "metrics" / "test_metrics.json").exists()
