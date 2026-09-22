import json

import numpy as np
import pytest
from PIL import Image

from workspace.surface_contact_bc.data import make_dataset
from workspace.surface_contact_bc.visualize import main, visualize_episode


def test_expert_visualizer_writes_plot_and_reloadable_artifacts(tmp_path):
    output, metrics = visualize_episode(tmp_path / "view", seed=5)
    assert metrics["task_success"] and metrics["clean_success"]
    for name in ("rollout.png", "context.json", "metrics.json", "source.json", "episode.npz"):
        assert (output / name).stat().st_size > 0
    with np.load(output / "episode.npz", allow_pickle=False) as trace:
        assert trace["observations"].shape == (151, 11)
        assert trace["actions"].shape == (150, 2)
        assert trace["normal_force"].shape == (151,)
    assert json.loads((output / "context.json").read_text())["seed"] == 5
    with pytest.raises(FileExistsError):
        visualize_episode(output, seed=5)


def test_dataset_visualizer_cli_and_gif(tmp_path, capsys):
    dataset = tmp_path / "demos"
    make_dataset(dataset, train_episodes=1, val_episodes=1, test_episodes=1)
    output = tmp_path / "view"
    main(["--dataset", str(dataset), "--split", "val", "--episode", "0",
          "--output", str(output), "--gif", "--fps", "1"])
    assert json.loads(capsys.readouterr().out)["clean_success"]
    source = json.loads((output / "source.json").read_text())
    assert source["episode_id"] == "val_000000" and len(source["manifest_sha256"]) == 64
    with Image.open(output / "rollout.gif") as image:
        assert image.n_frames >= 5


def test_visualizer_rejects_missing_episode(tmp_path):
    dataset = tmp_path / "demos"
    make_dataset(dataset, train_episodes=1, val_episodes=0, test_episodes=0)
    with pytest.raises(ValueError, match="Episode index"):
        visualize_episode(tmp_path / "view", dataset=dataset, episode=1)
