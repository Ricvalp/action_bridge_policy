"""Small end-to-end checks, not scientific performance assertions."""

import json

import numpy as np
import pytest
import torch

from workspace.surface_contact_bc.config import method_config
from workspace.surface_contact_bc.data import make_dataset, EpisodeWindows
from workspace.surface_contact_bc.env import Context
from workspace.surface_contact_bc.evaluation import rollout, episode_metrics
from workspace.surface_contact_bc.train import train, load_checkpoint, tensor_windows


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("contact-training") / "data"
    make_dataset(root, train_episodes=4, val_episodes=2, test_episodes=2)
    return root


def tiny_config(method):
    return method_config(method) | dict(steps=3, batch_size=8, hidden_dim=32,
        h_emb_dim=32, unet_channels=[16, 32], num_inference_steps=2,
        val_every=2, val_windows=8, threads=1)


@pytest.fixture(scope="module")
def full_checkpoint(dataset, tmp_path_factory):
    output, _ = train(tiny_config("action_bridge_contact_frame_full"), dataset,
                       tmp_path_factory.mktemp("trained-reference") / "run", progress=False)
    return output / "best.pt"


@pytest.mark.parametrize("method", ["action_bridge_isotropic", "action_bridge_contact_frame_no_kl",
    "diffusion_world", "diffusion_contact_frame", "diffusion_residual"])
def test_training_reload_and_real_state_rollout(dataset, tmp_path, method, full_checkpoint):
    output, summary = train(tiny_config(method), dataset, tmp_path / method,
                            reference_checkpoint=full_checkpoint, progress=False)
    assert np.isfinite(summary["best_val_action_mse"])
    for filename in ("config.json", "provenance.json", "metrics.csv", "best.pt", "latest.pt"):
        assert (output / filename).exists()
    model, normalizer, payload = load_checkpoint(output / "best.pt")
    windows = EpisodeWindows(dataset, split="test")
    context = Context.from_dict(windows.episodes[0]["context"])
    context.duration = .16  # Exercise actual simulation; do not claim task success.
    trace = rollout(model, normalizer, context, n_exec=2, seed=101)
    second = rollout(model, normalizer, context, n_exec=2, seed=101)
    np.testing.assert_array_equal(trace["actions"], second["actions"])
    assert np.isfinite(trace["observations"]).all()
    assert len(trace["actions"]) == 4
    metrics = episode_metrics(trace, context)
    assert "task_success" in metrics and "clean_success" in metrics
    json.dumps(metrics, allow_nan=False)
    assert payload["provenance"]["dataset_episodes_sha256"]


def test_reference_only_uses_exact_fitted_parameters(full_checkpoint, dataset):
    controlled, _, payload = load_checkpoint(full_checkpoint)
    reference, _, _ = load_checkpoint(full_checkpoint, reference_only=True)
    for key in payload["model"]:
        torch.testing.assert_close(controlled.state_dict()[key], reference.state_dict()[key])
    data = tensor_windows(EpisodeWindows(dataset), "cpu")
    _, diagnostics = reference.generate(data["obs_hist"][:2], data["act_hist"][:2], diagnostics=True)
    assert torch.count_nonzero(diagnostics["path_kl"]) == 0


def test_validation_never_refits_normalization(dataset, tmp_path):
    config = tiny_config("action_bridge_contact_frame_full") | {"train_episodes": 2}
    output, _ = train(config, dataset, tmp_path / "run", progress=False)
    _, normalizer, _ = load_checkpoint(output / "best.pt")
    assert normalizer.to_dict() == EpisodeWindows(dataset, limit=2).normalizer.to_dict()
