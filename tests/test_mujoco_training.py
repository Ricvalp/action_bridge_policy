from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch
from phi_mujoco.offline import EpisodeData, get_integration, write_processed_bundle

from action_bridge.config import apply_overrides, load_config
from action_bridge.training import train_mujoco
from action_bridge.training.common import (
    build_dataset,
    build_model,
    load_config_from_checkpoint,
)
from action_bridge.training.train_mujoco import train


def _cache(root, integration_name, seed_offset=0):
    integration = get_integration(integration_name)
    state_dim = integration.spec.observations["state"].shape[0]
    episodes = []
    for index in range(3):
        rng = np.random.default_rng(index + seed_offset)
        steps = 10
        success = np.zeros(steps, dtype=bool)
        success[-1] = True
        episodes.append(
            EpisodeData(
                episode_index=index,
                seed=index,
                observations={
                    "state": rng.normal(size=(steps + 1, state_dim)).astype(np.float32)
                },
                actions=rng.uniform(-0.8, 0.8, (steps, 7)).astype(np.float32),
                action_history_padding=np.zeros(7, dtype=np.float32),
                rewards=success.astype(np.float64),
                terminated=success,
                truncated=np.zeros(steps, dtype=bool),
                success=success,
                termination_reason="success",
            )
        )
    return write_processed_bundle(
        root,
        integration=integration,
        episodes=episodes,
        splits={"train": [0, 1], "val": [2]},
    )


def test_diffusion_uses_identical_windows_and_normalization(tmp_path):
    bundle = _cache(tmp_path / "cache", "robomimic_square")
    datasets = []
    for name in ("mujoco_robomimic_square", "mujoco_robomimic_square_diffusion"):
        config = apply_overrides(
            load_config(name),
            [f"data.cache_root={bundle.root}", "logging.progress=false"],
        )
        datasets.append(build_dataset(config, split="train"))
    bridge, diffusion = datasets
    assert bridge.window_config == diffusion.window_config
    assert bridge.split_plan == diffusion.split_plan
    assert bridge.normalization == diffusion.normalization
    assert len(bridge) == len(diffusion)
    for index in range(len(bridge)):
        for name, value in bridge[index].items():
            np.testing.assert_array_equal(value, diffusion[index][name])


def test_diffusion_trains_and_resumes_with_shared_trainer(tmp_path):
    pytest.importorskip("diffusers")
    bundle = _cache(tmp_path / "cache", "robomimic_square")
    config = apply_overrides(
        load_config("mujoco_robomimic_square_diffusion"),
        [
            f"data.cache_root={bundle.root}",
            f"output_dir={tmp_path / 'runs'}",
            "run_id=diffusion",
            "device=cpu",
            "model.unet_channels=[8,16,32]",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.time_emb_dim=16",
            "model.num_inference_steps=5",
            "optim.batch_size=2",
            "optim.max_steps=2",
            "logging.progress=false",
            "logging.log_every_steps=1",
            "logging.eval_every_steps=1",
            "logging.validation_max_batches=1",
            "eval.batch_size=2",
            "eval.offline_max_batches=1",
        ],
    )
    run = train(config)
    checkpoint = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["config"]["checkpoint_metric"] == "val_action_mse"
    assert checkpoint["online_evaluation"]["policy_type"] == "diffusion"
    assert checkpoint["online_evaluation"]["action_horizon"] == 8
    assert checkpoint["online_evaluation"]["actions_per_plan"] == 4
    assert checkpoint["config"]["data"]["normalization"]["source_episode_indices"] == [0, 1]
    with (run / "metrics" / "train_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(float(row["loss"]) == float(row["noise_mse"]) for row in rows)
    assert all("action_mse" not in row and "path_kl" not in row for row in rows)
    with (run / "metrics" / "val_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert checkpoint["best_metric"] == min(float(row["action_mse"]) for row in rows)
    assert all("noise_mse" not in row for row in rows)
    resumed = load_config_from_checkpoint(run / "checkpoints" / "latest.pt")
    resumed.optim.max_steps = 3
    assert train(resumed) == run
    assert torch.load(run / "checkpoints" / "latest.pt", weights_only=False)["step"] == 3


def test_batch_order_is_shared_across_bridge_and_diffusion(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    bundle = _cache(tmp_path / "cache", "robomimic_square")
    observed = []
    loss_function = train_mujoco.model_loss

    def record_batch(model, batch, loss_config, *, global_step):
        observed.append((batch["episode_index"].tolist(), batch["time_index"].tolist()))
        return loss_function(model, batch, loss_config, global_step=global_step)

    monkeypatch.setattr(train_mujoco, "model_loss", record_batch)
    orders = []
    for name in ("mujoco_robomimic_square", "mujoco_robomimic_square_diffusion"):
        observed.clear()
        config = apply_overrides(load_config(name), [
            f"data.cache_root={bundle.root}", f"output_dir={tmp_path / 'runs'}",
            f"run_id={name}", "device=cpu", "seed=42",
            "model.hidden_dim=16", "model.h_emb_dim=16",
            "model.unet_channels=[8,16,32]", "model.time_emb_dim=16",
            "model.num_inference_steps=2", "optim.batch_size=2",
            "optim.max_steps=4", "logging.progress=false",
            "logging.eval_every_steps=2", "logging.validation_max_batches=1",
            "eval.batch_size=2", "eval.offline_max_batches=1",
        ])
        train(config)
        orders.append(list(observed))
    assert len(orders[0]) == 4
    assert orders[0] == orders[1]


@pytest.mark.parametrize(
    ("task", "latent_type"),
    [("square", "none"), ("tool_hang", "continuous")],
)
def test_train_robomimic_without_offline_test_split(tmp_path, task, latent_type):
    bundle = _cache(tmp_path / "cache", f"robomimic_{task}")
    config = apply_overrides(
        load_config(f"mujoco_robomimic_{task}"),
        [
            f"data.cache_root={bundle.root}",
            f"output_dir={tmp_path / 'runs'}",
            "run_id=smoke",
            "device=cpu",
            "chunk_horizon=4",
            "eval.actions_per_plan=2",
            f"model.latent_type={latent_type}",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.encoder_depth=1",
            "model.control_depth=1",
            "optim.batch_size=2",
            "optim.max_steps=2",
            "logging.progress=false",
            "logging.eval_every_steps=1",
            "logging.validation_max_batches=1",
            "eval.batch_size=2",
            "eval.offline_max_batches=1",
        ],
    )

    run = train(config)

    checkpoint = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert checkpoint["step"] == 2
    assert np.isfinite(checkpoint["best_metric"])
    assert checkpoint["config"]["checkpoint_metric"] == "val_action_mse"
    with (run / "metrics" / "val_metrics.csv").open() as stream:
        validation_rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in validation_rows] == [1, 2]
    assert all("val_loss" not in row for row in validation_rows)
    assert checkpoint["best_metric"] == min(
        float(row["action_mse"]) for row in validation_rows
    )
    assert checkpoint["config"]["data"]["normalization"]["source_episode_indices"] == [
        0,
        1,
    ]
    assert checkpoint["online_evaluation"]["integration"]["name"] == f"robomimic_{task}"
    assert (run / "metrics" / "val_metrics.json").is_file()
    assert not (run / "metrics" / "test_metrics.json").exists()
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["phi_mujoco"]["version"] == "0.2.0"
    assert provenance["action_bridge"]["lock_sha256"]

    # A programmatic caller must not silently resume on another same-shaped cache.
    changed = _cache(tmp_path / "different-cache", f"robomimic_{task}", seed_offset=10)
    resume_config = apply_overrides(
        load_config(f"mujoco_robomimic_{task}"),
        [
            f"data.cache_root={changed.root}",
            f"resume_from={run / 'checkpoints' / 'latest.pt'}",
            "device=cpu",
            "chunk_horizon=4",
            "eval.actions_per_plan=2",
        ],
    )
    with pytest.raises(ValueError, match="online metadata disagrees"):
        train(resume_config)

    resume_config = load_config_from_checkpoint(run / "checkpoints" / "latest.pt")
    resume_config.optim.max_steps = 3
    assert train(resume_config) == run
    resumed = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert resumed["step"] == 3
    assert resumed["config"]["checkpoint_metric"] == "val_action_mse"
    assert resumed["best_metric"] <= checkpoint["best_metric"]


def test_best_checkpoint_uses_validation_action_mse_not_training_loss(
    tmp_path, monkeypatch
):
    bundle = _cache(tmp_path / "cache", "robomimic_square")
    config = apply_overrides(
        load_config("mujoco_robomimic_square"),
        [
            f"data.cache_root={bundle.root}",
            f"output_dir={tmp_path / 'runs'}",
            "run_id=selection",
            "device=cpu",
            "model.latent_type=continuous",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.encoder_depth=1",
            "model.control_depth=1",
            "optim.batch_size=2",
            "optim.max_steps=2",
            "logging.progress=false",
            "logging.log_every_steps=1",
            "logging.eval_every_steps=1",
            "logging.validation_max_batches=1",
            "eval.batch_size=2",
            "eval.offline_max_batches=1",
        ],
    )
    scores = iter([0.1, 0.3, 0.3])  # Two validation calls and the final report.
    evaluations = []

    def evaluate(model, dataset, config, device, *, output_dir=None, max_batches=0):
        assert dataset.split == "val"
        evaluations.append((output_dir, max_batches))
        return {"action_mse": next(scores)}

    loss_function = train_mujoco.model_loss

    def decreasing_loss(model, batch, loss_config, *, global_step):
        result = loss_function(model, batch, loss_config, global_step=global_step)
        # Loss improves, while the held-out inference score deliberately worsens.
        result["loss"] = result["loss"] * 0 + (3 - global_step)
        return result

    logged = []
    monkeypatch.setattr(train_mujoco, "evaluate_mujoco_offline", evaluate)
    monkeypatch.setattr(train_mujoco, "model_loss", decreasing_loss)
    monkeypatch.setattr(
        train_mujoco,
        "log_wandb_scalars",
        lambda run, metrics, *, step, prefix: logged.append((prefix, step, metrics)),
    )

    run = train(config)

    best = torch.load(run / "checkpoints" / "best.pt", weights_only=False)
    latest = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert best["step"] == 1
    assert latest["step"] == 2
    assert best["best_metric"] == latest["best_metric"] == 0.1
    assert best["config"]["checkpoint_metric"] == "val_action_mse"
    assert evaluations == [(None, 1), (None, 1), (run, 1)]
    with (run / "metrics" / "val_metrics.csv").open() as stream:
        assert list(csv.DictReader(stream)) == [
            {"step": "1", "action_mse": "0.1"},
            {"step": "2", "action_mse": "0.3"},
        ]
    with (run / "metrics" / "train_metrics.csv").open() as stream:
        assert [float(row["loss"]) for row in csv.DictReader(stream)] == [2.0, 1.0]
    assert any(
        prefix == "val" and step == 1 and metrics["action_mse"] == 0.1
        for prefix, step, metrics in logged
    )


@pytest.mark.parametrize("old_metric", [None, "val_loss"])
def test_resume_rejects_checkpoint_with_other_selection_metric(tmp_path, old_metric):
    config = load_config("mujoco_robomimic_square")
    if old_metric is not None:
        config.checkpoint_metric = old_metric
    checkpoint = tmp_path / "old-selection.pt"
    torch.save({"config": config.to_dict(), "best_metric": -1.0}, checkpoint)
    config.resume_from = str(checkpoint)
    config.device = "cpu"
    with pytest.raises(ValueError, match="checkpoint_metric"):
        train(config)


def test_square_continuous_latent_configuration_has_about_five_million_parameters():
    config = apply_overrides(
        load_config("mujoco_robomimic_square"),
        [
            "model.latent_type=continuous",
            "model.hidden_dim=736",
            "model.h_emb_dim=736",
            "chunk_horizon=8",
            "eval.actions_per_plan=4",
        ],
    )
    model = build_model(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_069_631
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == 5_069_624
    )


@pytest.mark.parametrize("fail_training", [False, True])
def test_background_simulation_lifecycle(tmp_path, monkeypatch, fail_training):
    bundle = _cache(tmp_path / "cache", "robomimic_square")
    config = apply_overrides(
        load_config("mujoco_robomimic_square"),
        [
            f"data.cache_root={bundle.root}",
            f"output_dir={tmp_path / 'runs'}",
            "run_id=async-hooks",
            "device=cpu",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "optim.batch_size=2",
            "optim.max_steps=4",
            "logging.progress=false",
            "logging.eval_every_steps=4",
            "logging.validation_max_batches=1",
            "eval.offline_max_batches=1",
            "logging.sim_eval_enabled=true",
            "logging.sim_eval_every_steps=2",
        ],
    )
    events = []

    class Evaluator:
        def __init__(self, config, run_dir, wandb_run):
            events.append("init")

        def poll(self):
            events.append("poll")

        def submit(self, model, optimizer, config, step, best_mse):
            events.append(("submit", step))
            return True

        def finish(self):
            events.append("finish")

        def close(self):
            events.append("close")

    monkeypatch.setattr(train_mujoco, "AsyncMujocoEvaluator", Evaluator)
    if fail_training:
        def fail(*args, **kwargs):
            raise RuntimeError("training failed")

        monkeypatch.setattr(train_mujoco, "model_loss", fail)
        with pytest.raises(RuntimeError, match="training failed"):
            train(config)
        assert events == ["init", "poll", "close"]
    else:
        train(config)
        assert events == [
            "init", "poll", "poll", ("submit", 2), "poll", "poll",
            "finish", ("submit", 4), "finish", "close",
        ]


def test_wandb_scalars_use_chart_step_not_global_history_step():
    class Run:
        def __init__(self):
            self.rows = []

        def log(self, payload):
            self.rows.append(payload)

    run = Run()
    train_mujoco.log_wandb_scalars(
        run, {"loss": 0.2, "ignored": "text"}, step=100, prefix="train"
    )
    train_mujoco.log_wandb_scalars(
        run, {"action_mse": 0.1}, step=100, prefix="val"
    )
    assert run.rows == [
        {"train/loss": 0.2, "train/step": 100},
        {"val/action_mse": 0.1, "val/step": 100},
    ]
