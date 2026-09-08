from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from phi_mujoco.offline import EpisodeData, get_integration, write_processed_bundle

from action_bridge.config import apply_overrides, load_config
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
