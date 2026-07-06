import torch

from action_bridge.config import apply_overrides, load_config
from action_bridge.eval.eval_toy import default_eval_output_dir, evaluate_toy_model
from action_bridge.training.common import build_dataset, build_model


def test_default_eval_output_dir_uses_eval_root(tmp_path):
    path = default_eval_output_dir({"output_dir": str(tmp_path), "run_id": "toy delayed run"})
    assert path.parent == tmp_path / "eval"
    assert path.name.startswith("toy_delayed_run_")


def test_closed_loop_eval_reports_success_metrics():
    config = apply_overrides(
        load_config("toy_delayed_categorical"),
        [
            "device=cpu",
            "trajectory_len=16",
            "chunk_horizon=4",
            "data.num_contexts=8",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.z_embed_dim=4",
            "optim.batch_size=8",
            "eval.closed_loop=true",
            "eval.closed_loop_episodes=2",
            "eval.n_exec=2",
        ],
    )
    dataset = build_dataset(config, split="test")
    model = build_model(config)
    metrics = evaluate_toy_model(model, dataset, config, torch.device("cpu"), output_dir=None, max_batches=1)
    assert "closed_loop_success_rate" in metrics
    assert "closed_loop_goal_error" in metrics
    assert "closed_loop_chunk_boundary_discontinuity" in metrics
