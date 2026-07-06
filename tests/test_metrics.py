import torch

from action_bridge.eval.metrics import compute_toy_metrics


def _batch(actions):
    b, h, _ = actions.shape
    positions = torch.cat([torch.tensor([[[0.2, 0.7]]]).repeat(b, 1, 1), torch.tensor([[[0.3, 0.7]]]).repeat(b, h, 1)], dim=1)
    positions[:, 1:] = positions[:, :1] + torch.cumsum(actions, dim=1)
    return {
        "act_hist": torch.zeros(b, 2, 2),
        "future_actions": actions,
        "future_positions": positions,
        "context": {
            "goal": positions[:, -1],
            "obstacle_center": torch.tensor([[0.5, 0.5]]).repeat(b, 1),
            "obstacle_radius": torch.full((b,), 0.13),
        },
    }


def test_metrics_detect_collision_and_modes():
    actions = torch.zeros(2, 4, 2)
    actions[0, :, 0] = 0.1
    actions[0, :, 1] = -0.05
    actions[1, :, 0] = 0.1
    actions[1, :, 1] = -0.15
    batch = _batch(actions)
    metrics = compute_toy_metrics(actions, batch, benchmark="toy_delayed")
    assert "collision_rate" in metrics
    assert "hybrid_rate" in metrics
    assert metrics["goal_error"] < 1e-6
