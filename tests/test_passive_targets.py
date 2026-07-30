import torch

from action_bridge.training.passive_targets import (
    contact_direction_projection,
    ema_smoothed_velocity_projection,
)


def test_ema_smoothed_velocity_is_causal_and_dynamically_consistent():
    q_seq = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]])
    p_seq = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])

    out = ema_smoothed_velocity_projection(q_seq, p_seq, ema_decay=0.5)

    expected_p = torch.tensor([[[0.5, 0.0], [1.25, 0.0], [2.125, 0.0]]])
    assert torch.allclose(out["p_bar_next"], expected_p)
    assert torch.allclose(out["q_bar_next"], q_seq[:, :-1] + expected_p)
    assert torch.all(out["projection_coefficient"] == 0.5)


def test_contact_direction_damps_lateral_velocity_only_near_t():
    q_seq = torch.zeros(2, 2, 2)
    p_seq = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    future_obs_raw = torch.tensor(
        [
            [[140.0, 256.0, 200.0, 256.0, 0.0]],
            [[500.0, 500.0, 200.0, 256.0, 0.0]],
        ]
    )

    out = contact_direction_projection(
        q_seq,
        p_seq,
        future_obs_raw,
        lambda_parallel=1.0,
        lambda_perp=0.0,
        contact_distance=18.0,
        contact_temperature=2.0,
    )

    near = out["p_bar_next"][0, 0]
    far = out["p_bar_next"][1, 0]
    assert out["contact_score"][0, 0] > 0.95
    assert out["contact_score"][1, 0] < 0.01
    assert near[0] > 0.99
    assert near[1] < 0.05
    assert torch.allclose(far, torch.tensor([1.0, 1.0]), atol=1e-2)
    assert torch.allclose(out["q_bar_next"], q_seq[:, :-1] + out["p_bar_next"])
