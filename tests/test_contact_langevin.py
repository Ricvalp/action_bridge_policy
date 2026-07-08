import torch

from action_bridge.config import apply_overrides, load_config
from action_bridge.data.action_coordinates import ActionCoordinateAdapter
from action_bridge.eval.rollout import generate_chunk
from action_bridge.models.references import ContactLangevinReference
from action_bridge.training.common import build_model


def test_absolute_action_roundtrip():
    adapter = ActionCoordinateAdapter("absolute_action", dt=1.0, action_dim=2)
    batch = {
        "actions": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        "prev_actions": torch.tensor([[[0.0, 0.0], [0.5, 0.5]]]),
    }
    q_seq = adapter.build_q_sequence(batch)
    assert torch.allclose(adapter.decode_raw_actions(q_seq), batch["actions"])


def test_absolute_from_delta_roundtrip():
    adapter = ActionCoordinateAdapter("absolute_from_delta", dt=1.0, action_dim=2)
    batch = {
        "actions": torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]),
        "prev_actions": torch.tensor([[[0.0, 0.0], [0.5, 0.5]]]),
        "current_position": torch.tensor([[2.0, 3.0]]),
    }
    q_seq = adapter.build_q_sequence(batch)
    assert torch.allclose(adapter.decode_raw_actions(q_seq), batch["actions"])


def test_contact_damping_decreases_velocity_without_control():
    ref = ContactLangevinReference(action_dim=2, h_emb_dim=4, gamma_const=0.25, potential_type="none")
    q = torch.zeros(3, 2)
    p = torch.tensor([[1.0, 0.0], [0.0, -2.0], [3.0, 4.0]])
    h = torch.zeros(3, 4)
    _, p_next, _ = ref.reference_step(q, p, h, 0)
    assert torch.all(torch.linalg.norm(p_next, dim=-1) <= torch.linalg.norm(p, dim=-1) + 1e-6)


def test_contact_rollout_shape():
    config = apply_overrides(
        load_config("toy_delayed_contact_absolute_from_delta"),
        [
            "obs_history=2",
            "action_history=2",
            "chunk_horizon=4",
            "model.hidden_dim=16",
            "model.h_emb_dim=16",
            "model.z_embed_dim=4",
            "model.num_categories=2",
        ],
    )
    model = build_model(config)
    obs_hist = torch.zeros(5, 2, 4)
    act_hist = torch.zeros(5, 2, 2)
    out = generate_chunk(model, obs_hist, act_hist, deterministic=True)
    assert out["actions"].shape == (5, 4, 2)
    assert out["q_seq"].shape == (5, 5, 2)
    assert out["p_seq"].shape == (5, 5, 2)
    assert out["path_kl_energy"].shape == (5,)


def test_zero_control_kl_formula_is_zero():
    u = torch.zeros(4, 2)
    kl = 0.5 * (u**2).sum(dim=-1).mean()
    assert kl.item() == 0.0
