import torch

from action_bridge.models.references import BrownianReference, ContinuationReference


def test_brownian_reference_returns_previous_action():
    ref = BrownianReference(action_dim=2, sigma=0.1)
    a = torch.tensor([[1.0, -2.0]])
    prev = torch.zeros_like(a)
    h = torch.zeros(1, 4)
    mu, log_sigma = ref(a, prev, h, 0)
    assert torch.allclose(mu, a)
    assert log_sigma.shape == a.shape


def test_continuation_reference_uses_action_velocity():
    ref = ContinuationReference(action_dim=2, sigma=0.1, alpha=0.5)
    a = torch.tensor([[2.0, 1.0]])
    prev = torch.tensor([[1.0, 3.0]])
    h = torch.zeros(1, 4)
    mu, _ = ref(a, prev, h, 0)
    assert torch.allclose(mu, torch.tensor([[2.5, 0.0]]), atol=1e-5)
