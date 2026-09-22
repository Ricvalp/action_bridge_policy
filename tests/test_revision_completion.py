"""Causal completion and whole-plan prior checks, independent of a simulator."""

import pytest
import torch

from action_bridge.plan_revision.completion import LearnedCompletion, complete_plan


def batch_for(horizon, dim, obs_dim=5, obs_history=2, act_history=2, batch_size=3):
    return {
        "obs_hist": torch.randn(batch_size, obs_history, obs_dim),
        "act_hist": torch.randn(batch_size, act_history, dim),
        "future_actions": torch.randn(batch_size, horizon, dim),
        "valid_mask": torch.ones(batch_size, horizon),
    }


@pytest.mark.parametrize("horizon,dim,obs_dim,obs_history,act_history", [(16, 2, 5, 2, 2), (9, 6, 7, 3, 3)])
def test_portable_loss_completion_and_reload(horizon, dim, obs_dim, obs_history, act_history):
    torch.manual_seed(7)
    config = dict(obs_dim=obs_dim, action_dim=dim, obs_history=obs_history,
                  action_history=act_history, horizon=horizon)
    model = LearnedCompletion(**config)
    batch = batch_for(horizon, dim, obs_dim, obs_history, act_history)
    loss = model.loss(batch)
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    old = torch.randn(3, horizon, dim)
    for executed in (1, horizon // 2, horizon - 1):
        generated = model.complete(old, executed, batch["obs_hist"], batch["act_hist"], torch.arange(3))
        assert generated.shape == old.shape
        assert torch.equal(generated[:, :horizon - executed], old[:, executed:])
        assert generated.isfinite().all()
    model.precision_scale.fill_(0.3)
    reloaded = LearnedCompletion(**config)
    reloaded.load_state_dict(model.state_dict())
    assert torch.equal(reloaded.precision_scale, model.precision_scale)
    for a, b in zip(model.coefficients(batch["obs_hist"], batch["act_hist"]),
                    reloaded.coefficients(batch["obs_hist"], batch["act_hist"])):
        assert torch.equal(a, b)


def test_repeat_and_fixed_damped_formula_and_source_immutability():
    old = torch.arange(8.0).reshape(1, 8, 1)
    original = old.clone()
    obs, hist = torch.zeros(1, 2, 3), torch.zeros(1, 2, 1)
    repeat = complete_plan(old, 4, obs, hist, 0)
    fixed = complete_plan(old, 4, obs, hist, 1)
    torch.testing.assert_close(repeat[0, :, 0], torch.tensor([4., 5., 6., 7., 7., 7., 7., 7.]))
    torch.testing.assert_close(fixed[0, 4:, 0], 7 + torch.tensor([.8, .64, .512, .4096]).cumsum(0))
    assert torch.equal(old, original)
    assert torch.equal(complete_plan(old, 0, obs, hist, 0), old)
    with pytest.raises(ValueError, match="bootstrap"):
        complete_plan(old, 8, obs, hist, 0)


def test_learned_uses_current_tail_indices_not_extrapolated_embeddings():
    class KnownCoefficients:
        horizon, action_dim, robot_dt = 8, 1, 1.0

        def coefficients(self, obs_hist, act_hist):
            index = torch.arange(8.0).reshape(1, 8, 1)
            return index, torch.ones_like(index), torch.ones_like(index)

    old = torch.zeros(1, 8, 1)
    generated = complete_plan(old, 3, torch.zeros(1, 2, 3), torch.zeros(1, 2, 1), 2, KnownCoefficients())
    # With K=gamma=1, q_next = m(index) exactly.
    torch.testing.assert_close(generated[0, :, 0], torch.tensor([0., 0., 0., 0., 0., 5., 6., 7.]))


def test_future_labels_cannot_affect_coefficients_or_prior():
    model = LearnedCompletion(5, 2, 2, 2, 16)
    batch = batch_for(16, 2)
    initial = model.coefficients(batch["obs_hist"], batch["act_hist"])
    prior = model.prior(batch["obs_hist"], batch["act_hist"], torch.ones(2))
    batch["future_actions"].fill_(10000)
    for before, after in zip(initial, model.coefficients(batch["obs_hist"], batch["act_hist"])):
        assert torch.equal(before, after)
    other = model.prior(batch["obs_hist"], batch["act_hist"], torch.ones(2))
    assert torch.equal(prior["precision"], other["precision"])
    assert torch.equal(prior["mean"], other["mean"])


@pytest.mark.parametrize("horizon,dim", [(16, 2), (9, 6)])
def test_prior_mean_matches_innovation_normal_equations(horizon, dim):
    model = LearnedCompletion(5, dim, 2, 2, horizon).double()
    batch = {key: value.double() for key, value in batch_for(horizon, dim).items()}
    variance = torch.linspace(0.4, 1., dim, dtype=torch.float64)
    ridge = 0.05
    prior = model.prior(batch["obs_hist"], batch["act_hist"], variance, ridge=ridge)
    at_mean = dict(batch, future_actions=prior["mean"].reshape(-1, horizon, dim).detach().requires_grad_())
    residual = model.residuals(at_mean)
    anchor = batch["act_hist"][:, -1:, :]
    energy = 0.5 * (residual.square() / variance).sum()
    energy = energy + 0.5 * ridge * (at_mean["future_actions"] - anchor).square().sum()
    gradient, = torch.autograd.grad(energy, at_mean["future_actions"])
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-11, rtol=0)
    rates = torch.linalg.eigvalsh(prior["precision"])
    assert (rates >= 1e-3 - 1e-10).all() and (rates <= 4 + 1e-10).all()
    assert (torch.linalg.eigvalsh(prior["mobility"]) > 0).all()
    assert prior["mean"].shape == (3, horizon * dim)


def test_banded_mobility_is_spd_and_relaxation_rates_are_bounded():
    model = LearnedCompletion(4, 3, 2, 2, 9)
    batch = batch_for(9, 3, obs_dim=4)
    result = model.prior(batch["obs_hist"], batch["act_hist"], torch.full((3,), .01), mobility_smoothing=.7)
    mobility = result["mobility"]
    assert torch.linalg.eigvalsh(mobility).min() > 0
    assert not torch.equal(mobility, torch.eye(27))
    chol = torch.linalg.cholesky(mobility)
    # L^T P L has the same eigenvalues as D P.
    rates = torch.linalg.eigvalsh(chol.T @ result["precision"] @ chol)
    assert rates.min() > 0 and rates.max() < 4.00001


def test_partial_horizons_and_unsupported_semantics_fail_explicitly():
    model = LearnedCompletion(5, 2, 2, 2, 16)
    batch = batch_for(16, 2)
    batch["valid_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="fully valid"):
        model.loss(batch)
    del batch["valid_mask"]
    with pytest.raises(ValueError, match="fully valid"):
        model.residuals(batch)
    with pytest.raises(ValueError, match="absolute_target"):
        LearnedCompletion(5, 2, 2, 2, 16, action_semantics="delta")
    with pytest.raises(ValueError, match="absolute_target"):
        complete_plan(torch.randn(1, 8, 2), 4, torch.randn(1, 2, 5), torch.randn(1, 2, 2), 0,
                      action_semantics="quaternion")


def test_gain_bounds_satisfy_robot_index_stability():
    model = LearnedCompletion(5, 2, 2, 2, 16, robot_dt=.2)
    batch = batch_for(16, 2)
    m, stiffness, damping = model.coefficients(batch["obs_hist"], batch["act_hist"])
    dt = model.robot_dt
    assert (m.abs() < model.attractor_limit).all()
    assert ((dt * damping > 0) & (dt * damping < 2)).all()
    assert (dt**2 * stiffness < 4 - 2 * dt * damping).all()
    assert (stiffness > 0).all()
