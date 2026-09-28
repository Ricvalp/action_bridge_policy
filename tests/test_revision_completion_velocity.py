"""Whole-retained-chunk slope fitting changes only learned-tail initialization."""
import pytest
import torch

from action_bridge.plan_revision.completion import (
    COMPLETION_VELOCITY_WEIGHTINGS, LearnedCompletion, complete_plan, completion_velocity,
)


FITS = COMPLETION_VELOCITY_WEIGHTINGS[1:]


@pytest.mark.parametrize("weighting", FITS)
@pytest.mark.parametrize("length,dt", [(2, .2), (8, 1.), (32, .1)])
def test_matches_weighted_least_squares_with_free_intercept(weighting, length, dt):
    generator = torch.Generator().manual_seed(17)
    old = torch.randn(3, length + 5, 4, generator=generator, dtype=torch.float64)
    index = torch.arange(length, dtype=torch.float64)
    weights = {"uniform": torch.ones_like(index), "linear": index + 1,
               "exp_half": 2 ** (-2 * (1 - index / (length - 1))),
               "exp_quarter": 2 ** (-4 * (1 - index / (length - 1)))}[weighting]
    design = torch.stack([torch.ones_like(index), index * dt], dim=-1)
    expected = torch.stack([
        torch.linalg.lstsq(design * weights.sqrt()[:, None], targets * weights.sqrt()[:, None]).solution[1]
        for targets in old[:, 5:]
    ])
    actual = completion_velocity(old, 5, robot_dt=dt, weighting=weighting)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("weighting", FITS)
def test_fit_uses_interior_but_never_executed_prefix(weighting):
    old = torch.arange(16., dtype=torch.float64).reshape(1, 16, 1)
    initial = completion_velocity(old, 8, weighting=weighting)
    changed = old.clone()
    changed[:, :8] = 100000.
    torch.testing.assert_close(completion_velocity(changed, 8, weighting=weighting), initial)
    changed[:, 9] += 2.
    assert not torch.allclose(completion_velocity(changed, 8, weighting=weighting), initial)
    torch.testing.assert_close(completion_velocity(changed, 8, weighting="last_pair"), initial)


@pytest.mark.parametrize("weighting", FITS)
def test_constant_linear_short_and_noisy_chunks(weighting):
    old = torch.arange(16.).reshape(1, 16, 1) * torch.tensor([1., .25])[None, None]
    expected = torch.tensor([[1., .25]])
    torch.testing.assert_close(completion_velocity(old, 8, weighting=weighting), expected)
    torch.testing.assert_close(completion_velocity(old, 14, weighting=weighting), expected)
    torch.testing.assert_close(completion_velocity(old, 15, weighting=weighting), torch.zeros_like(expected))
    constant = torch.full_like(old, 123.)
    torch.testing.assert_close(completion_velocity(constant, 8, weighting=weighting), torch.zeros_like(expected))
    noisy = old.clone()
    noisy[:, -1, 1] += 4.
    fitted_error = (completion_velocity(noisy, 8, weighting=weighting) - expected).norm()
    pair_error = (completion_velocity(noisy, 8, weighting="last_pair") - expected).norm()
    assert fitted_error < pair_error


def test_recent_weights_follow_a_bend_more_than_uniform():
    time = torch.arange(8., dtype=torch.float64)
    old = torch.stack([time, time.square()], dim=-1)[None]
    uniform = completion_velocity(old, 0, weighting="uniform")
    half = completion_velocity(old, 0, weighting="exp_half")
    quarter = completion_velocity(old, 0, weighting="exp_quarter")
    assert uniform[0, 1] < half[0, 1] < quarter[0, 1]
    torch.testing.assert_close(quarter[:, 0], torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_shape_gradient_and_translation(dtype):
    old = torch.arange(16., dtype=dtype).reshape(1, 16, 1).expand(2, -1, 3).clone().requires_grad_()
    velocity = completion_velocity(old, 8, weighting="linear")
    assert velocity.dtype == dtype and velocity.device == old.device and velocity.shape == (2, 3)
    torch.testing.assert_close(velocity, torch.ones_like(velocity))
    velocity.float().sum().backward()
    assert torch.isfinite(old.grad).all() and torch.equal(old.grad[:, :8], torch.zeros_like(old.grad[:, :8]))
    translated = completion_velocity(old.detach() + 32, 8, weighting="linear")
    torch.testing.assert_close(translated, velocity.detach())


class DampedCoefficients:
    horizon, action_dim, robot_dt = 16, 2, 1.

    def coefficients(self, obs_hist, act_hist):
        shape = (len(obs_hist), self.horizon, self.action_dim)
        return torch.zeros(shape), torch.zeros(shape), torch.full(shape, .2)

    @staticmethod
    def direct_tail(old, obs_hist, act_hist):
        return torch.full((len(old), 8, 2), 7.)


DampedCoefficients.direct_tail.execute = 8


@pytest.mark.parametrize("weighting", FITS)
def test_exact_anchor_overlap_and_only_learned_mode_changes(weighting):
    old = torch.arange(16.).reshape(1, 16, 1).expand(4, -1, 2).clone()
    old[:, -1, 1] += 4.
    before = old.clone()
    obs, history, modes = torch.zeros(4, 2, 5), torch.zeros(4, 2, 2), torch.arange(4)
    reference = DampedCoefficients()
    result = complete_plan(old, 8, obs, history, modes, reference, velocity_weighting=weighting)
    baseline = complete_plan(old, 8, obs, history, modes, reference, velocity_weighting="last_pair")
    torch.testing.assert_close(result[:, :8], old[:, 8:], atol=0, rtol=0)
    torch.testing.assert_close(result[[0, 1, 3]], baseline[[0, 1, 3]], atol=0, rtol=0)
    fitted = completion_velocity(old[2:3], 8, weighting=weighting)
    expected_tail = old[2, -1] + (.8 ** torch.arange(1., 9.)).cumsum(0)[:, None] * fitted
    torch.testing.assert_close(result[2, 8:], expected_tail)
    assert not torch.equal(result[2, 8:], baseline[2, 8:])
    assert torch.equal(old, before)


def test_model_weights_coefficients_and_prior_are_unchanged():
    model = LearnedCompletion(5, 2, 2, 2, 8, hidden_dim=8).eval()
    old, obs, history = torch.randn(2, 8, 2), torch.randn(2, 2, 5), torch.randn(2, 2, 2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    prior = model.prior(obs, history, torch.ones(2))
    for weighting in FITS:
        model.complete(old, 4, obs, history, 2, velocity_weighting=weighting)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    after = model.prior(obs, history, torch.ones(2))
    for key in ("precision", "mean", "mobility"):
        torch.testing.assert_close(prior[key], after[key], atol=0, rtol=0)


@pytest.mark.parametrize("kwargs", [{"weighting": "unknown"}, {"robot_dt": 0.},
                                     {"robot_dt": float("nan")}, {"robot_dt": float("inf")}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        completion_velocity(torch.zeros(1, 8, 2), 4, **kwargs)


def test_empty_retained_chunk_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        completion_velocity(torch.zeros(1, 8, 2), 8)
