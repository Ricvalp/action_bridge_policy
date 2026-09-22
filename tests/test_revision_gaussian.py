"""Short mathematical checks, not learning experiments."""

import math

import pytest
import torch

from action_bridge.plan_revision.gaussian import GaussianReference


@pytest.fixture(autouse=True)
def small_torch_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def reference(kind="ou", *, batch=2, n=3, mobility=None, dtype=torch.float64):
    generator = torch.Generator().manual_seed(25)
    factor = torch.randn(batch, n, n, generator=generator, dtype=dtype)
    precision = factor @ factor.transpose(-1, -2) / n + 0.4 * torch.eye(n, dtype=dtype)
    mean = torch.randn(batch, n, generator=generator, dtype=dtype)
    return GaussianReference(precision, mean, kind=kind, mobility=mobility)


def dense_matrices(ref):
    n, b = ref.n, ref.batch_size
    eye = torch.eye(n, dtype=torch.float64).expand(b, -1, -1)
    zero = torch.zeros_like(eye)
    sigma = math.sqrt(2 * ref.temperature * (ref.gamma if ref.kind == "kinetic" else 1))
    noise = sigma * torch.linalg.cholesky(ref.mobility)
    if ref.kind == "kinetic":
        dynamics = torch.cat(
            (torch.cat((zero, eye), -1), torch.cat((-ref.precision, -ref.gamma * ref.mobility), -1)), -2
        )
        noise = torch.cat((zero, noise), -2)
    else:
        dynamics = zero if ref.kind == "brownian" else -ref.mobility @ ref.precision
    return dynamics, noise


def dense_transition(ref, x, dt, reverse=False):
    dynamics, noise = dense_matrices(ref)
    if reverse:
        dynamics = -dynamics
    n = ref.state_dim
    zero = torch.zeros_like(dynamics)
    block = torch.cat(
        (torch.cat((dynamics, noise @ noise.transpose(-1, -2)), -1),
         torch.cat((zero, -dynamics.transpose(-1, -2)), -1)), -2
    )
    exp = torch.matrix_exp(dt * block)
    phi = exp[:, :n, :n]
    cov = exp[:, :n, n:] @ phi.transpose(-1, -2)
    mean = ref.center + (phi @ (x - ref.center).unsqueeze(-1)).squeeze(-1)
    return mean, cov, phi


def test_brownian_known_bridge_and_both_control_signs():
    ref = GaussianReference(torch.zeros(2, 3, 3), torch.ones(2, 3), kind="brownian", temperature=0.18)
    sigma = 0.6
    x0 = torch.tensor([[1., -2., 4.], [0.4, 3., 0.]], dtype=torch.float64)
    x1 = -x0
    x = 0.7 * x0
    t = torch.tensor([0.2, 0.65], dtype=torch.float64)
    mean, covariance = ref.bridge(x0, x1, t)
    torch.testing.assert_close(mean, (1 - t[:, None]) * x0 + t[:, None] * x1)
    torch.testing.assert_close(covariance, (sigma**2 * t * (1 - t))[:, None, None] * torch.eye(3))
    plus, minus = ref.controls(x, x0, x1, t)
    torch.testing.assert_close(plus, (x1 - x) / (sigma * (1 - t[:, None])))
    torch.testing.assert_close(minus, (x0 - x) / (sigma * t[:, None]))


@pytest.mark.parametrize("kind", ["ou", "kinetic"])
@pytest.mark.parametrize("reverse", [False, True])
def test_modal_transition_matches_independent_dense_exponential(kind, reverse):
    ref = reference(kind)
    x = torch.linspace(-0.8, 1.1, ref.batch_size * ref.state_dim).reshape(2, -1).double()
    mean, covariance = ref.transition(x, 0.27, reverse=reverse)
    expected_mean, expected_covariance, _ = dense_transition(ref, x, 0.27, reverse)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(covariance, expected_covariance, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("kind", ["ou", "kinetic", "brownian"])
def test_bridge_matches_joint_gaussian_conditioning_and_score_autograd(kind):
    ref = reference(kind)
    x0 = torch.randn(2, ref.state_dim, dtype=torch.float64)
    x1 = torch.randn_like(x0)
    mean, covariance = ref.bridge(x0, x1, 0.31)
    m0, q0, _ = dense_transition(ref, x0, 0.31)
    m1, q1, _ = dense_transition(ref, x0, 1.0)
    _, _, phir = dense_transition(ref, x0, 0.69)
    cross = q0 @ phir.transpose(-1, -2)
    gain = torch.linalg.solve(q1, cross.transpose(-1, -2)).transpose(-1, -2)
    expected = m0 + (gain @ (x1 - m1).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(mean, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(covariance, q0 - gain @ cross.transpose(-1, -2), rtol=1e-9, atol=1e-10)
    x = ref.sample_bridge(x0, x1, 0.31, generator=torch.Generator().manual_seed(4)).requires_grad_()
    next_mean, next_cov, _ = dense_transition(ref, x, 0.69)
    plus_score = torch.autograd.grad(torch.distributions.MultivariateNormal(next_mean, next_cov).log_prob(x1).sum(), x)[0]
    minus_score = torch.autograd.grad(torch.distributions.MultivariateNormal(m0, q0).log_prob(x).sum(), x)[0]
    _, noise = dense_matrices(ref)
    plus, minus = ref.controls(x, x0, x1, 0.31)
    torch.testing.assert_close(plus, (noise.transpose(-1, -2) @ plus_score.unsqueeze(-1)).squeeze(-1), rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(minus, (noise.transpose(-1, -2) @ minus_score.unsqueeze(-1)).squeeze(-1), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("kind", ["ou", "kinetic", "brownian"])
def test_noncommuting_spd_mobility_dense_fallback(kind):
    mobility = torch.tensor([[1.3, 0.2, 0.0], [0.2, 1.0, -0.1], [0.0, -0.1, 0.9]])
    ref = reference(kind, mobility=mobility)
    x = torch.randn(2, ref.state_dim, dtype=torch.float64)
    actual_mean, actual_cov = ref.transition(x, 0.18)
    mean, cov, _ = dense_transition(ref, x, 0.18)
    torch.testing.assert_close(actual_mean, mean)
    torch.testing.assert_close(actual_cov, cov)
    bridge = ref.sample_bridge(x, -x, 0.4, generator=torch.Generator().manual_seed(8))
    assert torch.isfinite(bridge).all()
    plus, minus = ref.controls(bridge, x, -x, 0.4)
    assert plus.shape == minus.shape == (2, 3)
    assert torch.isfinite(ref.step(x, plus, 0.01)).all()


@pytest.mark.parametrize("dt", [1e-6, 0.0024, 0.2])
def test_kinetic_finite_duration_covariance_full_rank_and_correlated(dt):
    ref = reference("kinetic")
    x = torch.zeros(2, ref.state_dim, dtype=torch.float64)
    _, covariance = ref.transition(x, dt)
    assert (torch.linalg.eigvalsh(covariance) > 0).all()
    assert (covariance[:, :ref.n, ref.n:].diagonal(dim1=-2, dim2=-1) > 0).all()
    _, noise = dense_matrices(ref)
    assert torch.linalg.matrix_rank(noise @ noise.transpose(-1, -2)).tolist() == [ref.n] * 2
    direct_control = ref.noise_control(torch.ones(2, ref.n, dtype=torch.float64))
    assert torch.count_nonzero(direct_control[:, :ref.n]) == 0


def test_kinetic_forward_reverse_kinematic_signs_no_velocity_parity_flip():
    ref = reference("kinetic")
    x = torch.randn(2, ref.state_dim, dtype=torch.float64)
    dt = 1e-6
    forward, _ = ref.transition(x, dt)
    reverse, _ = ref.transition(x, dt, reverse=True)
    velocity = x[:, ref.n:]
    torch.testing.assert_close((forward[:, :ref.n] - x[:, :ref.n]) / dt, velocity, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close((reverse[:, :ref.n] - x[:, :ref.n]) / dt, -velocity, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ref.drift(x)[:, :ref.n], velocity)


@pytest.mark.parametrize("kind", ["ou", "kinetic", "brownian"])
@pytest.mark.parametrize("reverse", [False, True])
def test_held_control_exact_integral_and_seeded_stochastic_step(kind, reverse):
    ref = reference(kind)
    x = torch.randn(2, ref.state_dim, dtype=torch.float64)
    u = torch.randn(2, ref.n, dtype=torch.float64)
    dt = 0.07
    # Shared seeded noise cancels and exposes the exact held-control mean gain.
    controlled = ref.step(x, u, dt, reverse, torch.Generator().manual_seed(18))
    uncontrolled = ref.step(x, torch.zeros_like(u), dt, reverse, torch.Generator().manual_seed(18))
    dynamics, noise = dense_matrices(ref)
    if reverse:
        dynamics = -dynamics
    d, n = ref.state_dim, ref.n
    augmented = torch.zeros(2, d+n, d+n, dtype=torch.float64)
    augmented[:, :d, :d] = dynamics
    augmented[:, :d, d:] = noise
    gain = torch.matrix_exp(dt * augmented)[:, :d, d:]
    expected = (gain @ u.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(controlled - uncontrolled, expected, rtol=2e-9, atol=2e-10)
    assert not torch.equal(uncontrolled, ref.transition(x, dt, reverse)[0])


@pytest.mark.parametrize("kind", ["ou", "kinetic", "brownian"])
def test_bridge_endpoints_and_zero_duration_are_exact(kind):
    ref = reference(kind)
    x0 = torch.randn(2, ref.state_dim, dtype=torch.float64)
    x1 = torch.randn_like(x0)
    t = torch.tensor([0., 1.])
    expected = torch.stack((x0[0], x1[1]))
    mean, covariance = ref.bridge(x0, x1, t)
    torch.testing.assert_close(mean, expected)
    assert torch.count_nonzero(covariance) == 0
    torch.testing.assert_close(ref.sample_bridge(x0, x1, t), expected)
    torch.testing.assert_close(ref.step(x0, torch.ones(2, ref.n), 0.0), x0)
    with pytest.raises(ValueError, match="strictly inside"):
        ref.controls(x0, x0, x1, t)


@pytest.mark.parametrize("horizon,action_dim", [(16, 2), (9, 6)])
def test_dimension_portability_and_auxiliary_endpoint_convention(horizon, action_dim):
    n = horizon * action_dim
    ref = reference("kinetic", n=n, dtype=torch.float32)
    actions = torch.randn(2, n)
    state = ref.augment(actions, generator=torch.Generator().manual_seed(10))
    assert state.shape == (2, 2*n)
    assert ref.n == n and ref.state_dim == 2*n
    torch.testing.assert_close(ref.positions(state), actions)
    expected_velocity = math.sqrt(ref.temperature) * torch.randn(2, n, generator=torch.Generator().manual_seed(10))
    torch.testing.assert_close(state[:, n:], expected_velocity)
    assert not torch.equal(state[:, n:], ref.augment(actions)[:, n:])
    sampled = ref.sample_bridge(state, -state, torch.tensor([0.2, 0.8]))
    assert sampled.dtype == actions.dtype
    assert torch.isfinite(sampled).all()
    controls = ref.controls(sampled, state, -state, torch.tensor([0.2, 0.8]))
    assert all(item.shape == actions.shape for item in controls)


def test_bad_coefficients_and_times_fail_clearly():
    mean = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="positive definite"):
        GaussianReference(-torch.eye(3).repeat(2, 1, 1), mean)
    with pytest.raises(ValueError, match="temperature"):
        GaussianReference(torch.eye(3).repeat(2, 1, 1), mean, temperature=0)
    ref = reference()
    with pytest.raises(ValueError, match="nonnegative"):
        ref.transition(mean, -0.1)
    with pytest.raises(ValueError, match="shape"):
        ref.transition(torch.zeros(2, 4), 0.2)
    with pytest.raises(ValueError, match="scalar or"):
        ref.transition(mean, torch.ones(2, 1))


def test_ou_continuous_brownian_limit_and_cached_coefficients():
    mean = torch.randn(2, 3, dtype=torch.float64)
    ref = GaussianReference(1e-9 * torch.eye(3).repeat(2, 1, 1).double(), mean)
    x0, x1 = mean + 2, mean - 1
    bridge_mean, covariance = ref.bridge(x0, x1, 0.4)
    torch.testing.assert_close(bridge_mean, 0.6*x0 + 0.4*x1, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(covariance, (0.1*0.4*0.6)*torch.eye(3).repeat(2, 1, 1).double())
    first = ref._moments(0.25)
    second = ref._moments(0.25)
    assert all(a is b for a, b in zip(first, second))
