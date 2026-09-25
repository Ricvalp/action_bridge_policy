"""Reference ablations change generation dynamics, not completed source plans."""

import copy

import pytest
import torch

from action_bridge.plan_revision.cache import reference_for, reference_kind_for, refresh_coupling
from action_bridge.plan_revision.gaussian import GaussianReference
from action_bridge.plan_revision.models import build_policy


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(kind="learned", method="sb_ou"):
    return dict(reference_kind=kind, method=method, horizon=4, execute=2,
                action_dim=2, obs_dim=5, obs_history=2, action_history=2,
                channels=[8, 16], history_dim=16, hidden_dim=16, time_dim=8,
                num_inference_steps=4, temperature=.05, revision_gamma=2.,
                mobility_smoothing=0., batch_size=2, source_std=.01,
                endpoint_std=.001, coupling_steps=3)


def records():
    generator = torch.Generator().manual_seed(42)
    factor = torch.randn(3, 8, 8, generator=generator)
    return dict(
        precision=factor @ factor.transpose(-1, -2) / 8 + .2 * torch.eye(8),
        prior_mean=torch.randn(3, 8, generator=generator),
        obs_hist=torch.randn(3, 2, 5, generator=generator),
        act_hist=torch.randn(3, 2, 2, generator=generator),
        source_actions=torch.randn(3, 4, 2, generator=generator),
        future_actions=torch.randn(3, 4, 2, generator=generator),
        completion_id=torch.tensor([0, 1, 2]),
        has_previous_plan=torch.tensor([False, True, True]),
        valid_mask=torch.ones(3, 4, dtype=torch.bool),
    )


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
@pytest.mark.parametrize("explicit", [False, True])
def test_default_reference_is_numerically_unchanged(method, explicit):
    cfg, batch = config(method=method), records()
    if not explicit:
        del cfg["reference_kind"]
    actual = reference_for(batch, cfg)
    expected = GaussianReference(batch["precision"], batch["prior_mean"],
                                 kind="kinetic" if method == "sb_kinetic" else "ou",
                                 temperature=cfg["temperature"], gamma=cfg["revision_gamma"])
    assert actual.kind == expected.kind
    torch.testing.assert_close(actual.precision, expected.precision, rtol=0, atol=0)
    initial = actual.augment(batch["source_actions"].flatten(1),
                             generator=torch.Generator().manual_seed(4))
    control = torch.full((3, 8), .1)
    for reverse in (False, True):
        left = actual.step(initial, control, .2, reverse,
                           generator=torch.Generator().manual_seed(6))
        right = expected.step(initial, control, .2, reverse,
                              generator=torch.Generator().manual_seed(6))
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_brownian_removes_attraction_and_is_independent_of_prior_center():
    cfg, batch = config("brownian"), records()
    shifted = copy.deepcopy(batch)
    shifted["prior_mean"] += 100
    first, second = reference_for(batch, cfg), reference_for(shifted, cfg)
    assert first.kind == "brownian"
    assert torch.count_nonzero(first.precision) == 0
    initial = batch["source_actions"].flatten(1)
    target = batch["future_actions"].flatten(1)
    assert torch.count_nonzero(first.drift(initial)) == 0
    for reverse in (False, True):
        mean, covariance = first.transition(initial, .3, reverse=reverse)
        torch.testing.assert_close(mean, initial, rtol=0, atol=0)
        torch.testing.assert_close(covariance, (.3 * 2 * cfg["temperature"] *
                                              torch.eye(8)).expand(3, 8, 8))
        other_mean, other_covariance = second.transition(initial, .3, reverse=reverse)
        torch.testing.assert_close(other_mean, mean, rtol=0, atol=1e-6)
        torch.testing.assert_close(other_covariance, covariance, rtol=0, atol=0)
    bridge, covariance = first.bridge(initial, target, .4)
    other_bridge, other_covariance = second.bridge(initial, target, .4)
    torch.testing.assert_close(bridge, .6 * initial + .4 * target)
    torch.testing.assert_close(other_bridge, bridge, rtol=0, atol=1e-6)
    torch.testing.assert_close(other_covariance, covariance, rtol=0, atol=0)
    controls = first.controls(bridge, initial, target, .4)
    other_controls = second.controls(bridge, initial, target, .4)
    for left, right in zip(controls, other_controls):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-6)


def test_isotropic_reference_matches_each_contexts_trace_and_preserves_center():
    batch = records()
    original = copy.deepcopy(batch)
    ref = reference_for(batch, config("isotropic_ou"))
    expected_rate = batch["precision"].double().diagonal(dim1=-2, dim2=-1).mean(-1)
    assert ref.kind == "ou"
    torch.testing.assert_close(ref.mean, batch["prior_mean"].double(), rtol=0, atol=0)
    torch.testing.assert_close(ref.precision, expected_rate[:, None, None] * torch.eye(8))
    torch.testing.assert_close(torch.linalg.eigvalsh(ref.precision), expected_rate[:, None].expand(3, 8))
    torch.testing.assert_close(ref.precision.diagonal(dim1=-2, dim2=-1).sum(-1),
                               batch["precision"].double().diagonal(dim1=-2, dim2=-1).sum(-1))
    assert not torch.equal(ref.precision, batch["precision"].double())
    initial = batch["source_actions"].flatten(1)
    torch.testing.assert_close(ref.drift(initial),
                               (-expected_rate[:, None] * (initial.double() - ref.mean)).float())
    for key in batch:
        torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["brownian", "isotropic_ou"])
@pytest.mark.parametrize("method", ["sb_kinetic", "ddim", "fm_paired"])
def test_nonlearned_variants_require_first_order_bridge(kind, method):
    cfg = config(kind, method)
    with pytest.raises(ValueError, match="requires method=sb_ou"):
        reference_kind_for(cfg)
    with pytest.raises(ValueError, match="requires method=sb_ou"):
        reference_for(records(), cfg)


@pytest.mark.parametrize("kind", ["brownian", "isotropic_ou"])
def test_reference_ablation_rejects_changed_noise_geometry(kind):
    cfg, batch = config(kind), records()
    with pytest.raises(ValueError, match="mobility_smoothing=0"):
        reference_kind_for(cfg | {"mobility_smoothing": .1})
    for mobility in (2 * torch.eye(8), torch.zeros(3, 8, 8), torch.eye(3)):
        with pytest.raises(ValueError, match="identity mobility"):
            reference_for(batch | {"mobility": mobility}, cfg)
    for mobility in (torch.eye(8), torch.eye(8).expand(3, 8, 8)):
        ref = reference_for(batch | {"mobility": mobility}, cfg)
        torch.testing.assert_close(ref.mobility, torch.eye(8).double().expand(3, 8, 8))


def test_invalid_reference_variant_is_not_silently_ignored():
    with pytest.raises(ValueError, match="reference_kind must be"):
        reference_kind_for(config("typo"))


@pytest.mark.parametrize("kind", ["brownian", "isotropic_ou"])
def test_bridge_training_and_exact_forward_reverse_sampling_remain_finite(kind):
    cfg, batch = config(kind), records()
    ref = reference_for(batch, cfg)
    policy = build_policy(cfg)
    initial = batch["source_actions"].flatten(1)
    target = batch["future_actions"].flatten(1)
    for direction in ("forward", "reverse"):
        loss = policy.loss(batch, ref, direction=direction,
                           generator=torch.Generator().manual_seed(18))["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        state, info = policy.rollout(target if direction == "reverse" else initial,
                                    batch["obs_hist"], batch["act_hist"], batch["completion_id"],
                                    ref, reverse=direction == "reverse", trace=True,
                                    generator=torch.Generator().manual_seed(19))
        assert state.shape == initial.shape
        assert torch.isfinite(state).all()
        assert torch.isfinite(info["control_energy"]).all()
        assert info["revision_states"].shape == (3, 5, 8)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in policy.parameters())
    generated, diagnostics = policy.sample(batch["obs_hist"], batch["act_hist"],
                                           batch["completion_id"], source_actions=batch["source_actions"],
                                           reference=ref, generator=torch.Generator().manual_seed(20))
    assert generated.shape == batch["source_actions"].shape
    assert torch.isfinite(generated).all()
    assert diagnostics["nfe"] == cfg["num_inference_steps"]


@pytest.mark.parametrize("kind", ["brownian", "isotropic_ou"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_coupling_refresh_uses_same_ablation_for_both_directions(kind, direction):
    cfg, source_records = config(kind), records()
    policy = build_policy(cfg)
    references_seen = []

    class Snapshot:
        def rollout(self, state, obs, actions, mode, reference, **kwargs):
            references_seen.append(reference)
            return policy.rollout(state, obs, actions, mode, reference, **kwargs)

    coupling, diagnostics = refresh_coupling(source_records, 3, "cpu", cfg, None,
                                             Snapshot(), direction)
    assert len(references_seen) == 2
    assert diagnostics["generated_endpoint"] == ("source" if direction == "forward" else "target")
    for ref in references_seen:
        assert ref.kind == ("brownian" if kind == "brownian" else "ou")
        if kind == "brownian":
            assert torch.count_nonzero(ref.precision) == 0
        else:
            rates = ref.precision.diagonal(dim1=-2, dim2=-1)
            torch.testing.assert_close(rates, rates[:, :1].expand_as(rates), rtol=0, atol=0)
    assert torch.isfinite(coupling["x0"]).all()
    assert torch.isfinite(coupling["x1"]).all()
    for key in ("source_actions", "completion_id", "has_previous_plan", "precision", "prior_mean"):
        torch.testing.assert_close(coupling[key], source_records[key][coupling["record_id"]], rtol=0, atol=0)
