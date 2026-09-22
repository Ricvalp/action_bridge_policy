from __future__ import annotations

import pytest
import torch

from action_bridge.plan_revision.ot import batched_local_ot_pair, local_ot_pair, sinkhorn_coupling


def test_uniform_marginals_and_forbidden_edges():
    torch.manual_seed(2)
    cost = torch.rand(8, 8)
    mode = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    allowed = mode[:, None] == mode[None]
    coupling = sinkhorn_coupling(cost, allowed)
    torch.testing.assert_close(coupling.sum(0), torch.full((8,), 1 / 8), atol=2e-7, rtol=0)
    torch.testing.assert_close(coupling.sum(1), torch.full((8,), 1 / 8), atol=2e-7, rtol=0)
    assert (coupling[~allowed] == 0).all()


def test_identity_only_is_reported_without_relaxing_contexts():
    source, target = torch.randn(8, 9, 6), torch.randn(8, 9, 6)
    distance = torch.ones(8, 8) - torch.eye(8)
    si, ti, info = local_ot_pair(source, target, distance, torch.ones(8, 8, dtype=torch.bool), radius=.1)
    assert torch.equal(si, ti)
    assert info["inactive"]
    assert info["off_diagonal_mass"] == 0
    assert info["context_displacement"] == 0


def test_pairs_are_discrete_and_obey_completion_and_context_masks():
    source = torch.arange(8.).reshape(8, 1, 1)
    target = source.flip(0)
    context = torch.tensor([0., .01, .02, .03, 1., 1.01, 1.02, 1.03])
    distance = (context[:, None] - context[None]).abs()
    modes = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    si, ti, info = local_ot_pair(source, target, distance, torch.ones(8, 8, dtype=torch.bool),
                                completion_ids=modes, radius=.1, generator=torch.Generator().manual_seed(2))
    assert si.dtype == ti.dtype == torch.long
    assert torch.equal(ti, torch.arange(8))
    assert torch.equal(modes[si], modes[ti])
    assert (distance[si, ti] <= .1).all()
    assert info["max_context_displacement"] <= .1
    # Selected labels are exact recorded chunks, not barycentric mixtures.
    assert torch.equal(target[ti], target)


def test_anchor_selection_preserves_target_record():
    source = torch.randn(8, 2, 3)
    target = torch.randn_like(source)
    si, ti, info = local_ot_pair(source, target, torch.zeros(8, 8), torch.ones(8, 8, dtype=torch.bool),
                                target_indices=[0], generator=torch.Generator().manual_seed(1))
    assert si.shape == ti.shape == (1,)
    assert ti.item() == 0
    assert info["context_displacement"] == 0


def test_no_implicit_context_metric_or_infeasible_diagonal():
    with pytest.raises(ValueError, match="diagonal"):
        sinkhorn_coupling(torch.zeros(2, 2), ~torch.eye(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="context distances"):
        local_ot_pair(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), torch.zeros(2), torch.ones(2, 2, dtype=torch.bool))


def test_batched_sinkhorn_matches_independent_blocks_and_marginals():
    torch.manual_seed(5)
    costs = torch.rand(12, 8, 8)
    allowed = torch.ones_like(costs, dtype=torch.bool)
    modes = torch.randint(3, (12, 8))
    allowed &= modes[:, :, None] == modes[:, None, :]
    batched = sinkhorn_coupling(costs, allowed)
    separate = torch.stack([sinkhorn_coupling(cost, mask) for cost, mask in zip(costs, allowed)])
    torch.testing.assert_close(batched, separate, atol=2e-7, rtol=0)
    torch.testing.assert_close(batched.sum(-1), torch.full((12, 8), 1 / 8), atol=2e-7, rtol=0)
    torch.testing.assert_close(batched.sum(-2), torch.full((12, 8), 1 / 8), atol=2e-7, rtol=0)
    assert (batched[~allowed] == 0).all()


def test_batched_pairing_one_anchor_per_block_and_tensor_diagnostics():
    torch.manual_seed(8)
    source = torch.randn(12, 8, 9, 6)
    target = torch.randn_like(source)
    features = torch.randn(12, 8, 4)
    distance = torch.cdist(features, features)
    modes = torch.randint(3, (12, 8))
    si, ti, metrics = batched_local_ot_pair(
        source, target, distance, distance <= 1.5, radius=1.5,
        completion_ids=modes, target_indices=[0], generator=torch.Generator().manual_seed(3),
    )
    assert si.shape == ti.shape == (12, 1)
    assert (ti == 0).all()
    rows = torch.arange(12)
    assert torch.equal(modes[rows, si[:, 0]], modes[:, 0])
    assert (distance[rows, si[:, 0], 0] <= 1.5).all()
    assert all(value.shape == (12,) for value in metrics.values())


def test_nonconverged_uniform_marginals_fail_clearly():
    with pytest.raises(RuntimeError, match="uniform marginals"):
        sinkhorn_coupling(torch.tensor([[0., 0.], [0., 10.]]),
                          torch.ones(2, 2, dtype=torch.bool), iterations=1)


def test_pusht_pairer_batches_blocks_and_keeps_uniform_anchor_labels(monkeypatch):
    from action_bridge.data.revision_pusht import make_local_pairer
    from action_bridge.plan_revision.completion import complete_plan
    from action_bridge.plan_revision.contracts import take
    import action_bridge.plan_revision.ot as ot

    n, horizon, dim = 6, 4, 2
    records = {"old_actions": torch.arange(n * horizon * dim).float().reshape(n, horizon, dim),
               "future_actions": torch.randn(n, horizon, dim),
               "obs_hist": torch.randn(n, 2, 5), "act_hist": torch.randn(n, 2, dim)}
    # Several contexts tie and topk-like candidates omit the actual anchor.
    neighborhood = {"neighbors": torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3],
                                                 [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                    "features": torch.zeros(n, 10), "radius": 0.}
    cfg = dict(ot_block_size=3, execute=2, source_std=0., endpoint_std=.001,
               ot_entropy=.1, ot_context_weight=1.)
    anchor_ids = torch.tensor([4, 1, 5])
    batch = take(records, anchor_ids)
    batch.update(record_id=anchor_ids, completion_id=torch.tensor([0, 1, 0]))
    batch["future_actions"] += .123  # The exact already-dequantized endpoint draw.
    histories, labels = batch["obs_hist"].clone(), batch["future_actions"].clone()
    calls = []

    def select_original(source, target, distances, compatible, **kwargs):
        calls.append((source.clone(), target.clone(), kwargs["completion_ids"].clone()))
        torch.testing.assert_close(target[:, 0], labels)
        assert kwargs["target_indices"] == [0]
        ids = torch.zeros(len(source), 1, dtype=torch.long)
        return ids, ids, {"context_displacement": torch.zeros(len(source))}

    monkeypatch.setattr(ot, "batched_local_ot_pair", select_original)
    pairer = make_local_pairer(records, neighborhood, cfg)
    result, metrics = pairer(batch, None, "cpu")
    assert len(calls) == 1
    assert calls[0][0].shape == (3, 3, horizon, dim)
    assert torch.equal(calls[0][2], batch["completion_id"][:, None].expand(-1, 3))
    expected = complete_plan(records["old_actions"][anchor_ids], 2, histories,
                             batch["act_hist"], batch["completion_id"])
    torch.testing.assert_close(result["source_actions"], expected)
    torch.testing.assert_close(result["future_actions"], labels)
    torch.testing.assert_close(result["obs_hist"], histories)
    assert metrics["ot/context_displacement"].ndim == 0
