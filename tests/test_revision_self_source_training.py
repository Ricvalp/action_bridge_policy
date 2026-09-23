"""Block-source artifacts and exact resume, with tiny generic tensor records."""

import copy
import json

import pytest
import torch

pytest.importorskip("diffusers")

from action_bridge.plan_revision import checkpoints, training
from test_revision_training import assert_tree_equal, setup


@pytest.fixture(autouse=True)
def cpu_resources(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def self_source_case(method):
    config, records, _, dependencies, metadata = setup(method)
    config.update(protocol="self_source_v1", training_blocks=4, rounds=4,
                  phase_updates=1, source_self_probabilities=[.1, .5, 1., 1.],
                  source_seed=17, prior_ridge=.05, max_rate=4.,
                  mobility_smoothing=0., coupling_refresh_every=1)
    records = {key: value for key, value in records.items()
               if key not in {"old_actions", "prior_mean", "precision"}}
    records["episode_id"] = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    records["time_index"] = torch.tensor([0, 2, 4, 0, 2, 0, 2])
    records["startup_actions"] = records["act_hist"][:, -1:, :].expand(-1, config["horizon"], -1).clone()
    metadata = {key: value for key, value in metadata.items() if key != "source_hash"}
    return config, records, dependencies, metadata


def pairer_factory(records):
    return lambda batch, completion, device: (batch, {"ot_test": 0.})


@pytest.mark.parametrize("method", ["fm_paired", "fm_local_ot", "sb_ou", "sb_kinetic"])
def test_midblock_resume_rebuilds_original_sources_exactly(tmp_path, method):
    config, records, dependencies, metadata = self_source_case(method)
    kwargs = {"pairer_factory": pairer_factory} if method == "fm_local_ot" else {}
    original_windows = copy.deepcopy(records)
    whole = training.train(records, config, tmp_path / "whole", metadata, dependencies, "cpu", **kwargs)
    output = tmp_path / "resume"
    partial = training.train(records, config, output, metadata, dependencies, "cpu", stop_after=5, **kwargs)
    assert partial["source_block"] == 2
    provenance = partial["source_provenance"]
    snapshot_path = output / provenance["snapshot_path"]
    original_snapshot_hash = checkpoints.digest(snapshot_path)
    # Test only artifacts may be removed; production checkpoints are untouched.
    (output / provenance["cache_path"]).unlink()
    torch.manual_seed(984)
    resumed = training.train(records, config, output, metadata, dependencies, "cpu", **kwargs)
    assert resumed["complete"] and resumed["source_block"] == 3
    assert checkpoints.digest(snapshot_path) == original_snapshot_hash
    for key in ("model", "ema", "optimizers", "coupling_cache", "opposite_snapshot", "rng"):
        assert_tree_equal(whole[key], resumed[key])
    assert_tree_equal(records, original_windows)
    rows = [json.loads(line) for line in (output / "source_replay.jsonl").read_text().splitlines()]
    assert [row["block"] for row in rows] == [0, 1, 2, 2, 3]
    assert rows[-2]["rebuilt"]
    assert rows[2]["records_sha256"] == rows[3]["records_sha256"]
    for block in range(4):
        path = output / "sources" / f"block_{block:03d}"
        snapshot = checkpoints.load(path / "snapshot.pt")
        artifact = torch.load(path / "sources.pt", weights_only=False)
        assert snapshot["source_identity"]["p_self"] == config["source_self_probabilities"][block]
        assert artifact["provenance"]["snapshot_sha256"] == checkpoints.digest(path / "snapshot.pt")
        assert all(not tensor.requires_grad for tensor in artifact["records"].values()
                   if isinstance(tensor, torch.Tensor))


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
def test_source_law_fixed_for_round_but_coupling_snapshots_change(tmp_path, monkeypatch, method):
    config, records, dependencies, metadata = self_source_case(method)
    config.update(updates=16, phase_updates=2)
    calls = []
    real_refresh = training.refresh_coupling

    def refresh(source, count, device, settings, completion, opposite, direction):
        calls.append((checkpoints.content_digest(source), direction, opposite is None))
        return real_refresh(source, count, device, settings, completion, opposite, direction)

    monkeypatch.setattr(training, "refresh_coupling", refresh)
    state = training.train(records, config, tmp_path, metadata, dependencies, "cpu")
    for block in range(4):
        block_calls = calls[block * 4:(block + 1) * 4]
        assert len({call[0] for call in block_calls}) == 1
        assert [call[1] for call in block_calls] == ["reverse", "reverse", "forward", "forward"]
        assert [call[2] for call in block_calls] == ([True, True, False, False] if block == 0 else [False] * 4)
    assert len({calls[block * 4][0] for block in range(4)}) == 4
    assert state["cache_provenance"]["source_hash"] == state["source_provenance"]["version"]
    assert state["cache_provenance"]["source_block"] == 3
    assert state["cache_provenance"]["snapshot_sha256"]
    initial = checkpoints.load(tmp_path / "sources" / "block_000" / "snapshot.pt")["ema"]
    assert any(not torch.equal(initial[key], state["ema"][key]) for key in initial)


def test_missing_source_snapshot_is_not_replaced_with_new_ema(tmp_path):
    config, records, dependencies, metadata = self_source_case("fm_paired")
    state = training.train(records, config, tmp_path, metadata, dependencies, "cpu", stop_after=1)
    (tmp_path / state["source_provenance"]["snapshot_path"]).unlink()
    with pytest.raises(ValueError, match="Missing immutable source snapshot"):
        training.train(records, config, tmp_path, metadata, dependencies, "cpu")


def test_local_pairer_is_rebuilt_only_for_new_block(tmp_path):
    config, records, dependencies, metadata = self_source_case("fm_local_ot")
    versions = []

    def factory(source):
        versions.append(checkpoints.content_digest(source))
        return pairer_factory(source)

    training.train(records, config, tmp_path, metadata, dependencies, "cpu", pairer_factory=factory)
    assert len(versions) == len(set(versions)) == 4


def test_ddim_is_independent_of_source_curriculum(tmp_path, monkeypatch):
    config, records, _, metadata = self_source_case("ddim")

    def unexpected(*args, **kwargs):
        raise AssertionError("DDIM must not use a block source snapshot")

    monkeypatch.setattr(training, "_block_sources", unexpected)
    state = training.train(records, config, tmp_path, metadata, {}, "cpu")
    assert state["complete"]
    assert state["source_provenance"] is None
    assert not (tmp_path / "sources").exists()
