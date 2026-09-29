"""The diagnostic uses one reached context, not candidate physics rollouts."""
import json

import numpy as np
import pytest
import torch

from action_bridge.plan_revision import checkpoints
from action_bridge.scripts import diagnose_pusht_source_sensitivity as cli
from test_revision_generation_trace import scenario


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cli_records_paired_interventions_identity_and_all_samples(tmp_path, monkeypatch):
    policy, config, metadata, dependencies = scenario("sb_ou", monkeypatch)
    checkpoint = tmp_path / "best.pt"
    checkpoints.save(checkpoint, dict(config=config, metadata=metadata, dependencies=dependencies,
                                     ema=policy.state_dict(), direction="forward", step=25))
    before = checkpoint.read_bytes()
    calls = []
    original = cli.evaluate

    def rollout(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "evaluate", rollout)
    monkeypatch.setattr(cli, "render_source_sensitivity", lambda *args, **kwargs: {"sources": "test.png"})
    output = tmp_path / "diagnostic"
    args = ["--checkpoint", str(checkpoint), "--seed", "71", "--replan", "1",
            "--samples", "5", "--batch-size", "2", "--sampling-seed", "32",
            "--amplitudes", "-5", "5", "--families", "translate_x", "bow_y",
            "--output-dir", str(output), "--no-progress"]
    assert cli.main(args) == 0
    assert len(calls) == 1 and calls[0]["render"] is False
    assert checkpoint.read_bytes() == before
    report = json.loads((output / "source_sensitivity.json").read_text())
    assert report["replan"] == 1 and report["robot_step"] == 2
    assert report["execute"] == report["elapsed_count"] == 2
    assert report["checkpoint_sha256"] == checkpoints.digest(checkpoint)
    assert report["weights"] == "ema" and report["checkpoint_step"] == 25
    assert not report["simulated_candidates"] and not report["physics_snapshot"]
    assert len(report["variant_labels"]) == 5
    assert "not redrawn" in report["source_noise"]
    assert "not success" in report["interpretation"]
    with np.load(output / "source_sensitivity.npz", allow_pickle=False) as arrays:
        assert arrays["sources_raw"].shape == (5, 4, 2)
        assert arrays["candidates_raw"].shape == (5, 5, 4, 2)
        assert arrays["revision_states_raw"].shape[:2] == (5, 5)
        np.testing.assert_array_equal(arrays["candidates_raw"][0], arrays["repeat_baseline_raw"])
        assert arrays["variant_labels"].tolist() == report["variant_labels"]
    with pytest.raises(FileExistsError):
        cli.main(args)


@pytest.mark.parametrize("extra", [
    ["--samples", "0"], ["--batch-size", "0"], ["--threads", "0"],
    ["--seed", "-1"], ["--replan", "-1"], ["--sampling-seed", "-1"],
    ["--amplitudes", "nan"], ["--amplitudes", "0"], ["--amplitudes", "5", "5"],
    ["--families", "translate_x", "translate_x"],
])
def test_invalid_options_fail_before_checkpoint_access(extra):
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", "does-not-exist.pt", *extra])


@pytest.mark.parametrize("method,region,replan,message", [
    ("ddim", "all", 1, "SB or FM"),
    ("sb_ou", "retained", 0, "no retained"),
    ("sb_ou", "all", 100, "step limit"),
])
def test_rejects_inapplicable_diagnostics_before_simulation(tmp_path, monkeypatch, capsys,
                                                         method, region, replan, message):
    policy, config, metadata, dependencies = scenario(method, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoints.save(checkpoint, dict(config=config, metadata=metadata, dependencies=dependencies,
                                     ema=policy.state_dict(), direction="forward", step=1))
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: pytest.fail("must not start simulator"))
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--checkpoint", str(checkpoint), "--region", region, "--replan", str(replan)])
    assert message in capsys.readouterr().err
