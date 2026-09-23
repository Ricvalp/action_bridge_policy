"""W&B integration checks with a local SDK stub: no accounts or uploads."""
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from action_bridge.plan_revision.tracking import Tracker, TrackingOptions, preserve_rng


def random_draw():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


@pytest.fixture
def sdk(monkeypatch):
    runs = []

    def init(**kwargs):
        random_draw()
        run = SimpleNamespace(id=kwargs["id"], kwargs=kwargs, metrics=[], rows=[], exits=[])

        def log(row):
            random_draw()
            run.rows.append(row)

        def finish(*, exit_code):
            random_draw()
            run.exits.append(exit_code)

        run.define_metric = lambda *args, **kwargs: run.metrics.append((args, kwargs))
        run.log, run.finish = log, finish
        runs.append(run)
        return run

    def image(path, *, caption):
        random_draw()
        return {"path": path, "caption": caption}

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Image=image))
    return runs


def tracker(output, **kwargs):
    return Tracker(output, {"method": "ddim", "updates": 10}, {"dataset_sha256": "abc"},
                   TrackingOptions(**kwargs), group="test-experiment", name="test-ddim")


def test_disabled_or_unused_tracking_does_not_import_or_create_output(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    output = tmp_path / "disabled"
    with tracker(output) as tracking:
        tracking.log(1, {"train/loss": 1.})
        tracking.log_evaluation(1, {"success_rate": .2})
        tracking.images(1, [tmp_path / "missing.png"])
        assert not tracking.images_due(5000)
        assert not tracking.images_due(1, final=True)
    with tracker(output, enabled=True):
        pass  # E.g. the stage already has a completed checkpoint.
    assert not output.exists()


def test_logging_and_images_share_custom_optimizer_step(tmp_path, sdk):
    with tracker(tmp_path, enabled=True, image_count=2) as tracking:
        tracking.log(5, {"train/loss": 1.})
        tracking.log(5, {"val/success_rate": .2})
        tracking.images(5, [tmp_path / f"chunk-{index}.png" for index in range(3)])
    assert len(sdk) == 1
    run = sdk[0]
    assert run.metrics == [(("train_step",), {}), (("*",), {"step_metric": "train_step"}),
                           (("sim_eval/checkpoint_step",), {}),
                           (("sim_eval/*",), {"step_metric": "sim_eval/checkpoint_step"})]
    assert [row["train_step"] for row in run.rows] == [5, 5, 5]
    assert len(run.rows[-1]["examples/action_chunks"]) == 2
    assert run.kwargs["config"] == {"method": "ddim", "updates": 10,
                                    "metadata": {"dataset_sha256": "abc"}}
    assert run.exits == [0]


def test_preserve_rng_also_restores_after_errors():
    with preserve_rng():
        expected = random_draw()
    with pytest.raises(RuntimeError), preserve_rng():
        random_draw()
        raise RuntimeError("plot failed")
    assert random_draw() == expected


def test_sdk_calls_preserve_rng(tmp_path, sdk):
    with preserve_rng():
        expected = random_draw()
    with tracker(tmp_path, enabled=True) as tracking:
        tracking.log(1, {"train/loss": .5})
        tracking.images(1, [tmp_path / "chunk.png"])
        tracking.log_evaluation(1, {"success_rate": .2})
    assert random_draw() == expected


def test_late_evaluation_uses_its_own_axis_and_never_uploads_videos(tmp_path, sdk):
    with tracker(tmp_path, enabled=True) as tracking:
        tracking.log(20000, {"train/loss": .1})
        tracking.log_evaluation(10000, {"success_rate": .8, "max_coverage": .9})
        tracking.log(20100, {"train/loss": .09})
    assert sdk[0].rows == [
        {"train_step": 20000, "train/loss": .1},
        {"sim_eval/checkpoint_step": 10000, "sim_eval/success_rate": .8, "sim_eval/max_coverage": .9},
        {"train_step": 20100, "train/loss": .09},
    ]


def test_online_restarts_reuse_run_id(tmp_path, sdk):
    for step in [5, 10]:
        with tracker(tmp_path, enabled=True) as tracking:
            tracking.log(step, {"train/loss": 1 / step})
    assert sdk[0].id == sdk[1].id
    assert all(run.kwargs["resume"] == "allow" for run in sdk)
    assert json.loads((tmp_path / "wandb_run.json").read_text())["id"] == sdk[0].id


def test_offline_runs_are_separate_and_do_not_change_online_resume_id(tmp_path, sdk):
    with tracker(tmp_path, enabled=True) as tracking:
        tracking.log(1, {"train/loss": 1.})
    online_id = json.loads((tmp_path / "wandb_run.json").read_text())["id"]
    for step in [5, 10]:
        with tracker(tmp_path, enabled=True, mode="offline") as tracking:
            tracking.log(step, {"train/loss": 1.})
    assert len({run.id for run in sdk}) == 3
    assert all("resume" not in run.kwargs for run in sdk[1:])
    assert json.loads((tmp_path / "wandb_run.json").read_text())["id"] == online_id


def test_failure_finishes_run_with_failure_status(tmp_path, sdk):
    with pytest.raises(ValueError):
        with tracker(tmp_path, enabled=True) as tracking:
            tracking.log(1, {"train/loss": 1.})
            raise ValueError("training failed")
    assert sdk[0].exits == [1]


def test_missing_sdk_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(RuntimeError, match="requires wandb"):
        with tracker(tmp_path, enabled=True) as tracking:
            tracking.log(1, {"train/loss": 1.})


def test_image_schedule_and_zero_image_count(tmp_path):
    tracking = tracker(tmp_path, enabled=True, images_every=10)
    assert tracking.images_due(10)
    assert not tracking.images_due(11)
    assert tracking.images_due(11, final=True)
    assert not tracker(tmp_path, enabled=True, image_count=0).images_due(10, final=True)


@pytest.mark.parametrize("kwargs", [{"image_count": -1}, {"images_every": 0}, {"mode": "bad"}])
def test_invalid_options_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        TrackingOptions(**kwargs)
