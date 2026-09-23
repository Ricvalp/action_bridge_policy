"""The HPC preflight uses the real async worker and decodes its local MP4."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from action_bridge.scripts import sb_pusht_smoke as smoke


@pytest.mark.parametrize("name", ["NVIDIA B200", "NVIDIA RTX 5000 Ada Generation"])
def test_h200_guard_rejects_other_allocations(monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: name)
    with pytest.raises(RuntimeError, match="Expected an H200"):
        smoke.check_h200()


def test_h200_guard_rejects_missing_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no usable CUDA"):
        smoke.check_h200()


def test_h200_guard_checks_visible_device_and_matrix_math(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    get_name = Mock(return_value="NVIDIA H200")
    monkeypatch.setattr(torch.cuda, "get_device_name", get_name)
    matrix = torch.full((32, 32), 1.)
    allocate = Mock(return_value=matrix)
    synchronize = Mock()
    monkeypatch.setattr(torch, "ones", allocate)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    assert smoke.check_h200() == "NVIDIA H200"
    get_name.assert_called_once_with(3)
    allocate.assert_called_once_with((32, 32), device="cuda:3")
    synchronize.assert_called_once_with(3)


def test_real_headless_async_worker_and_mp4(tmp_path):
    for package in ("diffusers", "gym_pusht", "pymunk", "imageio_ffmpeg"):
        pytest.importorskip(package)
    output = tmp_path / "smoke"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", DISPLAY="",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", WANDB_MODE="disabled")
    command = [sys.executable, "-m", "action_bridge.scripts.sb_pusht_smoke",
               "--output-dir", str(output)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
    report = json.loads((output / "smoke.json").read_text())
    assert report["passed"] and report["gpu"] is None
    assert report["evaluation_device"] == "cpu"
    assert report["metrics"]["episodes"] == 1
    assert report["metrics"]["episode_length"] <= 8
    assert Path(report["video"]).is_file()
    assert report["frame_shape"][-1] == 3
    assert list(output.glob("sim_eval/*/worker.log"))
    assert not list(output.glob("sim_eval/*/checkpoint.pt"))
    # A second invocation must not overwrite the original smoke's artifacts.
    repeat = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    assert repeat.returncode != 0
    assert json.loads((output / "smoke.json").read_text()) == report
