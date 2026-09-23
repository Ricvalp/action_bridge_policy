"""Slurm launch contracts; no scheduler, GPU or cluster connection required."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from action_bridge.configs.sb_pusht import METHODS


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "reference", *METHODS, "evaluate", "smoke")


@pytest.mark.parametrize("stage", STAGES)
def test_sbatch_h200_resources_and_shell_syntax(stage):
    path = ROOT / "hpc" / f"sb_pusht_{stage}_h200_1gpu.sbatch"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "#SBATCH --partition=gpuq\n" in text
    assert "#SBATCH --gres=gpu:1\n" in text
    assert "#SBATCH --cpus-per-task=8\n" in text
    assert "aiq" not in text and "B200" not in text
    assert "uv sync" not in text and "pip install" not in text
    if stage in METHODS:
        assert "#SBATCH --time=10:00:00\n" in text
        assert '"H200" not in name.upper()' in text


@pytest.mark.parametrize("stage", STAGES)
def test_job_launches_only_its_stage_with_headless_environment(tmp_path, stage):
    python = tmp_path / ".venv-sb-pusht" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['LAUNCH_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')
""")
    python.chmod(0o755)
    run = tmp_path / "shared run"
    (run / "reference").mkdir(parents=True)
    (run / "windows.pt").touch()
    (run / "reference" / "latest.pt").touch()
    log = tmp_path / "launch.jsonl"
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID="42",
               SB_PUSHT_RUN_ROOT=str(run), PUSHT_DATASET=str(tmp_path / "replay.zarr"),
               CUDA_VISIBLE_DEVICES="3", LAUNCH_LOG=str(log), WANDB_MODE="offline")
    subprocess.run(["bash", str(ROOT / "hpc" / f"sb_pusht_{stage}_h200_1gpu.sbatch")],
                   env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    call = calls[-1]
    for name, value in (("SDL_VIDEODRIVER", "dummy"), ("SDL_AUDIODRIVER", "dummy"),
                        ("MPLBACKEND", "Agg"), ("CUDA_VISIBLE_DEVICES", "3")):
        assert call["env"][name] == value
    assert Path(call["env"]["XDG_CACHE_HOME"]).is_relative_to(tmp_path)
    if stage == "smoke":
        assert call["args"][:2] == ["-m", "action_bridge.scripts.sb_pusht_smoke"]
        assert "--require-h200" in call["args"]
        return
    assert calls[0]["args"] == ["-"]  # GPU preflight runs before the CLI.
    assert call["args"][:3] == ["-m", "action_bridge.scripts.sb_pusht", stage]
    assert call["args"][call["args"].index("--run-root") + 1] == str(run)
    if stage in METHODS:
        assert "--sim-eval" in call["args"] and "--eval-videos" in call["args"]
        assert call["args"][call["args"].index("--eval-device") + 1] == "cpu"
        assert call["args"][call["args"].index("--eval-every") + 1] == "10000"
        assert call["args"][call["args"].index("--eval-threads") + 1] == "2"
    if stage in (*METHODS, "reference"):
        assert "--wandb" in call["args"]
        assert call["args"][call["args"].index("--wandb-mode") + 1] == "offline"


def test_reviser_fails_before_launch_when_reference_missing(tmp_path):
    (tmp_path / "windows.pt").touch()
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), SB_PUSHT_RUN_ROOT=str(tmp_path))
    result = subprocess.run(["bash", str(ROOT / "hpc/sb_pusht_sb_ou_h200_1gpu.sbatch")], env=env)
    assert result.returncode != 0
