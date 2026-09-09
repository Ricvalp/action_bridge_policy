#!/usr/bin/env bash
# Submit from action_bridge_policy after creating hpc/logs.
# Tests the actual Robosuite/Square rendering path used during evaluation.
# Robosuite uses numeric CUDA_VISIBLE_DEVICES as EGL indices; remapped or UUID
# allocations need a cluster-confirmed device mapping instead of guessing GPU 0.
# No environment installation or dataset is needed: use the training .venv.
#SBATCH --job-name=mujoco_egl_smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/workspace/.cache"
mkdir -p "$XDG_CACHE_HOME"

.venv/bin/python - <<'PY'
import os
from pathlib import Path

allocated = os.environ.get("SLURM_JOB_GPUS", "")
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"SLURM_JOB_GPUS={allocated or '<unset>'}")
print(f"CUDA_VISIBLE_DEVICES={visible or '<unset>'}")
if not visible.isdecimal():
    raise SystemExit(
        "This one-GPU Robosuite smoke test needs one numeric CUDA_VISIBLE_DEVICES "
        "value. Ask the cluster administrators how to select the allocated EGL "
        "device with this allocation; do not replace it blindly with 0."
    )
if not allocated.isdecimal() or int(allocated) != int(visible):
    raise SystemExit(
        "The Slurm GPU index is missing, nonnumeric, or differs from the CUDA-visible "
        "index. Ask the cluster administrators to confirm EGL device mapping "
        "before rendering; this test will not guess."
    )
egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID", visible)
if not egl_device.isdecimal() or int(egl_device) != int(visible):
    raise SystemExit("MUJOCO_EGL_DEVICE_ID must match the allocated CUDA-visible GPU.")
print(f"Robosuite EGL device index: {egl_device}")

import imageio.v2 as imageio
import mujoco
import numpy as np
from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner
from phi_mujoco.integrations import get_integration


class NeutralPolicy:
    def reset(self, *, integration, seed):
        self.action_shape = integration.action.shape

    def predict(self, inputs):
        return np.zeros(self.action_shape, dtype=np.float32)


output = Path("workspace/experiments/mujoco") / f"egl-smoke-{os.environ['SLURM_JOB_ID']}"
result = EvaluationRunner(
    get_integration("robomimic_square"),
    NeutralPolicy(),
    EvaluationConfig(
        episodes=1,
        max_steps=2,
        run_directory=output,
        record_video=True,
        render_width=320,
        render_height=240,
    ),
).run()
video_path = result.video_paths[0]
with imageio.get_reader(video_path) as reader:
    frame = reader.get_data(0)
assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
assert frame.std() > 0, "EGL returned a blank frame"
imageio.imwrite(output / "frame.png", frame)
print(f"MuJoCo: {mujoco.__version__}")
print(f"Frame: {(output / 'frame.png').resolve()}")
print(f"Video: {video_path.resolve()} ({video_path.stat().st_size} bytes)")
print("PASS: headless EGL rendering and MP4 encoding/decoding succeeded.")
PY
