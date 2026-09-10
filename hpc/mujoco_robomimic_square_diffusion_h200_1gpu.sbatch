#!/usr/bin/env bash
# Setup: MUJOCO.md, "DDIM diffusion baseline".
# Submit from action_bridge_policy after creating hpc/logs and logging into W&B.
# Videos stay disabled until the EGL smoke test succeeds on Peano.
#SBATCH --job-name=square_diffusion
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"

test "$(git -C ../phi-mujoco rev-parse HEAD)" = 528bdf97471493a9d63da424872141ff1279be4b
test -z "$(git -C ../phi-mujoco status --porcelain)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/workspace/.cache"
mkdir -p "$XDG_CACHE_HOME"

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; install the cu128 environment first.")
name = torch.cuda.get_device_name(0)
if "H200" not in name.upper():
    raise SystemExit(f"Expected an H200 allocation, got {name}")
print(f"GPU: {name}; PyTorch: {torch.__version__}", flush=True)
PY

run_id="square-diffusion-h8-exec4-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"

uv run --frozen --no-sync --extra cu128 --extra robomimic --extra diffusion \
  python -m action_bridge.training.train_mujoco \
  --config-name mujoco_robomimic_square_diffusion \
  data.cache_root="$PWD/../phi-mujoco/datasets/processed/robomimic_square-20260908T122032Z" \
  run_id="$run_id" \
  device=cuda \
  chunk_horizon=8 \
  eval.actions_per_plan=4 \
  inference.n_exec=4 \
  model.num_train_timesteps=100 \
  model.num_inference_steps=20 \
  optim.batch_size=256 \
  optim.max_steps=100000 \
  logging.wandb.enabled=true \
  logging.sim_eval_enabled=true \
  logging.sim_eval_every_steps=5000 \
  logging.sim_eval_episodes=40 \
  logging.sim_eval_num_workers=8 \
  logging.sim_eval_worker_threads=1 \
  logging.sim_eval_n_exec=4 \
  logging.sim_eval_success_videos="${SIM_EVAL_VIDEOS:-0}" \
  logging.sim_eval_failure_videos="${SIM_EVAL_VIDEOS:-0}" \
  logging.sim_eval_video_backend=egl
