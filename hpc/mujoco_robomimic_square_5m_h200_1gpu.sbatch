#!/usr/bin/env bash
# Setup and transfer instructions: MUJOCO.md, "Run on the HPC".
# Submit from action_bridge_policy after creating hpc/logs.
# Log in once on the HPC before submitting: .venv/bin/wandb login
# gpuq + gpu:1 is the H200 allocation used by the existing jobs here.
#SBATCH --job-name=square_latent_5m
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"

# Keep the backend at the exact revision tested for this experiment.
test "$(git -C ../phi-mujoco rev-parse HEAD)" = 528bdf97471493a9d63da424872141ff1279be4b
test -z "$(git -C ../phi-mujoco status --porcelain)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
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

run_id="square-latent-5m-h8-exec4-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"

# The MuJoCo-only environment is prepared on the login node, not in the job.
uv run --frozen --no-sync --extra cu128 --extra robomimic \
  python -m action_bridge.training.train_mujoco \
  --config-name mujoco_robomimic_square \
  data.cache_root="$PWD/../phi-mujoco/datasets/processed/robomimic_square-20260908T122032Z" \
  run_id="$run_id" \
  device=cuda \
  model.latent_type=continuous \
  model.hidden_dim=736 \
  model.h_emb_dim=736 \
  chunk_horizon=8 \
  eval.actions_per_plan=4 \
  inference.latent_commitment=chunk \
  optim.batch_size=256 \
  optim.max_steps=100000 \
  logging.wandb.enabled=true
