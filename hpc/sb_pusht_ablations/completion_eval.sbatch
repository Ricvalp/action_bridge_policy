#!/usr/bin/env bash
# Usage: sbatch hpc/sb_pusht_ablations/completion_eval.sbatch CHECKPOINT_ROOT METHOD [best.pt]
# Compare three completion modes without retraining. Metrics/videos are local,
# not W&B runs. CHECKPOINT_ROOT contains METHOD/best.pt (or another filename).
#SBATCH --job-name=pusht_ablation_completion
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: sbatch $0 CHECKPOINT_ROOT METHOD [CHECKPOINT_NAME]" >&2
  exit 2
fi
method="$2"
case "$method" in
  fm_paired|fm_local_ot|sb_ou|sb_kinetic) ;;
  *) echo "Completion evaluation needs an FM/SB method, got: $method" >&2; exit 2 ;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
checkpoint_root="$(realpath "$1")"
checkpoint="$checkpoint_root/$method/${3:-best.pt}"
test -f "$checkpoint"
python="$PWD/.venv-sb-pusht/bin/python"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/workspace/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$MPLCONFIGDIR"

"$python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; prepare the cu128 environment first.")
name = torch.cuda.get_device_name(0)
if "H200" not in name.upper():
    raise SystemExit(f"Expected an H200 allocation, got {name}")
print(f"GPU allocation: {name}; evaluation runs in four CPU workers.", flush=True)
PY

output="$checkpoint_root/completion_eval/$method-$(date -u +%Y%m%dT%H%M%S%NZ)"
mkdir -p "$output"
for completion in repeat fixed_damped learned_dissipative; do
  "$python" -m "action_bridge.scripts.eval_pusht_$method" \
    --checkpoint "$checkpoint" \
    --completion "$completion" \
    --device cpu \
    --workers 4 \
    --threads 1 \
    --episodes "${EVAL_EPISODES:-200}" \
    --seed "${EVAL_SEED:-1000000}" \
    --save-videos \
    --no-progress \
    --output-dir "$output/$completion"
done
echo "Completion comparison: $output"
