#!/usr/bin/env bash
# Usage: sbatch hpc/sb_pusht_ablations/reference.sbatch VARIANT
# Submit after this variant's prepare job; shared by its four revisers.
#SBATCH --job-name=pusht_ablation_reference
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: sbatch $0 VARIANT" >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
source hpc/sb_pusht_ablations/common.sh "$1"
test -f "$run_root/windows.pt"

exec "$python" -m action_bridge.scripts.sb_pusht reference \
  "${common_args[@]}" \
  "${tracking_args[@]}" \
  --device cuda
