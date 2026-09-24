#!/usr/bin/env bash
# Usage: sbatch hpc/sb_pusht_ablations/prepare.sbatch VARIANT
# Export SB_PUSHT_CAMPAIGN_ROOT first; see this directory's README.
#SBATCH --job-name=pusht_ablation_prepare
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: sbatch $0 VARIANT" >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
source hpc/sb_pusht_ablations/common.sh "$1"

exec "$python" -m action_bridge.scripts.sb_pusht prepare \
  "${common_args[@]}" \
  --device cpu
