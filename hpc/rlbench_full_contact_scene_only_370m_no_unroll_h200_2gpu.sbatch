#!/usr/bin/env bash
#SBATCH --job-name=rlb_scene_370m_no_unroll
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the action_bridge_policy directory.}"
export UNROLL_WEIGHT=0.0
export UNROLL_TAG=no_unroll
export RUN_VARIANT=no_unroll
exec bash hpc/rlbench_full_contact_scene_only_370m_h200_2gpu.sbatch
