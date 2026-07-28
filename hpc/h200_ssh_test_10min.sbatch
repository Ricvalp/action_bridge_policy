#!/bin/bash

#SBATCH --job-name=h200-ssh-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Uncomment and edit only if your cluster explicitly requires an account.
##SBATCH --account=rvalperga

set -euo pipefail

mkdir -p logs/slurm

echo "Job ID: ${SLURM_JOB_ID}"
echo "Allocated node(s): ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo
echo "Try from your login node while this job is running:"
echo "  ssh ${SLURM_JOB_NODELIST}"
echo
echo "If SSH is blocked, try:"
echo "  srun --jobid=${SLURM_JOB_ID} --pty bash"
echo
echo "Holding allocation for 10 minutes..."

sleep 600
