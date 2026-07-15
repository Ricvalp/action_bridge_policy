#!/bin/bash
#
# Reserve one H200 for 10 minutes so you can test whether direct SSH to the
# allocated compute node is allowed on your cluster.
#
# Submit with:
#   sbatch hpc/h200_ssh_test_10min.sbatch
#
# Then watch the allocation with:
#   squeue -u "$USER"
#
# If the job is running, get the node name with:
#   squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
#
# Try SSH:
#   ssh <NODE_NAME>
#
# If direct SSH is blocked, try the Slurm-supported interactive shell:
#   srun --jobid=<JOB_ID> --pty bash

#SBATCH --job-name=h200-ssh-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Uncomment and edit these if your cluster requires them.
##SBATCH --partition=<h200_partition>
##SBATCH --account=<account_name>

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
