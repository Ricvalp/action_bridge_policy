#!/usr/bin/env bash
# Run on your laptop while connected to the VPN, not on Peano.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 RUN_DIRECTORY WORKSTATION_SSH_ALIAS" >&2
  exit 2
fi

run_name="$1"
workstation="$2"
if [[ ! "$run_name" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
  echo "RUN_DIRECTORY must be a directory name, not a full path." >&2
  exit 2
fi
if [[ ! "$workstation" =~ ^[[:alnum:]][[:alnum:]_.@-]*$ ]]; then
  echo "WORKSTATION_SSH_ALIAS must be an SSH alias or user@hostname." >&2
  exit 2
fi

hpc_run="/hpc/home/phi/rvalperga/action_bridge_policy/workspace/sb_pusht/$run_name"
staging="$HOME/Downloads/sb-pusht/$run_name"
destination="/home/rvalperga/action_bridge_policy/workspace/sb_pusht/from-hpc/$run_name"
checkpoint_filters=(
  --include='*/'
  --include='best.pt'
  --include='latest.pt'
  --include='best_eval.json'
  --include='manifest.json'
  --exclude='*'
)

echo "HPC -> laptop: $staging"
mkdir -p "$staging"
rsync -avPm "${checkpoint_filters[@]}" "peano:$hpc_run/" "$staging/"

echo "Laptop -> workstation: $destination"
ssh "$workstation" "mkdir -p '$destination'"
rsync -avPm "${checkpoint_filters[@]}" "$staging/" "$workstation:$destination/"

echo "Done. Checkpoints: $workstation:$destination"
echo "The laptop copy is kept at $staging."
