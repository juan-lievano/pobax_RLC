#!/usr/bin/env bash
# Submit with:
#
#   MANIFEST=/nas/.../results/batch_12345/probe_manifest.txt
#   N=$(wc -l < $MANIFEST)
#   sbatch --array=0-$((N-1)) --export=ALL,MANIFEST=$MANIFEST slurm/train/probe_array_launch.sh
#
# (The exact command is printed by make_probe_manifest.py.)
#
#SBATCH --job-name=probe_array
#SBATCH --output=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.log
#SBATCH --error=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

set -euo pipefail

if [[ -z "${MANIFEST:-}" ]]; then
  echo "ERROR: MANIFEST env var not set. Pass it via --export=ALL,MANIFEST=/path/to/probe_manifest.txt"
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest file not found: $MANIFEST"
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# Manifest is 1-indexed for sed; task IDs are 0-indexed
RUN_DIR="$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")"

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: No entry for task_id=$TASK_ID in $MANIFEST"
  exit 3
fi

export RUN_DIR

echo "Selected run (task_id=$TASK_ID):"
echo "  RUN_DIR=$RUN_DIR"
echo "  MANIFEST=$MANIFEST"

bash /nas/ucb/juanlievano/pobax_RLC/slurm/train/probe_run_one.sh
