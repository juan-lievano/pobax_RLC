#!/usr/bin/env bash
# Submit with dependency on training array:
#
#   TRAIN_JOB=$(sbatch --array=0-159 --parsable array_launch.sh)
#   sbatch --array=0-159 --dependency=aftercorr:$TRAIN_JOB probe_array_launch.sh
#
# aftercorr: probe task i starts as soon as training task i succeeds (1:1 mapping).
# Use afterok:$TRAIN_JOB instead if you prefer to wait for all 160 to finish first.
#
#SBATCH --job-name=pobax_probes
#SBATCH --output=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.log
#SBATCH --error=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

set -euo pipefail

GRID_FILE="/nas/ucb/juanlievano/pobax_RLC/slurm/ppo/grid.tsv"

if [[ ! -f "$GRID_FILE" ]]; then
  echo "ERROR: grid file not found at $GRID_FILE"
  echo "Did you run: python slurm/ppo/make_grid.py  (from repo root)?"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# grid.tsv has 1 header line starting with '#'
LINE_NUM=$(( TASK_ID + 2 ))  # +1 for 1-indexed sed, +1 to skip header
LINE="$(sed -n "${LINE_NUM}p" "$GRID_FILE" || true)"

if [[ -z "$LINE" ]]; then
  echo "ERROR: No line for task_id=$TASK_ID (line_num=$LINE_NUM) in $GRID_FILE"
  exit 2
fi

IFS=$'\t' read -r ENV_NAME TOTAL_STEPS HIDDEN_SIZE DOUBLE_CRITIC MEMORYLESS ENTROPY_COEFF <<< "$LINE"

export ENV_NAME TOTAL_STEPS HIDDEN_SIZE DOUBLE_CRITIC MEMORYLESS ENTROPY_COEFF

echo "Selected grid row (task_id=$TASK_ID):"
echo "  ENV_NAME=$ENV_NAME"
echo "  TOTAL_STEPS=$TOTAL_STEPS"
echo "  HIDDEN_SIZE=$HIDDEN_SIZE"
echo "  DOUBLE_CRITIC=$DOUBLE_CRITIC"
echo "  MEMORYLESS=$MEMORYLESS"
echo "  ENTROPY_COEFF=$ENTROPY_COEFF"

bash /nas/ucb/juanlievano/pobax_RLC/slurm/ppo/probe_run_one.sh
