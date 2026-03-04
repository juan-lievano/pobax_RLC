#!/usr/bin/env bash
# Submit with dependency on training array:
#
#   TRAIN_JOB=$(sbatch --array=0-359 --parsable slurm/dqn/array_launch.sh)
#   sbatch --array=0-359 --dependency=aftercorr:$TRAIN_JOB slurm/dqn/probe_array_launch.sh
#
#SBATCH --job-name=dqn_probes
#SBATCH --output=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.log
#SBATCH --error=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

set -euo pipefail

GRID_FILE="/nas/ucb/juanlievano/pobax_RLC/slurm/dqn/grid.tsv"

if [[ ! -f "$GRID_FILE" ]]; then
  echo "ERROR: grid file not found at $GRID_FILE"
  echo "Did you run: python slurm/dqn/make_grid.py  (from repo root)?"
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

IFS=$'\t' read -r ENV_NAME LR TRACE_LENGTH HIDDEN_SIZE NUM_ENVS BUFFER_BATCH_SIZE MEMORYLESS <<< "$LINE"

export ENV_NAME LR TRACE_LENGTH HIDDEN_SIZE NUM_ENVS BUFFER_BATCH_SIZE MEMORYLESS

echo "Selected grid row (task_id=$TASK_ID):"
echo "  ENV_NAME=$ENV_NAME"
echo "  LR=$LR"
echo "  TRACE_LENGTH=$TRACE_LENGTH"
echo "  HIDDEN_SIZE=$HIDDEN_SIZE"
echo "  NUM_ENVS=$NUM_ENVS"
echo "  BUFFER_BATCH_SIZE=$BUFFER_BATCH_SIZE"
echo "  MEMORYLESS=$MEMORYLESS"

bash /nas/ucb/juanlievano/pobax_RLC/slurm/dqn/probe_run_one.sh
