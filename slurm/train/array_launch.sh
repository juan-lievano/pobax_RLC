#!/usr/bin/env bash
#SBATCH --job-name=train_array
#SBATCH --output=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.log
#SBATCH --error=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

GRID_FILE="${GRID_FILE:-/nas/ucb/juanlievano/pobax_RLC/slurm/train/grid_final.tsv}"

if [[ ! -f "$GRID_FILE" ]]; then
  echo "ERROR: grid file not found at $GRID_FILE"
  echo "Did you run: python slurm/train/make_grid_final.py  (from repo root)?"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# grid_final.tsv has 1 header line starting with '#'
LINE_NUM=$(( TASK_ID + 2 ))  # +1 for 1-indexed sed, +1 to skip header
LINE="$(sed -n "${LINE_NUM}p" "$GRID_FILE" || true)"

if [[ -z "$LINE" ]]; then
  echo "ERROR: No line for task_id=$TASK_ID (line_num=$LINE_NUM) in $GRID_FILE"
  exit 2
fi

IFS=$'\t' read -r ALGO ENV_NAME HIDDEN_SIZE MEMORYLESS TOTAL_STEPS \
                   DOUBLE_CRITIC ENTROPY_COEFF LR TRACE_LENGTH NUM_ENVS BUFFER_BATCH_SIZE <<< "$LINE"

# Compute a batch dir name shared by all tasks in this array submission.
# Falls back to a local timestamp when not running under SLURM.
BATCH_DIR="batch_${SLURM_ARRAY_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}_$(date +%Y%m%d)"

export ALGO ENV_NAME HIDDEN_SIZE MEMORYLESS TOTAL_STEPS
export DOUBLE_CRITIC ENTROPY_COEFF LR TRACE_LENGTH NUM_ENVS BUFFER_BATCH_SIZE
export BATCH_DIR

echo "Selected grid row (task_id=$TASK_ID):"
echo "  ALGO=$ALGO"
echo "  ENV_NAME=$ENV_NAME"
echo "  HIDDEN_SIZE=$HIDDEN_SIZE"
echo "  MEMORYLESS=$MEMORYLESS"
echo "  TOTAL_STEPS=$TOTAL_STEPS"
echo "  DOUBLE_CRITIC=$DOUBLE_CRITIC"
echo "  ENTROPY_COEFF=$ENTROPY_COEFF"
echo "  LR=$LR"
echo "  TRACE_LENGTH=$TRACE_LENGTH"
echo "  NUM_ENVS=$NUM_ENVS"
echo "  BUFFER_BATCH_SIZE=$BUFFER_BATCH_SIZE"
echo "  BATCH_DIR=$BATCH_DIR"

case "$ALGO" in
  ppo)
    bash /nas/ucb/juanlievano/pobax_RLC/slurm/train/run_ppo.sh
    ;;
  dqn)
    bash /nas/ucb/juanlievano/pobax_RLC/slurm/train/run_dqn.sh
    ;;
  transformer_xl)
    bash /nas/ucb/juanlievano/pobax_RLC/slurm/train/run_transformer_xl.sh
    ;;
  *)
    echo "ERROR: unknown ALGO='$ALGO' (expected ppo, dqn, or transformer_xl)"
    exit 3
    ;;
esac
