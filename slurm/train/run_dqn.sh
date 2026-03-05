#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by array_launch.sh):
#   ENV_NAME, LR, TRACE_LENGTH, HIDDEN_SIZE, NUM_ENVS, BUFFER_BATCH_SIZE,
#   MEMORYLESS, BATCH_DIR

# ---- Fixed constants ----
N_SEEDS=8
SEED=2026
NUM_CHECKPOINTS=10
: "${TOTAL_STEPS:?TOTAL_STEPS not set -- must be exported by array_launch.sh}"
BUFFER_SIZE=1000000
EPSILON_FINISH=0.05        # fixed (sweep showed low sensitivity)

# All schedule params below are in SCAN STEPS (vectorised env steps, +1 per _env_step).
# 1 scan step = NUM_ENVS individual transitions.
TRAINING_INTERVAL=10       # gradient update every 10 scan steps
TARGET_UPDATE_INTERVAL=62  # target copy every 62 scan steps
LEARNING_STARTS=156        # start learning after 156 scan steps
EPSILON_ANNEAL_TIME=$(( TOTAL_STEPS / 2 / NUM_ENVS ))  # anneal over first half of scan steps
NUM_EVAL_ENVS=64
PLATFORM=gpu

# ---- Environment setup ----
export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobaxRLC
set -u

cd /nas/ucb/juanlievano/pobax_RLC

MEM_TAG="$([ "$MEMORYLESS" = "True" ] && echo "dqn" || echo "drqn")"
STUDY_NAME="dqn_${ENV_NAME}_lr${LR}_tr${TRACE_LENGTH}_h${HIDDEN_SIZE}_ne${NUM_ENVS}_bbs${BUFFER_BATCH_SIZE}_${MEM_TAG}_ac"

echo "=== DQN Run config ==="
echo "Job ID:               ${SLURM_JOB_ID:-N/A}"
echo "Host:                 $(hostname)"
echo "Start time:           $(date)"
echo "BATCH_DIR:            $BATCH_DIR"
echo "ENV_NAME:             $ENV_NAME"
echo "LR:                   $LR"
echo "EPSILON_FINISH:       $EPSILON_FINISH"
echo "TRACE_LENGTH:         $TRACE_LENGTH"
echo "HIDDEN_SIZE:          $HIDDEN_SIZE"
echo "NUM_ENVS:             $NUM_ENVS"
echo "BUFFER_BATCH_SIZE:    $BUFFER_BATCH_SIZE"
echo "MEMORYLESS:           $MEMORYLESS  (mode: $MEM_TAG)"
echo "TOTAL_STEPS:          $TOTAL_STEPS"
echo "BUFFER_SIZE:          $BUFFER_SIZE"
echo "EPSILON_ANNEAL_TIME:  $EPSILON_ANNEAL_TIME"
echo "STUDY_NAME:           $STUDY_NAME"
echo "======================"

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader || true

# ---- Divisibility preflight ----
DIVISOR=$(( NUM_ENVS * NUM_CHECKPOINTS ))
REM=$(( TOTAL_STEPS % DIVISOR ))
if [[ "$REM" -ne 0 ]]; then
  echo "ERROR: total_steps=$TOTAL_STEPS not divisible by $DIVISOR"
  exit 2
fi

echo "Running $MEM_TAG: study_name=${BATCH_DIR}/${STUDY_NAME}"

CMD=(srun python -m pobax.algos.dqn
  --env "$ENV_NAME"
  --hidden_size "$HIDDEN_SIZE"
  --total_steps "$TOTAL_STEPS"
  --num_envs "$NUM_ENVS"
  --trace_length "$TRACE_LENGTH"
  --buffer_size "$BUFFER_SIZE"
  --buffer_batch_size "$BUFFER_BATCH_SIZE"
  --training_interval "$TRAINING_INTERVAL"
  --target_update_interval "$TARGET_UPDATE_INTERVAL"
  --learning_starts "$LEARNING_STARTS"
  --epsilon_anneal_time "$EPSILON_ANNEAL_TIME"
  --epsilon_finish "$EPSILON_FINISH"
  --lr "$LR"
  --n_seeds "$N_SEEDS"
  --seed "$SEED"
  --platform "$PLATFORM"
  --num_eval_envs "$NUM_EVAL_ENVS"
  --action_concat
  --study_name "${BATCH_DIR}/${STUDY_NAME}"
  --save_checkpoints
  --num_checkpoints "$NUM_CHECKPOINTS"
)

if [ "$MEMORYLESS" = "True" ]; then
  CMD+=(--memoryless)
fi

# shellcheck disable=SC2068
${CMD[@]}

echo "--- GPU info (post-training) ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true

echo "End time: $(date)"
