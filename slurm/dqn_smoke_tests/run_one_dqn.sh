#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by array_launch_dqn.sh):
#   ENV_NAME, LR, EPSILON_FINISH, TRACE_LENGTH, HIDDEN_SIZE

# ---- Fixed constants ----
NUM_ENVS=32
N_SEEDS=3
SEED=2026
TOTAL_STEPS=2000000
BUFFER_SIZE=100000
TRAINING_INTERVAL=10
TARGET_UPDATE_INTERVAL=2000
LEARNING_STARTS=10000
EPSILON_ANNEAL_TIME=1000000
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

echo "=== Run config ==="
echo "Job ID:               ${SLURM_JOB_ID:-N/A}"
echo "Host:                 $(hostname)"
echo "Start time:           $(date)"
echo "ENV_NAME:             $ENV_NAME"
echo "LR:                   $LR"
echo "EPSILON_FINISH:       $EPSILON_FINISH"
echo "TRACE_LENGTH:         $TRACE_LENGTH"
echo "HIDDEN_SIZE:          $HIDDEN_SIZE"
echo "NUM_ENVS:             $NUM_ENVS"
echo "TOTAL_STEPS:          $TOTAL_STEPS"
echo "BUFFER_SIZE:          $BUFFER_SIZE"
echo "EPSILON_ANNEAL_TIME:  $EPSILON_ANNEAL_TIME"
echo "=================="

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader || true

# ---- Study name ----
LR_TAG="$(printf "%s" "$LR" | sed 's/\./p/g; s/e-0*/e/g')"
EPS_TAG="$(printf "%s" "$EPSILON_FINISH" | sed 's/\./p/g')"
STUDY_NAME="${ENV_NAME}_drqn_lr${LR_TAG}_eps${EPS_TAG}_tr${TRACE_LENGTH}_h${HIDDEN_SIZE}_s${N_SEEDS}"

echo "Running DRQN: study_name=$STUDY_NAME"

CMD=(srun python -m pobax.algos.dqn
  --env "$ENV_NAME"
  --hidden_size "$HIDDEN_SIZE"
  --total_steps "$TOTAL_STEPS"
  --num_envs "$NUM_ENVS"
  --trace_length "$TRACE_LENGTH"
  --buffer_size "$BUFFER_SIZE"
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
  --study_name "$STUDY_NAME"
)

# shellcheck disable=SC2068
${CMD[@]}

echo "--- GPU info (post-training) ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true

echo "End time: $(date)"
