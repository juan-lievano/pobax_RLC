#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by array_launch.sh):
#   ENV_NAME, TOTAL_STEPS, HIDDEN_SIZE, DOUBLE_CRITIC, MEMORYLESS,
#   ENTROPY_COEFF, LR, BATCH_DIR

# ---- Fixed constants ----
NUM_ENVS=64
NUM_STEPS=128
NUM_CHECKPOINTS=20
UPDATE_EPOCHS=4
N_SEEDS=5
SEED=2026
NUM_MINIBATCHES=8
NUM_EVAL_ENVS=256
STEPS_LOG_FREQ=32
PLATFORM=gpu

# Lambdas constant for all envs
LAMBDA0="0.9"
LAMBDA1="0.5"
LD_WEIGHT="0.25"

# Action concat: always on
ACTION_CONCAT=true

# ---- Your environment setup ----
export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobaxRLC
set -u

cd /nas/ucb/juanlievano/pobax_RLC

echo "=== PPO Run config ==="
echo "Job ID:          ${SLURM_JOB_ID:-N/A}"
echo "Host:            $(hostname)"
echo "Start time:      $(date)"
echo "BATCH_DIR:       $BATCH_DIR"
echo "ENV_NAME:        $ENV_NAME"
echo "TOTAL_STEPS:     $TOTAL_STEPS"
echo "HIDDEN_SIZE:     $HIDDEN_SIZE"
echo "DOUBLE_CRITIC:   $DOUBLE_CRITIC"
echo "MEMORYLESS:      $MEMORYLESS"
echo "ENTROPY_COEFF:   $ENTROPY_COEFF"
echo "LR:              $LR"
echo "NUM_ENVS:        $NUM_ENVS"
echo "NUM_STEPS:       $NUM_STEPS"
echo "NUM_CHECKPOINTS: $NUM_CHECKPOINTS"
echo "UPDATE_EPOCHS:   $UPDATE_EPOCHS"
echo "N_SEEDS:         $N_SEEDS"
echo "SEED:            $SEED"
echo "======================"

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader || true

# ---- Divisibility preflight ----
DIVISOR=$(( NUM_ENVS * NUM_STEPS * NUM_CHECKPOINTS ))
REM=$(( TOTAL_STEPS % DIVISOR ))
echo "Divisibility check: TOTAL_STEPS % (NUM_ENVS*NUM_STEPS*NUM_CHECKPOINTS) = ${REM} (divisor=${DIVISOR})"
if [[ "$REM" -ne 0 ]]; then
  echo "ERROR: total_steps=$TOTAL_STEPS is not divisible by $DIVISOR."
  exit 2
fi

# ---- Bool flags ----
BOOL_FLAGS=""
[[ "$DOUBLE_CRITIC" == "true" ]] && BOOL_FLAGS+=" --double_critic"
[[ "$ACTION_CONCAT" == "true" ]] && BOOL_FLAGS+=" --action_concat"
[[ "$MEMORYLESS"    == "true" ]] && BOOL_FLAGS+=" --memoryless"

# ---- Study name ----
ENT_TAG="$(printf "%s" "$ENTROPY_COEFF" | sed 's/\./p/g')"
[[ "$DOUBLE_CRITIC" == "true" ]] && _DC="_dc" || _DC=""
_AC="_ac"  # always on
[[ "$MEMORYLESS"    == "true" ]] && _ML="_ml" || _ML=""
STUDY_NAME="${ENV_NAME}_h${HIDDEN_SIZE}${_DC}${_AC}${_ML}_ent${ENT_TAG}_s${N_SEEDS}_ts${TOTAL_STEPS}"

echo "Running: python -m pobax.algos.ppo (study_name=${BATCH_DIR}/${STUDY_NAME})"
CMD=(srun python -m pobax.algos.ppo
  --env "$ENV_NAME"
  --hidden_size "$HIDDEN_SIZE"
  $BOOL_FLAGS
  --total_steps "$TOTAL_STEPS"
  --num_envs "$NUM_ENVS"
  --num_steps "$NUM_STEPS"
  --num_minibatches "$NUM_MINIBATCHES"
  --update_epochs "$UPDATE_EPOCHS"
  --entropy_coeff "$ENTROPY_COEFF"
  --n_seeds "$N_SEEDS"
  --seed "$SEED"
  --lr "$LR"
  --ld_weight "$LD_WEIGHT"
  --lambda0 "$LAMBDA0"
  --lambda1 "$LAMBDA1"
  --platform "$PLATFORM"
  --num_eval_envs "$NUM_EVAL_ENVS"
  --steps_log_freq "$STEPS_LOG_FREQ"
  --save_checkpoints
  --num_checkpoints "$NUM_CHECKPOINTS"
  --study_name "${BATCH_DIR}/${STUDY_NAME}"
)

# shellcheck disable=SC2068
${CMD[@]}

echo "--- GPU info (post-training) ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true

echo "End time: $(date)"
