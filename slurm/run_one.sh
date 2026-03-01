#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by array launcher):
#   ENV_NAME, TOTAL_STEPS, HIDDEN_SIZE, DOUBLE_CRITIC, MEMORYLESS, ENTROPY_COEFF

# ---- Fixed constants (your spec) ----
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
LR=0.0025

# Lambdas constant for all envs (your spec)
LAMBDA0="0.9"
LAMBDA1="0.5"
LD_WEIGHT="0.25"

# Action concat: keep as you currently do (always true in your scripts)
ACTION_CONCAT=true

# ---- Your environment setup ----
export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobax310
set -u

cd /nas/ucb/juanlievano/pobax_RLC

echo "=== Run config ==="
echo "Job ID:          ${SLURM_JOB_ID:-N/A}"
echo "Host:            $(hostname)"
echo "Start time:      $(date)"
echo "ENV_NAME:        $ENV_NAME"
echo "TOTAL_STEPS:     $TOTAL_STEPS"
echo "HIDDEN_SIZE:     $HIDDEN_SIZE"
echo "DOUBLE_CRITIC:   $DOUBLE_CRITIC"
echo "MEMORYLESS:      $MEMORYLESS"
echo "ENTROPY_COEFF:   $ENTROPY_COEFF"
echo "NUM_ENVS:        $NUM_ENVS"
echo "NUM_STEPS:       $NUM_STEPS"
echo "NUM_CHECKPOINTS: $NUM_CHECKPOINTS"
echo "UPDATE_EPOCHS:   $UPDATE_EPOCHS"
echo "N_SEEDS:         $N_SEEDS"
echo "SEED:            $SEED"
echo "=================="

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader || true

# ---- Divisibility preflight (your constraint) ----
DIVISOR=$(( NUM_ENVS * NUM_STEPS * NUM_CHECKPOINTS ))  # 163840
REM=$(( TOTAL_STEPS % DIVISOR ))
echo "Divisibility check: TOTAL_STEPS % (NUM_ENVS*NUM_STEPS*NUM_CHECKPOINTS) = ${REM} (divisor=${DIVISOR})"
if [[ "$REM" -ne 0 ]]; then
  echo "ERROR: total_steps=$TOTAL_STEPS is not divisible by $DIVISOR (num_envs=$NUM_ENVS, num_steps=$NUM_STEPS, num_checkpoints=$NUM_CHECKPOINTS)."
  exit 2
fi

# ---- Env-type defaults (keep other params as you currently do by env type) ----
# LR=""
# case "$ENV_NAME" in
  # compass_world_*|marquee_*)
    # LR="2.5e-03"
    # ;;
  # rocksample_*)
    # LR="2.5e-04"
    # ;;
  # *)
    # echo "ERROR: Unknown env type for ENV_NAME='$ENV_NAME' (expected compass_world_*, marquee_*, rocksample_*)."
    # exit 3
    # ;;
# esac

# ---- Bool flags (now memoryless applies to ALL envs) ----
BOOL_FLAGS=""
[[ "$DOUBLE_CRITIC" == "true" ]] && BOOL_FLAGS+=" --double_critic"
[[ "$ACTION_CONCAT" == "true" ]] && BOOL_FLAGS+=" --action_concat"
[[ "$MEMORYLESS" == "true" ]] && BOOL_FLAGS+=" --memoryless"

# ---- Study name includes ALL gridded params ----
# helper: turn 0.05 -> 0p05, 0.2 -> 0p2
ENT_TAG="$(printf "%s" "$ENTROPY_COEFF" | sed 's/\./p/g')"
[[ "$DOUBLE_CRITIC" == "true" ]] && _DC="_dc" || _DC=""
[[ "$ACTION_CONCAT" == "true" ]] && _AC="_ac" || _AC=""
[[ "$MEMORYLESS" == "true" ]] && _ML="_ml" || _ML=""
STUDY_NAME="${ENV_NAME}_h${HIDDEN_SIZE}${_DC}${_AC}${_ML}_ent${ENT_TAG}_s${N_SEEDS}_ts${TOTAL_STEPS}"

echo "Running: python -m pobax.algos.ppo (study_name=$STUDY_NAME)"
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
  --study_name "$STUDY_NAME"
)

# execute
# shellcheck disable=SC2068
${CMD[@]}

echo "--- GPU info (post-training) ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true

echo "End time: $(date)"
