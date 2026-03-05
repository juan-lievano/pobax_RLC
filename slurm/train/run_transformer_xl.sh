#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by array_launch.sh):
#   ENV_NAME, HIDDEN_SIZE, DOUBLE_CRITIC, ENTROPY_COEFF, LR,
#   TOTAL_STEPS, BATCH_DIR

# ---- Fixed constants ----
NUM_ENVS=64
NUM_STEPS=128
NUM_CHECKPOINTS=20
UPDATE_EPOCHS=4
N_SEEDS=8
SEED=2026
NUM_MINIBATCHES=8
NUM_EVAL_ENVS=256
STEPS_LOG_FREQ=32
PLATFORM=gpu

# Lambdas / LD weight constants for all envs
# (kept hardcoded by design for this sweep family)
LAMBDA0="0.7"
LAMBDA1="0.95"
LD_WEIGHT="0.25"

# Action concat: always on
ACTION_CONCAT=true

# Transformer-specific constants
EMBED_SIZE=256
NUM_HEADS=8
QKV_FEATURES=256
NUM_LAYERS=2
WINDOW_MEM=128
WINDOW_GRAD=64
GATING=true
GATING_BIAS=2.0

# ---- Your environment setup ----
export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobaxRLC
set -u

cd /nas/ucb/juanlievano/pobax_RLC

echo "=== GTrXL Run config ==="
echo "Job ID:          ${SLURM_JOB_ID:-N/A}"
echo "Host:            $(hostname)"
echo "Start time:      $(date)"
echo "BATCH_DIR:       $BATCH_DIR"
echo "ENV_NAME:        $ENV_NAME"
echo "TOTAL_STEPS:     $TOTAL_STEPS"
echo "HIDDEN_SIZE:     $HIDDEN_SIZE"
echo "DOUBLE_CRITIC:   $DOUBLE_CRITIC"
echo "ENTROPY_COEFF:   $ENTROPY_COEFF"
echo "LR:              $LR"
echo "EMBED_SIZE:      $EMBED_SIZE"
echo "NUM_HEADS:       $NUM_HEADS"
echo "NUM_LAYERS:      $NUM_LAYERS"
echo "WINDOW_MEM:      $WINDOW_MEM"
echo "WINDOW_GRAD:     $WINDOW_GRAD"
echo "NUM_ENVS:        $NUM_ENVS"
echo "NUM_STEPS:       $NUM_STEPS"
echo "NUM_CHECKPOINTS: $NUM_CHECKPOINTS"
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
[[ "$GATING"        == "true" ]] && BOOL_FLAGS+=" --gating"

# ---- Study name ----
ENT_TAG="$(printf "%s" "$ENTROPY_COEFF" | sed 's/\./p/g')"
[[ "$DOUBLE_CRITIC" == "true" ]] && _DC="_dc" || _DC=""
_AC="_ac"  # always on
STUDY_NAME="gtrxl_${ENV_NAME}_h${HIDDEN_SIZE}_e${EMBED_SIZE}${_DC}${_AC}_ent${ENT_TAG}_s${N_SEEDS}_ts${TOTAL_STEPS}"

echo "Running: python -m pobax.algos.transformer_xl (study_name=${BATCH_DIR}/${STUDY_NAME})"
CMD=(srun python -m pobax.algos.transformer_xl
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
  --embed_size "$EMBED_SIZE"
  --num_heads "$NUM_HEADS"
  --qkv_features "$QKV_FEATURES"
  --num_layers "$NUM_LAYERS"
  --window_mem "$WINDOW_MEM"
  --window_grad "$WINDOW_GRAD"
  --gating_bias "$GATING_BIAS"
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
