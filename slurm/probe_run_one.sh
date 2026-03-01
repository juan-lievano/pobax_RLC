#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by probe_array_launch.sh):
#   ENV_NAME, TOTAL_STEPS, HIDDEN_SIZE, DOUBLE_CRITIC, MEMORYLESS, ENTROPY_COEFF

# ---- Fixed constants (must match run_one.sh exactly) ----
N_SEEDS=5
ACTION_CONCAT=true

# ---- Probe pipeline hyperparams ----
N_TIMESTEPS=10000   # training budget: all probes capped to this many timesteps
H_IDX=0
EPOCHS=80
BATCH_SIZE=1024

# ---- Your environment setup ----
export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobax310
set -u

cd /nas/ucb/juanlievano/pobax_RLC

# ---- Reconstruct STUDY_NAME (identical logic to run_one.sh) ----
ENT_TAG="$(printf "%s" "$ENTROPY_COEFF" | sed 's/\./p/g')"
[[ "$DOUBLE_CRITIC" == "true" ]] && _DC="_dc" || _DC=""
[[ "$ACTION_CONCAT"  == "true" ]] && _AC="_ac" || _AC=""
[[ "$MEMORYLESS"     == "true" ]] && _ML="_ml" || _ML=""
STUDY_NAME="${ENV_NAME}_h${HIDDEN_SIZE}${_DC}${_AC}${_ML}_ent${ENT_TAG}_s${N_SEEDS}_ts${TOTAL_STEPS}"

RUN_DIR="/nas/ucb/juanlievano/pobax_RLC/results/${STUDY_NAME}"
OUT_DIR="/nas/ucb/juanlievano/pobax_RLC/probe_results/${STUDY_NAME}"

echo "=== Probe pipeline config ==="
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Host:        $(hostname)"
echo "Start time:  $(date)"
echo "STUDY_NAME:   $STUDY_NAME"
echo "RUN_DIR:      $RUN_DIR"
echo "OUT_DIR:      $OUT_DIR"
echo "N_TIMESTEPS:  $N_TIMESTEPS"
echo "=============================="

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: results directory not found: $RUN_DIR"
  echo "Has the training job completed for this experiment?"
  exit 3
fi

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

srun python scripts/bayesian_belief_probes/run_probe_pipeline.py \
  --run_dir     "$RUN_DIR"       \
  --out_dir     "$OUT_DIR"       \
  --n_timesteps "$N_TIMESTEPS"   \
  --h_idx       "$H_IDX"         \
  --epochs      "$EPOCHS"        \
  --batch_size  "$BATCH_SIZE"

echo "End time: $(date)"
