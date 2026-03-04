#!/usr/bin/env bash
set -euo pipefail

# Expected env vars (set by probe_array_launch.sh):
#   RUN_DIR  — absolute path to the timestamped run directory

# ---- Probe pipeline hyperparams ----
N_TIMESTEPS=10000
H_IDX=0
EPOCHS=80
BATCH_SIZE=1024

# ---- Your environment setup ----
export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobaxRLC
set -u

cd /nas/ucb/juanlievano/pobax_RLC

# Derive OUT_DIR by replacing the results/ prefix with probe_results/
OUT_DIR="${RUN_DIR/\/results\//\/probe_results\/}"

echo "=== Probe pipeline config ==="
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Host:        $(hostname)"
echo "Start time:  $(date)"
echo "RUN_DIR:     $RUN_DIR"
echo "OUT_DIR:     $OUT_DIR"
echo "N_TIMESTEPS: $N_TIMESTEPS"
echo "=============================="

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run directory not found: $RUN_DIR"
  exit 3
fi

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

srun python scripts/bayesian_belief_probes/run_probe_pipeline.py \
  --run_dir     "$RUN_DIR"     \
  --out_dir     "$OUT_DIR"     \
  --n_timesteps "$N_TIMESTEPS" \
  --h_idx       "$H_IDX"       \
  --epochs      "$EPOCHS"      \
  --batch_size  "$BATCH_SIZE"  \
  --no_viz

echo "End time: $(date)"
