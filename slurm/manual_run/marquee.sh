#!/bin/bash
#SBATCH --job-name=marquee_40_16
#SBATCH --output=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%j.log
#SBATCH --error=/nas/ucb/juanlievano/pobax_RLC/slurm/logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

ENV_NAME=marquee_40_16
HIDDEN_SIZE=64
DOUBLE_CRITIC=true
ACTION_CONCAT=true
TOTAL_STEPS=256000000

[[ "$DOUBLE_CRITIC" == "true" ]] && _DC="_dc" || _DC=""
[[ "$ACTION_CONCAT" == "true" ]] && _AC="_ac" || _AC=""
STUDY_NAME="${ENV_NAME}_h${HIDDEN_SIZE}${_DC}${_AC}_$((TOTAL_STEPS / 1000000))M"

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobax310
set -u

cd /nas/ucb/juanlievano/pobax_RLC

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader

echo "Running: python -m pobax.algos.ppo"
BOOL_FLAGS=""
[[ "$DOUBLE_CRITIC" == "true" ]] && BOOL_FLAGS+=" --double_critic"
[[ "$ACTION_CONCAT" == "true" ]] && BOOL_FLAGS+=" --action_concat"

srun python -m pobax.algos.ppo \
  --env "$ENV_NAME" \
  --hidden_size "$HIDDEN_SIZE" \
  $BOOL_FLAGS \
  --total_steps "$TOTAL_STEPS" \
  --num_envs 1024 \
  --num_steps 32 \
  --num_minibatches 8 \
  --update_epochs 3 \
  --n_seeds 5 \
  --seed 2024 \
  --lr 2.5e-03 \
  --ld_weight 0.25 \
  --lambda0 0.9 \
  --lambda1 0.5 \
  --platform gpu \
  --num_eval_envs 256 \
  --steps_log_freq 32 \
  --save_checkpoints \
  --num_checkpoints 10 \
  --study_name "$STUDY_NAME"

echo "--- GPU info (post-training) ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "End time: $(date)"
