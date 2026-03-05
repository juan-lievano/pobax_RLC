#!/usr/bin/env python3
"""Generate slurm/train/grid_transformer_xl.tsv with GTrXL rows.

Same 11-column format as grid.tsv so array_launch.sh can read it unchanged.
TRACE_LENGTH, NUM_ENVS, BUFFER_BATCH_SIZE are set to "0" (unused by transformer).

Hyperparameter choices:
  - LR: sweep 1e-4, 5e-4, 1e-3 (same spread as DQN/PPO)
  - DOUBLE_CRITIC: both true and false
  - ENTROPY_COEFF: sweep 0.01 and 0.1

TOTAL_STEPS = 50,135,040  (~50M, divisible by 64*128*20 = 163,840)

Usage (from repo root):
    python slurm/train/make_grid_transformer_xl.py
Then submit:
    GRID_FILE=/nas/ucb/juanlievano/pobax_RLC/slurm/train/grid_transformer_xl.tsv \\
      sbatch --array=0-N slurm/train/array_launch.sh
"""
from itertools import product

# ---- Envs ----
ENVS = [
    "marquee_40_16",
    "compass_world_8",
    "compass_world_10",
    "rocksample_5_5",
    "rocksample_7_8",
]

# ---- GTrXL grid ----
LRS = ["1e-4", "5e-4", "1e-3"]
DOUBLE_CRITICS = ["false", "true"]
ENTROPY_COEFFS = ["0.01", "0.1"]
HIDDEN_SIZE = "128"

# ~50M steps, divisible by 64 * 128 * 20 = 163,840
TOTAL_STEPS = 50_135_040

# Must match --num_checkpoints in run_transformer_xl.sh
NUM_CHECKPOINTS = 20
NUM_ENVS_FIXED = 64
NUM_STEPS_FIXED = 128

OUT_PATH = "slurm/train/grid_transformer_xl.tsv"

HEADER = (
    "# ALGO\tENV_NAME\tHIDDEN_SIZE\tMEMORYLESS\tTOTAL_STEPS\t"
    "DOUBLE_CRITIC\tENTROPY_COEFF\tLR\tTRACE_LENGTH\tNUM_ENVS\tBUFFER_BATCH_SIZE"
)


def check_divisibility(total_steps: int, num_envs: int, num_steps: int,
                        num_checkpoints: int) -> bool:
    """total_steps % (num_envs * num_steps * num_checkpoints) == 0"""
    return total_steps % (num_envs * num_steps * num_checkpoints) == 0


def transformer_rows() -> list:
    rows = []
    skipped = []

    for env, lr, dc, ent in product(ENVS, LRS, DOUBLE_CRITICS, ENTROPY_COEFFS):
        if not check_divisibility(TOTAL_STEPS, NUM_ENVS_FIXED, NUM_STEPS_FIXED, NUM_CHECKPOINTS):
            skipped.append(f"GTrXL env={env} lr={lr} dc={dc} ent={ent}")
            continue
        rows.append((
            "transformer_xl", env, HIDDEN_SIZE, "False", str(TOTAL_STEPS),
            dc, ent, lr,
            "0",   # TRACE_LENGTH (unused)
            "0",   # NUM_ENVS (unused; actual num_envs is hardcoded in run_transformer_xl.sh)
            "0",   # BUFFER_BATCH_SIZE (unused)
        ))

    if skipped:
        print(f"WARNING: {len(skipped)} rows skipped (divisibility check failed):")
        for s in skipped:
            print(f"  {s}")

    return rows


def main() -> None:
    rows = transformer_rows()
    dc_true = [r for r in rows if r[5] == "true"]
    dc_false = [r for r in rows if r[5] == "false"]
    ent_001 = [r for r in rows if r[6] == "0.01"]
    ent_01 = [r for r in rows if r[6] == "0.1"]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"  double_critic=false rows: {len(dc_false)}")
    print(f"  double_critic=true  rows: {len(dc_true)}")
    print(f"  entropy=0.01 rows:        {len(ent_001)}")
    print(f"  entropy=0.1  rows:        {len(ent_01)}")
    print(f"\nSubmit:")
    print(f"  GRID_FILE=/nas/ucb/juanlievano/pobax_RLC/slurm/train/grid_transformer_xl.tsv \\")
    print(f"    sbatch --array=0-{len(rows)-1} slurm/train/array_launch.sh")


if __name__ == "__main__":
    main()
