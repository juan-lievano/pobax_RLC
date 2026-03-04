#!/usr/bin/env python3
"""Generate slurm/train/grid.tsv with DQN/DRQN rows.

Columns:
  ALGO  ENV_NAME  HIDDEN_SIZE  MEMORYLESS  TOTAL_STEPS  DOUBLE_CRITIC  ENTROPY_COEFF
  LR  TRACE_LENGTH  NUM_ENVS  BUFFER_BATCH_SIZE

Note: action_concat is always enabled in run_dqn.sh (not a grid column).

Hyperparameter choices informed by prior sweep results:
  - LR: 5e-4 and 1e-3 are clear winners; 1e-4 borderline; 5e-5 fails.
  - Trace length: 100 best from prior sweep; 200 to explore longer context.
  - Hidden size: 128 consistently works; 256 mostly fails; 64 untested (skipped).
  - Epsilon: fixed at 0.05 in run_dqn.sh (0.10 showed similar results).
  - num_envs and buffer_batch_size: swept here since prior results didn't isolate them.

Divisibility constraint (enforced below):
  total_steps must satisfy (total_steps // num_envs) % num_checkpoints == 0,
  i.e. total_steps % (num_envs * num_checkpoints) == 0.

Usage (from repo root):
    python slurm/train/make_grid.py
Then submit:
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

# ---- DQN grid ----
# LR: 5e-4 and 1e-3 are clear winners from prior sweep; include 1e-4 for coverage.
DQN_LRS                = ["1e-4", "5e-4", "1e-3"]
# Trace length: 100 is best from prior sweep; 200 to explore longer context.
DQN_TRACE_LENGTHS_DRQN = ["100", "200"]
# Hidden size: 128 consistently best; 256 mostly failed; 64 untested.
DQN_HIDDEN_SIZES       = ["128"]
# num_envs and buffer_batch_size: not isolated in prior sweep, swept here.
DQN_NUM_ENVS_LIST      = ["32", "64"]
DQN_BUFFER_BATCH_SIZES = ["64", "128"]
DQN_TOTAL_STEPS        = 256_000_000

# Must match --num_checkpoints passed in run_dqn.sh.
NUM_CHECKPOINTS = 20

OUT_PATH = "slurm/train/grid.tsv"

HEADER = (
    "# ALGO\tENV_NAME\tHIDDEN_SIZE\tMEMORYLESS\tTOTAL_STEPS\t"
    "DOUBLE_CRITIC\tENTROPY_COEFF\tLR\tTRACE_LENGTH\tNUM_ENVS\tBUFFER_BATCH_SIZE"
)


def check_divisibility(total_steps: int, num_envs: int, num_checkpoints: int,
                        num_steps: int = 1) -> bool:
    """Check that total_steps evenly divides into checkpoints.

    DQN:  num_updates = total_steps // num_envs           → num_steps=1  (default)
    PPO:  num_updates = total_steps // (num_envs * num_steps)

    In both cases the assert in the training code is: num_updates % num_checkpoints == 0,
    which is equivalent to: total_steps % (num_envs * num_steps * num_checkpoints) == 0.
    """
    return total_steps % (num_envs * num_steps * num_checkpoints) == 0


def dqn_rows() -> list:
    rows = []
    skipped = []

    # DRQN rows (memoryless=False): trace_length is swept
    for env, lr, tl, hs, ne, bbs in product(
        ENVS, DQN_LRS, DQN_TRACE_LENGTHS_DRQN, DQN_HIDDEN_SIZES,
        DQN_NUM_ENVS_LIST, DQN_BUFFER_BATCH_SIZES
    ):
        if not check_divisibility(DQN_TOTAL_STEPS, int(ne), NUM_CHECKPOINTS):
            skipped.append(f"DRQN env={env} ne={ne} ts={DQN_TOTAL_STEPS}")
            continue
        rows.append((
            "dqn", env, hs, "False", str(DQN_TOTAL_STEPS),
            "false", "0.0", lr, tl, ne, bbs
        ))

    # DQN rows (memoryless=True): trace_length fixed at 100 (ignored by flat buffer)
    for env, lr, hs, ne, bbs in product(
        ENVS, DQN_LRS, DQN_HIDDEN_SIZES, DQN_NUM_ENVS_LIST, DQN_BUFFER_BATCH_SIZES
    ):
        if not check_divisibility(DQN_TOTAL_STEPS, int(ne), NUM_CHECKPOINTS):
            skipped.append(f"DQN env={env} ne={ne} ts={DQN_TOTAL_STEPS}")
            continue
        rows.append((
            "dqn", env, hs, "True", str(DQN_TOTAL_STEPS),
            "false", "0.0", lr, "100", ne, bbs
        ))

    if skipped:
        print(f"WARNING: {len(skipped)} rows skipped (divisibility check failed):")
        for s in skipped:
            print(f"  {s}")

    return rows


def main() -> None:
    rows = dqn_rows()
    drqn = [r for r in rows if r[3] == "False"]
    dqn  = [r for r in rows if r[3] == "True"]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"  DRQN rows: {len(drqn)}")
    print(f"  DQN (memoryless) rows: {len(dqn)}")
    print(f"\nSubmit: sbatch --array=0-{len(rows)-1} slurm/train/array_launch.sh")


if __name__ == "__main__":
    main()
