#!/usr/bin/env python3
"""Generate slurm/dqn/grid.tsv — one row per DRQN/DQN experiment.

360 total:
  240 DRQN (memoryless=False): 5 envs × 3 lrs × 2 trace_lengths × 2 hidden_sizes
                                × 2 num_envs × 2 buffer_batch_sizes
  120 DQN  (memoryless=True):  5 envs × 3 lrs × 2 hidden_sizes
                                × 2 num_envs × 2 buffer_batch_sizes
                                (trace_length fixed at 100, ignored by flat buffer)

Fixed: epsilon_finish=0.05, total_steps=256M, study_name=dqn_cluster_run_1

Usage (from repo root):
    python slurm/dqn/make_grid.py
Then submit:
    sbatch --array=0-359%32 slurm/dqn/array_launch.sh
    # Increase %32 to %64 or %96 if more GPUs are available.
"""
from itertools import product

ENVS = [
    "compass_world_8",
    "compass_world_10",
    "rocksample_5_5",
    "rocksample_7_8",
    "marquee_40_16",
]

LRS                = ["1e-4", "5e-4", "1e-3"]
TRACE_LENGTHS_DRQN = ["100", "200"]   # swept for DRQN; DQN uses fixed placeholder
HIDDEN_SIZES       = ["64", "128"]
NUM_ENVS_LIST      = ["32", "64"]
BUFFER_BATCH_SIZES = ["64", "128"]

OUT_PATH = "slurm/dqn/grid.tsv"


def main() -> None:
    # DRQN rows (memoryless=False): trace_length is swept
    drqn_rows = [
        (env, lr, tl, hs, ne, bbs, "False")
        for env, lr, tl, hs, ne, bbs in product(
            ENVS, LRS, TRACE_LENGTHS_DRQN, HIDDEN_SIZES, NUM_ENVS_LIST, BUFFER_BATCH_SIZES
        )
    ]

    # DQN rows (memoryless=True): trace_length fixed at 100 (ignored by flat buffer)
    dqn_rows = [
        (env, lr, "100", hs, ne, bbs, "True")
        for env, lr, hs, ne, bbs in product(
            ENVS, LRS, HIDDEN_SIZES, NUM_ENVS_LIST, BUFFER_BATCH_SIZES
        )
    ]

    rows = drqn_rows + dqn_rows

    assert len(drqn_rows) == 240, f"Expected 240 DRQN rows, got {len(drqn_rows)}"
    assert len(dqn_rows) == 120, f"Expected 120 DQN rows, got {len(dqn_rows)}"
    assert len(rows) == 360, f"Expected 360 total rows, got {len(rows)}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tLR\tTRACE_LENGTH\tHIDDEN_SIZE\tNUM_ENVS\tBUFFER_BATCH_SIZE\tMEMORYLESS\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}  ({len(drqn_rows)} DRQN + {len(dqn_rows)} DQN)")
    print(f"Submit: sbatch --array=0-359%32 slurm/dqn/array_launch.sh")


if __name__ == "__main__":
    main()
