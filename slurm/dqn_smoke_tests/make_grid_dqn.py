#!/usr/bin/env python3
"""Generate slurm/dqn_smoke_tests/grid_dqn.tsv — one row per DRQN experiment.

144 total: 3 envs × 4 lrs × 2 eps_finish × 3 trace_lengths × 2 hidden_sizes

Changes from previous sweep (96 runs, 5M steps):
  - envs:         drop compass_world_6 (too easy); add compass_world_10
  - lr:           drop 2.5e-4 (mediocre everywhere); add 5e-5 (rocksample wants low LR)
  - trace_length: add 100 (longer memory horizon, unexplored)
  - hidden_size:  drop 64; add 256 (more capacity for harder envs)
  - total_steps:  5M → 64M

Usage (from repo root):
    python slurm/dqn_smoke_tests/make_grid_dqn.py
Then submit (32 concurrent at a time):
    sbatch --array=0-143%32 slurm/dqn_smoke_tests/array_launch_dqn.sh
"""
from itertools import product

ENVS = [
    "compass_world_8",
    "compass_world_10",
    "rocksample_5_5",
]

LRS              = ["5e-5", "1e-4", "5e-4", "1e-3"]
EPSILON_FINISHES = ["0.05", "0.1"]
TRACE_LENGTHS    = ["20", "50", "100"]
HIDDEN_SIZES     = ["128", "256"]

OUT_PATH = "slurm/dqn_smoke_tests/grid_dqn.tsv"


def main() -> None:
    rows = []
    for env, lr, eps, tl, hs in product(ENVS, LRS, EPSILON_FINISHES, TRACE_LENGTHS, HIDDEN_SIZES):
        rows.append((env, lr, eps, tl, hs))

    assert len(rows) == 144, f"Expected 144 rows, got {len(rows)}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tLR\tEPSILON_FINISH\tTRACE_LENGTH\tHIDDEN_SIZE\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Submit (32 concurrent): sbatch --array=0-143%32 slurm/dqn_smoke_tests/array_launch_dqn.sh")


if __name__ == "__main__":
    main()
