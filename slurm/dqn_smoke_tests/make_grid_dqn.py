#!/usr/bin/env python3
"""Generate slurm/dqn_smoke_tests/grid_dqn.tsv — one row per DRQN experiment.

64 total: 4 envs × 2 lrs × 2 eps_finish × 2 trace_lengths × 2 hidden_sizes

Usage (from repo root):
    python slurm/dqn_smoke_tests/make_grid_dqn.py
Then submit (32 concurrent at a time):
    sbatch --array=0-63%32 slurm/dqn_smoke_tests/array_launch_dqn.sh
"""
from itertools import product

ENVS = [
    "tmaze_5",
    "compass_world_6",
    "compass_world_8",
    "rocksample_5_5",
]

LRS              = ["2.5e-4", "2.5e-3"]
EPSILON_FINISHES = ["0.05", "0.1"]
TRACE_LENGTHS    = ["20", "50"]
HIDDEN_SIZES     = ["64", "128"]

OUT_PATH = "slurm/dqn_smoke_tests/grid_dqn.tsv"


def main() -> None:
    rows = []
    for env, lr, eps, tl, hs in product(ENVS, LRS, EPSILON_FINISHES, TRACE_LENGTHS, HIDDEN_SIZES):
        rows.append((env, lr, eps, tl, hs))

    assert len(rows) == 64, f"Expected 64 rows, got {len(rows)}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tLR\tEPSILON_FINISH\tTRACE_LENGTH\tHIDDEN_SIZE\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Submit (32 concurrent): sbatch --array=0-63%32 slurm/dqn_smoke_tests/array_launch_dqn.sh")


if __name__ == "__main__":
    main()
