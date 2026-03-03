#!/usr/bin/env python3
"""Generate slurm/dqn_smoke_tests/grid_dqn.tsv — one row per DRQN experiment.

96 total: 3 envs × 4 lrs × 2 eps_finish × 2 trace_lengths × 2 hidden_sizes

LR sweep rationale: 2.5e-3 failed everywhere (too high); adding 1e-4 and 5e-4
to bracket the known-good 2.5e-4 and probe one decade higher with 1e-3.

Usage (from repo root):
    python slurm/dqn_smoke_tests/make_grid_dqn.py
Then submit (32 concurrent at a time):
    sbatch --array=0-95%32 slurm/dqn_smoke_tests/array_launch_dqn.sh
"""
from itertools import product

ENVS = [
    "compass_world_6",
    "compass_world_8",
    "rocksample_5_5",
]

LRS              = ["1e-4", "2.5e-4", "5e-4", "1e-3"]
EPSILON_FINISHES = ["0.05", "0.1"]
TRACE_LENGTHS    = ["20", "50"]
HIDDEN_SIZES     = ["64", "128"]

OUT_PATH = "slurm/dqn_smoke_tests/grid_dqn.tsv"


def main() -> None:
    rows = []
    for env, lr, eps, tl, hs in product(ENVS, LRS, EPSILON_FINISHES, TRACE_LENGTHS, HIDDEN_SIZES):
        rows.append((env, lr, eps, tl, hs))

    assert len(rows) == 96, f"Expected 96 rows, got {len(rows)}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tLR\tEPSILON_FINISH\tTRACE_LENGTH\tHIDDEN_SIZE\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Submit (32 concurrent): sbatch --array=0-95%32 slurm/dqn_smoke_tests/array_launch_dqn.sh")


if __name__ == "__main__":
    main()
