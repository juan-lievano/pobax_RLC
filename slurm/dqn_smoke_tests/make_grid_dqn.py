#!/usr/bin/env python3
"""Generate slurm/dqn_smoke_tests/grid_dqn.tsv — one row per DQN/DRQN combo.

Usage (from repo root):
    python slurm/dqn_smoke_tests/make_grid_dqn.py
Then submit all 16 jobs in parallel:
    sbatch --array=0-15 slurm/dqn_smoke_tests/array_launch_dqn.sh
"""
from itertools import product

# ---- Grid spec (4 × 2 × 2 = 16 rows) ----
ENVS = [
    "tmaze_5",
    "compass_world_6",
    "compass_world_8",
    "rocksample_5_5",
]

LRS = ["2.5e-4", "2.5e-3"]
EPSILON_FINISHES = ["0.05", "0.1"]
MEMORYLESS = ["false"]   # "false" = DRQN (GRU); add "true" for FF DQN later

OUT_PATH = "slurm/dqn_smoke_tests/grid_dqn.tsv"


def main() -> None:
    rows = []
    for env, lr, eps, ml in product(ENVS, LRS, EPSILON_FINISHES, MEMORYLESS):
        rows.append((env, lr, eps, ml))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tLR\tEPSILON_FINISH\tMEMORYLESS\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Submit with:  sbatch --array=0-{len(rows) - 1} slurm/dqn_smoke_tests/array_launch_dqn.sh")


if __name__ == "__main__":
    main()
