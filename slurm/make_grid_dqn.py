#!/usr/bin/env python3
"""Generate slurm/grid_dqn.tsv — one row per DQN/DRQN hyperparameter combo.

Usage (from repo root):
    python slurm/make_grid_dqn.py
Then submit:
    sbatch --array=0-<N_ROWS-1> slurm/array_launch_dqn.sh
"""
from itertools import product

# ---- Grid spec ----
ENVS = [
    "compass_world_8",
]

LRS = ["2.5e-4", "2.5e-3"]
EPSILON_FINISHES = ["0.05", "0.1"]
MEMORYLESS = ["false"]   # "false" = DRQN (GRU), "true" = DQN (FF)

OUT_PATH = "slurm/grid_dqn.tsv"


def main() -> None:
    rows = []
    for env, lr, eps, ml in product(ENVS, LRS, EPSILON_FINISHES, MEMORYLESS):
        rows.append((env, lr, eps, ml))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tLR\tEPSILON_FINISH\tMEMORYLESS\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Submit with:  sbatch --array=0-{len(rows) - 1} slurm/array_launch_dqn.sh")


if __name__ == "__main__":
    main()
