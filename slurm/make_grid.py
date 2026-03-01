# File: slurm/make_grid.py
#!/usr/bin/env python3
from itertools import product

# ---- Grid spec ----
ENVS = [
    "marquee_40_16",
    "compass_world_8",
    "compass_world_10",
    "rocksample_5_5",
    "rocksample_7_8",
]

HIDDEN_SIZES = [64, 128]
DOUBLE_CRITIC = ["true", "false"]
MEMORYLESS = ["true", "false"]
ENTROPY_COEFFS = ["0.2", "0.05"]  # as strings for stable formatting/parsing

# ---- Fixed constants (must match run_one.sh) ----
NUM_ENVS = 64
NUM_STEPS = 128
NUM_CHECKPOINTS = 20
DIVISOR = NUM_ENVS * NUM_STEPS * NUM_CHECKPOINTS  # 163,840

# Two shared total_steps options (multiples of DIVISOR), near 5M and near 128M:
TOTAL_STEPS_OPTIONS = [
    5_242_880,     # DIVISOR * 32
    127_959_040,   # DIVISOR * 781  (~128M)
]

OUT_PATH = "slurm/grid.tsv"


def is_valid(total_steps: int) -> bool:
    return (total_steps % DIVISOR) == 0


def main() -> None:
    rows = []
    for env, hs, dc, ml, ent, ts in product(
        ENVS, HIDDEN_SIZES, DOUBLE_CRITIC, MEMORYLESS, ENTROPY_COEFFS, TOTAL_STEPS_OPTIONS
    ):
        if not is_valid(ts):
            continue
        rows.append((env, str(ts), str(hs), dc, ml, ent))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# ENV_NAME\tTOTAL_STEPS\tHIDDEN_SIZE\tDOUBLE_CRITIC\tMEMORYLESS\tENTROPY_COEFF\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"Divisor enforced: {DIVISOR} (= {NUM_ENVS}*{NUM_STEPS}*{NUM_CHECKPOINTS})")
    print(f"Total_steps options: {TOTAL_STEPS_OPTIONS}")
    print(f"Entropy grid: {ENTROPY_COEFFS}")


if __name__ == "__main__":
    main()
