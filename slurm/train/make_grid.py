#!/usr/bin/env python3
"""Generate slurm/train/grid.tsv with both PPO and DQN rows.

Unified columns (all present for every row; inapplicable columns get a neutral default):
  ALGO  ENV_NAME  HIDDEN_SIZE  MEMORYLESS  TOTAL_STEPS  DOUBLE_CRITIC  ENTROPY_COEFF
  LR  TRACE_LENGTH  NUM_ENVS  BUFFER_BATCH_SIZE

Usage (from repo root):
    python slurm/train/make_grid.py
Then submit:
    sbatch --array=0-N slurm/train/array_launch.sh
"""
from itertools import product

# ---- Shared envs ----
ENVS = [
    "marquee_40_16",
    "compass_world_8",
    "compass_world_10",
    "rocksample_5_5",
    "rocksample_7_8",
]

# ---- PPO grid ----
PPO_HIDDEN_SIZES   = [64, 128]
PPO_DOUBLE_CRITIC  = ["true", "false"]
PPO_MEMORYLESS     = ["true", "false"]
PPO_ENTROPY_COEFFS = ["0.2", "0.05"]
PPO_LR             = "0.0025"   # fixed for PPO

# PPO fixed constants (must match run_ppo.sh)
PPO_NUM_ENVS    = 64
PPO_NUM_STEPS   = 128
PPO_NUM_CKPTS   = 20
PPO_DIVISOR     = PPO_NUM_ENVS * PPO_NUM_STEPS * PPO_NUM_CKPTS  # 163,840

PPO_TOTAL_STEPS_OPTIONS = [
    5_242_880,    # DIVISOR * 32  (~5M)
    127_959_040,  # DIVISOR * 781 (~128M)
]

# ---- DQN grid ----
DQN_LRS                = ["1e-4", "5e-4", "1e-3"]
DQN_TRACE_LENGTHS_DRQN = ["100", "200"]   # swept for DRQN; DQN uses fixed placeholder
DQN_HIDDEN_SIZES       = ["64", "128"]
DQN_NUM_ENVS_LIST      = ["32", "64"]
DQN_BUFFER_BATCH_SIZES = ["64", "128"]
DQN_TOTAL_STEPS        = "256000000"

OUT_PATH = "slurm/train/grid.tsv"

HEADER = (
    "# ALGO\tENV_NAME\tHIDDEN_SIZE\tMEMORYLESS\tTOTAL_STEPS\t"
    "DOUBLE_CRITIC\tENTROPY_COEFF\tLR\tTRACE_LENGTH\tNUM_ENVS\tBUFFER_BATCH_SIZE"
)


def ppo_rows() -> list:
    rows = []
    for env, hs, dc, ml, ent, ts in product(
        ENVS, PPO_HIDDEN_SIZES, PPO_DOUBLE_CRITIC, PPO_MEMORYLESS,
        PPO_ENTROPY_COEFFS, PPO_TOTAL_STEPS_OPTIONS
    ):
        if ts % PPO_DIVISOR != 0:
            continue
        rows.append((
            "ppo",          # ALGO
            env,            # ENV_NAME
            str(hs),        # HIDDEN_SIZE
            ml,             # MEMORYLESS
            str(ts),        # TOTAL_STEPS
            dc,             # DOUBLE_CRITIC
            ent,            # ENTROPY_COEFF
            PPO_LR,         # LR (fixed for PPO)
            "0",            # TRACE_LENGTH (unused for PPO)
            "0",            # NUM_ENVS (unused; run_ppo.sh uses its own constant)
            "0",            # BUFFER_BATCH_SIZE (unused for PPO)
        ))
    return rows


def dqn_rows() -> list:
    # DRQN rows (memoryless=False): trace_length is swept
    drqn = [
        ("dqn", env, hs, "False", DQN_TOTAL_STEPS, "false", "0.0",
         lr, tl, ne, bbs)
        for env, lr, tl, hs, ne, bbs in product(
            ENVS, DQN_LRS, DQN_TRACE_LENGTHS_DRQN,
            DQN_HIDDEN_SIZES, DQN_NUM_ENVS_LIST, DQN_BUFFER_BATCH_SIZES
        )
    ]
    # DQN rows (memoryless=True): trace_length fixed at 100 (ignored by flat buffer)
    dqn = [
        ("dqn", env, hs, "True", DQN_TOTAL_STEPS, "false", "0.0",
         lr, "100", ne, bbs)
        for env, lr, hs, ne, bbs in product(
            ENVS, DQN_LRS, DQN_HIDDEN_SIZES, DQN_NUM_ENVS_LIST, DQN_BUFFER_BATCH_SIZES
        )
    ]
    return drqn + dqn


def main() -> None:
    ppo = ppo_rows()
    dqn = dqn_rows()
    rows = ppo + dqn

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"  PPO rows: {len(ppo)}")
    print(f"  DQN rows: {len(dqn)}  ({len(dqn) - len([r for r in dqn if r[3]=='True'])} DRQN"
          f" + {len([r for r in dqn if r[3]=='True'])} DQN)")
    print(f"  Grand total: {len(rows)}")
    print(f"\nSubmit: sbatch --array=0-{len(rows)-1} slurm/train/array_launch.sh")


if __name__ == "__main__":
    main()
