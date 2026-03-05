#!/usr/bin/env python3
"""Generate slurm/train/grid_final.tsv — unified grid of best hyperparameters.

60 rows covering 8 algo/architecture combos on 5 POMDP environments,
with hand-picked best hparams from prior sweeps.

Columns (same 11-column format as grid.tsv / grid_transformer_xl.tsv):
  ALGO  ENV_NAME  HIDDEN_SIZE  MEMORYLESS  TOTAL_STEPS  DOUBLE_CRITIC
  ENTROPY_COEFF  LR  TRACE_LENGTH  NUM_ENVS  BUFFER_BATCH_SIZE

Grid composition (60 rows):
  PPO+GRU           5 envs =  5
  PPO+FF            5 envs =  5
  LambdaDisc+GRU    5 envs =  5
  LambdaDisc+FF     5 envs =  5
  Transformer+PPO   5 envs x 2 ent = 10
  Transformer+LD    5 envs x 2 ent = 10
  DRQN              5 envs x 2 step budgets = 10
  DQN               5 envs x 2 step budgets = 10

Usage:
    python slurm/train/make_grid_final.py
Then submit:
    GRID_FILE=/path/to/grid_final.tsv sbatch --array=0-59 slurm/train/array_launch.sh
"""

ENVS = [
    "compass_world_8",
    "compass_world_10",
    "marquee_40_16",
    "rocksample_5_5",
    "rocksample_7_8",
]

# Step budgets
PPO_STEPS = 52_019_200       # divisible by 64*128*10 = 81,920
DQN_STEPS_SHORT = 52_019_200 # divisible by 64*10=640 and 32*10=320
DQN_STEPS_LONG = 100_000_000 # divisible by 64*10=640 and 32*10=320

NUM_CHECKPOINTS_PPO = 10     # num_envs=64, num_steps=128
NUM_CHECKPOINTS_DQN = 10     # num_steps=1 for DQN

OUT_PATH = "slurm/train/grid_final.tsv"

HEADER = (
    "# ALGO\tENV_NAME\tHIDDEN_SIZE\tMEMORYLESS\tTOTAL_STEPS\t"
    "DOUBLE_CRITIC\tENTROPY_COEFF\tLR\tTRACE_LENGTH\tNUM_ENVS\tBUFFER_BATCH_SIZE"
)


def check_divisibility_ppo(total_steps: int) -> bool:
    """PPO/Transformer: total_steps % (64 * 128 * 10) == 0"""
    return total_steps % (64 * 128 * NUM_CHECKPOINTS_PPO) == 0


def check_divisibility_dqn(total_steps: int, num_envs: int) -> bool:
    """DQN: total_steps % (num_envs * 10) == 0"""
    return total_steps % (num_envs * NUM_CHECKPOINTS_DQN) == 0


def build_rows() -> list:
    rows = []

    # ------------------------------------------------------------------
    # 1. PPO+GRU (5 rows)
    # ALGO=ppo, MEMORYLESS=false, DC=false, LR=2.5e-3
    # ------------------------------------------------------------------
    ppo_gru = [
        # (env, hs, ent)
        ("compass_world_8",  "64",  "0.05"),
        ("compass_world_10", "64",  "0.05"),
        ("marquee_40_16",    "128", "0.05"),
        ("rocksample_5_5",   "128", "0.05"),
        ("rocksample_7_8",   "128", "0.05"),
    ]
    for env, hs, ent in ppo_gru:
        rows.append((
            "ppo", env, hs, "false", str(PPO_STEPS),
            "false", ent, "2.5e-3", "0", "0", "0"
        ))

    # ------------------------------------------------------------------
    # 2. PPO+FF (5 rows)
    # ALGO=ppo, MEMORYLESS=true, DC=false, LR=2.5e-3
    # ------------------------------------------------------------------
    ppo_ff = [
        ("compass_world_8",  "128", "0.05"),
        ("compass_world_10", "128", "0.05"),
        ("marquee_40_16",    "128", "0.05"),
        ("rocksample_5_5",   "64",  "0.20"),
        ("rocksample_7_8",   "64",  "0.20"),
    ]
    for env, hs, ent in ppo_ff:
        rows.append((
            "ppo", env, hs, "true", str(PPO_STEPS),
            "false", ent, "2.5e-3", "0", "0", "0"
        ))

    # ------------------------------------------------------------------
    # 3. LambdaDisc+GRU (5 rows)
    # ALGO=ppo, MEMORYLESS=false, DC=true, LR=2.5e-3
    # ------------------------------------------------------------------
    ld_gru = [
        ("compass_world_8",  "128", "0.05"),
        ("compass_world_10", "128", "0.05"),
        ("marquee_40_16",    "128", "0.05"),
        ("rocksample_5_5",   "64",  "0.05"),
        ("rocksample_7_8",   "128", "0.05"),
    ]
    for env, hs, ent in ld_gru:
        rows.append((
            "ppo", env, hs, "false", str(PPO_STEPS),
            "true", ent, "2.5e-3", "0", "0", "0"
        ))

    # ------------------------------------------------------------------
    # 4. LambdaDisc+FF (5 rows)
    # ALGO=ppo, MEMORYLESS=true, DC=true, LR=2.5e-3
    # ------------------------------------------------------------------
    ld_ff = [
        ("compass_world_8",  "128", "0.05"),
        ("compass_world_10", "128", "0.05"),
        ("marquee_40_16",    "64",  "0.05"),
        ("rocksample_5_5",   "128", "0.05"),
        ("rocksample_7_8",   "128", "0.05"),
    ]
    for env, hs, ent in ld_ff:
        rows.append((
            "ppo", env, hs, "true", str(PPO_STEPS),
            "true", ent, "2.5e-3", "0", "0", "0"
        ))

    # ------------------------------------------------------------------
    # 5. Transformer+PPO (10 rows: 5 envs x 2 entropy values)
    # ALGO=transformer_xl, DC=false, hs=128, LR=1e-4
    # ------------------------------------------------------------------
    for env in ENVS:
        for ent in ["0.1", "0.01"]:
            rows.append((
                "transformer_xl", env, "128", "False", str(PPO_STEPS),
                "false", ent, "1e-4", "0", "0", "0"
            ))

    # ------------------------------------------------------------------
    # 6. Transformer+LambdaDisc (10 rows: 5 envs x 2 entropy values)
    # ALGO=transformer_xl, DC=true, hs=128, LR=1e-4
    # ------------------------------------------------------------------
    for env in ENVS:
        for ent in ["0.1", "0.01"]:
            rows.append((
                "transformer_xl", env, "128", "False", str(PPO_STEPS),
                "true", ent, "1e-4", "0", "0", "0"
            ))

    # ------------------------------------------------------------------
    # 7. DRQN (10 rows: 5 envs x 2 step budgets)
    # ALGO=dqn, MEMORYLESS=False, hs=128
    # ------------------------------------------------------------------
    drqn_hparams = {
        # env: (lr, trace_length, num_envs, buffer_batch_size)
        "compass_world_8":  ("1e-3", "100", "64", "64"),
        "compass_world_10": ("1e-3", "100", "64", "64"),
        "marquee_40_16":    ("1e-3", "100", "64", "64"),
        "rocksample_5_5":   ("1e-4", "200", "64", "128"),
        "rocksample_7_8":   ("1e-4", "200", "64", "64"),
    }
    for env in ENVS:
        lr, tl, ne, bbs = drqn_hparams[env]
        for ts in [DQN_STEPS_SHORT, DQN_STEPS_LONG]:
            rows.append((
                "dqn", env, "128", "False", str(ts),
                "false", "0.0", lr, tl, ne, bbs
            ))

    # ------------------------------------------------------------------
    # 8. DQN memoryless (10 rows: 5 envs x 2 step budgets)
    # ALGO=dqn, MEMORYLESS=True, hs=128
    # ------------------------------------------------------------------
    dqn_hparams = {
        # env: (lr, num_envs, buffer_batch_size)
        "marquee_40_16":    ("5e-4", "32", "128"),
        "compass_world_8":  ("1e-3", "64", "128"),
        "compass_world_10": ("1e-3", "64", "128"),
        "rocksample_5_5":   ("1e-3", "64", "128"),
        "rocksample_7_8":   ("1e-3", "64", "128"),
    }
    for env in ENVS:
        lr, ne, bbs = dqn_hparams[env]
        for ts in [DQN_STEPS_SHORT, DQN_STEPS_LONG]:
            rows.append((
                "dqn", env, "128", "True", str(ts),
                "false", "0.0", lr, "100", ne, bbs
            ))

    return rows


def validate(rows: list) -> None:
    """Check divisibility constraints for all rows."""
    errors = []
    for i, r in enumerate(rows):
        algo, env = r[0], r[1]
        ts = int(r[4])
        if algo in ("ppo", "transformer_xl"):
            if not check_divisibility_ppo(ts):
                errors.append(f"Row {i}: {algo} {env} ts={ts} not divisible by 64*128*10")
        elif algo == "dqn":
            num_envs = int(r[9])
            if not check_divisibility_dqn(ts, num_envs):
                errors.append(f"Row {i}: dqn {env} ts={ts} ne={num_envs} not divisible by {num_envs}*10")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError(f"{len(errors)} divisibility check(s) failed")
    print("All divisibility checks passed.")


def main() -> None:
    rows = build_rows()
    validate(rows)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    # Summary
    ppo_rows = [r for r in rows if r[0] == "ppo"]
    txl_rows = [r for r in rows if r[0] == "transformer_xl"]
    dqn_rows = [r for r in rows if r[0] == "dqn"]
    drqn_rows = [r for r in dqn_rows if r[3] == "False"]
    dqn_ml_rows = [r for r in dqn_rows if r[3] == "True"]

    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")
    print(f"  PPO/LambdaDisc (GRU+FF): {len(ppo_rows)}")
    print(f"  Transformer (PPO+LD):    {len(txl_rows)}")
    print(f"  DRQN:                    {len(drqn_rows)}")
    print(f"  DQN (memoryless):        {len(dqn_ml_rows)}")
    print(f"\nSubmit:")
    print(f"  GRID_FILE=/path/to/grid_final.tsv sbatch --array=0-{len(rows)-1} slurm/train/array_launch.sh")


if __name__ == "__main__":
    main()
