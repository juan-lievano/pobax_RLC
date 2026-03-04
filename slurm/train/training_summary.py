#!/usr/bin/env python3
"""Unified summary table for a batch directory (PPO and DQN runs).

Scans `dir` at depths 1 and 2, loads every valid Orbax checkpoint, and prints
one row per run.  Invalid or old-format checkpoints are silently skipped.

Usage:
    python slurm/train/training_summary.py results/batch_12345_20260304/
    python slurm/train/training_summary.py results/batch_12345/ --pattern rocksample
    python slurm/train/training_summary.py results/ --verbose
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint


# ── helpers ──────────────────────────────────────────────────────────────────

def _restore(path: Path, verbose: bool) -> dict | None:
    try:
        return orbax.checkpoint.PyTreeCheckpointer().restore(path)
    except Exception as exc:
        if verbose:
            print(f"  [WARN] could not load {path}: {exc}", file=sys.stderr)
        return None


def _eval_return(result: dict) -> tuple[float, float, int]:
    fe = result.get("final_eval", result.get("final_eval_metric", {}))
    rets = np.asarray(fe.get("returned_episode_returns", []))
    mask = np.asarray(fe.get("returned_episode", []), dtype=bool)
    completed = rets[mask]
    if completed.size == 0:
        return float("nan"), float("nan"), 0
    return float(completed.mean()), float(completed.std()), int(completed.size)


def _runtime_str(result: dict) -> str:
    rt = result.get("total_runtime", None)
    if rt is None:
        return "?"
    secs = float(np.asarray(rt).ravel()[0])
    return f"{secs / 60:.1f}m"


def _scalar(v, default=0):
    if isinstance(v, (list, np.ndarray)):
        arr = np.asarray(v).ravel()
        return arr[0] if arr.size > 0 else default
    return v if v is not None else default


def _candidate_dirs(results_root: Path):
    """Yield directories to try loading as Orbax checkpoints (depth 1 and 2)."""
    for d1 in sorted(results_root.iterdir()):
        if not d1.is_dir():
            continue
        yield d1
        for d2 in sorted(d1.iterdir()):
            if d2.is_dir() and not d2.name.startswith("checkpoint_"):
                yield d2


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir",
                    help="Batch or results directory to scan.")
    ap.add_argument("--pattern", default="",
                    help="Only include runs whose env name contains this substring.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print warnings for unreadable checkpoints.")
    args = ap.parse_args()

    results_root = Path(args.dir).resolve()
    if not results_root.exists():
        sys.exit(f"ERROR: directory not found: {results_root}")

    rows = []
    for candidate in _candidate_dirs(results_root):
        result = _restore(candidate, args.verbose)
        if result is None:
            continue

        cfg = result.get("args", {})
        if not cfg or not cfg.get("env"):
            continue

        env = str(cfg.get("env", "?"))
        if args.pattern and args.pattern not in env:
            continue

        algo = str(cfg.get("algo", "ppo"))
        lr   = f"{float(_scalar(cfg.get('lr'), 0)):.2e}"
        hs   = int(_scalar(cfg.get("hidden_size"), 0))
        ml   = bool(_scalar(cfg.get("memoryless"), False))

        mean_ret, std_ret, n_ep = _eval_return(result)
        rt = _runtime_str(result)

        row = {
            "env": env, "algo": algo, "ml": ml, "hs": hs, "lr": lr,
            "mean_ret": mean_ret, "std_ret": std_ret, "n_ep": n_ep, "runtime": rt,
        }

        if algo == "ppo":
            row["dc"]  = bool(_scalar(cfg.get("double_critic"), False))
            row["ac"]  = bool(_scalar(cfg.get("action_concat"), False))
            row["ent"] = f"{float(_scalar(cfg.get('entropy_coeff'), 0)):.3f}"
            row["ts"]  = int(_scalar(cfg.get("total_steps"), 0))
            row["ns"]  = int(_scalar(cfg.get("n_seeds"), 1))
        else:  # dqn
            row["tr"]  = int(_scalar(cfg.get("trace_length"), 0))
            row["ne"]  = int(_scalar(cfg.get("num_envs"), 0))
            row["bbs"] = int(_scalar(cfg.get("buffer_batch_size"), 0))

        rows.append(row)

    if not rows:
        sys.exit(f"No results found under {results_root}.")

    print(f"\nFound {len(rows)} run(s) in {results_root}\n")

    rows.sort(key=lambda r: (r["env"], r["algo"], r["ml"], r["lr"], r["hs"]))

    env_w = max(len(r["env"]) for r in rows)

    # Unified header
    header = (
        f"{'ENV':<{env_w}}  {'ALGO':>5}  {'MEM':>4}  {'HS':>3}  {'LR':<8}  "
        f"{'MEAN_RET':>10}  {'STD':>8}  {'N_EP':>5}  {'TIME':>7}  EXTRAS"
    )
    sep = "─" * (len(header) + 20)
    print(header)
    print(sep)

    prev_env = None
    for r in rows:
        if prev_env and r["env"] != prev_env:
            print()
        prev_env = r["env"]

        mode_s  = ("DQN" if r["ml"] else "DRQN") if r["algo"] == "dqn" else ("PPO-ML" if r["ml"] else "PPO")
        mean_s  = f"{r['mean_ret']:>10.2f}" if not np.isnan(r["mean_ret"]) else f"{'no eps':>10}"
        std_s   = f"{r['std_ret']:>8.2f}"   if not np.isnan(r["std_ret"])  else f"{'':>8}"

        if r["algo"] == "ppo":
            ts_s = f"{r['ts'] / 1e6:.0f}M" if r["ts"] > 0 else "?"
            dc_s = "Y" if r["dc"] else "N"
            ac_s = "Y" if r["ac"] else "N"
            extras = f"ent={r['ent']} dc={dc_s} ac={ac_s} steps={ts_s} ns={r['ns']}"
        else:
            extras = f"tr={r['tr']} ne={r['ne']} bbs={r['bbs']}"

        print(
            f"{r['env']:<{env_w}}  {mode_s:>5}  {str(r['ml'])[0]:>4}  {r['hs']:>3}  "
            f"{r['lr']:<8}  {mean_s}  {std_s}  {r['n_ep']:>5}  {r['runtime']:>7}  {extras}"
        )

    print(sep)
    print(f"\nTotal runs: {len(rows)}")
    n_ppo = sum(1 for r in rows if r["algo"] == "ppo")
    n_dqn = sum(1 for r in rows if r["algo"] != "ppo")
    print(f"  PPO: {n_ppo}  DQN/DRQN: {n_dqn}")
    print("MEM=memoryless flag  HS=hidden_size  MEAN_RET: greedy policy. 'no eps' = all timed out.")


if __name__ == "__main__":
    main()
