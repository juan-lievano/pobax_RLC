#!/usr/bin/env python3
"""Print a sorted summary table of DQN/DRQN experiment results.

Scans results/ for directories whose name contains the given pattern, loads
each Orbax checkpoint, and prints one row per completed run.

Usage (from repo root):
    python slurm/dqn_smoke_tests/summarize_results.py
    python slurm/dqn_smoke_tests/summarize_results.py --results_dir /path/to/results
    python slurm/dqn_smoke_tests/summarize_results.py --pattern _drqn   # old smoke-test runs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint


# ── helpers ──────────────────────────────────────────────────────────────────

def _restore(path: Path) -> dict | None:
    try:
        return orbax.checkpoint.PyTreeCheckpointer().restore(path)
    except Exception as exc:
        print(f"  [WARN] could not load {path}: {exc}", file=sys.stderr)
        return None


def _eval_return(result: dict) -> tuple[float, float, int]:
    """(mean, std, n_completed) from the saved final_eval block."""
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


def _scalar(v):
    """Unwrap a list/array to a plain scalar (sweep hparams are stored as lists)."""
    if isinstance(v, (list, np.ndarray)):
        v = np.asarray(v).ravel()[0]
    return v


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results",
                    help="Root results directory (default: results/)")
    ap.add_argument("--pattern", default="dqn_",
                    help="Substring to filter study-name directories (default: 'dqn_')")
    args = ap.parse_args()

    results_root = Path(args.results_dir).resolve()
    if not results_root.exists():
        sys.exit(f"ERROR: results dir not found: {results_root}")

    run_dirs = []
    for study_dir in sorted(results_root.iterdir()):
        if not study_dir.is_dir() or args.pattern not in study_dir.name:
            continue
        for run_dir in sorted(study_dir.iterdir()):
            if run_dir.is_dir():
                run_dirs.append((study_dir.name, run_dir))

    if not run_dirs:
        sys.exit(f"No matching run directories found under {results_root} "
                 f"with pattern '{args.pattern}'.")

    print(f"\nFound {len(run_dirs)} run(s) matching '{args.pattern}' in {results_root}\n")

    rows = []
    for _study_name, run_dir in run_dirs:
        result = _restore(run_dir)
        if result is None:
            continue

        cfg = result.get("args", {})
        env        = cfg.get("env", "?")
        lr         = f"{float(_scalar(cfg.get('lr', 0))):.2e}"
        tr         = int(_scalar(cfg.get("trace_length", 0)))
        hs         = int(_scalar(cfg.get("hidden_size", 0)))
        num_envs   = int(_scalar(cfg.get("num_envs", 32)))
        bbs        = int(_scalar(cfg.get("buffer_batch_size", 128)))
        memoryless = bool(_scalar(cfg.get("memoryless", False)))

        mean_ret, std_ret, n_ep = _eval_return(result)
        rt = _runtime_str(result)

        rows.append({
            "env":      env,
            "mem":      memoryless,
            "lr":       lr,
            "tr":       tr,
            "hs":       hs,
            "ne":       num_envs,
            "bbs":      bbs,
            "mean_ret": mean_ret,
            "std_ret":  std_ret,
            "n_ep":     n_ep,
            "runtime":  rt,
        })

    if not rows:
        print("No results could be loaded.")
        return

    # Sort: env → memoryless → lr → tr → hs → num_envs → buffer_batch_size
    rows.sort(key=lambda r: (r["env"], r["mem"], r["lr"], r["tr"], r["hs"], r["ne"], r["bbs"]))

    # ── print table ──────────────────────────────────────────────────────────
    env_w = max(len(r["env"]) for r in rows)
    header = (
        f"{'ENV':<{env_w}}  {'MODE':>4}  {'LR':<8}  {'TR':>3}  {'HS':>3}  "
        f"{'NE':>2}  {'BBS':>3}  "
        f"{'MEAN_RET':>10}  {'STD':>8}  {'N_EP':>5}  {'TIME':>7}"
    )
    sep = "─" * len(header)
    print(header)
    print(sep)

    prev_env = None
    for r in rows:
        if prev_env and r["env"] != prev_env:
            print()
        prev_env = r["env"]

        mode_s = "DQN" if r["mem"] else "DRQN"
        mean_s = f"{r['mean_ret']:>10.2f}" if not np.isnan(r["mean_ret"]) else f"{'no eps':>10}"
        std_s  = f"{r['std_ret']:>8.2f}"  if not np.isnan(r["std_ret"])  else f"{'':>8}"

        print(
            f"{r['env']:<{env_w}}  {mode_s:>4}  {r['lr']:<8}  "
            f"{r['tr']:>3}  {r['hs']:>3}  "
            f"{r['ne']:>2}  {r['bbs']:>3}  "
            f"{mean_s}  {std_s}  {r['n_ep']:>5}  {r['runtime']:>7}"
        )

    print(sep)
    print(f"\nTotal runs: {len(rows)}")
    print("MODE=DQN/DRQN  TR=trace_length  HS=hidden_size  NE=num_envs  BBS=buffer_batch_size")
    print("N_EP=completed eval episodes  MEAN_RET: greedy policy. 'no eps' = all timed out.")


if __name__ == "__main__":
    main()
