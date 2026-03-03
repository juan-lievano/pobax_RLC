#!/usr/bin/env python3
"""Print a sorted summary table of DQN/DRQN smoke-test results.

Scans results/ for directories whose name contains '_drqn' (or '_dqn' if you
add memoryless runs later), loads each Orbax checkpoint, and prints one row
per completed run with the final greedy-eval return.

Usage (from repo root):
    python slurm/dqn_smoke_tests/summarize_results.py
    python slurm/dqn_smoke_tests/summarize_results.py --results_dir /path/to/results
    python slurm/dqn_smoke_tests/summarize_results.py --pattern compass_world
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint


# ── helpers ──────────────────────────────────────────────────────────────────

def _restore(path: Path) -> dict | None:
    """Restore an Orbax checkpoint; return None on failure."""
    try:
        return orbax.checkpoint.PyTreeCheckpointer().restore(path)
    except Exception as exc:
        print(f"  [WARN] could not load {path}: {exc}", file=sys.stderr)
        return None


def _eval_return(result: dict) -> tuple[float, float, int]:
    """Extract (mean, std, n_completed) from the final_eval block."""
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
    return f"{secs / 60:.1f} min"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results",
                    help="Root results directory (default: results/)")
    ap.add_argument("--pattern", default="_drqn",
                    help="Substring to filter study-name directories (default: '_drqn')")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        sys.exit(f"ERROR: results dir not found: {results_root}")

    # Collect all run dirs whose study name matches the pattern
    run_dirs = []
    for study_dir in sorted(results_root.iterdir()):
        if not study_dir.is_dir():
            continue
        if args.pattern not in study_dir.name:
            continue
        for run_dir in sorted(study_dir.iterdir()):
            if run_dir.is_dir():
                run_dirs.append((study_dir.name, run_dir))

    if not run_dirs:
        sys.exit(f"No matching run directories found under {results_root} "
                 f"with pattern '{args.pattern}'.")

    print(f"\nFound {len(run_dirs)} run(s) matching '{args.pattern}' in {results_root}\n")

    rows = []
    for study_name, run_dir in run_dirs:
        result = _restore(run_dir)
        if result is None:
            continue

        cfg = result.get("args", {})
        env       = cfg.get("env", "?")
        lr        = cfg.get("lr", ["?"])
        eps_fin   = cfg.get("epsilon_finish", ["?"])
        memless   = cfg.get("memoryless", False)
        n_seeds   = cfg.get("n_seeds", "?")
        tot_steps = cfg.get("total_steps", "?")

        # lr / epsilon_finish are stored as lists (sweep arrays); grab first elem
        lr_val  = lr[0]  if isinstance(lr,  (list, np.ndarray)) else lr
        eps_val = eps_fin[0] if isinstance(eps_fin, (list, np.ndarray)) else eps_fin

        mean_ret, std_ret, n_ep = _eval_return(result)
        rt = _runtime_str(result)

        rows.append({
            "env":       env,
            "algo":      "DQN" if memless else "DRQN",
            "lr":        f"{float(lr_val):.2e}",
            "eps_fin":   f"{float(eps_val):.2f}",
            "mean_ret":  mean_ret,
            "std_ret":   std_ret,
            "n_ep":      n_ep,
            "seeds":     n_seeds,
            "steps":     tot_steps,
            "runtime":   rt,
            "study":     study_name,
        })

    if not rows:
        print("No results could be loaded.")
        return

    # Sort: env, algo, lr, eps
    rows.sort(key=lambda r: (r["env"], r["algo"], r["lr"], r["eps_fin"]))

    # ── print table ──────────────────────────────────────────────────────────
    col_w = {
        "env": max(len(r["env"]) for r in rows),
        "algo": 4,
        "lr": 8,
        "eps": 6,
        "mean": 10,
        "std": 8,
        "n_ep": 6,
        "rt": 9,
    }

    header = (
        f"{'ENV':<{col_w['env']}}  "
        f"{'ALGO':<{col_w['algo']}}  "
        f"{'LR':<{col_w['lr']}}  "
        f"{'EPS_FIN':<{col_w['eps']}}  "
        f"{'MEAN_RET':>{col_w['mean']}}  "
        f"{'STD_RET':>{col_w['std']}}  "
        f"{'N_EP':>{col_w['n_ep']}}  "
        f"{'RUNTIME':>{col_w['rt']}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    prev_env = None
    for r in rows:
        if prev_env and r["env"] != prev_env:
            print()  # blank line between env groups
        prev_env = r["env"]

        mean_s = f"{r['mean_ret']:>10.2f}" if not np.isnan(r["mean_ret"]) else f"{'no eps':>10}"
        std_s  = f"{r['std_ret']:>8.2f}"  if not np.isnan(r["std_ret"])  else f"{'':>8}"

        print(
            f"{r['env']:<{col_w['env']}}  "
            f"{r['algo']:<{col_w['algo']}}  "
            f"{r['lr']:<{col_w['lr']}}  "
            f"{r['eps_fin']:<{col_w['eps']}}  "
            f"{mean_s}  "
            f"{std_s}  "
            f"{r['n_ep']:>{col_w['n_ep']}}  "
            f"{r['runtime']:>{col_w['rt']}}"
        )

    print(sep)
    print(f"\nTotal runs: {len(rows)}")
    print("MEAN_RET = mean episodic return across all seeds and eval envs "
          "(greedy policy, completed episodes only).")
    print("'no eps' = no episodes completed (greedy policy timed out on all envs).")


if __name__ == "__main__":
    main()
