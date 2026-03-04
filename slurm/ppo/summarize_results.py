#!/usr/bin/env python3
"""Print a sorted summary table of PPO experiment results.

Scans results/ for directories whose name contains the given pattern, loads
each Orbax checkpoint, and prints one row per completed run.

Usage (from repo root):
    python slurm/ppo/summarize_results.py
    python slurm/ppo/summarize_results.py --results_dir /path/to/results
    python slurm/ppo/summarize_results.py --pattern compass_world_8
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
    ap.add_argument("--pattern", default="",
                    help="Substring to filter study-name directories "
                         "(default: '' = all non-DQN dirs)")
    args = ap.parse_args()

    results_root = Path(args.results_dir).resolve()
    if not results_root.exists():
        sys.exit(f"ERROR: results dir not found: {results_root}")

    run_dirs = []
    for study_dir in sorted(results_root.iterdir()):
        if not study_dir.is_dir():
            continue
        # Skip DQN results
        if study_dir.name.startswith("dqn_"):
            continue
        if args.pattern and args.pattern not in study_dir.name:
            continue
        # Each PPO study dir contains one timestamped subdir
        subdirs = [d for d in study_dir.iterdir() if d.is_dir()
                   and not d.name.startswith("checkpoint_")]
        for sub in subdirs:
            run_dirs.append((study_dir.name, sub))

    if not run_dirs:
        sys.exit(f"No matching PPO run directories found under {results_root}.")

    print(f"\nFound {len(run_dirs)} PPO run(s) in {results_root}\n")

    rows = []
    for _study_name, run_dir in run_dirs:
        result = _restore(run_dir)
        if result is None:
            continue

        cfg = result.get("args", {})
        env        = cfg.get("env", "?")
        lr         = f"{float(_scalar(cfg.get('lr', 0))):.2e}"
        hs         = int(_scalar(cfg.get("hidden_size", 0)))
        dc         = bool(_scalar(cfg.get("double_critic", False)))
        ml         = bool(_scalar(cfg.get("memoryless", False)))
        ac         = bool(_scalar(cfg.get("action_concat", False)))
        ent        = f"{float(_scalar(cfg.get('entropy_coeff', 0))):.3f}"
        ts         = int(_scalar(cfg.get("total_steps", 0)))
        n_seeds    = int(_scalar(cfg.get("n_seeds", 1)))

        mean_ret, std_ret, n_ep = _eval_return(result)
        rt = _runtime_str(result)

        rows.append({
            "env":      env,
            "lr":       lr,
            "hs":       hs,
            "dc":       dc,
            "ml":       ml,
            "ac":       ac,
            "ent":      ent,
            "ts":       ts,
            "ns":       n_seeds,
            "mean_ret": mean_ret,
            "std_ret":  std_ret,
            "n_ep":     n_ep,
            "runtime":  rt,
        })

    if not rows:
        print("No results could be loaded.")
        return

    # Sort: env → ml → dc → ac → ent → lr → hs
    rows.sort(key=lambda r: (r["env"], r["ml"], r["dc"], r["ac"], r["ent"], r["lr"], r["hs"]))

    # ── print table ──────────────────────────────────────────────────────────
    env_w = max(len(r["env"]) for r in rows)
    header = (
        f"{'ENV':<{env_w}}  {'MODE':>6}  {'LR':<8}  {'HS':>3}  "
        f"{'DC':>2}  {'AC':>2}  {'ENT':<6}  {'STEPS':>9}  {'NS':>2}  "
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

        mode_s = "PPO" if not r["ml"] else "PPO-ML"
        mean_s = f"{r['mean_ret']:>10.2f}" if not np.isnan(r["mean_ret"]) else f"{'no eps':>10}"
        std_s  = f"{r['std_ret']:>8.2f}"  if not np.isnan(r["std_ret"])  else f"{'':>8}"
        dc_s   = "Y" if r["dc"] else "N"
        ac_s   = "Y" if r["ac"] else "N"
        ts_s   = f"{r['ts'] / 1e6:.0f}M"

        print(
            f"{r['env']:<{env_w}}  {mode_s:>6}  {r['lr']:<8}  {r['hs']:>3}  "
            f"{dc_s:>2}  {ac_s:>2}  {r['ent']:<6}  {ts_s:>9}  {r['ns']:>2}  "
            f"{mean_s}  {std_s}  {r['n_ep']:>5}  {r['runtime']:>7}"
        )

    print(sep)
    print(f"\nTotal runs: {len(rows)}")
    print("MODE=PPO/PPO-ML(memoryless)  DC=double_critic  AC=action_concat  NS=n_seeds")
    print("N_EP=completed eval episodes  MEAN_RET: greedy policy. 'no eps' = all timed out.")


if __name__ == "__main__":
    main()
