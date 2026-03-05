"""
export_results.py  –  Aggregate all probe metrics and episodic rewards into
a single gzip-compressed JSON file suitable for local analysis.

Traverses a probe_results directory (e.g. probe_results/cluster_run_1/) and
for each run collects:

  1. Run config  (from run_config.json)
  2. Probe metrics per seed  (from checkpoint_*/h*_s*_metrics.json)
     - All scalar metrics for all probe types (tv, mean_kl_bits, etc.)
     - belief_sanity block where present (Marquee / goal-tracking envs)
     - mean_pred_belief vectors  [only with --include_beliefs]
  3. Per-episode returns per seed  (from checkpoint_*/h*_trajectories.npz)
     - ep_returns = (rewards * masks).sum(axis=-1)  → one float per episode

Output schema (JSON):
  {
    "source_dir": str,
    "created": str,           # ISO timestamp
    "n_runs": int,
    "runs": [
      {
        "run_id": str,        # directory name
        "config": { env_name, hidden_size, memoryless, ... },
        "checkpoints": {
          "<ckpt_idx>": {
            "seeds": {
              "<seed_idx>": {
                "metrics": {
                  "mlp_from_rnn_hidden":    { tv, mean_kl_bits, ... },
                  "linear_from_rnn_hidden": { ... },
                  "analytic_mean_constant_kl": { ... }
                },
                "belief_sanity": { ... },   # omitted if not present
                "mean_ep_return": float,    # mean over episodes; omitted if no NPZ
                "std_ep_return": float,     # std over episodes; omitted if no NPZ
                "mean_pred_belief": { ... } # omitted unless --include_beliefs
              }
            }
          }
        }
      }
    ]
  }

Estimated output sizes (cluster_run_1, 160 runs):
  without --include_beliefs:  ~80 MB raw → ~15–20 MB gzipped
  with    --include_beliefs:  ~700 MB raw → ~150 MB gzipped (CompassWorld adds ~600 MB)

Pandas loading recipe:
  import gzip, json, pandas as pd

  with gzip.open("all_results.json.gz") as f:
      data = json.load(f)

  metric_rows, reward_rows = [], []
  for run in data["runs"]:
      cfg = run["config"]
      for ckpt_str, ckpt in run["checkpoints"].items():
          for seed_str, sd in ckpt["seeds"].items():
              base = {**cfg, "run_id": run["run_id"],
                      "ckpt": int(ckpt_str), "seed": int(seed_str)}
              for probe, mdict in sd.get("metrics", {}).items():
                  metric_rows.append({**base, "probe": probe, **mdict})
              mean_ret = sd.get("mean_ep_return")
              if mean_ret is not None:
                  reward_rows.append({**base,
                      "mean_ep_return": mean_ret,
                      "std_ep_return":  sd.get("std_ep_return", 0.0)})

  metrics_df = pd.DataFrame(metric_rows)
  rewards_df = pd.DataFrame(reward_rows)

Usage:
    # Dry run: estimate output size without writing anything
    python export_results.py \\
        --results_dir probe_results/cluster_run_1 \\
        --out         all_results.json.gz \\
        --dry_run

    # Full collection (run on the server where NPZs are present):
    python export_results.py \\
        --results_dir probe_results/cluster_run_1 \\
        --out         probe_results/cluster_run_1/all_results.json.gz

    # Include mean_pred_belief vectors (~600 MB extra for CompassWorld):
    python export_results.py \\
        --results_dir probe_results/cluster_run_1 \\
        --out         all_results_with_beliefs.json.gz \\
        --include_beliefs
"""

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_config(run_dir: Path) -> Optional[dict]:
    p = run_dir / "run_config.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _ckpt_dirs(run_dir: Path) -> List[Path]:
    return sorted(
        [d for d in run_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]),
    )


def _load_seed_metrics(path: Path, include_beliefs: bool) -> Optional[dict]:
    """Load one h*_s*_metrics.json; strip mean_pred_belief unless requested."""
    try:
        with open(path) as f:
            r = json.load(f)
    except Exception as e:
        print(f"    [warn] could not read {path}: {e}", file=sys.stderr)
        return None

    out: dict = {"metrics": r.get("metrics", {})}
    if "belief_sanity" in r:
        out["belief_sanity"] = r["belief_sanity"]
    if include_beliefs and "mean_pred_belief" in r:
        out["mean_pred_belief"] = r["mean_pred_belief"]
    return out


def _load_ep_returns(npz_path: Path) -> Optional[Dict[str, dict]]:
    """Compute per-episode return summary stats for each seed from an NPZ trajectory file.

    Rewards are shaped [n_seeds, n_traj, max_len]; masks mark valid timesteps.
    Returns {seed_idx_str: {"mean_ep_return": float, "std_ep_return": float}}
    or None if file is absent.
    """
    if not npz_path.exists():
        return None
    try:
        d = np.load(npz_path, allow_pickle=False)
        rewards = d["rewards"]                           # [n_seeds, n_traj, T]
        masks   = d["masks"].astype(np.float32)          # [n_seeds, n_traj, T]
        ep_ret  = (rewards * masks).sum(axis=-1)         # [n_seeds, n_traj]
        return {
            str(s): {
                "mean_ep_return": float(np.mean(ep_ret[s])),
                "std_ep_return": float(np.std(ep_ret[s])),
            }
            for s in range(ep_ret.shape[0])
        }
    except Exception as e:
        print(f"    [warn] could not load {npz_path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Size estimation (dry run)
# ---------------------------------------------------------------------------

def estimate_size(results_dir: Path, include_beliefs: bool) -> None:
    """Sample a few runs and checkpoint directories to estimate output size."""
    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "run_config.json").exists()
    ])
    n_total = len(run_dirs)
    sample_n = min(4, n_total)
    print(f"Dry run: sampling {sample_n}/{n_total} runs, 3 checkpoints each ...")

    # Discover full checkpoint count from first run
    n_ckpts_full = len(_ckpt_dirs(run_dirs[0])) if run_dirs else 1

    sample_runs = []
    for run_dir in run_dirs[:sample_n]:
        config = _load_config(run_dir)
        if config is None:
            continue
        ckpts: dict = {}
        for ckpt_dir in _ckpt_dirs(run_dir)[:3]:
            ckpt_idx = int(ckpt_dir.name.split("_")[1])
            seeds: dict = {}
            for mp in sorted(ckpt_dir.glob("h*_s*_metrics.json")):
                parts = mp.stem.split("_")
                try:
                    s_idx = str(int(parts[1][1:]))
                except (IndexError, ValueError):
                    continue
                rec = _load_seed_metrics(mp, include_beliefs)
                if rec:
                    seeds.setdefault(s_idx, {}).update(rec)
            ep = _load_ep_returns(ckpt_dir / "h0_trajectories.npz")
            if ep:
                for s, v in ep.items():
                    seeds.setdefault(s, {}).update(v)
            if seeds:
                ckpts[str(ckpt_idx)] = {"seeds": seeds}
        sample_runs.append({"run_id": run_dir.name, "config": config,
                             "checkpoints": ckpts})

    sample_bytes = len(json.dumps(sample_runs, separators=(",", ":")).encode())
    bytes_per_run = sample_bytes / max(sample_n, 1)
    # Scale from 3 sampled checkpoints to the full number
    bytes_per_run_full = bytes_per_run * (n_ckpts_full / 3)
    est_raw = bytes_per_run_full * n_total
    est_gz  = est_raw / 6   # gzip is roughly 6× on this kind of mixed data

    print(f"\nEstimated output size ({n_total} runs, {n_ckpts_full} ckpts each):")
    print(f"  uncompressed JSON:  ~{est_raw/1024**2:.0f} MB")
    print(f"  gzip compressed:    ~{est_gz/1024**2:.0f} MB  (6× assumed)")
    if not include_beliefs:
        print("\n  Note: mean_pred_belief arrays are excluded.")
        print("  Add --include_beliefs to include them (~600 MB extra for CompassWorld).")


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

def collect(results_dir: Path, include_beliefs: bool = False) -> dict:
    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "run_config.json").exists()
    ])
    print(f"Found {len(run_dirs)} run directories under {results_dir}")

    runs = []
    n_metric_records  = 0
    n_ep_return_seeds = 0
    n_missing_npz     = 0

    for i, run_dir in enumerate(run_dirs):
        config = _load_config(run_dir)
        if config is None:
            print(f"  [warn] no run_config.json in {run_dir.name} — skipping",
                  file=sys.stderr)
            continue

        ckpt_list  = _ckpt_dirs(run_dir)
        checkpoints: dict = {}

        for ckpt_dir in ckpt_list:
            ckpt_idx = int(ckpt_dir.name.split("_")[1])
            seeds: dict = {}

            # Probe metrics (one JSON per seed, named h{h}_s{s}_metrics.json)
            for metric_path in sorted(ckpt_dir.glob("h*_s*_metrics.json")):
                parts = metric_path.stem.split("_")   # ["h0", "s3", "metrics"]
                try:
                    s_idx = str(int(parts[1][1:]))    # "s3" → "3"
                except (IndexError, ValueError):
                    continue
                rec = _load_seed_metrics(metric_path, include_beliefs)
                if rec is not None:
                    seeds.setdefault(s_idx, {}).update(rec)
                    n_metric_records += 1

            # Per-episode return stats from NPZ trajectory file
            ep_returns = _load_ep_returns(ckpt_dir / "h0_trajectories.npz")
            if ep_returns is None:
                n_missing_npz += 1
            else:
                for s_idx, stats in ep_returns.items():
                    seeds.setdefault(s_idx, {}).update(stats)
                    n_ep_return_seeds += 1

            if seeds:
                checkpoints[str(ckpt_idx)] = {"seeds": seeds}

        runs.append({
            "run_id":      run_dir.name,
            "config":      config,
            "checkpoints": checkpoints,
        })
        print(f"  [{i+1:3d}/{len(run_dirs)}] {run_dir.name}  "
              f"({len(ckpt_list)} ckpts, "
              f"npz={'ok' if n_missing_npz == 0 else 'some missing'})")

    result = {
        "source_dir": str(results_dir.resolve()),
        "created":    datetime.now(timezone.utc).isoformat(),
        "n_runs":     len(runs),
        "runs":       runs,
    }

    print(f"\nCollection summary:")
    print(f"  runs collected:               {len(runs)}")
    print(f"  metric records (ckpt×seed):   {n_metric_records}")
    print(f"  seed-ckpts with ep_returns:   {n_ep_return_seeds}")
    print(f"  checkpoint dirs without NPZ:  {n_missing_npz}")

    return result


# ---------------------------------------------------------------------------
# Write gzip JSON
# ---------------------------------------------------------------------------

def write_gzip_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSerialising to JSON ...", flush=True)
    raw = json.dumps(data, separators=(",", ":")).encode()
    print(f"  uncompressed: {len(raw)/1024**2:.1f} MB")
    print(f"Compressing (level 6) and writing to {out_path} ...", flush=True)
    with gzip.open(out_path, "wb", compresslevel=6) as f:
        f.write(raw)
    gz_bytes = out_path.stat().st_size
    print(f"  compressed:   {gz_bytes/1024**2:.1f} MB  "
          f"({len(raw)/gz_bytes:.1f}× ratio)")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results_dir", required=True, type=Path,
                   help="Root probe results directory to traverse.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output path (should end in .json.gz).")
    p.add_argument("--dry_run", action="store_true",
                   help="Estimate output size without writing anything.")
    p.add_argument("--include_beliefs", action="store_true",
                   help="Also store mean_pred_belief vectors per seed/checkpoint/probe "
                        "(adds ~600 MB for CompassWorld environments).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run:
        estimate_size(args.results_dir, args.include_beliefs)
    else:
        data = collect(args.results_dir, include_beliefs=args.include_beliefs)
        write_gzip_json(data, args.out)
