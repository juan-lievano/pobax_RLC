#!/usr/bin/env python3
"""Pre-processing step before submitting the probe SLURM array.

Scans a batch directory for run dirs, optionally filters by reward threshold,
and writes a probe_manifest.txt listing the absolute paths to probe.

Usage:
    python slurm/train/make_probe_manifest.py results/batch_12345/
    python slurm/train/make_probe_manifest.py results/batch_12345/ --config example.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint

PROBING_CONFIGS_DIR = Path(__file__).parent.parent.parent / "scripts" / "bayesian_belief_probes" / "probing_configs"


def _resolve_run_dir(path: Path) -> Path | None:
    """Return the timestamped child dir (contains checkpoint_* subdirs), or None."""
    try:
        if any(d.is_dir() and d.name.startswith("checkpoint_") for d in path.iterdir()):
            return path
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
    except PermissionError:
        pass
    return None


def _load_run_info(run_dir: Path) -> tuple[str | None, float | None]:
    """Load checkpoint once, return (env_name, mean_return)."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    try:
        result = checkpointer.restore(run_dir)
    except Exception as e:
        print(f"  [WARN] could not load {run_dir}: {e}", file=sys.stderr)
        return None, None

    env = str(result.get("args", {}).get("env", "")) or None

    fe = result.get("final_eval", result.get("final_eval_metric", {}))
    rets = np.asarray(fe.get("returned_episode_returns", []))
    mask = np.asarray(fe.get("returned_episode", []), dtype=bool)
    completed = rets[mask]
    mean_ret = float(completed.mean()) if completed.size > 0 else None

    return env, mean_ret


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_dir", type=Path,
                    help="Batch results directory to scan (e.g. results/batch_12345/).")
    ap.add_argument("--config", default=None,
                    help="Filename (not full path) of a JSON in probing_configs/ "
                         "specifying minimum mean return per env.")
    args = ap.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not batch_dir.exists():
        sys.exit(f"ERROR: batch_dir not found: {batch_dir}")

    # Load optional reward thresholds
    thresholds: dict = {}
    if args.config:
        cfg_path = PROBING_CONFIGS_DIR / args.config
        if not cfg_path.exists():
            sys.exit(f"ERROR: probing config not found: {cfg_path}")
        with open(cfg_path) as f:
            thresholds = json.load(f)
        print(f"Using probing config: {cfg_path}")
        for env, thr in sorted(thresholds.items()):
            if not env.startswith("_"):
                print(f"  {env}: min_return >= {thr}")

    # Scan batch_dir for study-name subdirs (depth-1 subdirs that are not checkpoint_*)
    study_dirs = sorted(
        d for d in batch_dir.iterdir()
        if d.is_dir() and not d.name.startswith("checkpoint_")
        and d.name != "probe_manifest.txt"
    )

    if not study_dirs:
        sys.exit(f"No subdirectories found in {batch_dir}")

    included = []
    skipped = []

    for study_dir in study_dirs:
        run_dir = _resolve_run_dir(study_dir)
        if run_dir is None:
            print(f"  [SKIP] could not resolve run dir in {study_dir.name}", file=sys.stderr)
            skipped.append((study_dir.name, "no run dir"))
            continue

        # Check reward threshold if config provided (single restore)
        if thresholds:
            env, mean_ret = _load_run_info(run_dir)
            if env and env in thresholds:
                threshold = thresholds[env]
                if mean_ret is None:
                    print(f"  [WARN] no eval data for {study_dir.name}, including anyway")
                elif mean_ret < threshold:
                    print(f"  [SKIP] {study_dir.name}: mean_return={mean_ret:.3f} < {threshold}")
                    skipped.append((study_dir.name, f"mean_return={mean_ret:.3f} < {threshold}"))
                    continue
                else:
                    print(f"  [PASS] {study_dir.name}: mean_return={mean_ret:.3f} >= {threshold}")

        included.append(run_dir)

    manifest_path = batch_dir / "probe_manifest.txt"
    with open(manifest_path, "w") as f:
        for p in included:
            f.write(str(p) + "\n")

    n_included = len(included)
    n_skipped = len(skipped)
    print(f"\nManifest written to: {manifest_path}")
    print(f"  Included: {n_included}")
    print(f"  Skipped:  {n_skipped}")

    if n_included > 0:
        print(f"\nTo submit the probe array, run:")
        print(f"  MANIFEST={manifest_path}")
        print(f"  sbatch --array=0-{n_included - 1} --export=ALL,MANIFEST=$MANIFEST \\")
        print(f"    slurm/train/probe_array_launch.sh")
    else:
        print("\nNo runs to probe.")


if __name__ == "__main__":
    main()
