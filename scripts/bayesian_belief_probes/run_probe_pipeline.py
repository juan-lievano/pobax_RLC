"""
Orchestrator for the Bayesian belief probe pipeline.

For each checkpoint in the run directory:
  1. Sample trajectories for all seeds at once (vmapped) → NPZ
  2. Train probes for each seed (sequential Python loop) → JSON per seed
  3. Visualize metric curves + env-specific grids

Usage:
    python run_probe_pipeline.py \\
        --run_dir     results/rocksample_11_11_h256_dc_ac_5M \\
        --n_timesteps 10000 \\
        [--n_traj     1000] \\
        [--out_dir    probe_results/my_exp] \\
        [--h_idx 0] \\
        [--force]
        [--no_viz]
        [--probing_config example.json]

--run_dir can point either directly to the timestamped run directory
(e.g. results/.../rocksample_11_11_seed(2026)_time(...)) or to its
parent directory.  If the parent contains exactly one subdirectory it
is resolved automatically, so you never have to copy the hash string.

Batch-dir mode: if --run_dir contains multiple non-checkpoint subdirs,
each is treated as a separate run and processed in sequence.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).parent))
from sample_trajectories import sample_and_save
from train_probes import train_and_save
from visualize import visualize_results

PROBING_CONFIGS_DIR = Path(__file__).parent / "probing_configs"


def _resolve_run_dir(path: Path) -> Path:
    """Accept either the timestamped run dir or its parent.

    If `path` directly contains checkpoint_* subdirs it is returned as-is.
    Otherwise, if it contains exactly one subdirectory, that child is
    returned (the common case where the parent is the experiment folder and
    the child is the single seed/time-stamped run inside it).
    """
    if any(d.is_dir() and d.name.startswith("checkpoint_") for d in path.iterdir()):
        return path
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        print(f"  auto-resolved run_dir → {subdirs[0]}")
        return subdirs[0]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No subdirectories found in {path}")
    names = "\n  ".join(d.name for d in sorted(subdirs))
    raise ValueError(
        f"{path} contains {len(subdirs)} subdirectories; pass the specific one:\n  {names}"
    )


def _is_batch_dir(path: Path) -> bool:
    """Return True if path contains multiple non-checkpoint subdirs (batch mode)."""
    subdirs = [d for d in path.iterdir()
               if d.is_dir() and not d.name.startswith("checkpoint_")]
    return len(subdirs) > 1


def _load_probing_config(config_name: str) -> dict:
    config_path = PROBING_CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Probing config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def _check_reward_threshold(run_dir: Path, env_name: str, threshold: float) -> tuple[bool, float]:
    """Load main checkpoint and check if mean return meets threshold.
    Returns (passes, mean_return).
    """
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    try:
        result = checkpointer.restore(run_dir)
    except Exception as e:
        print(f"  [WARN] could not load checkpoint for threshold check: {e}")
        return True, float("nan")

    fe = result.get("final_eval", result.get("final_eval_metric", {}))
    rets = np.asarray(fe.get("returned_episode_returns", []))
    mask = np.asarray(fe.get("returned_episode", []), dtype=bool)
    completed = rets[mask]
    if completed.size == 0:
        print(f"  [WARN] no completed episodes found for threshold check, skipping filter")
        return True, float("nan")
    mean_ret = float(completed.mean())
    return mean_ret >= threshold, mean_ret


def _discover_checkpoints(run_dir: Path):
    """Return checkpoint subdirs sorted numerically by index."""
    dirs = [
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_")
    ]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def _process_one_run(run_dir_raw: Path, out_dir: Path, args) -> None:
    """Process a single run: sample trajectories, train probes, optionally visualize."""
    run_dir = _resolve_run_dir(run_dir_raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load run metadata from the main Orbax checkpoint
    # -----------------------------------------------------------------------
    print(f"Loading run metadata from {run_dir} ...")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    main_results = checkpointer.restore(run_dir)
    run_args = main_results["args"]

    env_name      = str(run_args["env"])
    hidden_size   = int(run_args["hidden_size"])
    n_seeds       = int(run_args["n_seeds"])
    algo          = str(run_args.get("algo", "ppo"))
    double_critic = bool(run_args.get("double_critic", False))
    memoryless    = bool(run_args.get("memoryless", False))
    action_concat = bool(run_args.get("action_concat", False))

    def _scalar(v, t, default):
        """Coerce an orbax-restored value (may be list/array/scalar) to a Python scalar."""
        if isinstance(v, (list, tuple)):
            v = v[0] if v else default
        try:
            return t(v)
        except (TypeError, ValueError):
            return default

    entropy_coeff    = _scalar(run_args.get("entropy_coeff"),    float, 0.0)
    total_steps      = _scalar(run_args.get("total_steps"),      int,   0)
    # DQN-specific (safe defaults for PPO runs)
    lr               = _scalar(run_args.get("lr"),               float, 0.0)
    trace_length     = _scalar(run_args.get("trace_length"),     int,   0)
    buffer_batch_size = _scalar(run_args.get("buffer_batch_size"), int, 0)
    num_envs         = _scalar(run_args.get("num_envs"),         int,   0)

    print(
        f"  algo={algo}, env={env_name}, hidden_size={hidden_size}, "
        f"n_seeds={n_seeds}, double_critic={double_critic}, "
        f"memoryless={memoryless}, action_concat={action_concat}, "
        f"entropy_coeff={entropy_coeff}, total_steps={total_steps}"
    )

    # -----------------------------------------------------------------------
    # Optional reward threshold check (probing_config)
    # -----------------------------------------------------------------------
    if args.probing_config:
        probing_cfg = _load_probing_config(args.probing_config)
        threshold = probing_cfg.get(env_name)
        if threshold is not None:
            passes, mean_ret = _check_reward_threshold(run_dir, env_name, threshold)
            if not passes:
                print(f"  SKIP: mean_return={mean_ret:.3f} < threshold={threshold} for {env_name}")
                return
            print(f"  PASS: mean_return={mean_ret:.3f} >= threshold={threshold} for {env_name}")

    # Save a run_config.json so visualize.py can label figures when run standalone
    config = {
        "algo":             algo,
        "env_name":         env_name,
        "hidden_size":      hidden_size,
        "n_seeds":          n_seeds,
        "double_critic":    double_critic,
        "memoryless":       memoryless,
        "action_concat":    action_concat,
        "entropy_coeff":    entropy_coeff,
        "total_steps":      total_steps,
        "h_idx":            args.h_idx,
        # DQN-specific (zero for PPO runs)
        "lr":               lr,
        "trace_length":     trace_length,
        "buffer_batch_size": buffer_batch_size,
        "num_envs":         num_envs,
    }
    config_path = out_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved run_config.json → {config_path}")

    # -----------------------------------------------------------------------
    # Discover checkpoint directories
    # -----------------------------------------------------------------------
    ckpt_dirs = _discover_checkpoints(run_dir)
    if not ckpt_dirs:
        print(f"No checkpoint_* directories found in {run_dir}. "
              f"Re-run training with --save_checkpoints to enable probing. Exiting.")
        return
    print(f"Found {len(ckpt_dirs)} checkpoint(s).")

    # -----------------------------------------------------------------------
    # Process each checkpoint
    # -----------------------------------------------------------------------
    for ckpt_dir in ckpt_dirs:
        ckpt_idx = int(ckpt_dir.name.split("_")[1])
        ckpt_out = out_dir / f"checkpoint_{ckpt_idx}"
        ckpt_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[checkpoint {ckpt_idx}]")

        # --- Step 1: sample trajectories (all seeds at once) ---
        npz_path = ckpt_out / f"h{args.h_idx}_trajectories.npz"
        if npz_path.exists() and not args.force:
            print(f"  NPZ already exists, skipping sampling: {npz_path}")
        else:
            sample_and_save(
                ckpt_path=ckpt_dir,
                h_idx=args.h_idx,
                n_traj=args.n_traj,
                max_len=args.max_len,
                out=npz_path,
                seed=args.seed,
                hidden_size=hidden_size,
                env_name=env_name,
                double_critic=double_critic,
                memoryless=memoryless,
                action_concat=action_concat,
                algo=algo,
                env_seed=int(run_args["seed"]),
            )

        # --- Step 2: train probes for each seed sequentially ---
        # (Seeds are stored in the NPZ's leading dimension; we slice per seed here.)
        for s_idx in range(n_seeds):
            json_path = ckpt_out / f"h{args.h_idx}_s{s_idx}_metrics.json"
            if json_path.exists() and not args.force:
                print(f"  JSON already exists, skipping probe training: {json_path}")
                continue
            print(f"  training probes for seed {s_idx}/{n_seeds - 1}...")
            train_and_save(
                npz=npz_path,
                seed_idx=s_idx,
                checkpoint_idx=ckpt_idx,
                hparam_idx=args.h_idx,
                out=json_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                mlp_lr=args.mlp_lr,
                mlp_weight_decay=args.mlp_wd,
                linear_lr=args.linear_lr,
                linear_weight_decay=args.linear_wd,
                max_train_steps=args.n_timesteps,
            )

    # -----------------------------------------------------------------------
    # Step 3: visualize (skipped with --no_viz)
    # -----------------------------------------------------------------------
    if not args.no_viz:
        print("\nGenerating visualizations...")
        figures_dir = out_dir / "figures"
        visualize_results(out_dir, figures_dir, h_idx=args.h_idx, config=config)
    print(f"\nDone. Results in {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run_dir",  required=True, type=Path,
                   help="Top-level Orbax run directory (contains checkpoint_* subdirs), "
                        "its parent, or a batch directory with multiple run subdirs.")
    p.add_argument("--n_traj",      type=int, default=1000,
                   help="Trajectories to sample per checkpoint per seed (default 1000).")
    p.add_argument("--n_timesteps", type=int, default=10000,
                   help="Training budget: probe training is capped at this many timesteps "
                        "per seed per checkpoint, making all probes see the same amount of data "
                        "regardless of env or checkpoint (default 10000).")
    p.add_argument("--max_len",  type=int, default=1000,
                   help="Maximum steps per trajectory (default 1000).")
    p.add_argument("--out_dir",  type=Path, default=None,
                   help="Root directory for probe pipeline outputs. "
                        "Defaults to probe_results/<parent-dir-of-run_dir>.")
    p.add_argument("--h_idx",   type=int, default=0,
                   help="Hparam index to use (default 0).")
    p.add_argument("--seed",    type=int, default=0,
                   help="Base RNG seed for trajectory sampling.")
    p.add_argument("--force",   action="store_true",
                   help="Overwrite existing NPZ/JSON files.")
    p.add_argument("--no_viz",  action="store_true",
                   help="Skip visualization step (useful on server; visualize locally).")
    p.add_argument("--probing_config", type=str, default=None,
                   help="Filename (not full path) of a JSON in "
                        "scripts/bayesian_belief_probes/probing_configs/ specifying "
                        "minimum mean episodic return required to probe each env.")
    # Probe training hyper-params (forwarded to train_and_save)
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--mlp_lr",     type=float, default=1e-3)
    p.add_argument("--mlp_wd",     type=float, default=1e-4)
    p.add_argument("--linear_lr",  type=float, default=1e-2)
    p.add_argument("--linear_wd",  type=float, default=1.0)
    args = p.parse_args()

    raw_dir = args.run_dir.resolve()

    # -----------------------------------------------------------------------
    # Batch-dir mode: multiple run subdirs in the top-level dir
    # -----------------------------------------------------------------------
    if _is_batch_dir(raw_dir):
        subdirs = sorted(
            d for d in raw_dir.iterdir()
            if d.is_dir() and not d.name.startswith("checkpoint_")
        )
        print(f"Batch mode: found {len(subdirs)} run(s) in {raw_dir}")
        for run_subdir in subdirs:
            print(f"\n{'='*60}")
            print(f"Processing run: {run_subdir.name}")
            print(f"{'='*60}")
            run_out_dir = (args.out_dir or Path("probe_results") / raw_dir.name) / run_subdir.name
            try:
                _process_one_run(run_subdir, run_out_dir, args)
            except Exception as e:
                print(f"  ERROR processing {run_subdir.name}: {e}", file=sys.stderr)
        return

    # -----------------------------------------------------------------------
    # Single-run mode
    # -----------------------------------------------------------------------
    # out_dir is derived from the path the user typed (before auto-resolution),
    # so it stays clean even when we descend into a timestamped child.
    out_dir = (args.out_dir or Path("probe_results") / raw_dir.name).resolve()
    _process_one_run(raw_dir, out_dir, args)


if __name__ == "__main__":
    main()
