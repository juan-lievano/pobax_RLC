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

--run_dir can point either directly to the timestamped run directory
(e.g. results/.../rocksample_11_11_seed(2026)_time(...)) or to its
parent directory.  If the parent contains exactly one subdirectory it
is resolved automatically, so you never have to copy the hash string.
"""
import argparse
import json
import sys
from pathlib import Path

import orbax.checkpoint

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).parent))
from sample_trajectories import sample_and_save
from train_probes import train_and_save
from visualize import visualize_results


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


def _discover_checkpoints(run_dir: Path):
    """Return checkpoint subdirs sorted numerically by index."""
    dirs = [
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_")
    ]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run_dir",  required=True, type=Path,
                   help="Top-level Orbax run directory (contains checkpoint_* subdirs).")
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
    # Probe training hyper-params (forwarded to train_and_save)
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--mlp_lr",     type=float, default=1e-3)
    p.add_argument("--mlp_wd",     type=float, default=1e-4)
    p.add_argument("--linear_lr",  type=float, default=1e-2)
    p.add_argument("--linear_wd",  type=float, default=1.0)
    args = p.parse_args()

    raw_dir = args.run_dir.resolve()
    # out_dir is derived from the path the user typed (before auto-resolution),
    # so it stays clean even when we descend into a timestamped child.
    out_dir = (args.out_dir or Path("probe_results") / raw_dir.name).resolve()
    run_dir = _resolve_run_dir(raw_dir)
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

    entropy_coeff = _scalar(run_args.get("entropy_coeff"), float, 0.0)
    total_steps   = _scalar(run_args.get("total_steps"),   int,   0)

    print(
        f"  env={env_name}, hidden_size={hidden_size}, "
        f"n_seeds={n_seeds}, double_critic={double_critic}, "
        f"memoryless={memoryless}, action_concat={action_concat}, "
        f"entropy_coeff={entropy_coeff}, total_steps={total_steps}"
    )

    # Save a run_config.json so visualize.py can label figures when run standalone
    config = {
        "env_name":      env_name,
        "hidden_size":   hidden_size,
        "n_seeds":       n_seeds,
        "double_critic": double_critic,
        "memoryless":    memoryless,
        "action_concat": action_concat,
        "entropy_coeff": entropy_coeff,
        "total_steps":   total_steps,
        "h_idx":         args.h_idx,
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
        print(f"No checkpoint_* directories found in {run_dir}. Exiting.")
        return
    print(f"Found {len(ckpt_dirs)} checkpoints.")

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
    # Step 3: visualize
    # -----------------------------------------------------------------------
    print("\nGenerating visualizations...")
    figures_dir = out_dir / "figures"
    visualize_results(out_dir, figures_dir, h_idx=args.h_idx, config=config)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
