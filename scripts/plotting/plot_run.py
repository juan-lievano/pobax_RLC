"""
Quick plot of a single raw training run.

Usage:
    python scripts/plotting/plot_run.py <path_to_run_dir>

Where <path_to_run_dir> is the top-level Orbax directory for a run, e.g.:
    results/compass_world_10_checkpoints/compass_world_8_seed(2020)_time(...)_hash
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint

from pobax.definitions import PROJECT_ROOT_DIR


def load_run(run_path: Path) -> dict:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return checkpointer.restore(run_path)


def plot_run(run_path: Path, discounted: bool = False, smooth_window: int = 1):
    res = load_run(run_path)

    args = res['args']
    metric = res['out']['metric']

    # Shape: [n_hparams, n_seeds, n_updates, n_steps, n_envs]
    returns_key = 'returned_discounted_episode_returns' if discounted else 'returned_episode_returns'
    returns = metric[returns_key]          # [n_hparams, n_seeds, n_updates, n_steps, n_envs]
    ep_done  = metric['returned_episode']  # same shape, bool mask

    n_hparams, n_seeds, n_updates, n_steps, n_envs = returns.shape

    total_steps = args['total_steps']
    xs = np.linspace(0, total_steps, n_updates)

    fig, axes = plt.subplots(1, n_hparams, figsize=(6 * n_hparams, 4), squeeze=False)

    for h in range(n_hparams):
        ax = axes[0, h]

        # For each seed: mean return over update steps, averaging only over
        # completed episodes (where returned_episode is True).
        seed_curves = []
        for s in range(n_seeds):
            r = returns[h, s]   # [n_updates, n_steps, n_envs]
            d = ep_done[h, s]   # [n_updates, n_steps, n_envs]
            # Per update: average return over all (step, env) pairs where an
            # episode ended. Fall back to nan if no episode ended that update.
            with np.errstate(invalid='ignore'):
                per_update = np.where(d, r, np.nan)
                curve = np.nanmean(per_update.reshape(n_updates, -1), axis=-1)
            seed_curves.append(curve)

        seed_curves = np.stack(seed_curves)  # [n_seeds, n_updates]

        if smooth_window > 1:
            from scipy.signal import savgol_filter
            seed_curves = savgol_filter(seed_curves, smooth_window, 3, axis=-1)

        mean = seed_curves.mean(axis=0)
        std  = seed_curves.std(axis=0)

        ax.plot(xs, mean, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.3)
        ax.set_xlabel('Environment steps')
        ax.set_ylabel('Discounted return' if discounted else 'Episode return')
        ax.set_title(f"{args['env']}  |  hparam set {h}")
        ax.spines[['right', 'top']].set_visible(False)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_path = Path(sys.argv[1])
    if not run_path.is_absolute():
        run_path = Path(PROJECT_ROOT_DIR) / run_path

    discounted   = '--discounted' in sys.argv
    smooth       = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--smooth=')), 1)

    fig = plot_run(run_path, discounted=discounted, smooth_window=smooth)

    out_path = run_path / 'training_curve.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.show()
