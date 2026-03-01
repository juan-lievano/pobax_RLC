"""
Aggregate probe pipeline results across all 160 experiments.

Scans a probe_results/ root directory, loads the final-checkpoint metrics and
reward from each experiment, and produces:

  1. summary.csv            — one row per experiment, all final-checkpoint metrics
  2. cross_exp_scatter.png  — reward vs belief metric for all experiments (per probe)
  3. <env>_marginal.png     — per-env marginal effect of each hyperparameter on metrics

Usage:
    python collect_results.py \\
        --probe_results_dir /path/to/probe_results \\
        [--out_dir /path/to/probe_results/summary] \\
        [--h_idx 0]
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Probe display config (mirrors visualize.py)
PROBE_ORDER = [
    "mlp_from_rnn_hidden",
    "linear_from_rnn_hidden",
    "analytic_mean_constant_kl",
]
PROBE_SHORT = {
    "mlp_from_rnn_hidden":       "mlp",
    "linear_from_rnn_hidden":    "linear",
    "analytic_mean_constant_kl": "const",
}
PROBE_LABELS = {
    "mlp_from_rnn_hidden":       "MLP",
    "linear_from_rnn_hidden":    "Linear",
    "analytic_mean_constant_kl": "Const (baseline)",
}
_OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
PROBE_COLORS = {k: _OKABE[i] for i, k in enumerate(PROBE_ORDER)}

# All metrics (env-specific subsets applied later)
METRIC_KEYS_ALL = [
    "tv", "mean_kl_bits", "impossible_mass_overall",
    "impossible_mass_should_know", "argmax_match_rate", "mean_prob_on_true_location",
]
METRIC_KEYS_ROCKSAMPLE = ["tv", "mean_kl_bits", "impossible_mass_overall"]

METRIC_TITLES = {
    "tv":                          "TV ↓",
    "mean_kl_bits":                "KL (bits) ↓",
    "argmax_match_rate":           "Argmax Match ↑",
    "mean_prob_on_true_location":  "Prob@True ↑",
    "impossible_mass_overall":     "Impossible Mass ↓",
    "impossible_mass_should_know": "Imp. Mass (SK) ↓",
}

ENV_COLORS = {
    "marquee_40_16":   "#E69F00",
    "compass_world_8": "#56B4E9",
    "compass_world_10":"#009E73",
    "rocksample_5_5":  "#D55E00",
    "rocksample_7_8":  "#CC79A7",
}

HPARAM_DISPLAY = {
    "hidden_size":   "Hidden size",
    "double_critic": "Double critic",
    "memoryless":    "Memoryless",
    "entropy_coeff": "Entropy coeff",
    "total_steps":   "Total steps",
}


# ---------------------------------------------------------------------------
# Study-name parser (fallback when run_config.json is absent)
# ---------------------------------------------------------------------------

def _parse_study_name(name: str) -> dict:
    """
    Parse hparams from the STUDY_NAME convention:
      {env}_h{hidden}[_dc][_ac][_ml]_ent{ent_tag}_s{n_seeds}_ts{total_steps}
    """
    config: dict = {"study_name": name}
    # total_steps
    m = re.search(r"_ts(\d+)", name)
    config["total_steps"] = int(m.group(1)) if m else 0
    # n_seeds
    m = re.search(r"_s(\d+)_ts", name)
    config["n_seeds"] = int(m.group(1)) if m else 0
    # hidden_size
    m = re.search(r"_h(\d+)", name)
    config["hidden_size"] = int(m.group(1)) if m else 0
    # entropy_coeff
    m = re.search(r"_ent([0-9p]+)", name)
    config["entropy_coeff"] = float(m.group(1).replace("p", ".")) if m else float("nan")
    # boolean flags
    config["double_critic"] = "_dc" in name
    config["memoryless"]    = "_ml" in name
    config["action_concat"] = "_ac" in name
    # env_name: everything before _h{hidden}
    m = re.match(r"^(.+?)_h\d+", name)
    config["env_name"] = m.group(1) if m else name
    return config


# ---------------------------------------------------------------------------
# Per-experiment data loading
# ---------------------------------------------------------------------------

def _load_experiment(exp_dir: Path, h_idx: int = 0) -> Optional[dict]:
    """
    Load config + final-checkpoint metrics + final-checkpoint reward for one
    experiment directory (probe_results/{study_name}/).

    Returns None if the directory lacks enough data.
    """
    study_name = exp_dir.name

    # --- Config ---
    config_path = exp_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["study_name"] = study_name
    else:
        config = _parse_study_name(study_name)

    # --- Find final checkpoint ---
    ckpt_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not ckpt_dirs:
        return None
    final_dir  = ckpt_dirs[-1]
    final_ckpt = int(final_dir.name.split("_")[1])

    env_name = config.get("env_name", "")

    # --- Final checkpoint metrics (aggregated over seeds) ---
    json_paths = sorted(final_dir.glob(f"h{h_idx}_s*_metrics.json"))
    if not json_paths:
        return None

    seed_metrics: Dict[str, Dict[str, list]] = {}  # probe → metric → [seed values]
    for jp in json_paths:
        with open(jp) as f:
            rec = json.load(f)
        for probe, mdict in rec.get("metrics", {}).items():
            if probe not in seed_metrics:
                seed_metrics[probe] = {}
            for metric, val in mdict.items():
                seed_metrics[probe].setdefault(metric, []).append(val)

    # --- Final checkpoint reward ---
    npz_path = final_dir / f"h{h_idx}_trajectories.npz"
    reward_mean = reward_std = float("nan")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        if "rewards" in data and "masks" in data:
            rewards  = data["rewards"]
            masks    = data["masks"]
            ep_ret   = (rewards * masks).sum(axis=-1).mean(axis=1)  # [n_seeds]
            reward_mean = float(ep_ret.mean())
            reward_std  = float(ep_ret.std()) if len(ep_ret) > 1 else 0.0

    # Flatten metrics into (mean, std) per (probe, metric)
    flat_metrics: Dict[str, float] = {}
    for probe, mdict in seed_metrics.items():
        short = PROBE_SHORT.get(probe, probe)
        for metric, vals in mdict.items():
            clean = [v for v in vals if v == v]  # drop NaN
            flat_metrics[f"{short}_{metric}_mean"] = float(np.mean(clean)) if clean else float("nan")
            flat_metrics[f"{short}_{metric}_std"]  = float(np.std(clean))  if len(clean) > 1 else 0.0

    return {
        "study_name":   study_name,
        "env_name":     env_name,
        "hidden_size":  config.get("hidden_size",   0),
        "double_critic": config.get("double_critic", False),
        "memoryless":   config.get("memoryless",    False),
        "action_concat": config.get("action_concat", True),
        "entropy_coeff": config.get("entropy_coeff", float("nan")),
        "total_steps":  config.get("total_steps",   0),
        "n_seeds":      config.get("n_seeds",       len(json_paths)),
        "final_ckpt":   final_ckpt,
        "reward_mean":  reward_mean,
        "reward_std":   reward_std,
        # raw seed metrics for plotting (probe → metric → [vals])
        "_seed_metrics": seed_metrics,
        **flat_metrics,
    }


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_summary_csv(rows: List[dict], out_path: Path):
    import csv

    # Build column order: metadata first, then all metric columns sorted
    meta_cols = [
        "study_name", "env_name", "hidden_size", "double_critic", "memoryless",
        "action_concat", "entropy_coeff", "total_steps", "n_seeds",
        "final_ckpt", "reward_mean", "reward_std",
    ]
    metric_cols = sorted({k for r in rows for k in r if k not in meta_cols and not k.startswith("_")})
    cols = meta_cols + metric_cols

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    print(f"  saved {out_path}  ({len(rows)} experiments)")


# ---------------------------------------------------------------------------
# Cross-experiment scatter: reward vs belief metric
# ---------------------------------------------------------------------------

def plot_cross_experiment_scatter(rows: List[dict], out_dir: Path):
    """
    One figure per probe (MLP and linear; skip constant baseline).
    Columns = metrics.  Each point = one experiment.
    Colour = env.  Filled circle = recurrent, open triangle = memoryless.
    Size = total_steps (small = 5M, large = 128M).
    """
    probes = ["mlp_from_rnn_hidden", "linear_from_rnn_hidden"]
    envs   = sorted({r["env_name"] for r in rows})

    # Determine which metrics exist across all rows
    all_metrics = set()
    for r in rows:
        for probe in probes:
            short = PROBE_SHORT[probe]
            for mk in METRIC_KEYS_ALL:
                col = f"{short}_{mk}_mean"
                if col in r and r[col] == r[col]:   # not NaN
                    all_metrics.add(mk)
    metric_keys = [m for m in METRIC_KEYS_ALL if m in all_metrics]
    if not metric_keys:
        print("  no metric data for cross-experiment scatter — skipping")
        return

    for probe in probes:
        short = PROBE_SHORT[probe]
        n_cols = len(metric_keys)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 4.0),
                                 constrained_layout=True)
        if n_cols == 1:
            axes = [axes]

        # Determine reward range for x-axis alignment
        all_rewards = [r["reward_mean"] for r in rows if r["reward_mean"] == r["reward_mean"]]
        r_min = min(all_rewards) if all_rewards else 0
        r_max = max(all_rewards) if all_rewards else 1

        for ax, metric in zip(axes, metric_keys):
            for env in envs:
                env_rows = [r for r in rows if r["env_name"] == env]
                color = ENV_COLORS.get(env, "gray")

                # Recurrent and memoryless plotted separately for different markers
                for ml, marker, label_suffix in [(False, "o", "recurrent"),
                                                  (True,  "^", "memoryless")]:
                    sub = [r for r in env_rows if bool(r["memoryless"]) == ml]
                    if not sub:
                        continue
                    xs = np.array([r["reward_mean"] for r in sub])
                    ys = np.array([r.get(f"{short}_{metric}_mean", float("nan")) for r in sub])
                    # Bubble size proportional to log(total_steps)
                    sizes = np.array([40 if r.get("total_steps", 0) < 10_000_000 else 120
                                      for r in sub])
                    mask = np.isfinite(xs) & np.isfinite(ys)
                    if not mask.any():
                        continue
                    ax.scatter(xs[mask], ys[mask], c=color, marker=marker,
                               s=sizes[mask], alpha=0.75, linewidths=0.5,
                               edgecolors="white" if ml else "none",
                               label=f"{env} ({label_suffix})")

            ax.set_xlabel("Final mean reward", fontsize=9)
            ax.set_ylabel(METRIC_TITLES.get(metric, metric), fontsize=9)
            ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=9)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xlim(r_min - 0.05 * (r_max - r_min), r_max + 0.05 * (r_max - r_min))

        # Shared legend: one entry per env + one for each marker shape
        handles = []
        for env in envs:
            handles.append(Patch(color=ENV_COLORS.get(env, "gray"), label=env))
        from matplotlib.lines import Line2D
        handles += [
            Line2D([0], [0], marker="o", color="gray", linestyle="none",
                   markersize=7, label="recurrent"),
            Line2D([0], [0], marker="^", color="gray", linestyle="none",
                   markersize=7, label="memoryless"),
            Line2D([0], [0], marker="o", color="gray", linestyle="none",
                   markersize=5, label="5M steps"),
            Line2D([0], [0], marker="o", color="gray", linestyle="none",
                   markersize=10, label="128M steps"),
        ]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.18), ncol=min(len(handles), 5), fontsize=7)

        fig.suptitle(
            f"Final Reward vs Belief Metric — {PROBE_LABELS[probe]}\n"
            f"All experiments  (▲ = memoryless, ● = recurrent;  small = 5M steps, large = 128M)",
            fontsize=9,
        )

        out_path = out_dir / f"cross_exp_scatter_{short}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Per-env marginal effect figures
# ---------------------------------------------------------------------------

def plot_marginal_effects(rows: List[dict], out_dir: Path):
    """
    For each environment, show the marginal effect of each hyperparameter on
    each belief metric (MLP probe only) at the final checkpoint.

    Layout: rows = hyperparameters, cols = metrics.
    Each cell: two bars (one per hparam value), height = mean, error = std across exps.
    """
    hparam_vals = {
        "hidden_size":   [64, 128],
        "double_critic": [False, True],
        "memoryless":    [False, True],
        "entropy_coeff": [0.05, 0.2],
        "total_steps":   [5_242_880, 127_959_040],
    }
    hparam_labels = {
        "hidden_size":   {64: "h=64",    128: "h=128"},
        "double_critic": {False: "single", True: "double"},
        "memoryless":    {False: "recurrent", True: "memoryless"},
        "entropy_coeff": {0.05: "ent=0.05", 0.2: "ent=0.2"},
        "total_steps":   {5_242_880: "5M steps", 127_959_040: "128M steps"},
    }
    probe = "mlp_from_rnn_hidden"
    short = PROBE_SHORT[probe]

    envs = sorted({r["env_name"] for r in rows})
    for env in envs:
        env_rows = [r for r in rows if r["env_name"] == env]
        if not env_rows:
            continue

        # Metrics available for this env (skip NaN-only columns)
        if env.startswith("rocksample_"):
            metric_keys = METRIC_KEYS_ROCKSAMPLE
        else:
            metric_keys = METRIC_KEYS_ALL

        # Filter to metrics that actually have data
        avail = [mk for mk in metric_keys
                 if any(r.get(f"{short}_{mk}_mean", float("nan")) == r.get(f"{short}_{mk}_mean", float("nan"))
                        for r in env_rows)]
        if not avail:
            print(f"  no MLP metric data for {env} — skipping marginal effects")
            continue

        n_rows_fig = len(hparam_vals)
        n_cols_fig = len(avail)
        fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                                 figsize=(3.0 * n_cols_fig, 2.6 * n_rows_fig),
                                 constrained_layout=True)
        if n_rows_fig == 1:
            axes = axes[np.newaxis, :]
        if n_cols_fig == 1:
            axes = axes[:, np.newaxis]

        bar_colors = ["#56B4E9", "#E69F00"]

        for r_idx, (hparam, vals) in enumerate(hparam_vals.items()):
            for c_idx, metric in enumerate(avail):
                ax = axes[r_idx, c_idx]
                col = f"{short}_{metric}_mean"
                bar_means, bar_errs, bar_labels = [], [], []

                for v in vals:
                    # Match rows with this hparam value (approximate for floats)
                    if isinstance(v, float):
                        sub = [r for r in env_rows
                               if abs(float(r.get(hparam, float("nan"))) - v) < 1e-6]
                    else:
                        sub = [r for r in env_rows if r.get(hparam) == v]
                    vals_here = [r[col] for r in sub if r.get(col, float("nan")) == r.get(col, float("nan"))]
                    bar_means.append(float(np.mean(vals_here)) if vals_here else float("nan"))
                    bar_errs.append(float(np.std(vals_here))   if len(vals_here) > 1 else 0.0)
                    bar_labels.append(hparam_labels[hparam][v])

                xs = np.arange(len(vals))
                for xi, (bm, be, bc) in enumerate(zip(bar_means, bar_errs, bar_colors)):
                    if bm == bm:   # not NaN
                        ax.bar(xi, bm, yerr=be, color=bc, capsize=3, width=0.6,
                               error_kw={"linewidth": 1})
                ax.set_xticks(xs)
                ax.set_xticklabels(bar_labels, fontsize=7)
                ax.tick_params(axis="y", labelsize=7)
                ax.spines[["right", "top"]].set_visible(False)

                if c_idx == 0:
                    ax.set_ylabel(HPARAM_DISPLAY[hparam], fontsize=8, labelpad=4)
                if r_idx == 0:
                    ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=8)

        fig.suptitle(f"{env}  —  Marginal effect of each hyperparameter\n"
                     f"(MLP probe, final checkpoint, mean ± std across experiments)",
                     fontsize=9)

        out_path = out_dir / f"{env}_marginal.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Also emit a reward summary figure (final reward per experiment, grouped by env)
# ---------------------------------------------------------------------------

def plot_reward_summary(rows: List[dict], out_dir: Path):
    """Bar chart of final mean reward for every experiment, grouped by env."""
    envs = sorted({r["env_name"] for r in rows})
    fig, axes = plt.subplots(1, len(envs), figsize=(3.5 * len(envs), 5),
                             constrained_layout=True, sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        env_rows = sorted(
            [r for r in rows if r["env_name"] == env],
            key=lambda r: (r["memoryless"], r["hidden_size"],
                           r["double_critic"], r["entropy_coeff"], r["total_steps"])
        )
        ys     = np.array([r["reward_mean"] for r in env_rows])
        errs   = np.array([r["reward_std"]  for r in env_rows])
        xs     = np.arange(len(env_rows))
        colors = [("#D55E00" if r["memoryless"] else "#0072B2") for r in env_rows]
        ax.bar(xs, ys, yerr=errs, color=colors, capsize=2, width=0.8,
               error_kw={"linewidth": 0.8})
        ax.set_title(env, fontsize=9)
        ax.set_xticks([])
        ax.set_xlabel("Experiments", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Final mean reward", fontsize=9)

    legend_handles = [
        Patch(color="#0072B2", label="recurrent"),
        Patch(color="#D55E00", label="memoryless"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9)
    fig.suptitle("Final Checkpoint Mean Reward — All Experiments", fontsize=10)

    out_path = out_dir / "reward_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--probe_results_dir", required=True, type=Path,
                   help="Root directory containing one subdirectory per experiment "
                        "(probe_results/).")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Where to write outputs.  Defaults to probe_results_dir/summary/.")
    p.add_argument("--h_idx", type=int, default=0,
                   help="Hparam index to load (default 0).")
    args = p.parse_args()

    probe_results_dir = args.probe_results_dir.resolve()
    out_dir = (args.out_dir or probe_results_dir / "summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiment directories
    exp_dirs = sorted([
        d for d in probe_results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name != "summary"
        and any(d.iterdir())  # non-empty
    ])
    print(f"Found {len(exp_dirs)} candidate experiment directories in {probe_results_dir}")

    # Load each experiment
    rows = []
    for exp_dir in exp_dirs:
        result = _load_experiment(exp_dir, h_idx=args.h_idx)
        if result is None:
            print(f"  [skip] {exp_dir.name}  (no checkpoint data)")
            continue
        rows.append(result)
        print(f"  [ok]   {exp_dir.name}  reward={result['reward_mean']:.3f}")

    if not rows:
        print("No experiments with data found. Exiting.")
        return

    print(f"\nLoaded {len(rows)} experiments.")

    # 1. Summary CSV
    save_summary_csv(rows, out_dir / "summary.csv")

    # 2. Reward summary bar chart
    plot_reward_summary(rows, out_dir)

    # 3. Cross-experiment scatter
    plot_cross_experiment_scatter(rows, out_dir)

    # 4. Per-env marginal effects
    plot_marginal_effects(rows, out_dir)

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
