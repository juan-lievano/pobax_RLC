"""
statistics_on_exported_json.py  –  Figures and statistics from all_results.json.gz

Produces outputs for two goals:

  Goal 1  –  Do different algorithms learn belief representations at different
             rates?  Uses matched pairs (same env × hidden_size × entropy_coeff
             × total_steps × memoryless, different algo).

  Goal 2  –  Does belief quality correlate with reward?  Three analyses:
               a) raw scatter  (all ckpts, naive correlation)
               b) partial scatter  (checkpoint residualised — removes the
                  "both improve over time" confound)
               c) final-checkpoint cross-run scatter  (one point per run)

Output directory structure:
    out_dir/
        goal1_belief_learning_rates/
            {env}_learning_curves_{probe}.png
            {env}_final_ckpt_distribution_{probe}.png
            summary_stats.csv            ← pairwise Wilcoxon, all (env, probe, metric)
        goal2_belief_reward_correlation/
            raw_scatter_{probe}.png
            partial_corr_scatter_{probe}.png
            final_ckpt_scatter_{probe}.png
            correlation_heatmap.png      ← partial Spearman ρ, env × (probe × metric)
            summary_stats.csv            ← raw + partial Pearson/Spearman for all combos

Adding DQN later:
    Only one function needs to change: derive_algo() below.
    Add the check `if cfg.get("algo") == "dqn": return "DQN"` (or "DRQN").
    All figures and stats automatically include the new algorithm.

Usage:
    python statistics_on_exported_json.py \\
        --json_path /path/to/all_results.json.gz \\
        --out_dir   /path/to/output_directory
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

# Okabe-Ito palette (colourblind-safe).  Extend this dict when adding algos.
ALGO_COLORS = {
    "PPO":    "#0072B2",   # blue
    "DC-PPO": "#E69F00",   # orange  (lambda discrepancy / double critic)
    "DQN":    "#009E73",   # green   (future)
    "DRQN":   "#CC79A7",   # pink    (future)
}

ENV_ORDER = [
    "compass_world_8",
    "compass_world_10",
    "marquee_40_16",
    "rocksample_5_5",
    "rocksample_7_8",
]
ENV_LABELS = {
    "compass_world_8":  "CompassWorld 8",
    "compass_world_10": "CompassWorld 10",
    "marquee_40_16":    "Marquee 40×16",
    "rocksample_5_5":   "RockSample 5×5",
    "rocksample_7_8":   "RockSample 7×8",
}
ENV_COLORS = {
    "compass_world_8":  "#0072B2",
    "compass_world_10": "#E69F00",
    "marquee_40_16":    "#009E73",
    "rocksample_5_5":   "#D55E00",
    "rocksample_7_8":   "#CC79A7",
}

# Probes shown in figures (analytic baseline omitted from main plots)
PROBE_KEYS = ["mlp_from_rnn_hidden", "linear_from_rnn_hidden"]
PROBE_LABELS = {
    "mlp_from_rnn_hidden":    "MLP probe",
    "linear_from_rnn_hidden": "Linear probe",
}

PRIMARY_METRICS = ["tv", "mean_kl_bits"]   # shown in figures
ALL_METRICS = [                             # all go into CSVs
    "tv", "mean_kl_bits",
    "impossible_mass_overall", "impossible_mass_should_know",
    "argmax_match_rate", "mean_prob_on_true_location",
]
METRIC_LABELS = {
    "tv":                          "Total Variation ↓",
    "mean_kl_bits":                "Mean KL (bits) ↓",
    "impossible_mass_overall":     "Impossible Mass ↓",
    "impossible_mass_should_know": "Impossible Mass (SK) ↓",
    "argmax_match_rate":           "Should-Know Accuracy ↑",
    "mean_prob_on_true_location":  "Should-Know Mass ↑",
}


# ---------------------------------------------------------------------------
# Algorithm label  ←  ONLY function to modify when adding DQN
# ---------------------------------------------------------------------------

def derive_algo(cfg: dict) -> str:
    """Map a run config dict to a human-readable algorithm label.

    To add DQN support, insert before the PPO checks:

        algo_field = cfg.get("algo", "").lower()
        if algo_field == "dqn":   return "DQN"
        if algo_field == "drqn":  return "DRQN"
    """
    if cfg.get("double_critic", False):
        return "DC-PPO"
    return "PPO"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_build_dataframes(
    json_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all_results.json.gz and return (metrics_df, rewards_df).

    metrics_df columns:
        run_id, env_name, algo, hidden_size, double_critic, memoryless,
        action_concat, entropy_coeff, total_steps, hparam_id,
        ckpt, seed, probe, tv, mean_kl_bits, ...

    rewards_df columns:
        run_id, env_name, algo, hidden_size, double_critic, memoryless,
        action_concat, entropy_coeff, total_steps, hparam_id,
        ckpt, seed, mean_ep_return, std_ep_return, n_episodes

    hparam_id is a compact string of the non-algo hyperparameters, used for
    matched-pair grouping in Goal 1.
    """
    print(f"Loading {json_path} ...")
    opener = gzip.open if json_path.suffix == ".gz" else open
    with opener(json_path, "rb") as f:
        data = json.load(f)
    print(f"  {data['n_runs']} runs in file")

    metric_rows: List[dict] = []
    reward_rows: List[dict] = []

    for run in data["runs"]:
        cfg    = run["config"]
        run_id = run["run_id"]
        base = {
            "run_id":        run_id,
            "env_name":      cfg.get("env_name", "unknown"),
            "algo":          derive_algo(cfg),
            "hidden_size":   cfg.get("hidden_size"),
            "double_critic": cfg.get("double_critic", False),
            "memoryless":    cfg.get("memoryless", False),
            "action_concat": cfg.get("action_concat", True),
            "entropy_coeff": cfg.get("entropy_coeff"),
            "total_steps":   cfg.get("total_steps"),
            # Identifies the "matched" condition (everything except algo).
            # Two runs with the same hparam_id differ only in algorithm.
            "hparam_id": (
                f"h{cfg.get('hidden_size')}"
                f"_ent{str(cfg.get('entropy_coeff')).replace('.', 'p')}"
                f"_ts{cfg.get('total_steps')}"
                f"_ml{int(cfg.get('memoryless', False))}"
            ),
        }

        for ckpt_str, ckpt in run["checkpoints"].items():
            for seed_str, sd in ckpt["seeds"].items():
                row_base = {**base, "ckpt": int(ckpt_str), "seed": int(seed_str)}

                for probe, mdict in sd.get("metrics", {}).items():
                    metric_rows.append({**row_base, "probe": probe, **mdict})

                ep = sd.get("ep_returns")
                if ep:
                    reward_rows.append({**row_base,
                        "mean_ep_return": float(np.mean(ep)),
                        "std_ep_return":  float(np.std(ep)),
                        "n_episodes":     len(ep)})

    metrics_df = pd.DataFrame(metric_rows)
    rewards_df = pd.DataFrame(reward_rows)
    print(f"  metrics_df: {len(metrics_df):,} rows")
    print(f"  rewards_df: {len(rewards_df):,} rows")
    return metrics_df, rewards_df


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _sorted_algos(df: pd.DataFrame) -> List[str]:
    """Return algorithm labels present in df, in ALGO_COLORS order."""
    present = set(df["algo"].unique())
    ordered = [a for a in ALGO_COLORS if a in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _partial_residualize(
    x: np.ndarray, y: np.ndarray, covariate: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the linear effect of covariate from both x and y."""
    def _resid(arr: np.ndarray, cov: np.ndarray) -> np.ndarray:
        cov_c = cov - cov.mean()
        denom = np.dot(cov_c, cov_c)
        beta  = np.dot(cov_c, arr) / (denom + 1e-12) if denom > 0 else 0.0
        return arr - beta * cov_c
    return _resid(x, covariate), _resid(y, covariate)


# ---------------------------------------------------------------------------
# Goal 1 – Belief learning rates by algorithm
# ---------------------------------------------------------------------------

def goal1_learning_curves(
    env: str,
    metrics_df: pd.DataFrame,
    probe: str,
    out_dir: Path,
) -> None:
    """Metric curves over training checkpoints, one line per algorithm.

    Aggregates over all matched hparam combinations (hidden_size, entropy_coeff,
    total_steps) to show the typical learning trajectory per algo.
    Recurrent runs only (memoryless=False).

    Saved as: {env}_learning_curves_{mlp|linear}.png
    """
    df = metrics_df[
        (metrics_df["env_name"] == env) &
        (metrics_df["probe"] == probe) &
        (~metrics_df["memoryless"])
    ]
    if df.empty:
        return

    algos   = _sorted_algos(df)
    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5.5 * len(metrics), 4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        for algo in algos:
            curve = (df[df["algo"] == algo]
                     .groupby("ckpt")[metric]
                     .agg(["mean", "std"])
                     .reset_index()
                     .sort_values("ckpt"))
            if curve.empty:
                continue
            color = ALGO_COLORS.get(algo, "gray")
            ax.plot(curve["ckpt"], curve["mean"],
                    color=color, linewidth=2, label=algo)
            ax.fill_between(curve["ckpt"],
                            curve["mean"] - curve["std"],
                            curve["mean"] + curve["std"],
                            color=color, alpha=0.15)
        ax.set_xlabel("Checkpoint", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [mlines.Line2D([], [], color=ALGO_COLORS.get(a, "gray"),
                              linewidth=2, label=a) for a in algos]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.07), ncol=len(algos),
               fontsize=9, frameon=False)

    probe_label = PROBE_LABELS.get(probe, probe)
    env_label   = ENV_LABELS.get(env, env)
    fig.suptitle(
        f"{env_label}  ·  {probe_label}  ·  Belief learning curves by algorithm\n"
        f"recurrent runs only  ·  mean ±1 std over all seeds and hparam combos",
        fontsize=9, color="dimgray",
    )
    out_path = out_dir / f"{env}_learning_curves_{probe.split('_')[0]}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"      {out_path.name}")


def goal1_final_distribution(
    env: str,
    metrics_df: pd.DataFrame,
    probe: str,
    out_dir: Path,
) -> None:
    """Paired strip plot at the final checkpoint.

    Each dot = one matched hparam combination (same hidden_size, entropy_coeff,
    total_steps, averaged over seeds).  Lines connect matched pairs.
    Horizontal bar = mean across matched pairs.
    Recurrent runs only.

    Saved as: {env}_final_ckpt_distribution_{mlp|linear}.png
    """
    df = metrics_df[
        (metrics_df["env_name"] == env) &
        (metrics_df["probe"] == probe) &
        (~metrics_df["memoryless"])
    ]
    if df.empty:
        return

    df = df[df["ckpt"] == df["ckpt"].max()]
    # Average over seeds within each (hparam_id, algo)
    agg = (df.groupby(["hparam_id", "algo"])[PRIMARY_METRICS]
             .mean()
             .reset_index())

    algos   = _sorted_algos(agg)
    metrics = [m for m in PRIMARY_METRICS if m in agg.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3 * len(metrics), 4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    rng  = np.random.default_rng(seed=0)

    for ax, metric in zip(axes, metrics):
        # Connecting lines for matched pairs
        for hp in agg["hparam_id"].unique():
            hp_sub = agg[agg["hparam_id"] == hp].set_index("algo")
            present = [a for a in algos if a in hp_sub.index]
            if len(present) >= 2:
                xs = [algos.index(a) for a in present]
                ys = [hp_sub.loc[a, metric] for a in present]
                ax.plot(xs, ys, color="gray", linewidth=0.7, alpha=0.35, zorder=1)

        # Dots + mean bars
        for xi, algo in enumerate(algos):
            vals = agg.loc[agg["algo"] == algo, metric].values
            jitter = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(xi + jitter, vals,
                       color=ALGO_COLORS.get(algo, "gray"),
                       s=45, alpha=0.85, zorder=2)
            if len(vals):
                ax.hlines(vals.mean(), xi - 0.22, xi + 0.22,
                          colors="black", linewidth=2.2, zorder=3)

        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

    probe_label = PROBE_LABELS.get(probe, probe)
    env_label   = ENV_LABELS.get(env, env)
    fig.suptitle(
        f"{env_label}  ·  {probe_label}  ·  Final checkpoint belief quality\n"
        f"recurrent runs  ·  each dot = one hparam combo (avg over seeds)\n"
        f"lines connect matched pairs  ·  bar = mean",
        fontsize=8.5, color="dimgray",
    )
    out_path = out_dir / f"{env}_final_ckpt_distribution_{probe.split('_')[0]}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"      {out_path.name}")


def goal1_stats(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """Pairwise Wilcoxon signed-rank test on matched pairs at the final checkpoint.

    For every (env, probe, metric) and every pair of algorithms present,
    tests whether algo_1 and algo_2 differ significantly on matched pairs.
    Works for any number of algorithms (PPO, DC-PPO, future DQN, etc.).

    Saved as: summary_stats.csv
    """
    final = metrics_df[metrics_df["ckpt"] == metrics_df["ckpt"].max()]
    algos = _sorted_algos(metrics_df)
    rows  = []

    for env in metrics_df["env_name"].unique():
        for probe in PROBE_KEYS:
            for metric in ALL_METRICS:
                sub = final[
                    (final["env_name"] == env) &
                    (final["probe"]    == probe) &
                    (~final["memoryless"]) &
                    (final[metric].notna())
                ] if metric in final.columns else pd.DataFrame()
                if sub.empty:
                    continue

                # Aggregate to one value per (hparam_id, algo)
                agg = (sub.groupby(["hparam_id", "algo"])[metric]
                           .mean()
                           .unstack("algo"))

                for i, a1 in enumerate(algos):
                    for a2 in algos[i + 1:]:
                        if a1 not in agg.columns or a2 not in agg.columns:
                            continue
                        paired = agg[[a1, a2]].dropna()
                        if len(paired) < 4:   # too few pairs for meaningful test
                            continue
                        try:
                            _, pval = wilcoxon(paired[a1].values, paired[a2].values)
                        except Exception:
                            pval = float("nan")
                        rows.append({
                            "env":        env,
                            "probe":      probe,
                            "metric":     metric,
                            "algo_1":     a1,
                            "algo_2":     a2,
                            "mean_1":     round(float(paired[a1].mean()), 5),
                            "mean_2":     round(float(paired[a2].mean()), 5),
                            "delta":      round(float(paired[a2].mean() - paired[a1].mean()), 5),
                            "n_pairs":    len(paired),
                            "wilcoxon_p": round(pval, 5) if pval == pval else float("nan"),
                            "sig":        "***" if pval < 0.001 else
                                          ("**" if pval < 0.01 else
                                          ("*"  if pval < 0.05 else
                                          ("."  if pval < 0.1  else ""))),
                        })

    if rows:
        out_path = out_dir / "summary_stats.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"      {out_path.name}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Goal 2 – Belief quality vs reward
# ---------------------------------------------------------------------------

def goal2_raw_scatter(
    merged: pd.DataFrame, probe: str, out_dir: Path
) -> None:
    """Raw scatter: x = mean_ep_return, y = metric.  All checkpoints.

    Naively strong correlation expected because both reward and belief quality
    improve over training.  Compare with partial_corr_scatter to separate the
    training-progress confound.

    Saved as: raw_scatter_{mlp|linear}.png
    """
    df = merged[
        (merged["probe"] == probe) &
        (~merged["memoryless"]) &
        (merged["mean_ep_return"].notna())
    ]
    if df.empty:
        return

    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    envs    = [e for e in ENV_ORDER if e in df["env_name"].unique()]

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5.5 * len(metrics), 4.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        for env in envs:
            sub = df[df["env_name"] == env]
            if sub.empty:
                continue
            ax.scatter(sub["mean_ep_return"], sub[metric],
                       c=ENV_COLORS.get(env, "gray"),
                       alpha=0.2, s=10, linewidths=0)
        ax.set_xlabel("Mean episodic return", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(f"Raw  ·  {METRIC_LABELS.get(metric, metric)}", fontsize=9,
                     fontweight="bold")
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                              color=ENV_COLORS.get(e, "gray"), markersize=7,
                              label=ENV_LABELS.get(e, e)) for e in envs]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(envs), fontsize=8,
               frameon=False)

    probe_label = PROBE_LABELS.get(probe, probe)
    fig.suptitle(
        f"Reward vs Belief Quality  ·  {probe_label}  ·  raw (all checkpoints)\n"
        f"recurrent runs only  ·  each point = one (run, checkpoint, seed)\n"
        f"NOTE: correlation is inflated by training progress — see partial_corr_scatter",
        fontsize=8.5, color="dimgray",
    )
    out_path = out_dir / f"raw_scatter_{probe.split('_')[0]}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"      {out_path.name}")


def goal2_partial_scatter(
    merged: pd.DataFrame, probe: str, out_dir: Path
) -> None:
    """Partial scatter: both reward and metric residualised on checkpoint index.

    Removes the 'both improve over training' confound.  A non-zero correlation
    here means that, at the same training stage, runs with better beliefs also
    have higher reward — a more interesting claim.

    Residualisation is done per environment (checkpoints have different
    absolute reward scales across envs).

    Saved as: partial_corr_scatter_{mlp|linear}.png
    """
    df = merged[
        (merged["probe"] == probe) &
        (~merged["memoryless"]) &
        (merged["mean_ep_return"].notna())
    ].copy()
    if df.empty:
        return

    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    envs    = [e for e in ENV_ORDER if e in df["env_name"].unique()]

    # Build residualised frame per env
    res_rows: List[dict] = []
    for env in envs:
        sub = df[df["env_name"] == env]
        cov = sub["ckpt"].values.astype(float)
        rew = sub["mean_ep_return"].values
        for metric in metrics:
            if metric not in sub.columns:
                continue
            met  = sub[metric].values
            mask = np.isfinite(rew) & np.isfinite(met)
            if mask.sum() < 10:
                continue
            res_r, res_m = _partial_residualize(rew[mask], met[mask], cov[mask])
            for rr, rm in zip(res_r, res_m):
                res_rows.append({"env_name": env, "metric": metric,
                                 "res_reward": rr, "res_metric": rm})
    if not res_rows:
        return

    res_df = pd.DataFrame(res_rows)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5.5 * len(metrics), 4.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        sub = res_df[res_df["metric"] == metric]
        for env in envs:
            esub = sub[sub["env_name"] == env]
            if esub.empty:
                continue
            ax.scatter(esub["res_reward"], esub["res_metric"],
                       c=ENV_COLORS.get(env, "gray"),
                       alpha=0.2, s=10, linewidths=0)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xlabel("Reward  (checkpoint effect removed)", fontsize=9)
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)}\n(checkpoint effect removed)",
                      fontsize=9)
        ax.set_title(f"Partial  ·  {METRIC_LABELS.get(metric, metric)}", fontsize=9,
                     fontweight="bold")
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                              color=ENV_COLORS.get(e, "gray"), markersize=7,
                              label=ENV_LABELS.get(e, e)) for e in envs]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(envs), fontsize=8,
               frameon=False)

    probe_label = PROBE_LABELS.get(probe, probe)
    fig.suptitle(
        f"Reward vs Belief Quality  ·  {probe_label}  ·  partial (checkpoint residualised)\n"
        f"recurrent runs only  ·  each point = one (run, checkpoint, seed)\n"
        f"non-zero correlation here cannot be explained by training progress alone",
        fontsize=8.5, color="dimgray",
    )
    out_path = out_dir / f"partial_corr_scatter_{probe.split('_')[0]}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"      {out_path.name}")


def goal2_final_ckpt_scatter(
    merged: pd.DataFrame, probe: str, out_dir: Path
) -> None:
    """Cross-run scatter at the final checkpoint.

    One point per run (averaged over seeds).  Avoids the training-progress
    confound by construction.  Per-env linear fit overlaid.

    Saved as: final_ckpt_scatter_{mlp|linear}.png
    """
    df = merged[
        (merged["probe"] == probe) &
        (~merged["memoryless"]) &
        (merged["mean_ep_return"].notna())
    ]
    if df.empty:
        return

    df = df[df["ckpt"] == df["ckpt"].max()]
    df = (df.groupby(["run_id", "env_name", "algo"])[PRIMARY_METRICS + ["mean_ep_return"]]
           .mean()
           .reset_index())

    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    envs    = [e for e in ENV_ORDER if e in df["env_name"].unique()]

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5.5 * len(metrics), 4.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        for env in envs:
            sub = df[df["env_name"] == env]
            if sub.empty:
                continue
            color = ENV_COLORS.get(env, "gray")
            ax.scatter(sub["mean_ep_return"], sub[metric],
                       c=color, s=55, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
            if len(sub) >= 3:
                try:
                    coeffs = np.polyfit(sub["mean_ep_return"], sub[metric], 1)
                    xr = np.linspace(sub["mean_ep_return"].min(),
                                     sub["mean_ep_return"].max(), 60)
                    ax.plot(xr, np.polyval(coeffs, xr),
                            color=color, linewidth=1.3, alpha=0.7)
                except Exception:
                    pass
        ax.set_xlabel("Mean episodic return  (final ckpt)", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(f"Final ckpt  ·  {METRIC_LABELS.get(metric, metric)}", fontsize=9,
                     fontweight="bold")
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                              color=ENV_COLORS.get(e, "gray"), markersize=7,
                              label=ENV_LABELS.get(e, e)) for e in envs]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(envs), fontsize=8,
               frameon=False)

    probe_label = PROBE_LABELS.get(probe, probe)
    fig.suptitle(
        f"Reward vs Belief Quality  ·  {probe_label}  ·  final checkpoint  (cross-run)\n"
        f"recurrent runs only  ·  each point = one run (mean over seeds)"
        f"  ·  line = per-env linear fit",
        fontsize=8.5, color="dimgray",
    )
    out_path = out_dir / f"final_ckpt_scatter_{probe.split('_')[0]}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"      {out_path.name}")


def goal2_correlation_heatmap(merged: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of partial Spearman ρ (checkpoint residualised).

    Rows = environments.  Columns = probe × metric pairs.
    Cells annotated with ρ value and significance stars.

    Saved as: correlation_heatmap.png
    """
    df = merged[
        (~merged["memoryless"]) &
        (merged["mean_ep_return"].notna())
    ]
    probes  = [p for p in PROBE_KEYS    if p in df["probe"].unique()]
    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    envs    = [e for e in ENV_ORDER     if e in df["env_name"].unique()]
    if not probes or not metrics or not envs:
        return

    col_labels = [
        f"{PROBE_LABELS[p].split()[0]}\n{METRIC_LABELS[m].split()[0]}"
        for p in probes for m in metrics
    ]
    rho_mat  = np.full((len(envs), len(col_labels)), np.nan)
    pval_mat = np.full((len(envs), len(col_labels)), np.nan)

    for ri, env in enumerate(envs):
        ci = 0
        for probe in probes:
            for metric in metrics:
                sub = df[(df["env_name"] == env) & (df["probe"] == probe)]
                if sub.empty or metric not in sub.columns:
                    ci += 1
                    continue
                rew  = sub["mean_ep_return"].values
                met  = sub[metric].values
                cov  = sub["ckpt"].values.astype(float)
                mask = np.isfinite(rew) & np.isfinite(met)
                if mask.sum() < 10:
                    ci += 1
                    continue
                res_r, res_m = _partial_residualize(rew[mask], met[mask], cov[mask])
                try:
                    rho, pval = spearmanr(res_r, res_m)
                    rho_mat[ri, ci]  = rho
                    pval_mat[ri, ci] = pval
                except Exception:
                    pass
                ci += 1

    fig, ax = plt.subplots(
        figsize=(max(4.5, len(col_labels) * 1.5), max(3, len(envs) * 0.9)),
        constrained_layout=True,
    )
    im = ax.imshow(rho_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")

    for ri in range(len(envs)):
        for ci in range(len(col_labels)):
            rho  = rho_mat[ri, ci]
            pval = pval_mat[ri, ci]
            if np.isnan(rho):
                continue
            star = ("***" if pval < 0.001 else
                    ("**" if pval < 0.01 else
                    ("*"  if pval < 0.05 else "")))
            ax.text(ci, ri, f"{rho:.2f}{star}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(rho) > 0.55 else "black",
                    fontweight="bold" if star else "normal")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8.5)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([ENV_LABELS.get(e, e) for e in envs], fontsize=9)
    ax.set_title(
        "Reward–Belief Correlation  (partial Spearman ρ, checkpoint residualised)\n"
        "recurrent runs only  ·  *p<0.05  **p<0.01  ***p<0.001",
        fontsize=9, pad=10,
    )
    out_path = out_dir / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"      {out_path.name}")


def goal2_stats(merged: pd.DataFrame, out_dir: Path) -> None:
    """Save CSV with raw and partial Pearson/Spearman correlations.

    Reports separately for recurrent and memoryless runs so you can check
    whether the correlation holds without memory (if it does, it may be more
    about training quality than about belief representations per se).

    Saved as: summary_stats.csv
    """
    rows = []
    metrics = [m for m in ALL_METRICS if m in merged.columns]

    for env in merged["env_name"].unique():
        for probe in PROBE_KEYS:
            for ml in [False, True]:
                sub = merged[
                    (merged["env_name"]      == env) &
                    (merged["probe"]         == probe) &
                    (merged["memoryless"]    == ml) &
                    (merged["mean_ep_return"].notna())
                ]
                if sub.empty:
                    continue
                for metric in metrics:
                    if metric not in sub.columns:
                        continue
                    rew  = sub["mean_ep_return"].values
                    met  = sub[metric].values
                    cov  = sub["ckpt"].values.astype(float)
                    mask = np.isfinite(rew) & np.isfinite(met)
                    n    = int(mask.sum())
                    if n < 6:
                        continue

                    def _safe_corr(fn, a, b):
                        try:
                            r, p = fn(a, b)
                            return round(float(r), 5), round(float(p), 5)
                        except Exception:
                            return float("nan"), float("nan")

                    rho_r, p_rho_r = _safe_corr(spearmanr, rew[mask], met[mask])
                    r_r,   p_r_r   = _safe_corr(pearsonr,  rew[mask], met[mask])
                    res_rew, res_met = _partial_residualize(rew[mask], met[mask], cov[mask])
                    rho_p, p_rho_p = _safe_corr(spearmanr, res_rew, res_met)
                    r_p,   p_r_p   = _safe_corr(pearsonr,  res_rew, res_met)

                    rows.append({
                        "env":                   env,
                        "probe":                 probe,
                        "metric":                metric,
                        "memoryless":            ml,
                        "n":                     n,
                        "spearman_rho_raw":      rho_r,
                        "spearman_p_raw":        p_rho_r,
                        "pearson_r_raw":         r_r,
                        "pearson_p_raw":         p_r_r,
                        "spearman_rho_partial":  rho_p,
                        "spearman_p_partial":    p_rho_p,
                        "pearson_r_partial":     r_p,
                        "pearson_p_partial":     p_r_p,
                    })

    if rows:
        out_path = out_dir / "summary_stats.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"      {out_path.name}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json_path", required=True, type=Path,
                   help="Path to all_results.json.gz produced by export_results.py.")
    p.add_argument("--out_dir",   required=True, type=Path,
                   help="Root directory where all figures and CSVs are written.")
    args = p.parse_args()

    metrics_df, rewards_df = load_and_build_dataframes(args.json_path)
    if metrics_df.empty:
        print("No metric data found. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Goal 1 – algorithm comparison
    # -----------------------------------------------------------------------
    g1_dir = args.out_dir / "goal1_belief_learning_rates"
    g1_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Goal 1]  Belief learning rates by algorithm")
    print(f"          → {g1_dir}\n")

    envs = [e for e in ENV_ORDER if e in metrics_df["env_name"].unique()]
    for env in envs:
        print(f"  {ENV_LABELS.get(env, env)}")
        for probe in PROBE_KEYS:
            goal1_learning_curves(env, metrics_df, probe, g1_dir)
            goal1_final_distribution(env, metrics_df, probe, g1_dir)

    print("  Statistics (paired Wilcoxon) ...")
    goal1_stats(metrics_df, g1_dir)

    # -----------------------------------------------------------------------
    # Goal 2 – belief quality vs reward
    # -----------------------------------------------------------------------
    if rewards_df.empty:
        print("\n[Goal 2]  No reward data in JSON — skipping.")
        return

    g2_dir = args.out_dir / "goal2_belief_reward_correlation"
    g2_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Goal 2]  Belief quality vs reward")
    print(f"          → {g2_dir}\n")

    merged = metrics_df.merge(
        rewards_df[["run_id", "ckpt", "seed", "mean_ep_return"]],
        on=["run_id", "ckpt", "seed"],
        how="inner",
    )
    if merged.empty:
        print("  Could not merge metrics and rewards — skipping Goal 2.")
        return

    for probe in PROBE_KEYS:
        print(f"  {PROBE_LABELS[probe]}")
        goal2_raw_scatter(merged, probe, g2_dir)
        goal2_partial_scatter(merged, probe, g2_dir)
        goal2_final_ckpt_scatter(merged, probe, g2_dir)

    print("  Correlation heatmap ...")
    goal2_correlation_heatmap(merged, g2_dir)
    print("  Statistics CSV ...")
    goal2_stats(merged, g2_dir)

    print(f"\nAll outputs written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
