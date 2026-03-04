"""
Step 3 of the Bayesian belief probe pipeline.

Loads all metrics JSONs from a results directory, aggregates over seeds,
and produces:

  1. Metric curves figure:  6 subplots (one per metric), one line per probe
     type, ±1 std shaded band over seeds, x-axis = checkpoint index.

  2. Triangle grid figure (CompassWorld only, detected automatically):
     mean predicted belief at first / middle / last checkpoint for the
     MLP and linear probes.

  Additional env-specific figures are added via the if/elif dispatch at the
  bottom of this file.

Usage (standalone):
    python visualize.py --results_dir /tmp/probe_out --out_dir /tmp/figures
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from envs import get_env_handler


# ---------------------------------------------------------------------------
# Experiment label helper
# ---------------------------------------------------------------------------

def _make_experiment_label(config: dict) -> str:
    """Format a config dict into a compact one-line figure subtitle."""
    if not config:
        return ""
    env  = config.get("env_name", "?")
    h    = config.get("hidden_size", "?")
    ts   = config.get("total_steps", 0)
    ts_s = f"{ts / 1e6:.1f}M" if isinstance(ts, (int, float)) and ts > 0 else "?"
    if config.get("algo") == "dqn":
        mode = "DQN" if config.get("memoryless") else "DRQN"
        lr   = config.get("lr", "?")
        tr   = config.get("trace_length", "?")
        ne   = config.get("num_envs", "?")
        bbs  = config.get("buffer_batch_size", "?")
        return f"{env}  {mode}  h={h}  lr={lr}  tr={tr}  ne={ne}  bbs={bbs}  steps={ts_s}"
    else:
        dc   = "dc" if config.get("double_critic") else "no-dc"
        ml   = "memoryless" if config.get("memoryless") else "recurrent"
        ac   = "ac" if config.get("action_concat") else "no-ac"
        ent  = config.get("entropy_coeff", "?")
        return f"{env}  h={h}  {dc}  {ml}  {ac}  ent={ent}  steps={ts_s}"


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

PROBE_ORDER = [
    "mlp_from_rnn_hidden",
    "linear_from_rnn_hidden",
    "analytic_mean_constant_kl",
]

PROBE_LABELS = {
    "mlp_from_rnn_hidden":       "MLP (RNN hidden)",
    "linear_from_rnn_hidden":    "Linear (RNN hidden)",
    "analytic_mean_constant_kl": "Analytic Mean Constant",
}

# Okabe-Ito palette (colorblind-safe)
_OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
PROBE_COLORS = {k: _OKABE[i] for i, k in enumerate(PROBE_ORDER)}

METRIC_KEYS = [
    "tv",
    "mean_kl_bits",
    "impossible_mass_overall",
    "impossible_mass_should_know",
    "argmax_match_rate",
    "mean_prob_on_true_location",
]
# RockSample uses factored Bernoulli beliefs; "should-know" is defined per
# (step, rock) pair rather than over the full joint state.  Most certain pairs
# come from sample actions (p=0 mechanically), so argmax_match and
# impossible_mass_should_know are near-perfect across all checkpoints and
# carry no information about learning.  Show only the all-pairs metrics.
METRIC_KEYS_ROCKSAMPLE = [
    "tv",
    "mean_kl_bits",
    "impossible_mass_overall",
]
METRIC_TITLES = {
    "tv":                          "Total Variation ↓",
    "mean_kl_bits":                "Mean KL (bits) ↓",
    "argmax_match_rate":           "Should-Know Accuracy ↑",
    "mean_prob_on_true_location":  "Should-Know Mass ↑",
    "impossible_mass_overall":     "Impossible Mass (Overall) ↓",
    "impossible_mass_should_know": "Impossible Mass (SK) ↓",
}


# ---------------------------------------------------------------------------
# Data loading and aggregation
# ---------------------------------------------------------------------------

def load_all_jsons(results_dir: Path) -> List[dict]:
    jsons = []
    for p in sorted(results_dir.rglob("*_metrics.json")):
        with open(p) as f:
            jsons.append(json.load(f))
    return jsons


def aggregate(records: List[dict]) -> Dict:
    """
    Returns:
        {(h_idx, ckpt_idx, probe_name, metric_name): [value_seed0, value_seed1, ...]}
        and separately the set of checkpoint indices.
    """
    grouped = defaultdict(list)
    for r in records:
        h = r["hparam_idx"]
        c = r["checkpoint_idx"]
        for probe, mdict in r["metrics"].items():
            for metric, val in mdict.items():
                grouped[(h, c, probe, metric)].append(val)
    return dict(grouped)


def aggregate_mean_pred(records: List[dict]) -> Dict:
    """
    {(h_idx, ckpt_idx, probe_name): mean of mean_pred_belief across seeds}
    (simple average of mean_pred_belief vectors, one per seed)
    """
    accum: Dict = defaultdict(list)
    for r in records:
        h = r["hparam_idx"]
        c = r["checkpoint_idx"]
        for probe, vec in r.get("mean_pred_belief", {}).items():
            accum[(h, c, probe)].append(np.array(vec, dtype=np.float32))
    return {k: np.stack(v).mean(axis=0) for k, v in accum.items()}


def _load_rewards_by_ckpt(results_dir: Path, h_idx: int) -> Dict[int, np.ndarray]:
    """
    Load per-seed mean episodic return for each checkpoint directory.

    Returns:
        {ckpt_idx: array of shape [n_seeds]} — mean episodic return per seed,
        averaged over trajectories within that seed.
    """
    ckpt_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    result: Dict[int, np.ndarray] = {}
    for ckpt_dir in ckpt_dirs:
        ckpt_idx = int(ckpt_dir.name.split("_")[1])
        npz_path = ckpt_dir / f"h{h_idx}_trajectories.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=False)
        if "rewards" not in data or "masks" not in data:
            continue
        rewards = data["rewards"]          # [n_seeds, n_traj, max_len]
        masks   = data["masks"]            # [n_seeds, n_traj, max_len]
        ep_returns = (rewards * masks).sum(axis=-1)   # [n_seeds, n_traj]
        result[ckpt_idx] = ep_returns.mean(axis=1)    # [n_seeds]
    return result


def _build_metric_lookup(records: List[dict]) -> Dict:
    """
    Returns {(ckpt_idx, seed_idx, probe_name, metric_name): float}
    for fast per-point access in reward-vs-metric scatter.
    """
    lookup: Dict = {}
    for r in records:
        c, s = r["checkpoint_idx"], r["seed_idx"]
        for probe, mdict in r["metrics"].items():
            for metric, val in mdict.items():
                lookup[(c, s, probe, metric)] = val
    return lookup


# ---------------------------------------------------------------------------
# Metric curves
# ---------------------------------------------------------------------------

def plot_metric_curves(records: List[dict], out_dir: Path, h_idx: int = 0,
                       metric_keys: list = METRIC_KEYS, config: dict = None):
    grouped = aggregate(records)

    ckpt_indices = sorted({r["checkpoint_idx"] for r in records})
    probes = [p for p in PROBE_ORDER if any((h_idx, c, p, "tv") in grouped for c in ckpt_indices)]
    if not probes:
        print("  no probe data found — skipping metric curves")
        return

    hparams = sorted({r["hparam_idx"] for r in records})

    for h in hparams:
        n_metrics = len(metric_keys)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 * n_cols / 3, 7 * n_rows / 2),
                                 constrained_layout=True)
        axes_flat = np.array(axes).ravel()
        # Hide any unused subplot slots
        for ax in axes_flat[n_metrics:]:
            ax.axis("off")

        for ax, metric in zip(axes_flat, metric_keys):
            for probe in probes:
                means, stds, xs = [], [], []
                for c in ckpt_indices:
                    vals = grouped.get((h, c, probe, metric), [])
                    clean = [v for v in vals if v == v]  # drop NaN
                    if not clean:
                        continue
                    xs.append(c)
                    means.append(float(np.mean(clean)))
                    stds.append(float(np.std(clean)) if len(clean) > 1 else 0.0)

                if not xs:
                    continue
                xs_arr = np.array(xs)
                m_arr  = np.array(means)
                s_arr  = np.array(stds)
                color  = PROBE_COLORS.get(probe, "gray")
                ax.plot(xs_arr, m_arr, color=color, linewidth=2,
                        label=PROBE_LABELS.get(probe, probe))
                ax.fill_between(xs_arr, m_arr - s_arr, m_arr + s_arr,
                                color=color, alpha=0.2)

            ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=10)
            ax.set_xlabel("Checkpoint index")
            ax.spines[["right", "top"]].set_visible(False)

        legend_handles = [
            Patch(color=PROBE_COLORS.get(p, "gray"), label=PROBE_LABELS.get(p, p))
            for p in probes
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04), ncol=len(probes), fontsize=9)

        label = _make_experiment_label(config)
        if label:
            fig.suptitle(label, fontsize=8, color="dimgray", style="italic")

        suffix = f"_h{h}" if len(hparams) > 1 else ""
        out_path = out_dir / f"metric_curves{suffix}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# CompassWorld triangle grids
# ---------------------------------------------------------------------------

def _plot_compass_triangle_grids(
    records: List[dict], results_dir: Path, out_dir: Path, h_idx: int = 0,
    config: dict = None,
):
    mean_preds = aggregate_mean_pred(records)

    # Load belief_shape from the first matching NPZ in the results dir
    belief_shape = None
    for npz_path in results_dir.rglob("*.npz"):
        d = np.load(npz_path, allow_pickle=False)
        if "belief_shape" in d:
            belief_shape = tuple(int(x) for x in d["belief_shape"])
            break
    if belief_shape is None:
        print("  could not find belief_shape in any NPZ — skipping triangle grids")
        return

    env_name = records[0].get("env_name", "")
    try:
        handler = get_env_handler(env_name)
    except ValueError:
        print(f"  no handler for {env_name} — skipping triangle grids")
        return

    ckpt_indices = sorted({r["checkpoint_idx"] for r in records if r["hparam_idx"] == h_idx})
    if not ckpt_indices:
        return

    probe_rows = [p for p in ["mlp_from_rnn_hidden", "linear_from_rnn_hidden"]
                  if any((h_idx, c, p) in mean_preds for c in ckpt_indices)]
    if not probe_rows:
        return

    # Pick first, mid, last checkpoint
    col_ckpts = [
        ckpt_indices[0],
        ckpt_indices[len(ckpt_indices) // 2],
        ckpt_indices[-1],
    ]
    col_labels = ["Early", "Mid", "Late"]

    n_rows = len(probe_rows)
    n_cols = len(col_ckpts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Shared colour scale
    all_vals = []
    for probe in probe_rows:
        for c in col_ckpts:
            vec = mean_preds.get((h_idx, c, probe))
            if vec is not None:
                all_vals.extend(vec.tolist())
    vmin = float(min(all_vals)) if all_vals else 0.0
    vmax = float(max(all_vals)) if all_vals else 1.0

    for r_idx, probe in enumerate(probe_rows):
        for c_idx, (c, col_label) in enumerate(zip(col_ckpts, col_labels)):
            ax = axes[r_idx, c_idx]
            vec = mean_preds.get((h_idx, c, probe))
            if vec is None:
                ax.axis("off")
                continue
            title = f"{PROBE_LABELS.get(probe, probe)}\n{col_label} (ckpt {c})"
            handler.visualize_beliefs(vec, ax, title=title, vmin=vmin, vmax=vmax)

    label = _make_experiment_label(config)
    suptitle = f"Mean Predicted Belief — {env_name}"
    if label:
        suptitle = f"{suptitle}\n{label}"
    fig.suptitle(suptitle, fontsize=11)

    hparams = sorted({r["hparam_idx"] for r in records})
    suffix = f"_h{h_idx}" if len(hparams) > 1 else ""
    out_path = out_dir / f"triangle_grids{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Marquee belief bar charts
# ---------------------------------------------------------------------------

def _plot_marquee_belief_bars(
    records: List[dict], out_dir: Path, h_idx: int = 0, config: dict = None,
):
    """
    For Marquee: plot mean predicted belief (bar chart over goals) at
    early / mid / late checkpoints for MLP and linear probes.
    """
    mean_preds = aggregate_mean_pred(records)

    ckpt_indices = sorted({r["checkpoint_idx"] for r in records if r["hparam_idx"] == h_idx})
    if not ckpt_indices:
        return

    probe_rows = [p for p in ["mlp_from_rnn_hidden", "linear_from_rnn_hidden"]
                  if any((h_idx, c, p) in mean_preds for c in ckpt_indices)]
    if not probe_rows:
        return

    col_ckpts  = [ckpt_indices[0], ckpt_indices[len(ckpt_indices) // 2], ckpt_indices[-1]]
    col_labels = ["Early", "Mid", "Late"]

    n_rows = len(probe_rows)
    n_cols = len(col_ckpts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                              constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Shared y-axis max for easy comparison across panels
    all_vals = [
        v
        for probe in probe_rows
        for c in col_ckpts
        for v in (mean_preds.get((h_idx, c, probe), np.array([])).tolist())
    ]
    ymax = float(max(all_vals)) * 1.15 if all_vals else 1.0

    env_name = records[0].get("env_name", "marquee")
    try:
        from envs import get_env_handler
        handler = get_env_handler(env_name)
    except Exception:
        handler = None

    for r_idx, probe in enumerate(probe_rows):
        for c_idx, (c, col_label) in enumerate(zip(col_ckpts, col_labels)):
            ax = axes[r_idx, c_idx]
            vec = mean_preds.get((h_idx, c, probe))
            if vec is None:
                ax.axis("off")
                continue
            title = f"{PROBE_LABELS.get(probe, probe)}\n{col_label} (ckpt {c})"
            if handler is not None:
                handler.visualize_beliefs(vec, ax, title=title, vmax=ymax)
            else:
                n = len(vec)
                ax.bar(np.arange(n), vec, color="#56B4E9", edgecolor="none")
                ax.set_ylim(0, ymax)
                ax.set_title(title, fontsize=9)
                ax.spines[["right", "top"]].set_visible(False)

    label = _make_experiment_label(config)
    suptitle = f"Mean Predicted Belief — {env_name}"
    if label:
        suptitle = f"{suptitle}\n{label}"
    fig.suptitle(suptitle, fontsize=11)

    hparams = sorted({r["hparam_idx"] for r in records})
    suffix = f"_h{h_idx}" if len(hparams) > 1 else ""
    out_path = out_dir / f"belief_bars{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Belief sanity curves (Marquee / envs with extras_goal_idx)
# ---------------------------------------------------------------------------

def plot_belief_sanity_curves(records: List[dict], out_dir: Path, h_idx: int = 0,
                               config: dict = None):
    """
    Plot belief sanity metrics vs checkpoint index.

    Three lines:
      - final_argmax_correct:    fraction of episodes where belief[-1] argmax = true goal
      - final_mean_prob_true:    avg P(true_goal) at the final step
      - all_steps_mean_prob_true: avg P(true_goal) across all valid steps

    A value near 1.0 for final_argmax_correct confirms the analytical belief
    computation is correct; probe failure then reflects the hidden state.
    """
    # Only records that contain belief_sanity (i.e. Marquee-like envs)
    sane_records = [r for r in records if "belief_sanity" in r and r["hparam_idx"] == h_idx]
    if not sane_records:
        return

    ckpt_indices = sorted({r["checkpoint_idx"] for r in sane_records})

    keys = ["final_argmax_correct", "final_mean_prob_true", "all_steps_mean_prob_true"]
    labels = {
        "final_argmax_correct":     "Final argmax correct ↑",
        "final_mean_prob_true":     "Final mean P(true goal) ↑",
        "all_steps_mean_prob_true": "All-steps mean P(true goal) ↑",
    }
    colors = ["#0072B2", "#E69F00", "#009E73"]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for key, color in zip(keys, colors):
        xs, means, stds = [], [], []
        for c in ckpt_indices:
            vals = [r["belief_sanity"][key] for r in sane_records
                    if r["checkpoint_idx"] == c and key in r["belief_sanity"]
                    and r["belief_sanity"][key] == r["belief_sanity"][key]]  # drop NaN
            if not vals:
                continue
            xs.append(c)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        if not xs:
            continue
        xs_arr = np.array(xs)
        m_arr  = np.array(means)
        s_arr  = np.array(stds)
        ax.plot(xs_arr, m_arr, color=color, linewidth=2, label=labels[key])
        ax.fill_between(xs_arr, m_arr - s_arr, m_arr + s_arr, color=color, alpha=0.2)

    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Checkpoint index")
    ax.set_ylabel("Fraction / probability")
    ax.set_title("Belief Sanity Check (analytical beliefs vs true goal)")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(fontsize=9)

    label = _make_experiment_label(config)
    if label:
        fig.suptitle(label, fontsize=8, color="dimgray", style="italic")

    suffix = f"_h{h_idx}"
    out_path = out_dir / f"belief_sanity{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Episodic reward curve
# ---------------------------------------------------------------------------

def plot_reward_curves(results_dir: Path, out_dir: Path, h_idx: int = 0,
                       config: dict = None):
    """
    Load per-seed mean episodic return for each checkpoint and plot mean ± std.
    Saves reward_curve[_h{h}].png.
    """
    rewards_by_ckpt = _load_rewards_by_ckpt(results_dir, h_idx)
    if not rewards_by_ckpt:
        print("  no reward data found in NPZs — skipping reward curve")
        return

    xs     = sorted(rewards_by_ckpt)
    means  = [float(rewards_by_ckpt[c].mean()) for c in xs]
    stds   = [float(rewards_by_ckpt[c].std()) if len(rewards_by_ckpt[c]) > 1 else 0.0
              for c in xs]

    xs_arr = np.array(xs)
    m_arr  = np.array(means)
    s_arr  = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(xs_arr, m_arr, color="#0072B2", linewidth=2)
    ax.fill_between(xs_arr, m_arr - s_arr, m_arr + s_arr, color="#0072B2", alpha=0.2,
                    label="±1 std (seeds)")
    ax.set_xlabel("Checkpoint index")
    ax.set_ylabel("Average episodic return")
    ax.set_title("Average Episodic Return per Checkpoint")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(fontsize=9)

    label = _make_experiment_label(config)
    if label:
        fig.suptitle(label, fontsize=8, color="dimgray", style="italic")

    suffix = f"_h{h_idx}"
    out_path = out_dir / f"reward_curve{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Reward vs belief metrics scatter + stats
# ---------------------------------------------------------------------------

def plot_reward_vs_metrics(
    records: List[dict],
    results_dir: Path,
    out_dir: Path,
    h_idx: int = 0,
    config: dict = None,
    metric_keys: list = METRIC_KEYS,
):
    """
    Scatter: episodic reward vs each belief metric across all (seed, checkpoint) pairs.

    Each point is one (seed, checkpoint) pair.  Points are coloured by checkpoint
    progress (early=dark, late=bright).  A linear fit is overlaid on each panel.

    Produces two figures:
      reward_vs_metrics_h{h}.png            — raw y-axes (per subplot)
      reward_vs_metrics_normalized_h{h}.png — shared y-axis per metric column,
                                              starting at 0, so slopes are comparable

    Also saves reward_metric_stats_h{h}.json with:
        Pearson r, Spearman ρ, R² for every (probe, metric) combination.
    """
    rewards_by_ckpt = _load_rewards_by_ckpt(results_dir, h_idx)
    if not rewards_by_ckpt:
        print("  no reward data — skipping reward-vs-metric scatter")
        return

    h_records = [r for r in records if r["hparam_idx"] == h_idx]
    if not h_records:
        return

    lookup = _build_metric_lookup(h_records)

    present_probes = {key[2] for key in lookup}
    probes = [p for p in PROBE_ORDER if p in present_probes]
    if not probes:
        return

    # Checkpoint colour normalisation (darker = earlier)
    ckpt_list    = sorted(rewards_by_ckpt)
    n_ckpts      = len(ckpt_list)
    ckpt_to_norm = {c: i / max(n_ckpts - 1, 1) for i, c in enumerate(ckpt_list)}
    cmap         = plt.cm.viridis

    # ------------------------------------------------------------------
    # Pass 1: collect scatter data and compute stats (no plotting yet)
    # ------------------------------------------------------------------
    # scatter_data[(probe, metric)] = (xs_arr, ys_arr, colours_arr) | None
    scatter_data: Dict = {}
    stats_out:    Dict = {}

    for probe in probes:
        stats_out[probe] = {}
        for metric in metric_keys:
            xs, ys, colours = [], [], []
            for c, seed_rewards in rewards_by_ckpt.items():
                norm = ckpt_to_norm[c]
                for s_idx, rw in enumerate(seed_rewards):
                    val = lookup.get((c, s_idx, probe, metric))
                    if val is None or val != val:
                        continue
                    xs.append(float(rw))
                    ys.append(float(val))
                    colours.append(norm)

            if len(xs) < 3:
                scatter_data[(probe, metric)] = None
                continue

            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            scatter_data[(probe, metric)] = (xs_arr, ys_arr, np.array(colours))

            # Statistics (compatible with scipy pre- and post-1.9 API)
            pearson_r = spearman_rho = r_sq = float("nan")
            pearson_p = spearman_p = float("nan")
            try:
                from scipy.stats import pearsonr, spearmanr, linregress
                _pr = pearsonr(xs_arr, ys_arr)
                _sr = spearmanr(xs_arr, ys_arr)
                _lr = linregress(xs_arr, ys_arr)
                pearson_r    = float(getattr(_pr, "statistic", _pr[0]))
                pearson_p    = float(getattr(_pr, "pvalue",    _pr[1]))
                spearman_rho = float(getattr(_sr, "statistic", _sr[0]))
                spearman_p   = float(getattr(_sr, "pvalue",    _sr[1]))
                r_sq         = float(_lr.rvalue ** 2)
            except ImportError:
                corr = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
                pearson_r = corr
                r_sq      = corr ** 2

            stats_out[probe][metric] = {
                "n_points":     len(xs),
                "pearson_r":    pearson_r,
                "pearson_p":    pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p":   spearman_p,
                "r_squared":    r_sq,
            }

    # ------------------------------------------------------------------
    # Shared y-limits per metric column (for the normalised figure)
    # y starts at 0; top is the max value across all probes for that metric.
    # ------------------------------------------------------------------
    shared_ylims: Dict = {}
    for metric in metric_keys:
        all_ys = []
        for probe in probes:
            entry = scatter_data.get((probe, metric))
            if entry is not None:
                all_ys.extend(entry[1].tolist())   # entry[1] = ys_arr
        if all_ys:
            ymax = max(v for v in all_ys if v == v)
            shared_ylims[metric] = (0.0, ymax * 1.05)

    # ------------------------------------------------------------------
    # Pass 2: draw figures (raw axes, then normalised axes)
    # ------------------------------------------------------------------
    def _draw_fig(ylims_override, stem):
        n_rows = len(probes)
        n_cols = len(metric_keys)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.4 * n_cols, 3.4 * n_rows),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for r_idx, probe in enumerate(probes):
            for c_idx, metric in enumerate(metric_keys):
                ax = axes[r_idx, c_idx]
                entry = scatter_data.get((probe, metric))
                if entry is None:
                    ax.axis("off")
                    continue

                xs_arr, ys_arr, colours_arr = entry
                ax.scatter(xs_arr, ys_arr, c=colours_arr, cmap=cmap,
                           alpha=0.55, s=18, linewidths=0)

                try:
                    coeffs = np.polyfit(xs_arr, ys_arr, 1)
                    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 100)
                    ax.plot(x_line, np.polyval(coeffs, x_line),
                            color="crimson", linewidth=1.5, alpha=0.85)
                except Exception:
                    pass

                if ylims_override and metric in ylims_override:
                    ax.set_ylim(*ylims_override[metric])

                s      = stats_out[probe].get(metric, {})
                r_val  = s.get("pearson_r",    float("nan"))
                rho    = s.get("spearman_rho", float("nan"))
                r2     = s.get("r_squared",    float("nan"))
                r_tag   = f"{r_val:+.2f}" if r_val  == r_val  else "n/a"
                rho_tag = f"{rho:+.2f}"   if rho    == rho    else "n/a"
                r2_tag  = f"{r2:.2f}"     if r2     == r2     else "n/a"

                ax.set_xlabel("Mean episodic return", fontsize=8)
                ax.set_ylabel(METRIC_TITLES.get(metric, metric), fontsize=8)
                ax.set_title(
                    f"{PROBE_LABELS.get(probe, probe)}\n"
                    f"r={r_tag}  ρ={rho_tag}  R²={r2_tag}",
                    fontsize=8,
                )
                ax.tick_params(labelsize=7)
                ax.spines[["right", "top"]].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01)
        cbar.set_label("Checkpoint (early → late)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        label    = _make_experiment_label(config)
        suptitle = stem.replace("_", " ").title()
        if label:
            suptitle = f"{suptitle}\n{label}"
        fig.suptitle(suptitle, fontsize=9 if label else 11)

        out_path = out_dir / f"{stem}_h{h_idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    _draw_fig(None,           "reward_vs_metrics")
    _draw_fig(shared_ylims,   "reward_vs_metrics_normalized")

    # ------------------------------------------------------------------
    # Save stats JSON
    # ------------------------------------------------------------------
    env_name      = h_records[0].get("env_name", "")
    n_seeds_found = len({r["seed_idx"] for r in h_records})
    stats_record  = {
        "env_name":      env_name,
        "hparam_idx":    h_idx,
        "n_checkpoints": n_ckpts,
        "n_seeds":       n_seeds_found,
        "metric_keys":   metric_keys,
        "stats":         stats_out,
    }
    if config:
        stats_record["config"] = config

    stats_path = out_dir / f"reward_metric_stats_h{h_idx}.json"
    with open(stats_path, "w") as f:
        json.dump(stats_record, f, indent=2)
    print(f"  saved {stats_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_results(results_dir: Path, out_dir: Path, h_idx: int = 0,
                      config: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_all_jsons(results_dir)

    if not records:
        print(f"  no metrics JSONs found in {results_dir}")
        return

    # Load experiment config from JSON if not provided (standalone CLI usage)
    if config is None:
        config_path = results_dir / "run_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

    env_name = records[0].get("env_name", "")
    print(f"  found {len(records)} JSON records, env='{env_name}'")

    metric_keys = (
        METRIC_KEYS_ROCKSAMPLE if env_name.startswith("rocksample_")
        else METRIC_KEYS
    )

    # --- Generic metric curves (always) ---
    plot_metric_curves(records, out_dir, h_idx=h_idx, metric_keys=metric_keys,
                       config=config)

    # --- Belief sanity curves (when extras_goal_idx is in the NPZ, e.g. Marquee) ---
    plot_belief_sanity_curves(records, out_dir, h_idx=h_idx, config=config)

    # --- Episodic reward curve (always, when NPZs contain rewards) ---
    plot_reward_curves(results_dir, out_dir, h_idx=h_idx, config=config)

    # --- Reward vs belief metrics scatter + stats JSON ---
    plot_reward_vs_metrics(records, results_dir, out_dir, h_idx=h_idx,
                           config=config, metric_keys=metric_keys)

    # --- Env-specific visualizations ---
    if env_name.startswith("compass_world_"):
        _plot_compass_triangle_grids(records, results_dir, out_dir, h_idx=h_idx,
                                     config=config)

    elif env_name.startswith("marquee_"):
        _plot_marquee_belief_bars(records, out_dir, h_idx=h_idx, config=config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True, type=Path,
                   help="Directory containing checkpoint_*/h*_s*_metrics.json files.")
    p.add_argument("--out_dir",     required=True, type=Path,
                   help="Directory where figures are written.")
    p.add_argument("--h_idx",       type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_results(args.results_dir, args.out_dir, h_idx=args.h_idx)
