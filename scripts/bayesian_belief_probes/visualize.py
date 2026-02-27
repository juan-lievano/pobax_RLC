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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from envs import get_env_handler


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

PROBE_ORDER = [
    "mlp_from_rnn_hidden",
    "linear_from_rnn_hidden",
    "trained_constant_kl",
    "analytic_mean_constant_kl",
]

PROBE_LABELS = {
    "mlp_from_rnn_hidden":       "MLP (RNN hidden)",
    "linear_from_rnn_hidden":    "Linear (RNN hidden)",
    "trained_constant_kl":       "Trained Constant (KL)",
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


# ---------------------------------------------------------------------------
# Metric curves
# ---------------------------------------------------------------------------

def plot_metric_curves(records: List[dict], out_dir: Path, h_idx: int = 0):
    grouped = aggregate(records)

    ckpt_indices = sorted({r["checkpoint_idx"] for r in records})
    probes = [p for p in PROBE_ORDER if any((h_idx, c, p, "tv") in grouped for c in ckpt_indices)]
    if not probes:
        print("  no probe data found — skipping metric curves")
        return

    hparams = sorted({r["hparam_idx"] for r in records})

    for h in hparams:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        axes_flat = axes.ravel()

        for ax, metric in zip(axes_flat, METRIC_KEYS):
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

        suffix = f"_h{h}" if len(hparams) > 1 else ""
        out_path = out_dir / f"metric_curves{suffix}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# CompassWorld triangle grids
# ---------------------------------------------------------------------------

def _plot_compass_triangle_grids(
    records: List[dict], results_dir: Path, out_dir: Path, h_idx: int = 0
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

    fig.suptitle(f"Mean Predicted Belief — {env_name}", fontsize=13)

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
    records: List[dict], out_dir: Path, h_idx: int = 0
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

    fig.suptitle(f"Mean Predicted Belief — {env_name}", fontsize=13)

    hparams = sorted({r["hparam_idx"] for r in records})
    suffix = f"_h{h_idx}" if len(hparams) > 1 else ""
    out_path = out_dir / f"belief_bars{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_results(results_dir: Path, out_dir: Path, h_idx: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_all_jsons(results_dir)

    if not records:
        print(f"  no metrics JSONs found in {results_dir}")
        return

    env_name = records[0].get("env_name", "")
    print(f"  found {len(records)} JSON records, env='{env_name}'")

    # --- Generic metric curves (always) ---
    plot_metric_curves(records, out_dir, h_idx=h_idx)

    # --- Env-specific visualizations ---
    if env_name.startswith("compass_world_"):
        _plot_compass_triangle_grids(records, results_dir, out_dir, h_idx=h_idx)

    elif env_name.startswith("marquee_"):
        _plot_marquee_belief_bars(records, out_dir, h_idx=h_idx)


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
