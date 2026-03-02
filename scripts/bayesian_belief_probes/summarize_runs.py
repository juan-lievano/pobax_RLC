"""
summarize_runs.py  –  Cross-environment summary of probe results.

Produces one figure per hyperparameter combination
(hidden_size × double_critic × action_concat × entropy_coeff × total_steps).

Each figure layout:
  Rows    = environments (compass_world_8, compass_world_10, marquee_40_16,
             rocksample_5_5, rocksample_7_8)
  Columns = metrics  (Total Variation ↓,  Mean KL (bits) ↓)

  Lines per subplot:
    Blue  solid  = recurrent MLP probe
    Blue  dashed = recurrent Linear probe
    Red   solid  = memoryless MLP probe
    Red   dashed = memoryless Linear probe
  Shaded bands = ±1 std over seeds.

Usage:
    python summarize_runs.py \\
        --results_dir probe_results/cluster_run_1 \\
        --out_dir     probe_results/cluster_run_1/summary_figures
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

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

METRICS_TO_SHOW: List[Tuple[str, str]] = [
    ("tv",           "Total Variation ↓"),
    ("mean_kl_bits", "Mean KL (bits) ↓"),
]

# Primary comparison axis: recurrent (False) vs memoryless (True)
MEMORY_COLORS = {False: "#0072B2", True: "#D55E00"}  # blue / vermilion
MEMORY_LABELS = {False: "recurrent", True: "memoryless"}

# Secondary axis: probe type encoded as line style
PROBES_TO_SHOW: List[Tuple[str, str, str]] = [
    ("mlp_from_rnn_hidden",    "MLP",    "-"),
    ("linear_from_rnn_hidden", "Linear", "--"),
]


# ---------------------------------------------------------------------------
# Hyperparameter key helpers
# ---------------------------------------------------------------------------

def _hparam_key(config: dict) -> tuple:
    return (
        config.get("hidden_size"),
        config.get("double_critic"),
        config.get("action_concat"),
        config.get("entropy_coeff"),
        config.get("total_steps"),
    )


def _hparam_label(key: tuple) -> str:
    hidden, dc, ac, ent, steps = key
    dc_str = "double-critic" if dc else "no-dc"
    ac_str = "action-concat" if ac else "no-ac"
    ts_str = f"{steps / 1e6:.1f}M" if steps else "?"
    return f"h={hidden}  {dc_str}  {ac_str}  ent={ent}  steps={ts_str}"


def _hparam_filename(key: tuple) -> str:
    hidden, dc, ac, ent, steps = key
    dc_str  = "dc"    if dc else "nodc"
    ac_str  = "ac"    if ac else "noac"
    ent_str = str(ent).replace(".", "p")
    ts_str  = f"{int(steps // 1e6)}M" if steps else "?"
    return f"h{hidden}_{dc_str}_{ac_str}_ent{ent_str}_ts{ts_str}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run_config(run_dir: Path) -> Optional[dict]:
    p = run_dir / "run_config.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_metrics(run_dir: Path) -> List[dict]:
    records = []
    for p in sorted(run_dir.rglob("*_metrics.json")):
        try:
            with open(p) as f:
                records.append(json.load(f))
        except Exception:
            pass
    return records


def _aggregate(
    records: List[dict], probe: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate metric over seeds per checkpoint.

    Returns (xs, means, stds) where xs = sorted checkpoint indices.
    """
    by_ckpt: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        val = r.get("metrics", {}).get(probe, {}).get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        by_ckpt[r["checkpoint_idx"]].append(float(val))

    if not by_ckpt:
        return np.array([]), np.array([]), np.array([])

    xs    = np.array(sorted(by_ckpt))
    means = np.array([float(np.mean(by_ckpt[c])) for c in xs])
    stds  = np.array([
        float(np.std(by_ckpt[c])) if len(by_ckpt[c]) > 1 else 0.0
        for c in xs
    ])
    return xs, means, stds


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_figure(
    env_data: Dict[str, Dict[bool, List[dict]]],
    hparam_key: tuple,
    out_dir: Path,
) -> None:
    """Produce and save one summary figure.

    Args:
        env_data: {env_name: {is_memoryless: [records]}}
        hparam_key: tuple identifying this hyperparameter combination
        out_dir: where to save the figure
    """
    n_envs    = len(ENV_ORDER)
    n_metrics = len(METRICS_TO_SHOW)

    fig, axes = plt.subplots(
        n_envs, n_metrics,
        figsize=(5.5 * n_metrics, 3.0 * n_envs),
        constrained_layout=True,
    )
    # Always 2-D axes array
    if n_envs == 1:
        axes = axes[np.newaxis, :]
    if n_metrics == 1:
        axes = axes[:, np.newaxis]

    for row, env_name in enumerate(ENV_ORDER):
        env_runs = env_data.get(env_name, {})

        for col, (metric_key, metric_title) in enumerate(METRICS_TO_SHOW):
            ax = axes[row, col]
            ax.spines[["right", "top"]].set_visible(False)
            ax.tick_params(labelsize=7)

            has_any = False
            for is_ml in [False, True]:
                records = env_runs.get(is_ml)
                if not records:
                    continue
                color = MEMORY_COLORS[is_ml]
                for probe_key, _, linestyle in PROBES_TO_SHOW:
                    xs, means, stds = _aggregate(records, probe_key, metric_key)
                    if len(xs) == 0:
                        continue
                    lw = 2.0 if linestyle == "-" else 1.5
                    ax.plot(xs, means, color=color, linestyle=linestyle,
                            linewidth=lw)
                    ax.fill_between(xs, means - stds, means + stds,
                                    color=color, alpha=0.12)
                    has_any = True

            if not has_any:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray",
                        style="italic")

            if col == 0:
                ax.set_ylabel(ENV_LABELS.get(env_name, env_name),
                              fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(metric_title, fontsize=10, fontweight="bold",
                             pad=6)
            if row == n_envs - 1:
                ax.set_xlabel("Checkpoint", fontsize=8)

    # Shared legend
    legend_handles = []
    for is_ml in [False, True]:
        color = MEMORY_COLORS[is_ml]
        mem_label = MEMORY_LABELS[is_ml]
        for _, probe_label, linestyle in PROBES_TO_SHOW:
            lw = 2.0 if linestyle == "-" else 1.5
            legend_handles.append(
                mlines.Line2D(
                    [], [], color=color, linestyle=linestyle, linewidth=lw,
                    label=f"{probe_label} – {mem_label}",
                )
            )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_handles),
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(_hparam_label(hparam_key), fontsize=11, y=1.01,
                 color="dimgray", style="italic")

    out_path = out_dir / (_hparam_filename(hparam_key) + ".png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "run_config.json").exists()
    ])
    print(f"Found {len(run_dirs)} run directories under {results_dir}")

    # Build: {hparam_key: {env_name: {is_ml: records}}}
    grouped: Dict[tuple, Dict[str, Dict[bool, List[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run_dir in run_dirs:
        config = _load_run_config(run_dir)
        if config is None:
            continue
        key   = _hparam_key(config)
        env   = config.get("env_name", "unknown")
        is_ml = bool(config.get("memoryless", False))
        records = _load_metrics(run_dir)
        if records:
            grouped[key][env][is_ml].extend(records)
        else:
            print(f"  [warn] no metrics found in {run_dir.name}")

    print(f"Found {len(grouped)} unique hyperparameter sets\n")

    for hparam_key in sorted(grouped):
        env_data = grouped[hparam_key]
        present_envs = sorted(env_data)
        print(f"Plotting: {_hparam_label(hparam_key)}")
        print(f"  envs with data: {present_envs}")
        _make_figure(env_data, hparam_key, out_dir)

    print(f"\nDone. {len(grouped)} figures saved to {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results_dir", required=True, type=Path,
        help="Root directory containing all run subdirectories "
             "(each with run_config.json and checkpoint_*/).",
    )
    p.add_argument(
        "--out_dir", required=True, type=Path,
        help="Directory where summary figures are written.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.results_dir, args.out_dir)
