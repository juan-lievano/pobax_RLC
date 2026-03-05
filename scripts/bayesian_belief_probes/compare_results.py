"""Compare probe results from multiple experiments on the same plots.

Each experiment is a probe_results directory produced by run_probe_pipeline.py,
or all experiments can be loaded from an exported JSON (produced by export_results.py).

Results are grouped by environment; one figure per env.  Each figure shows
the MLP probe belief metrics + episodic reward over training progress.

Usage:
    python scripts/bayesian_belief_probes/compare_results.py \\
        --dirs probe_results/ppo_run1 probe_results/dqn_run2 \\
        --labels "PPO recurrent" "DRQN" \\
        --out_dir compare_figures/rocksample_vs_compass

    # auto-label from run_config.json (omit --labels):
    python scripts/bayesian_belief_probes/compare_results.py \\
        --dirs probe_results/ppo_run1 probe_results/dqn_run2

    # load from exported JSON (after downloading from server):
    python scripts/bayesian_belief_probes/compare_results.py \\
        --from_json export.json \\
        --labels "PPO rec" "DRQN" \\
        --out_dir figures/
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse metric definitions from visualize.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from visualize import (
    METRIC_KEYS, METRIC_KEYS_ROCKSAMPLE, METRIC_TITLES,
    aggregate, _load_rewards_by_ckpt,
)


# ── colours ──────────────────────────────────────────────────────────────────

# Okabe-Ito palette
_OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

# Probe shown in the comparison (best single probe)
_COMPARE_PROBE = "mlp_from_rnn_hidden"


# ── data loading ─────────────────────────────────────────────────────────────

def _auto_label(config: dict) -> str:
    """Build a short label from run_config.json."""
    algo = config.get("algo", "ppo")
    h    = config.get("hidden_size", "?")
    if algo == "dqn":
        mode = "DQN" if config.get("memoryless") else "DRQN"
        lr   = config.get("lr", "?")
        tr   = config.get("trace_length", "?")
        return f"{mode} h={h} lr={lr} tr={tr}"
    else:
        ml  = "ML" if config.get("memoryless") else "rec"
        ent = config.get("entropy_coeff", "?")
        return f"PPO-{ml} h={h} ent={ent}"


def load_experiment(probe_dir: Path, label: str | None) -> dict:
    config = {}
    config_path = probe_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    records = []
    for p in sorted(probe_dir.rglob("*_metrics.json")):
        with open(p) as f:
            records.append(json.load(f))

    env_name = config.get("env_name") or (records[0].get("env_name", "?") if records else "?")
    return {
        "label":    label or _auto_label(config),
        "config":   config,
        "records":  records,
        "env_name": str(env_name),
        "dir":      probe_dir,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _metric_curve(exp: dict, grouped: dict, metric: str, ckpt_indices: list,
                  color: str, ls: str, ax, show_label: bool):
    """Add one line to ax for the MLP probe metric."""
    xs, means, stds = [], [], []
    for c in ckpt_indices:
        vals = grouped.get((0, c, _COMPARE_PROBE, metric), [])
        clean = [v for v in vals if v == v]
        if not clean:
            continue
        xs.append(c / max(ckpt_indices[-1], 1))   # normalize to [0,1]
        means.append(float(np.mean(clean)))
        stds.append(float(np.std(clean)) if len(clean) > 1 else 0.0)
    if not xs:
        return
    xs_a = np.array(xs)
    m_a  = np.array(means)
    s_a  = np.array(stds)
    ax.plot(xs_a, m_a, color=color, linestyle=ls, linewidth=2,
            label=exp["label"] if show_label else None)
    ax.fill_between(xs_a, m_a - s_a, m_a + s_a, color=color, alpha=0.15)


def _reward_curve(exp: dict, color: str, ls: str, ax, show_label: bool):
    """Add mean episodic return curve to ax."""
    # Use preloaded rewards (from --from_json mode) or load from disk
    if exp.get("rewards_by_ckpt") is not None:
        rewards_by_ckpt = exp["rewards_by_ckpt"]
    elif exp.get("dir") is not None:
        rewards_by_ckpt = _load_rewards_by_ckpt(exp["dir"], h_idx=0)
    else:
        rewards_by_ckpt = {}
    if not rewards_by_ckpt:
        return
    xs_raw = sorted(rewards_by_ckpt)
    x_max  = max(xs_raw)
    xs = [c / max(x_max, 1) for c in xs_raw]
    means = [float(rewards_by_ckpt[c].mean()) for c in xs_raw]
    stds  = [float(rewards_by_ckpt[c].std()) if len(rewards_by_ckpt[c]) > 1 else 0.0
             for c in xs_raw]
    xs_a = np.array(xs)
    m_a  = np.array(means)
    s_a  = np.array(stds)
    ax.plot(xs_a, m_a, color=color, linestyle=ls, linewidth=2,
            label=exp["label"] if show_label else None)
    ax.fill_between(xs_a, m_a - s_a, m_a + s_a, color=color, alpha=0.15)


def plot_comparison_for_env(env_name: str, experiments: list, out_dir: Path):
    metric_keys = METRIC_KEYS_ROCKSAMPLE if env_name.startswith("rocksample_") else METRIC_KEYS
    all_metrics = metric_keys + ["reward"]
    n_cols = min(3, len(all_metrics))
    n_rows = (len(all_metrics) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    axes_flat = np.array(axes).ravel()
    for ax in axes_flat[len(all_metrics):]:
        ax.axis("off")

    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for exp_idx, exp in enumerate(experiments):
        color = _OKABE[exp_idx % len(_OKABE)]
        ls    = linestyles[exp_idx % len(linestyles)]
        records = exp["records"]
        if not records:
            continue

        grouped      = aggregate(records)
        ckpt_indices = sorted({r["checkpoint_idx"] for r in records})
        if not ckpt_indices:
            continue

        show_label = True   # only need label once per experiment for the legend

        for ax_idx, metric in enumerate(all_metrics):
            ax = axes_flat[ax_idx]
            if metric == "reward":
                _reward_curve(exp, color, ls, ax, show_label)
            else:
                _metric_curve(exp, grouped, metric, ckpt_indices, color, ls, ax, show_label)
            show_label = False   # avoid duplicate legend entries

    # Axis labels and titles
    for ax_idx, metric in enumerate(all_metrics):
        ax = axes_flat[ax_idx]
        ax.set_xlabel("Training progress", fontsize=9)
        ax.set_xlim(0, 1)
        title = "Episodic Return ↑" if metric == "reward" else METRIC_TITLES.get(metric, metric)
        ax.set_title(title, fontsize=10)
        ax.spines[["right", "top"]].set_visible(False)

    # Single shared legend
    handles, labels = [], []
    for ax in axes_flat[:len(all_metrics)]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04 * n_rows),
                   ncol=min(len(handles), 4), fontsize=9)

    fig.suptitle(f"Belief Probe Comparison — {env_name}  (MLP probe)", fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_{env_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── JSON loading (from export_results.py output) ─────────────────────────────

def _load_json_file(path: Path) -> dict:
    """Load a plain or gzip-compressed JSON file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_experiments_from_json(json_path: Path, labels: list | None) -> list:
    """Build experiments list from an export_results.py output file.

    The export JSON schema (from export_results.py):
      { "runs": [ { "run_id", "config", "checkpoints": {
          "<ckpt_idx>": { "seeds": {
              "<seed_idx>": { "metrics": {...}, "ep_returns": [...] }
          }}
      }} ] }
    """
    data = _load_json_file(json_path)
    runs = data.get("runs", data) if isinstance(data, dict) else data

    experiments = []
    for i, run in enumerate(runs):
        config = run.get("config", {})
        records = []
        rewards_by_ckpt: dict = {}

        for ckpt_str, ckpt_data in run.get("checkpoints", {}).items():
            ckpt_idx = int(ckpt_str)
            seeds_data = ckpt_data.get("seeds", {})
            seed_means = []

            for seed_str, seed_data in sorted(seeds_data.items(), key=lambda x: int(x[0])):
                seed_idx = int(seed_str)
                records.append({
                    "hparam_idx":    0,
                    "checkpoint_idx": ckpt_idx,
                    "seed_idx":      seed_idx,
                    "metrics":       seed_data.get("metrics", {}),
                })
                mean_ret = seed_data.get("mean_ep_return")
                if mean_ret is not None:
                    seed_means.append(float(mean_ret))

            if seed_means:
                rewards_by_ckpt[ckpt_idx] = np.array(seed_means)

        env_name = config.get("env_name") or (
            records[0].get("env_name", "?") if records else "?"
        )
        label = (labels[i] if labels and i < len(labels) else None) or _auto_label(config)
        experiments.append({
            "label":           label,
            "config":          config,
            "records":         records,
            "env_name":        str(env_name),
            "dir":             None,
            "rewards_by_ckpt": rewards_by_ckpt,
        })

    return experiments


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dirs", nargs="+", type=Path,
                        help="Probe results directories to compare.")
    source.add_argument("--from_json", type=Path, metavar="FILE",
                        help="Load all experiments from an exported JSON (or .json.gz) "
                             "produced by export_results.py.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Labels for each experiment (auto-generated if omitted).")
    p.add_argument("--out_dir", type=Path, default=Path("compare_figures"),
                   help="Output directory for comparison figures (default: compare_figures/).")
    p.add_argument("--envs", nargs="*", default=None,
                   help="Only compare these environments (default: all found).")
    args = p.parse_args()

    if args.from_json:
        experiments = load_experiments_from_json(args.from_json, args.labels)
    else:
        if args.labels and len(args.labels) != len(args.dirs):
            p.error("--labels must have the same number of entries as --dirs")
        labels = args.labels or [None] * len(args.dirs)
        experiments = [load_experiment(Path(d), lbl)
                       for d, lbl in zip(args.dirs, labels)]

    # Group by env
    by_env: dict = defaultdict(list)
    for exp in experiments:
        by_env[exp["env_name"]].append(exp)

    if args.envs:
        by_env = {k: v for k, v in by_env.items() if k in args.envs}

    if not by_env:
        print("No matching environments found. Check --dirs / --from_json and --envs.")
        return

    print(f"Comparing {len(experiments)} experiment(s) across {len(by_env)} env(s):")
    for env_name, exps in sorted(by_env.items()):
        print(f"  {env_name}: {[e['label'] for e in exps]}")
        plot_comparison_for_env(env_name, exps, args.out_dir)

    print(f"\nDone. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()
