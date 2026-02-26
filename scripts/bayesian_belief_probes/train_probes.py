"""
Step 2 of the Bayesian belief probe pipeline.

Fully generic: reads an NPZ produced by sample_trajectories.py, trains
MLP and linear probes (rnn_hidden → flat belief vector) using JAX/Optax
with KL-divergence loss, evaluates 6 metrics, and writes a JSON.

The script knows nothing about the env — it only sees hidden_size and
belief_dim (inferred from array shapes).

Usage (standalone):
    python train_probes.py --npz /tmp/trajs.npz --seed_idx 0 \\
        --checkpoint_idx 3 --hparam_idx 0 --out /tmp/metrics.json

Output JSON structure:
    {
      "checkpoint_idx": int,
      "hparam_idx": int,
      "seed_idx": int,
      "env_name": str,
      "hidden_size": int,
      "belief_dim": int,
      "metrics": {
          "mlp_from_rnn_hidden":       { "tv": float, "mean_kl_bits": float, ... },
          "linear_from_rnn_hidden":    { ... },
          "trained_constant_kl":       { ... },
          "analytic_mean_constant_kl": { ... }
      },
      "mean_pred_belief": {
          "mlp_from_rnn_hidden":       [float, ...],  # shape [belief_dim]
          ...
      }
    }
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# JAX / Optax probe models
# ---------------------------------------------------------------------------

def _init_mlp(rng, in_dim: int, hidden_layers, out_dim: int):
    sizes = [in_dim] + list(hidden_layers) + [out_dim]
    params = []
    for i in range(len(sizes) - 1):
        rng, k = jax.random.split(rng)
        fan_in = sizes[i]
        w = jax.random.normal(k, (fan_in, sizes[i + 1])) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((sizes[i + 1],))
        params.append({"w": w, "b": b})
    return params


def _mlp_forward(params, x):
    for i, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def _init_linear(rng, in_dim: int, out_dim: int):
    rng, k = jax.random.split(rng)
    w = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(1.0 / in_dim)
    b = jnp.zeros((out_dim,))
    return {"w": w, "b": b}


def _linear_forward(params, x):
    return x @ params["w"] + params["b"]


def _init_constant_logits(out_dim: int):
    return {"logits": jnp.zeros((out_dim,))}


def _constant_forward(params, x):
    return jnp.broadcast_to(params["logits"], (x.shape[0], params["logits"].shape[0]))


def _train_kl(init_params, apply_fn, X_train, Y_train_dist,
               lr=1e-3, weight_decay=0.0, epochs=80, batch_size=1024, eps=1e-12):
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train_dist = np.asarray(Y_train_dist, dtype=np.float32)
    n = X_train.shape[0]
    if n == 0:
        return init_params

    opt = optax.adamw(lr, weight_decay=weight_decay)
    params = init_params
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s, x, y):
        def loss_fn(params):
            logits = apply_fn(params, x)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            y_c = jnp.clip(y, eps, 1.0)
            y_n = y_c / jnp.sum(y_c, axis=-1, keepdims=True)
            return jnp.mean(jnp.sum(y_n * (jnp.log(y_n) - log_probs), axis=-1))
        grads = jax.grad(loss_fn)(p)
        updates, new_s = opt.update(grads, s, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_s

    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            params, opt_state = step(
                params, opt_state,
                jnp.asarray(X_train[idx]),
                jnp.asarray(Y_train_dist[idx]),
            )
    return params


def _predict(apply_fn, params, X_std, input_mean, input_std, eps=1e-12):
    X = (np.asarray(X_std, dtype=np.float32) - input_mean) / input_std
    logits = apply_fn(params, jnp.asarray(X))
    probs = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float64)
    probs = np.clip(probs, eps, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _evaluate_metrics(Y_pred, Y_true, tol=1e-9, eps=1e-12):
    Y_pred = np.clip(Y_pred, eps, None);  Y_pred /= Y_pred.sum(axis=1, keepdims=True)
    Y_true = np.clip(Y_true, eps, None);  Y_true /= Y_true.sum(axis=1, keepdims=True)

    tv = float(np.mean(0.5 * np.abs(Y_true - Y_pred).sum(axis=1)))

    # KL in bits: sum p * log2(p/q)
    kl = np.sum(Y_true * (np.log2(Y_true) - np.log2(Y_pred)), axis=1)
    mean_kl_bits = float(np.mean(kl))

    N = Y_true.shape[0]
    true_max = Y_true.max(axis=1)
    true_arg = Y_true.argmax(axis=1)
    knows = true_max >= (1.0 - tol)
    k_count = int(knows.sum())
    k_frac  = float(knows.mean()) if N > 0 else 0.0

    if k_count > 0:
        pred_arg = Y_pred.argmax(axis=1)
        argmax_match = float(np.mean(pred_arg[knows] == true_arg[knows]))
        mean_prob_true = float(np.mean(Y_pred[knows, true_arg[knows]]))
    else:
        argmax_match = float("nan")
        mean_prob_true = float("nan")

    impossible = Y_true <= tol
    imp_overall = float(np.mean(np.sum(Y_pred * impossible, axis=1)))
    if k_count > 0:
        imp_sk = float(np.mean(np.sum(Y_pred[knows] * impossible[knows], axis=1)))
    else:
        imp_sk = float("nan")

    return {
        "tv":                           tv,
        "mean_kl_bits":                 mean_kl_bits,
        "should_know_frac":             k_frac,
        "argmax_match_rate":            argmax_match,
        "mean_prob_on_true_location":   mean_prob_true,
        "impossible_mass_overall":      imp_overall,
        "impossible_mass_should_know":  imp_sk,
    }


# ---------------------------------------------------------------------------
# Train/test split helpers
# ---------------------------------------------------------------------------

def _split_by_trajectory(n_traj: int, max_len: int, masks: np.ndarray,
                          test_size=0.2, seed=0):
    """
    Returns boolean train/test masks over the valid rows
    (indices into hidden[masks] / beliefs[masks]).
    Split is trajectory-level to avoid data leakage.
    """
    traj_ids_all = np.repeat(np.arange(n_traj), max_len)
    valid_mask_flat = masks.reshape(-1)
    valid_traj_ids  = traj_ids_all[valid_mask_flat]

    unique = np.unique(valid_traj_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    split = int(round(len(unique) * (1.0 - test_size)))
    train_set = set(unique[:split].tolist())

    train_mask = np.array([t in train_set for t in valid_traj_ids])
    test_mask  = ~train_mask
    return train_mask, test_mask


def _standardize(X_train, X_test, eps=1e-12):
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std  = X_train.std(axis=0,  keepdims=True).astype(np.float32)
    std  = np.where(std < eps, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean[0], std[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_save(
    *,
    npz: Path,
    seed_idx: int,
    checkpoint_idx: int,
    hparam_idx: int,
    out: Path,
    # Optional probe hyper-params
    mlp_hidden_layers=(128,),
    mlp_lr=1e-3,
    mlp_weight_decay=1e-4,
    linear_lr=1e-2,
    linear_weight_decay=1.0,
    epochs=80,
    batch_size=1024,
    test_size=0.2,
    tol=1e-9,
    eps=1e-12,
    # Step-budget control: if set, subsample training rows to this many steps.
    # Useful for comparing probes across checkpoints where episode length varies.
    # The trajectory-level train/test split is done first, then rows are subsampled
    # within the train split so there is no data leakage.
    max_train_steps: Optional[int] = None,
):
    data = np.load(npz, allow_pickle=False)

    # Slice out this seed
    hidden  = data["hidden"][seed_idx]    # [n_traj, max_len, H]
    beliefs = data["beliefs"][seed_idx]   # [n_traj, max_len, belief_dim]
    masks   = data["masks"][seed_idx]     # [n_traj, max_len]

    # Read env_name from metadata stored in the NPZ (written by sample_trajectories)
    # env_name is stored as a scalar string array; fall back to "" if missing.
    env_name = str(data.get("env_name", np.array("")))

    n_traj, max_len, hidden_size = hidden.shape
    belief_dim = beliefs.shape[-1]

    n_valid = int(masks.sum())
    print(
        f"  seed {seed_idx}: n_traj={n_traj}, hidden_size={hidden_size}, "
        f"belief_dim={belief_dim}, valid_steps={n_valid}"
        + (f" (capped to max_train_steps={max_train_steps})" if max_train_steps else "")
    )

    # Extract valid rows (trajectory-level split)
    X_all = hidden[masks]    # [n_valid, H]
    Y_all = beliefs[masks]   # [n_valid, belief_dim]

    train_mask, test_mask = _split_by_trajectory(n_traj, max_len, masks, test_size, seed=0)
    X_train, X_test = X_all[train_mask], X_all[test_mask]
    Y_train, Y_test = Y_all[train_mask], Y_all[test_mask]

    # Subsample training rows to at most max_train_steps to equalize supervision
    # across checkpoints (episode lengths vary as the agent improves).
    if max_train_steps is not None and len(X_train) > max_train_steps:
        sub_rng = np.random.default_rng(0)
        idx = sub_rng.choice(len(X_train), max_train_steps, replace=False)
        idx.sort()
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        print(f"    subsampled train set: {len(idx)} / {train_mask.sum()} rows")

    X_train_std, X_test_std, h_mean, h_std = _standardize(X_train, X_test, eps)

    # Normalized train distribution for KL loss
    Y_train_dist = np.clip(Y_train, eps, None)
    Y_train_dist /= Y_train_dist.sum(axis=1, keepdims=True)

    # Analytic mean constant: mean of training beliefs
    kl_opt_constant = Y_train_dist.mean(axis=0)

    rng = jax.random.PRNGKey(0)
    out_dim = belief_dim
    in_dim  = hidden_size

    print("    training MLP probe...", flush=True)
    rng, k = jax.random.split(rng)
    mlp_params = _init_mlp(k, in_dim, mlp_hidden_layers, out_dim)
    mlp_params = _train_kl(mlp_params, _mlp_forward, X_train_std, Y_train_dist,
                            lr=mlp_lr, weight_decay=mlp_weight_decay,
                            epochs=epochs, batch_size=batch_size, eps=eps)

    print("    training linear probe...", flush=True)
    rng, k = jax.random.split(rng)
    lin_params = _init_linear(k, in_dim, out_dim)
    lin_params = _train_kl(lin_params, _linear_forward, X_train_std, Y_train_dist,
                            lr=linear_lr, weight_decay=linear_weight_decay,
                            epochs=epochs, batch_size=batch_size, eps=eps)

    print("    training constant KL probe...", flush=True)
    rng, k = jax.random.split(rng)
    dummy_X = np.zeros((X_train_std.shape[0], 1), dtype=np.float32)
    const_params = _init_constant_logits(out_dim)
    const_params = _train_kl(const_params, _constant_forward, dummy_X, Y_train_dist,
                              lr=1e-2, weight_decay=0.0,
                              epochs=epochs, batch_size=batch_size, eps=eps)
    const_probs = np.asarray(jax.nn.softmax(const_params["logits"]), dtype=np.float64)
    const_probs = np.clip(const_probs, eps, None)
    const_probs /= const_probs.sum()

    # Predictions on test set
    def _pred_mlp(X):
        return _predict(_mlp_forward, mlp_params, X, h_mean, h_std, eps)
    def _pred_lin(X):
        return _predict(_linear_forward, lin_params, X, h_mean, h_std, eps)
    def _pred_const(X):
        return np.tile(const_probs, (X.shape[0], 1))
    def _pred_kl_opt(X):
        p = np.clip(kl_opt_constant, eps, None); p /= p.sum()
        return np.tile(p, (X.shape[0], 1))

    print("    evaluating metrics...", flush=True)
    metrics = {
        "mlp_from_rnn_hidden":       _evaluate_metrics(_pred_mlp(X_test_std),  Y_test, tol, eps),
        "linear_from_rnn_hidden":    _evaluate_metrics(_pred_lin(X_test_std),  Y_test, tol, eps),
        "trained_constant_kl":       _evaluate_metrics(_pred_const(X_test_std), Y_test, tol, eps),
        "analytic_mean_constant_kl": _evaluate_metrics(_pred_kl_opt(X_test_std), Y_test, tol, eps),
    }

    # Mean predicted belief on the test set (for triangle grids in visualize.py)
    mean_pred_belief = {
        "mlp_from_rnn_hidden":       _pred_mlp(X_test_std).mean(axis=0).tolist(),
        "linear_from_rnn_hidden":    _pred_lin(X_test_std).mean(axis=0).tolist(),
        "trained_constant_kl":       const_probs.tolist(),
        "analytic_mean_constant_kl": (kl_opt_constant / kl_opt_constant.sum()).tolist(),
    }

    result = {
        "checkpoint_idx":  checkpoint_idx,
        "hparam_idx":      hparam_idx,
        "seed_idx":        seed_idx,
        "env_name":        env_name,
        "hidden_size":     hidden_size,
        "belief_dim":      belief_dim,
        "metrics":         metrics,
        "mean_pred_belief": mean_pred_belief,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz",            required=True, type=Path)
    p.add_argument("--seed_idx",       required=True, type=int)
    p.add_argument("--checkpoint_idx", required=True, type=int)
    p.add_argument("--hparam_idx",     required=True, type=int)
    p.add_argument("--out",            required=True, type=Path)
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--batch_size",     type=int,   default=1024)
    p.add_argument("--mlp_lr",         type=float, default=1e-3)
    p.add_argument("--mlp_wd",         type=float, default=1e-4)
    p.add_argument("--linear_lr",      type=float, default=1e-2)
    p.add_argument("--linear_wd",      type=float, default=1.0)
    p.add_argument("--n_steps",        type=int,   default=None,
                   help="Cap training rows to this many timesteps (default: use all).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_save(
        npz=args.npz,
        seed_idx=args.seed_idx,
        checkpoint_idx=args.checkpoint_idx,
        hparam_idx=args.hparam_idx,
        out=args.out,
        mlp_lr=args.mlp_lr,
        mlp_weight_decay=args.mlp_wd,
        linear_lr=args.linear_lr,
        linear_weight_decay=args.linear_wd,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_steps=args.n_steps,
    )
