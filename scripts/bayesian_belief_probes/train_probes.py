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
          "analytic_mean_constant_kl": { ... }
      },
      "mean_pred_belief": {
          "mlp_from_rnn_hidden":       [float, ...],  # shape [belief_dim]
          ...
      },
      "belief_sanity": {                        # only present when extras_goal_idx available
          "final_argmax_correct":    float,    # fraction of episodes where belief[-1] argmax = true goal
          "final_mean_prob_true":    float,    # avg P(true_goal) at the last step of each episode
          "all_steps_mean_prob_true": float,   # avg P(true_goal) across all valid steps
      }
    }

belief_sanity interpretation:
  final_argmax_correct ≈ 1.0  -> belief computation is correct; probe failure = hidden state issue
  final_argmax_correct ≈ 1/n_goals -> belief computation is broken
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



def _make_bce_loss(apply_fn):
    def loss_fn(params, x, y):
        return jnp.mean(optax.sigmoid_binary_cross_entropy(apply_fn(params, x), y))
    return loss_fn


def _make_kl_loss(apply_fn, eps=1e-12):
    def loss_fn(params, x, y):
        logits = apply_fn(params, x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        y_c = jnp.clip(y, eps, 1.0)
        y_n = y_c / jnp.sum(y_c, axis=-1, keepdims=True)
        return jnp.mean(jnp.sum(y_n * (jnp.log(y_n) - log_probs), axis=-1))
    return loss_fn


def _run_training(init_params, loss_fn, X_train, Y_train,
                  lr=1e-3, weight_decay=0.0, epochs=80, batch_size=1024):
    """Training loop optimised for GPU (and transparent on CPU).

    Data is loaded to device once to avoid per-batch host→device transfers.
    The inner batch loop is replaced with lax.scan to eliminate Python
    dispatch overhead per batch. The outer epoch loop stays in Python (80
    iterations, negligible overhead).
    """
    n = len(X_train)
    if n == 0:
        return init_params

    X_dev = jnp.asarray(X_train, dtype=jnp.float32)
    Y_dev = jnp.asarray(Y_train, dtype=jnp.float32)
    # If n < batch_size there would be 0 batches and lax.scan returns untrained
    # params silently.  Use all data as a single batch instead.
    if n < batch_size:
        batch_size = n
    n_batches = n // batch_size  # last partial batch dropped (< batch_size rows)

    opt = optax.adamw(lr, weight_decay=weight_decay)
    params = init_params
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s, x, y):
        grads = jax.grad(loss_fn)(p, x, y)
        updates, new_s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s

    @jax.jit
    def train_epoch(params, opt_state, rng_key):
        perm = jax.random.permutation(rng_key, n)[:n_batches * batch_size]
        X_b = X_dev[perm].reshape(n_batches, batch_size, X_dev.shape[-1])
        Y_b = Y_dev[perm].reshape(n_batches, batch_size, Y_dev.shape[-1])
        def scan_step(carry, xy):
            return step(carry[0], carry[1], *xy), None
        (params, opt_state), _ = jax.lax.scan(scan_step, (params, opt_state), (X_b, Y_b))
        return params, opt_state

    rng = jax.random.PRNGKey(0)
    for _ in range(epochs):
        rng, k = jax.random.split(rng)
        params, opt_state = train_epoch(params, opt_state, k)
    return params


def _predict(apply_fn, params, X_std, input_mean, input_std, eps=1e-12):
    X = (np.asarray(X_std, dtype=np.float32) - input_mean) / input_std
    logits = apply_fn(params, jnp.asarray(X))
    probs = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float64)
    probs = np.clip(probs, eps, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _predict_bernoulli(apply_fn, params, X_raw, input_mean, input_std, eps=1e-12):
    """Sigmoid output for K independent Bernoulli beliefs."""
    X = (np.asarray(X_raw, dtype=np.float32) - input_mean) / input_std
    logits = apply_fn(params, jnp.asarray(X))
    probs = np.asarray(jax.nn.sigmoid(logits), dtype=np.float64)
    return np.clip(probs, eps, 1.0 - eps)


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


def _evaluate_bernoulli_metrics(Y_pred, Y_true, tol=1e-9, eps=1e-12):
    """
    Metrics for K independent Bernoulli beliefs.

    Y_pred, Y_true: [N, K]  float64 values in (0, 1).
    All per-element metrics are averaged over (N*K) elements.

    Uses the same key names as _evaluate_metrics so the visualizer
    can plot both categorical and Bernoulli runs without changes.
    """
    # Y_pred comes from sigmoid so is theoretically in (0,1), but extreme logits
    # can produce exact 0/1 in floating point. Clip to prevent log2(0) = -inf
    # when the probe is maximally wrong.  beliefs (Y_true) are float32 from the
    # NPZ; cast to float64 so subsequent log2 calls are numerically correct.
    Y_pred = np.clip(np.asarray(Y_pred, dtype=np.float64), eps, 1.0 - eps)
    Y_true = np.asarray(Y_true, dtype=np.float64)

    # TV: mean absolute error per element (= component-wise TV of Bernoulli)
    tv = float(np.mean(np.abs(Y_true - Y_pred)))

    # KL(Bern(p_true) || Bern(p_pred)) in bits per element.
    # Mathematical convention: 0 * log2(0 / q) = 0.
    # We implement this with np.where masking so log2 is never called with 0,
    # rather than clipping p_true away from 0/1 (which would distort the metric).
    p, q = Y_true, Y_pred
    kl_elem = (
        np.where(p > 0, p * (np.log2(np.where(p > 0, p, 1.0)) - np.log2(q)), 0.0)
        + np.where(p < 1, (1.0 - p) * (np.log2(np.where(p < 1, 1.0 - p, 1.0)) - np.log2(1.0 - q)), 0.0)
    )
    mean_kl_bits = float(np.mean(kl_elem))

    # "Should know": (step, rock) pairs where p_true is exactly 0 or 1.
    # This happens after sampling (p=0) or a distance-0 check (p=0 or 1).
    certain = (Y_true <= tol) | (Y_true >= 1.0 - tol)
    certain_frac = float(certain.mean())

    # Impossible mass: prob assigned to the genuinely impossible morality.
    # For certain pairs only — uncertain pairs contribute 0 (no morality is impossible).
    #   p_true = 0 (rock definitely bad):  impossible = p_pred  (prob of "good")
    #   p_true = 1 (rock definitely good): impossible = 1-p_pred (prob of "bad")
    #   0 < p_true < 1 (uncertain):        impossible = 0
    impossible_prob = np.where(
        Y_true <= tol,       Y_pred,
        np.where(Y_true >= 1.0 - tol, 1.0 - Y_pred, 0.0)
    )
    imp_overall = float(impossible_prob.mean())  # averaged over ALL (step, rock) pairs

    if certain.any():
        pred_side = Y_pred >= 0.5
        true_side = Y_true >= 0.5
        argmax_match = float(np.mean(pred_side[certain] == true_side[certain]))
        # Prob assigned to the true morality for certain pairs
        prob_correct_side = np.where(Y_true >= 0.5, Y_pred, 1.0 - Y_pred)
        mean_prob_true = float(np.mean(prob_correct_side[certain]))
        imp_sk = float(impossible_prob[certain].mean())
    else:
        argmax_match = float("nan")
        mean_prob_true = float("nan")
        imp_sk = float("nan")

    return {
        "tv":                           tv,
        "mean_kl_bits":                 mean_kl_bits,
        "should_know_frac":             certain_frac,
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
# Belief sanity check (requires extras_goal_idx in NPZ)
# ---------------------------------------------------------------------------

def _check_belief_sanity(
    beliefs: np.ndarray,    # [n_traj, max_len, belief_dim]
    goal_idx: np.ndarray,   # [n_traj, max_len]  int32
    lengths: np.ndarray,    # [n_traj]            int32
) -> dict:
    """
    Verify that the analytical belief converges to the true goal by episode end.

    Computes three scalars:
      final_argmax_correct  – fraction of episodes where beliefs[t=-1].argmax() == true_goal
      final_mean_prob_true  – average P(true_goal) at the final valid step of each episode
      all_steps_mean_prob_true – average P(true_goal) over all valid (step, traj) pairs

    High final_argmax_correct (>0.8) confirms belief computation is correct and that
    probe failure reflects the hidden state, not label noise.
    """
    n_traj = len(lengths)
    final_correct = 0
    final_prob_true: list = []
    all_prob_true: list = []

    for i in range(n_traj):
        l = int(lengths[i])
        if l <= 0:
            continue
        for t in range(l):
            tg = int(goal_idx[i, t])
            all_prob_true.append(float(beliefs[i, t, tg]))
        # Final step: last belief recorded by compute_beliefs = beliefs[l-1]
        # (posterior incorporating flips 0..l-2; see compute_beliefs docstring)
        tg_final = int(goal_idx[i, l - 1])
        p_true = float(beliefs[i, l - 1, tg_final])
        final_prob_true.append(p_true)
        if int(np.argmax(beliefs[i, l - 1])) == tg_final:
            final_correct += 1

    n_eps = len(final_prob_true)
    return {
        "final_argmax_correct":     float(final_correct / n_eps) if n_eps > 0 else float("nan"),
        "final_mean_prob_true":     float(np.mean(final_prob_true)) if final_prob_true else float("nan"),
        "all_steps_mean_prob_true": float(np.mean(all_prob_true)) if all_prob_true else float("nan"),
    }


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
    lengths = data["lengths"][seed_idx]   # [n_traj]

    # Read env_name and belief_type from NPZ metadata.
    # Fall back to "" / "categorical" for NPZs written before these keys were added.
    env_name    = str(data.get("env_name",    np.array("")))
    belief_type = str(data.get("belief_type", np.array("categorical")))
    # Backward compat: infer belief_type from env_name for NPZs missing the key.
    # TODO: remove this patch once all NPZs have been re-sampled with belief_type stored.
    if belief_type == "categorical" and env_name.startswith("rocksample_"):
        belief_type = "bernoulli"

    # Belief sanity check: does the analytical belief converge to the true goal?
    # Requires extras_goal_idx (available for Marquee, not CompassWorld).
    belief_sanity = None
    if "extras_goal_idx" in data:
        goal_idx = data["extras_goal_idx"][seed_idx]  # [n_traj, max_len]
        belief_sanity = _check_belief_sanity(beliefs, goal_idx, lengths)
        print(
            f"  belief sanity: final_argmax_correct={belief_sanity['final_argmax_correct']:.3f}  "
            f"final_mean_prob_true={belief_sanity['final_mean_prob_true']:.3f}  "
            f"all_steps_mean_prob_true={belief_sanity['all_steps_mean_prob_true']:.3f}"
        )

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

    X_train_std, _, h_mean, h_std = _standardize(X_train, X_test, eps)

    rng = jax.random.PRNGKey(0)
    out_dim = belief_dim
    in_dim  = hidden_size

    if belief_type == "bernoulli":
        # -----------------------------------------------------------------
        # Bernoulli pipeline: BCE loss + sigmoid, per-element metrics
        # -----------------------------------------------------------------
        print(f"    [bernoulli] training MLP probe...", flush=True)
        rng, k = jax.random.split(rng)
        mlp_params = _init_mlp(k, in_dim, mlp_hidden_layers, out_dim)
        mlp_params = _run_training(mlp_params, _make_bce_loss(_mlp_forward), X_train_std, Y_train,
                                   lr=mlp_lr, weight_decay=mlp_weight_decay,
                                   epochs=epochs, batch_size=batch_size)

        print(f"    [bernoulli] training linear probe...", flush=True)
        rng, k = jax.random.split(rng)
        lin_params = _init_linear(k, in_dim, out_dim)
        lin_params = _run_training(lin_params, _make_bce_loss(_linear_forward), X_train_std, Y_train,
                                   lr=linear_lr, weight_decay=linear_weight_decay,
                                   epochs=epochs, batch_size=batch_size)

        # Constant baseline: mean p_i over training set for each rock
        bce_constant = Y_train.mean(axis=0)

        print("    evaluating metrics...", flush=True)
        # Compute test predictions once; reuse for both metrics and mean_pred_belief.
        # _predict_bernoulli receives raw X_test and standardizes internally using
        # h_mean/h_std — matching the standardization seen during training.
        mlp_pred  = _predict_bernoulli(_mlp_forward,    mlp_params, X_test, h_mean, h_std, eps)
        lin_pred  = _predict_bernoulli(_linear_forward, lin_params,  X_test, h_mean, h_std, eps)
        const_pred = np.tile(bce_constant, (X_test.shape[0], 1))

        metrics = {
            "mlp_from_rnn_hidden":       _evaluate_bernoulli_metrics(mlp_pred,   Y_test, tol, eps),
            "linear_from_rnn_hidden":    _evaluate_bernoulli_metrics(lin_pred,   Y_test, tol, eps),
            "analytic_mean_constant_kl": _evaluate_bernoulli_metrics(const_pred, Y_test, tol, eps),
        }

        mean_pred_belief = {
            "mlp_from_rnn_hidden":       mlp_pred.mean(axis=0).tolist(),
            "linear_from_rnn_hidden":    lin_pred.mean(axis=0).tolist(),
            "analytic_mean_constant_kl": bce_constant.tolist(),
        }

    else:
        # -----------------------------------------------------------------
        # Categorical pipeline: KL loss + softmax
        # -----------------------------------------------------------------
        # Normalize train distribution to simplex for KL loss
        Y_train_dist = np.clip(Y_train, eps, None)
        Y_train_dist /= Y_train_dist.sum(axis=1, keepdims=True)

        # Analytic mean constant: mean of training beliefs (already on simplex)
        kl_opt_constant = Y_train_dist.mean(axis=0)

        print("    training MLP probe...", flush=True)
        rng, k = jax.random.split(rng)
        mlp_params = _init_mlp(k, in_dim, mlp_hidden_layers, out_dim)
        mlp_params = _run_training(mlp_params, _make_kl_loss(_mlp_forward, eps), X_train_std, Y_train_dist,
                                   lr=mlp_lr, weight_decay=mlp_weight_decay,
                                   epochs=epochs, batch_size=batch_size)

        print("    training linear probe...", flush=True)
        rng, k = jax.random.split(rng)
        lin_params = _init_linear(k, in_dim, out_dim)
        lin_params = _run_training(lin_params, _make_kl_loss(_linear_forward, eps), X_train_std, Y_train_dist,
                                   lr=linear_lr, weight_decay=linear_weight_decay,
                                   epochs=epochs, batch_size=batch_size)

        print("    evaluating metrics...", flush=True)
        # Compute test predictions once; reuse for both metrics and mean_pred_belief.
        # _predict receives raw X_test and standardizes internally using h_mean/h_std
        # — matching the standardization seen during training.
        kl_constant_norm = np.clip(kl_opt_constant, eps, None)
        kl_constant_norm /= kl_constant_norm.sum()
        mlp_pred   = _predict(_mlp_forward,    mlp_params, X_test, h_mean, h_std, eps)
        lin_pred   = _predict(_linear_forward, lin_params,  X_test, h_mean, h_std, eps)
        const_pred = np.tile(kl_constant_norm, (X_test.shape[0], 1))

        metrics = {
            "mlp_from_rnn_hidden":       _evaluate_metrics(mlp_pred,   Y_test, tol, eps),
            "linear_from_rnn_hidden":    _evaluate_metrics(lin_pred,   Y_test, tol, eps),
            "analytic_mean_constant_kl": _evaluate_metrics(const_pred, Y_test, tol, eps),
        }

        mean_pred_belief = {
            "mlp_from_rnn_hidden":       mlp_pred.mean(axis=0).tolist(),
            "linear_from_rnn_hidden":    lin_pred.mean(axis=0).tolist(),
            "analytic_mean_constant_kl": kl_constant_norm.tolist(),
        }

    result = {
        "checkpoint_idx":   checkpoint_idx,
        "hparam_idx":       hparam_idx,
        "seed_idx":         seed_idx,
        "env_name":         env_name,
        "hidden_size":      hidden_size,
        "belief_dim":       belief_dim,
        "metrics":          metrics,
        "mean_pred_belief": mean_pred_belief,
        # Only present when extras_goal_idx is available in the NPZ (e.g. Marquee).
        # None for envs that don't expose the true goal index (e.g. CompassWorld).
        **({"belief_sanity": belief_sanity} if belief_sanity is not None else {}),
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
