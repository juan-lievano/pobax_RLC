"""
Diagnostic: run the belief probe pipeline on an old DiscreteActorCriticRNN checkpoint.

Old checkpoints store params under "final_train_state" with a 7-D hyperparam grid
(one axis per swept argument); we index [0,0,0,0,0,0,0] to get one weight set.
The model class (DiscreteActorCriticRNN) was removed from the codebase; it is
reconstructed here verbatim from git commit 1421ada.

Produces an NPZ + probe-metrics JSON in the same format as sample_trajectories.py +
train_probes.py, so results can be placed directly alongside the current pipeline's
output and compared visually.

Usage:
    python run_old_checkpoint_probe.py \\
        --ckpt_dir notes/unireps_checkpoint/marquee/lightbulbs_40_16_seed... \\
        --out_dir /tmp/old_ckpt_probe \\
        --n_traj 500 --max_len 200 \\
        --hidden_size 64 --double_critic

    # Then visualise with the existing visualize.py if desired.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jr
from jax._src.nn.initializers import constant, orthogonal
import numpy as np
import orbax.checkpoint

# Make scripts/bayesian_belief_probes importable
sys.path.insert(0, str(Path(__file__).parent))
from envs import get_env_handler
from envs.marquee import MarqueeHandler
from train_probes import train_and_save

from pobax.models.network import ScannedRNN


# ---------------------------------------------------------------------------
# DiscreteActorCriticRNN — verbatim from git commit 1421ada
# (removed during the ActorCritic refactor; reproduced here for checkpoint compat)
# ---------------------------------------------------------------------------

class DiscreteActor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size,
                      kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim,
                          kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=logits)


class Critic(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size,
                      kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class DiscreteActorCriticRNN(nn.Module):
    """Old model class, reconstructed from archive."""
    action_dim:   int
    hidden_size:  int  = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        pi = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)(embedding)

        if self.double_critic:
            critic = nn.vmap(
                Critic,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=None,
                out_axes=2,
                axis_size=2,
            )(hidden_size=self.hidden_size)
        else:
            critic = Critic(hidden_size=self.hidden_size)

        v = critic(embedding)
        return hidden, pi, jnp.squeeze(v, axis=-1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _iter_leaves(p: Any, path: Tuple = ()) -> Iterable[Tuple[Tuple, Any]]:
    if isinstance(p, dict):
        for k, v in p.items():
            yield from _iter_leaves(v, path + (k,))
    else:
        yield path, p


def _infer_first_dense_input_dim(params, hidden_size: int) -> Optional[int]:
    """
    Find the input dim of the observation-embedding Dense layer.

    The embedding is the only Dense kernel with:
      shape[1] == hidden_size  AND  shape[0] != hidden_size
    Actor/critic hidden layers have [hidden_size → hidden_size], GRU gates are
    [hidden_size → hidden_size] (or larger output), so they don't match.
    """
    for path, leaf in _iter_leaves(params):
        if not hasattr(leaf, "shape") or len(leaf.shape) != 2:
            continue
        if (leaf.shape[1] == hidden_size
                and leaf.shape[0] != hidden_size
                and "kernel" in str(path[-1]).lower()):
            return int(leaf.shape[0])
    return None


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def build_rollout_fn(
    model,
    env,
    env_params,
    hidden_size: int,
    max_len: int,
    action_concat: bool,
    action_dim: int,
    goals_jax,   # [n_goals, n_bulbs] int32
):
    """
    Returns a JIT-compiled vmapped rollout:
        rollout(params, rng_keys) -> (obs, act, goal_idx, hidden, mask, length, reward)
    All outputs have a leading n_traj dimension.
    """

    def rollout_single(rng_key, params):
        rng, rng_reset = jr.split(rng_key)
        obs, state = env.reset_env(rng_reset, env_params)
        hidden = ScannedRNN.initialize_carry(1, hidden_size)
        done   = jnp.array(False)
        # Old checkpoints initialise prev_action to one-hot of the NO-OP (last action)
        prev_action_onehot = jax.nn.one_hot(action_dim - 1, action_dim, dtype=jnp.float32)

        def step_fn(carry, _):
            rng, state, obs, hidden, done, prev_action_onehot = carry
            rng, k_step, k_act = jr.split(rng, 3)

            if action_concat:
                obs_model = jnp.concatenate([obs, prev_action_onehot])
            else:
                obs_model = obs
            obs_in  = obs_model[None, None, :].astype(jnp.float32)  # [1, 1, dim]
            done_in = done[None, None]                                # [1, 1]

            # Old model takes (obs, done) directly — not wrapped in Observation
            hidden_new, pi, _ = model.apply(params, hidden, (obs_in, done_in))
            action = pi.sample(seed=k_act)[0, 0].astype(jnp.int32)

            hidden_rec = jnp.where(done, hidden, hidden_new)
            mask_out   = jnp.logical_not(done)

            next_obs, next_state, reward, next_done, _ = env.step_env(
                k_step, state, action, env_params
            )
            reward_rec     = jnp.where(done, jnp.zeros_like(reward), reward)
            next_done_flag = jnp.logical_or(done, next_done)
            new_prev_action = jnp.where(
                done, prev_action_onehot,
                jax.nn.one_hot(action, action_dim, dtype=jnp.float32),
            )

            # Identify which goal the env is using (for belief sanity check)
            matches  = jnp.all(state.goal[None, :].astype(jnp.int32) == goals_jax, axis=1)
            goal_idx = jnp.argmax(matches).reshape(1).astype(jnp.float32)

            next_carry = (
                rng,
                jax.tree_util.tree_map(
                    lambda a, b: jnp.where(done, a, b), state, next_state
                ),
                jnp.where(done, obs, next_obs.astype(jnp.float32)),
                hidden_rec,
                next_done_flag,
                new_prev_action,
            )
            step_out = (
                obs,              # [obs_dim]
                action,           # []
                goal_idx,         # [1]
                hidden_rec[0],    # [hidden_size]  squeeze batch dim
                mask_out,         # []
                reward_rec,       # []
            )
            return next_carry, step_out

        init_carry = (rng, state, obs.astype(jnp.float32), hidden, done, prev_action_onehot)
        _, (obs_seq, act_seq, goal_seq, hid_seq, mask_seq, rew_seq) = lax.scan(
            step_fn, init_carry, None, max_len
        )
        length = mask_seq.sum(dtype=jnp.int32)
        return obs_seq, act_seq, goal_seq, hid_seq, mask_seq, length, rew_seq

    def rollout_batch(params, rng_keys):
        return jax.vmap(rollout_single, in_axes=(0, None))(rng_keys, params)

    return jax.jit(rollout_batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    ckpt_dir: Path,
    out_dir:  Path,
    n_traj:   int,
    max_len:  int,
    seed:     int,
    hidden_size:   int,
    double_critic: bool,
    epochs:        int,
    batch_size:    int,
):
    ckpt_dir = Path(ckpt_dir).resolve()
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint ------------------------------------------------
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_dir)

    # Try to read model config from saved args (may override CLI defaults)
    args_dict = restored.get("args", {})
    if "hidden_size" in args_dict:
        hidden_size   = int(args_dict["hidden_size"])
    if "double_critic" in args_dict:
        double_critic = bool(args_dict["double_critic"])
    # action_concat: read from checkpoint if available, else use CLI value
    action_concat_from_ckpt: Optional[bool] = None
    if "action_concat" in args_dict:
        action_concat_from_ckpt = bool(args_dict["action_concat"])
    print(f"  checkpoint: hidden_size={hidden_size}, double_critic={double_critic}")

    # 7-D hyperparam grid — take [0,0,0,0,0,0,0]
    params = jax.tree.map(
        lambda x: x[0, 0, 0, 0, 0, 0, 0],
        restored["final_train_state"]["params"],
    )

    # ---- Detect action_concat -------------------------------------------
    env_name_current = "marquee_40_16"
    _handler     = get_env_handler(env_name_current)
    assert isinstance(_handler, MarqueeHandler), "This script requires a MarqueeHandler"
    handler: MarqueeHandler = _handler
    env, env_params = handler.make_raw_env()
    obs_dim    = handler.obs_dim()      # n_bulbs = 40
    action_dim = handler.action_dim()   # n_bulbs + 1 = 41

    if action_concat_from_ckpt is not None:
        action_concat = action_concat_from_ckpt
        print(f"  action_concat={action_concat}  (from checkpoint args)")
    else:
        # Infer from first Dense kernel: embedding is the only [*, hidden_size] layer
        # where input dim != hidden_size.
        inferred = _infer_first_dense_input_dim(params, hidden_size)
        if inferred == obs_dim:
            action_concat = False
        elif inferred == obs_dim + action_dim:
            action_concat = True
        else:
            raise RuntimeError(
                f"First Dense input dim = {inferred}; expected {obs_dim} (no prev_action) "
                f"or {obs_dim + action_dim} (with prev_action). "
                f"Pass --action_concat / --no_action_concat to override."
            )
        print(f"  action_concat={action_concat}  (inferred from first Dense dim={inferred})")

    # ---- Build model ----------------------------------------------------
    model = DiscreteActorCriticRNN(
        action_dim=action_dim,
        hidden_size=hidden_size,
        double_critic=double_critic,
    )
    goals_jax = jnp.asarray(handler._goals_np, dtype=jnp.int32)

    # ---- Rollout --------------------------------------------------------
    rollout_fn = build_rollout_fn(
        model, env, env_params, hidden_size, max_len,
        action_concat, action_dim, goals_jax,
    )
    rng_keys = jr.split(jr.PRNGKey(seed), n_traj)

    print(f"  running {n_traj} trajectories (max_len={max_len})...", flush=True)
    out = rollout_fn(params, rng_keys)
    obs_np, act_np, goal_np, hid_np, mask_np, len_np, rew_np = jax.device_get(out)
    # shapes: [n_traj, max_len, *]

    avg_ep = float(len_np.mean())
    print(f"  avg episode length: {avg_ep:.1f}  avg return: {(rew_np * mask_np).sum(-1).mean():.1f}")

    # ---- Compute beliefs ------------------------------------------------
    print("  computing beliefs...", flush=True)
    belief_dim = handler.belief_dim()
    beliefs = np.zeros((n_traj, max_len, belief_dim), dtype=np.float32)
    for i in range(n_traj):
        beliefs[i] = handler.compute_beliefs(obs_np[i], act_np[i], int(len_np[i]))

    # ---- Save NPZ (wrapped in a fake seeds=1 dim for train_and_save) ---
    npz_path = out_dir / "trajectories.npz"
    np.savez_compressed(
        npz_path,
        obs            = obs_np[np.newaxis].astype(np.float32),   # [1, n_traj, max_len, obs_dim]
        actions        = act_np[np.newaxis],
        hidden         = hid_np[np.newaxis].astype(np.float32),
        beliefs        = beliefs[np.newaxis],
        masks          = mask_np[np.newaxis],
        lengths        = len_np[np.newaxis],
        rewards        = rew_np[np.newaxis].astype(np.float32),
        belief_shape   = np.array(handler.belief_shape(), dtype=np.int32),
        env_name       = np.array(env_name_current),
        action_concat  = np.array(action_concat),
        extras_goal_idx= goal_np[np.newaxis, :, :, 0].astype(np.int32),  # [1, n_traj, max_len]
    )
    print(f"  saved {npz_path}")

    # ---- Train probes ---------------------------------------------------
    json_path = out_dir / "metrics.json"
    train_and_save(
        npz=npz_path,
        seed_idx=0,
        checkpoint_idx=0,
        hparam_idx=0,
        out=json_path,
        epochs=epochs,
        batch_size=batch_size,
    )

    # ---- Print summary --------------------------------------------------
    with open(json_path) as f:
        result = json.load(f)

    print("\n=== Probe metrics ===")
    for probe, mdict in result["metrics"].items():
        tv  = mdict.get("tv",               float("nan"))
        ska = mdict.get("argmax_match_rate", float("nan"))
        skm = mdict.get("mean_prob_on_true_location", float("nan"))
        print(f"  {probe:30s}  TV={tv:.3f}  SK-acc={ska:.3f}  SK-mass={skm:.3f}")

    if "belief_sanity" in result:
        bs = result["belief_sanity"]
        print("\n=== Belief sanity ===")
        print(f"  final_argmax_correct    = {bs['final_argmax_correct']:.3f}")
        print(f"  final_mean_prob_true    = {bs['final_mean_prob_true']:.3f}")
        print(f"  all_steps_mean_prob_true= {bs['all_steps_mean_prob_true']:.3f}")


def _parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt_dir",      required=True,  type=Path,
                   help="Path to the orbax checkpoint directory.")
    p.add_argument("--out_dir",       required=True,  type=Path,
                   help="Directory for NPZ + metrics JSON output.")
    p.add_argument("--n_traj",        type=int,   default=500)
    p.add_argument("--max_len",       type=int,   default=200)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--hidden_size",   type=int,   default=64,
                   help="Hidden size (overridden by checkpoint args if present).")
    p.add_argument("--double_critic", action="store_true", default=False,
                   help="Double critic (overridden by checkpoint args if present).")
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--batch_size",    type=int,   default=1024)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        ckpt_dir      = args.ckpt_dir,
        out_dir       = args.out_dir,
        n_traj        = args.n_traj,
        max_len       = args.max_len,
        seed          = args.seed,
        hidden_size   = args.hidden_size,
        double_critic = args.double_critic,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
    )
