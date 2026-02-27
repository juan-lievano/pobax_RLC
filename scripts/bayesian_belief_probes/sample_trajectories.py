"""
Step 1 of the Bayesian belief probe pipeline.

Rolls out trajectories for ALL seeds simultaneously (vmapped) using a trained
checkpoint, computes analytical belief states, and saves everything to an NPZ.

The checkpoint params are expected to have leaves of shape [n_hparams, n_seeds, *].
We extract h_idx and vmap over the seeds dimension.

Output NPZ keys (all arrays have a leading n_seeds dimension):
    obs          [n_seeds, n_traj, max_len, obs_dim]
    actions      [n_seeds, n_traj, max_len]
    hidden       [n_seeds, n_traj, max_len, hidden_size]
    beliefs      [n_seeds, n_traj, max_len, belief_dim]   <- flat belief vector
    masks        [n_seeds, n_traj, max_len]               <- True = valid step
    lengths      [n_seeds, n_traj]
    belief_shape [k]                                      <- e.g. [G, G, 4]
    extras_*     [n_seeds, n_traj, max_len, *extra_shape] <- env-specific

Usage (standalone):
    python sample_trajectories.py \\
        --run_dir  results/my_run/checkpoint_dir_parent \\
        --ckpt_path results/my_run/checkpoint_dir_parent/checkpoint_0 \\
        --h_idx 0 --n_traj 500 --max_len 200 --out /tmp/trajs.npz
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from jax import lax
from jax import random as jr

# Make envs importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from envs import get_env_handler

from pobax.envs.wrappers.gymnax import Observation
from pobax.models.actor_critic import ActorCritic
from pobax.models.network import ScannedRNN


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------

def build_rollout_fn(
    model,
    env,
    env_params,
    hidden_size: int,
    max_len: int,
    memoryless: bool,
    extras_fn,
    extras_dim: int,
    action_concat: bool,
    action_dim: int,
):
    """
    Returns a JIT-compiled function:
        rollout(params_all_seeds, rng_keys)
            -> (obs, actions, extras, hidden, masks, lengths)

    params_all_seeds: pytree with leaves [n_seeds, *param_shape]
    rng_keys:         [n_traj]  PRNGKeys (same starting keys for all seeds)
    extras_fn:        state -> jnp.ndarray [extras_dim]  (env-specific extras)
    extras_dim:       int  (0 for no extras)
    action_concat:    if True, prepend one-hot prev_action to obs before the model
                      (must match how the checkpoint was trained)
    action_dim:       size of the action space (needed to build the one-hot vector)
    """

    def rollout_single(rng_key, params_one_seed):
        """Rollout one trajectory with one seed's params."""
        rng, rng_reset = jr.split(rng_key)
        obs, state = env.reset_env(rng_reset, env_params)

        if memoryless:
            hidden = jnp.zeros((1, hidden_size))
        else:
            hidden = ScannedRNN.initialize_carry(1, hidden_size)

        done = jnp.array(False)
        # Carry prev_action_onehot always; it is only used when action_concat=True.
        # Initialised to zeros (equivalent to ActionConcatWrapper.reset behaviour).
        prev_action_onehot = jnp.zeros(action_dim, dtype=jnp.float32)

        def step_fn(carry, _):
            rng, state, obs, hidden, done, prev_action_onehot = carry
            rng, k_step, k_act = jr.split(rng, 3)

            # Replicate ActionConcatWrapper: prepend prev_action to obs fed to model.
            if action_concat:
                obs_model = jnp.concatenate([obs, prev_action_onehot])
            else:
                obs_model = obs
            obs_in   = obs_model[None, None, :].astype(jnp.float32)  # [1, 1, obs_dim]
            done_in  = done[None, None]                                # [1, 1]
            obs_dict = Observation(obs=obs_in)  # action_mask=None is the default

            hidden_new, pi, _ = model.apply(params_one_seed, hidden, (obs_dict, done_in))
            action = pi.sample(seed=k_act)[0, 0].astype(jnp.int32)

            # Freeze hidden/obs/state after episode ends
            hidden_rec = jnp.where(done, hidden, hidden_new)
            mask_out   = jnp.logical_not(done)

            next_obs, next_state, _, next_done, _ = env.step_env(
                k_step, state, action, env_params
            )
            next_done_flag = jnp.logical_or(done, next_done)

            # One-hot of current action becomes prev_action for the next step.
            new_prev_action_onehot = jnp.eye(action_dim, dtype=jnp.float32)[action]
            # Freeze after episode ends (consistent with obs/hidden freeze above).
            new_prev_action_onehot = jnp.where(done, prev_action_onehot, new_prev_action_onehot)

            next_carry = (
                rng,
                jax.tree_util.tree_map(
                    lambda a, b: jnp.where(done, a, b), state, next_state
                ),
                jnp.where(done, obs, next_obs.astype(jnp.float32)),
                hidden_rec,
                next_done_flag,
                new_prev_action_onehot,
            )
            step_out = (
                obs,               # [obs_dim]        raw obs (no prev_action) for storage
                action,            # []               action taken
                extras_fn(state),  # [extras_dim]     env-specific extras
                hidden_rec[0],     # [hidden_size]    squeeze batch dim for storage
                mask_out,          # []               validity mask
            )
            return next_carry, step_out

        init_carry = (rng, state, obs.astype(jnp.float32), hidden, done, prev_action_onehot)
        _, (obs_seq, act_seq, extras_seq, hid_seq, mask_seq) = lax.scan(
            step_fn, init_carry, None, max_len
        )
        length = mask_seq.sum(dtype=jnp.int32)
        return obs_seq, act_seq, extras_seq, hid_seq, mask_seq, length

    def rollout_all_seeds(params_all_seeds, rng_keys):
        def _one_seed(params_one_seed):
            return jax.vmap(rollout_single, in_axes=(0, None))(rng_keys, params_one_seed)
        return jax.vmap(_one_seed)(params_all_seeds)

    return jax.jit(rollout_all_seeds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_and_save(
    *,
    ckpt_path: Path,
    h_idx: int,
    n_traj: int,
    max_len: int,
    out: Path,
    seed: int,
    # model args (passed explicitly so the orchestrator doesn't need to re-load)
    hidden_size: int,
    env_name: str,
    double_critic: bool,
    memoryless: bool,
    action_concat: bool,
):
    handler = get_env_handler(env_name)
    env, env_params = handler.make_raw_env()

    # Build model (structure must match the training run)
    model = ActorCritic(
        env_name,
        handler.action_dim(),
        hidden_size=hidden_size,
        double_critic=double_critic,
        memoryless=memoryless,
        is_discrete=True,
        is_image=False,
    )

    # Load checkpoint and extract params for this hparam set (all seeds)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = checkpointer.restore(ckpt_path)
    # ckpt is a nested dict with TrainState fields; params has leaves [n_hparams, n_seeds, *]
    params_all_seeds = jax.tree.map(lambda x: x[h_idx], ckpt["params"])

    n_seeds = jax.tree.leaves(params_all_seeds)[0].shape[0]
    print(
        f"  checkpoint: {ckpt_path.name}, n_seeds={n_seeds}, n_traj={n_traj}, "
        f"max_len={max_len}, action_concat={action_concat}"
    )

    # Build JIT-compiled rollout
    extras_fn  = handler.get_jax_extras_fn()
    extras_dim = handler.extras_flat_dim()
    rollout_fn = build_rollout_fn(
        model, env, env_params, hidden_size, max_len, memoryless,
        extras_fn, extras_dim,
        action_concat=action_concat,
        action_dim=handler.action_dim(),
    )

    # Same rng_keys for all seeds (same starting states, different policies)
    rng = jr.PRNGKey(seed)
    rng_keys = jr.split(rng, n_traj)

    print("  running vmapped rollout...", flush=True)
    rollout_out = rollout_fn(params_all_seeds, rng_keys)  # type: ignore[misc]
    obs_j, act_j, extras_j, hid_j, mask_j, len_j = rollout_out
    obs_np, act_np, extras_np, hid_np, mask_np, len_np = jax.device_get(
        (obs_j, act_j, extras_j, hid_j, mask_j, len_j)
    )
    # shapes: [n_seeds, n_traj, max_len, *] and [n_seeds, n_traj]

    # Compute analytical beliefs in numpy (env-specific, Python loop)
    print("  computing beliefs...", flush=True)
    beliefs = np.zeros(
        (n_seeds, n_traj, max_len, handler.belief_dim()), dtype=np.float32
    )
    for s in range(n_seeds):
        for i in range(n_traj):
            length = int(len_np[s, i])
            beliefs[s, i] = handler.compute_beliefs(
                obs_np[s, i], act_np[s, i], length
            )

    # Build NPZ payload
    payload = {
        "obs":           obs_np.astype(np.float32),
        "actions":       act_np,
        "hidden":        hid_np.astype(np.float32),
        "beliefs":       beliefs,
        "masks":         mask_np,
        "lengths":       len_np,
        "belief_shape":  np.array(handler.belief_shape(), dtype=np.int32),
        "env_name":      np.array(env_name),          # stored as a 0-d string array
        "action_concat": np.array(action_concat),     # metadata: was prev_action prepended?
    }

    # Env-specific extras: delegated to handler (e.g. positions+dir for CompassWorld,
    # goal index for Marquee).  extras_np shape: [n_seeds, n_traj, max_len, extras_dim]
    handler.unpack_extras(extras_np, payload)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir",   required=True, type=Path, help="Parent Orbax directory (holds the main checkpoint + args).")
    p.add_argument("--ckpt_path", required=True, type=Path, help="Path to a checkpoint_i subdirectory.")
    p.add_argument("--h_idx",     type=int, default=0,   help="Hparam index to use (default 0).")
    p.add_argument("--n_traj",    type=int, required=True, help="Number of trajectories per seed.")
    p.add_argument("--max_len",   type=int, default=200,  help="Max steps per trajectory.")
    p.add_argument("--out",       required=True, type=Path, help="Output .npz path.")
    p.add_argument("--seed",      type=int, default=0,    help="Base RNG seed.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    main_results = checkpointer.restore(args.run_dir)
    run_args = main_results["args"]

    sample_and_save(
        ckpt_path=args.ckpt_path,
        h_idx=args.h_idx,
        n_traj=args.n_traj,
        max_len=args.max_len,
        out=args.out,
        seed=args.seed,
        hidden_size=int(run_args["hidden_size"]),
        env_name=str(run_args["env"]),
        double_critic=bool(run_args.get("double_critic", False)),
        memoryless=bool(run_args.get("memoryless", False)),
        action_concat=bool(run_args.get("action_concat", False)),
    )
