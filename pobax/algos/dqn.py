"""DQN / DRQN implementation compatible with the pobax pipeline.

- memoryless=False  →  DRQN (GRU-based, trajectory replay buffer)
- memoryless=True   →  DQN  (feedforward, flat replay buffer)

Checkpoints are stored in the same [n_hparams, n_seeds, *param_shape]
structure as PPO, and the train() function is compatible with vmap_and_train.
"""
from typing import NamedTuple

import flashbax as fbx
import flax
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from pobax.algos.run_helper import vmap_and_train
from pobax.config import DQNHyperparams
from pobax.envs import get_env
from pobax.envs.wrappers.gymnax import Observation
from pobax.models import ScannedRNN, get_q_network_fn
from pobax.utils.sweep import get_grid_hparams, get_randomly_sampled_hparams


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TimeStep(NamedTuple):
    obs: jnp.ndarray       # current obs  [obs_dim]
    action: jnp.ndarray    # scalar
    reward: jnp.ndarray    # scalar
    done: jnp.ndarray      # scalar
    next_obs: jnp.ndarray  # next obs     [obs_dim]


class DQNTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


# ---------------------------------------------------------------------------
# Epsilon-greedy exploration
# ---------------------------------------------------------------------------

def eps_greedy_action(rng, q_vals, timesteps, epsilon_finish, args):
    """Linear epsilon schedule; greedy arg-max otherwise."""
    epsilon = jnp.where(
        timesteps < args.learning_starts,
        args.epsilon_start,
        jnp.clip(
            args.epsilon_start + (epsilon_finish - args.epsilon_start)
            * (timesteps - args.learning_starts) / args.epsilon_anneal_time,
            epsilon_finish,
            args.epsilon_start,
        ),
    )
    rng_explore, rng_greedy = jax.random.split(rng)
    greedy_action = jnp.argmax(q_vals, axis=-1)
    random_action = jax.random.randint(rng_explore, shape=greedy_action.shape,
                                       minval=0, maxval=q_vals.shape[-1])
    explore = jax.random.uniform(rng_greedy, shape=greedy_action.shape) < epsilon
    return jnp.where(explore, random_action, greedy_action)


# ---------------------------------------------------------------------------
# make_train
# ---------------------------------------------------------------------------

def make_train(args: DQNHyperparams, rand_key: jax.random.PRNGKey):
    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_env(args.env, env_key,
                              num_envs=args.num_envs,
                              gamma=args.gamma,
                              action_concat=args.action_concat)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    network_fn, action_size, is_image = get_q_network_fn(env, env_params)
    network = network_fn(args.env,
                         action_size,
                         hidden_size=args.hidden_size,
                         memoryless=args.memoryless,
                         is_image=is_image)

    num_updates = args.total_steps // args.training_interval

    # --- Dummy timestep shape (single sample, no batch or time dims) ---
    obs_dim = env.observation_space(env_params).spaces['obs'].shape
    dummy_timestep = TimeStep(
        obs=jnp.zeros(obs_dim, dtype=jnp.float32),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=bool),
        next_obs=jnp.zeros(obs_dim, dtype=jnp.float32),
    )

    # --- Build replay buffer (static, outside train()) ---
    if args.memoryless:
        # Flat buffer: stores individual transitions, add shape (num_envs, ...)
        buffer = fbx.make_flat_buffer(
            max_length=args.buffer_size,
            min_length=args.buffer_batch_size,
            sample_batch_size=args.buffer_batch_size,
            add_sequences=False,
            add_batch_size=args.num_envs,
        )
    else:
        # Trajectory buffer: stores sequences, add shape (num_envs, 1, ...)
        buffer = fbx.make_trajectory_buffer(
            add_batch_size=args.num_envs,
            sample_batch_size=args.buffer_batch_size,
            sample_sequence_length=args.trace_length,
            period=1,
            min_length_time_axis=args.trace_length,
            max_length_time_axis=args.buffer_size // args.num_envs,
        )

    buffer_add = buffer.add
    buffer_sample = buffer.sample
    buffer_can_sample = buffer.can_sample

    # ---------------------------------------------------------------------------
    def train(sweep_args_dict, rng):
        lr = sweep_args_dict['lr']
        epsilon_finish = sweep_args_dict['epsilon_finish']

        # ---- Init network ----
        rng, _rng = jax.random.split(rng)
        init_obs = env.dummy_observation(args.num_envs, env_params)
        init_x = (init_obs, jnp.zeros((1, args.num_envs)))
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
        network_params = network.init(_rng, init_hstate, init_x)

        if args.anneal_lr:
            def linear_schedule(count):
                frac = 1.0 - count / num_updates
                return lr * frac
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )

        train_state = DQNTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            target_network_params=network_params,
            timesteps=0,
            n_updates=0,
        )

        # ---- Init env ----
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        # ---- Init buffer ----
        buffer_state = buffer.init(dummy_timestep)

        # ----------------------------------------------------------------
        # Learn phase
        # ----------------------------------------------------------------
        def _learn_phase(train_state, buffer_state, rng):
            rng, _rng = jax.random.split(rng)
            experience = buffer_sample(buffer_state, _rng).experience

            if args.memoryless:
                # flat buffer wraps in ExperiencePair; use .first for the stored TimeStep
                ts = experience.first                           # our TimeStep
                obs = ts.obs                                    # [B, obs_dim]
                next_obs = ts.next_obs                          # [B, obs_dim]
                action = ts.action                              # [B]
                reward = ts.reward                              # [B]
                done = ts.done.astype(jnp.float32)             # [B]
                B = obs.shape[0]

                dummy_h = ScannedRNN.initialize_carry(B, args.hidden_size)
                obs_in = Observation(obs=obs[None])         # [1, B, obs_dim]
                next_obs_in = Observation(obs=next_obs[None])
                dummy_done = jnp.zeros((1, B))

                def loss_fn(params):
                    _, q_vals = network.apply(params, dummy_h, (obs_in, dummy_done))
                    q_vals = q_vals.squeeze(0)  # [B, A]
                    _, q_next = network.apply(train_state.target_network_params, dummy_h,
                                              (next_obs_in, dummy_done))
                    q_next = q_next.squeeze(0)  # [B, A]
                    target = reward + (1.0 - done) * args.gamma * jnp.max(q_next, axis=-1)
                    chosen_q = q_vals[jnp.arange(B), action]
                    return jnp.mean(jnp.square(chosen_q - jax.lax.stop_gradient(target)))

            else:
                # experience leaves: [B, T, *]
                obs = experience.obs                            # [B, T, obs_dim]
                next_obs = experience.next_obs                  # [B, T, obs_dim]
                action = experience.action                      # [B, T]
                reward = experience.reward                      # [B, T]
                done = experience.done.astype(jnp.float32)     # [B, T]
                B, T = obs.shape[0], obs.shape[1]
                half = T // 2

                # Transpose to [T, B, *] for ScannedRNN
                obs_tb = jnp.transpose(obs, (1, 0, 2))          # [T, B, obs_dim]
                next_obs_tb = jnp.transpose(next_obs, (1, 0, 2))
                done_tb = jnp.transpose(done, (1, 0))            # [T, B]
                action_tb = jnp.transpose(action, (1, 0))        # [T, B]
                reward_tb = jnp.transpose(reward, (1, 0))        # [T, B]

                init_h = ScannedRNN.initialize_carry(B, args.hidden_size)
                obs_in = Observation(obs=obs_tb)
                next_obs_in = Observation(obs=next_obs_tb)

                def loss_fn(params):
                    _, q_vals = network.apply(params, init_h, (obs_in, done_tb))
                    # q_vals: [T, B, A]
                    _, q_next = network.apply(train_state.target_network_params, init_h,
                                              (next_obs_in, done_tb))
                    target = reward_tb + (1.0 - done_tb) * args.gamma * jnp.max(q_next, axis=-1)
                    chosen_q = q_vals[jnp.arange(T)[:, None], jnp.arange(B)[None, :], action_tb]
                    # Use last half to reduce zero-init hidden state bias
                    return jnp.mean(jnp.square(chosen_q[half:] - jax.lax.stop_gradient(target[half:])))

            grads = jax.grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            return train_state

        # ----------------------------------------------------------------
        # Target network soft update
        # ----------------------------------------------------------------
        def _update_target(train_state):
            new_target = jax.tree.map(
                lambda tp, op: args.tau * op + (1.0 - args.tau) * tp,
                train_state.target_network_params,
                train_state.params,
            )
            return train_state.replace(target_network_params=new_target)

        # ----------------------------------------------------------------
        # Single environment step
        # ----------------------------------------------------------------
        def _env_step(runner_state, unused):
            train_state, buffer_state, env_state, last_obs, last_done, hstate, rng = runner_state

            rng, rng_act, rng_step = jax.random.split(rng, 3)

            ac_in = (
                jax.tree.map(lambda x: x[jnp.newaxis, ...], last_obs),
                last_done[jnp.newaxis, :],
            )
            new_hstate, q_vals = network.apply(train_state.params, hstate, ac_in)
            q_vals_squeezed = q_vals.squeeze(0)  # [num_envs, action_dim]

            action = eps_greedy_action(rng_act, q_vals_squeezed,
                                       train_state.timesteps, epsilon_finish, args)

            # Step env
            rng_steps = jax.random.split(rng_step, args.num_envs)
            obsv, env_state, reward, done, info = env.step(rng_steps, env_state, action, env_params)

            # Update timestep counter
            train_state = train_state.replace(timesteps=train_state.timesteps + args.num_envs)

            # Build transition (cast to buffer dtypes)
            timestep = TimeStep(
                obs=last_obs.obs.astype(jnp.float32),
                action=action.astype(jnp.int32),
                reward=reward.astype(jnp.float32),
                done=done.astype(bool),
                next_obs=obsv.obs.astype(jnp.float32),
            )

            # Trajectory buffer expects (batch, 1, ...) per step-add; flat buffer takes (batch, ...)
            if not args.memoryless:
                timestep_buffered = jax.tree.map(lambda x: x[:, None], timestep)
            else:
                timestep_buffered = timestep

            buffer_state = buffer_add(buffer_state, timestep_buffered)

            # Conditionally learn
            can_learn = jnp.logical_and(
                buffer_can_sample(buffer_state),
                train_state.timesteps % args.training_interval == 0,
            )
            rng, rng_learn = jax.random.split(rng)

            train_state = jax.lax.cond(
                can_learn,
                lambda ts: _learn_phase(ts, buffer_state, rng_learn),
                lambda ts: ts,
                train_state,
            )

            # Conditionally update target network
            do_target_update = train_state.timesteps % args.target_update_interval == 0
            train_state = jax.lax.cond(
                do_target_update,
                _update_target,
                lambda ts: ts,
                train_state,
            )

            runner_state = (train_state, buffer_state, env_state, obsv, done, new_hstate, rng)
            return runner_state, info

        # ----------------------------------------------------------------
        # Build initial runner state
        # ----------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            buffer_state,
            env_state,
            obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_hstate,
            _rng,
        )

        # ----------------------------------------------------------------
        # Main training loop (with optional checkpointing)
        # ----------------------------------------------------------------
        if args.save_checkpoints:
            assert num_updates % args.num_checkpoints == 0, (
                f"num_updates ({num_updates}) must be divisible by "
                f"num_checkpoints ({args.num_checkpoints})"
            )
            ckpt_interval = num_updates // args.num_checkpoints

            def _checkpoint_update_step(runner_state, _):
                runner_state, metrics = jax.lax.scan(
                    _env_step, runner_state, None, ckpt_interval
                )
                return runner_state, (metrics, runner_state[0])

            runner_state, (metric_chunks, ckpt_train_states) = jax.lax.scan(
                _checkpoint_update_step, runner_state, None, args.num_checkpoints
            )
            metric = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), metric_chunks)
        else:
            runner_state, metric = jax.lax.scan(
                _env_step, runner_state, None, num_updates
            )

        # ----------------------------------------------------------------
        # Final greedy eval
        # ----------------------------------------------------------------
        final_train_state = runner_state[0]

        rng, _rng = jax.random.split(runner_state[-1])
        reset_rng = jax.random.split(_rng, args.num_envs)
        eval_obsv, eval_env_state = env.reset(reset_rng, env_params)
        eval_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        def _eval_step(eval_runner_state, unused):
            ts, es, obs, done, hs, rng_ = eval_runner_state
            rng_, rng_step = jax.random.split(rng_)
            ac_in = (
                jax.tree.map(lambda x: x[jnp.newaxis, ...], obs),
                done[jnp.newaxis, :],
            )
            new_hs, q_vals = network.apply(ts.params, hs, ac_in)
            action = jnp.argmax(q_vals.squeeze(0), axis=-1)
            rng_steps = jax.random.split(rng_step, args.num_envs)
            next_obs, next_es, reward, next_done, info = env.step(rng_steps, es, action, env_params)
            return (ts, next_es, next_obs, next_done, new_hs, rng_), info

        eval_runner_state = (final_train_state, eval_env_state, eval_obsv,
                             jnp.zeros(args.num_envs, dtype=bool), eval_hstate, _rng)
        _, eval_traj_info = jax.lax.scan(
            _eval_step, eval_runner_state, None, env_params.max_steps_in_episode
        )

        res = {
            "runner_state": runner_state,
            "metric": metric,
            "final_eval_metric": eval_traj_info,
        }
        if args.save_checkpoints:
            res['ckpt_train_states'] = ckpt_train_states
        return res

    return train


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args: DQNHyperparams):
    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    train_fn = make_train(args, make_train_rng)

    if args.sweep_type == 'grid':
        hparams, _ = get_grid_hparams(args)
    elif args.sweep_type == 'random':
        _rng, rng = jax.random.split(rng)
        hparams = get_randomly_sampled_hparams(_rng, args, n_samples=args.n_random_hparams)
    else:
        raise NotImplementedError

    vmap_and_train(args, train_fn, hparams, rng)


if __name__ == "__main__":
    args = DQNHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)
    main(args)
