"""
Marquee environment (formerly known as LightBulbs — that name is deprecated).

A cooperative POMDP where the robot must toggle binary bulbs to match a hidden
goal array, with a human co-player that flips one mismatched bulb toward the
goal each step.  The robot only observes the current bulb array (not the goal).

Configuration is loaded from a JSON file with keys:
    "size"          : int     — number of bulbs
    "goals"         : list    — list of goal arrays (each of length `size`)
    "distribution"  : list    — probability of each goal being selected
    "robot_noop"    : bool    — whether the robot can pass its turn
"""
from functools import partial
from typing import Tuple

import chex
import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams
from jax import random
import json


@chex.dataclass
class MarqueeState:
    bulbs: chex.Array  # current state of bulbs array [size]
    goal: chex.Array   # the hidden goal array [size]
    t: chex.Array      # step counter


class Marquee(Environment):
    """
    Marquee (formerly LightBulbs) environment.

    The robot toggles bulbs to match a hidden goal; a human co-player helps by
    flipping one mismatched bulb each step.  Episode ends when bulbs == goal.
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: path to JSON config with keys "size", "goals",
                         "distribution", "robot_noop".
        """
        self.size, self.goals, self.goal_distribution, self.robot_noop = \
            self._load_config(config_path)

    def _load_config(self, path: str):
        with open(path) as f:
            data = json.load(f)

        size = data["size"]
        goals = jnp.asarray(data["goals"], jnp.int32)
        dist = jnp.asarray(data["distribution"], jnp.float32)
        robot_noop = data["robot_noop"]

        if goals.shape[1] != size:
            raise ValueError(
                f"goal length {goals.shape[1]} != size {size}"
            )
        if dist.shape[0] != goals.shape[0]:
            raise ValueError("distribution length must match number of goals")
        if not isinstance(robot_noop, bool):
            raise TypeError("'robot_noop' must be a JSON boolean (true/false)")

        return size, goals, dist / jnp.sum(dist), robot_noop

    @property
    def num_goals(self) -> int:
        return int(self.goals.shape[0])

    def observation_space(self, env_params: EnvParams):
        return gymnax.environments.spaces.Box(0, 1, (self.size,))

    def action_space(self, env_params: EnvParams):
        """
        Actions are bulb indices to toggle.
        If robot_noop is True, action `size` means no-op.
        """
        if self.robot_noop:
            return gymnax.environments.spaces.Discrete(self.size + 1)
        else:
            return gymnax.environments.spaces.Discrete(self.size)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def get_obs(self, state: MarqueeState) -> jnp.ndarray:
        """The robot only observes the current bulb array."""
        return state.bulbs

    @staticmethod
    def human_policy(
        key,
        state: jnp.ndarray,
        goal: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Flip exactly one randomly-chosen mismatched bit toward the goal.
        If bulbs already match goal, return state unchanged.
        """
        mask = state != goal
        has_mismatch = jnp.any(mask)

        def _flip(_):
            probs = mask.astype(jnp.float32)
            probs /= jnp.sum(probs)
            idx = random.choice(key, state.shape[0], p=probs)
            return state.at[idx].set(goal[idx])

        return jax.lax.cond(
            has_mismatch,
            _flip,
            lambda _: state,
            operand=None,
        )

    def transition(
        self,
        key: chex.PRNGKey,
        state: MarqueeState,
        action: int,
    ):
        """
        1. Robot toggles bulb at `action` (or no-ops if action == size).
        2. Human flips one mismatched bulb toward goal.
        3. Check done; compute reward.
        """
        # 1. Robot move / no-op
        def _noop(bulbs_and_idx):
            bulbs, _ = bulbs_and_idx
            return bulbs

        def _toggle(bulbs_and_idx):
            bulbs, idx = bulbs_and_idx
            return bulbs.at[idx].set(1 - bulbs[idx])

        bulbs_after_robot = jax.lax.cond(
            action == self.size,
            _noop,
            _toggle,
            (state.bulbs, action),
        )

        # 2. Human responds
        key, subkey = random.split(key)
        bulbs_after_human = self.human_policy(subkey, bulbs_after_robot, state.goal)

        # 3. Check done
        done = jnp.all(bulbs_after_human == state.goal)

        # 4. Reward: -1 per step until done (0 at terminal)
        reward = jnp.where(done, 0, -1)

        next_state = MarqueeState(
            bulbs=bulbs_after_human,
            goal=state.goal,
            t=state.t + 1,
        )
        return next_state, reward, done

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self,
        key: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[jnp.ndarray, MarqueeState]:
        key, k_bulbs, k_goal = random.split(key, 3)

        bulbs = jax.random.bernoulli(k_bulbs, p=0.5, shape=(self.size,)).astype(
            jnp.int32
        )
        theta = jax.random.choice(k_goal, self.goals.shape[0], p=self.goal_distribution)
        goal = self.goals[theta]

        state = MarqueeState(bulbs=bulbs, goal=goal, t=jnp.int32(0))
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: MarqueeState,
        action: int,
        params: EnvParams,
    ):
        key, subkey = random.split(key)
        next_state, reward, done = self.transition(subkey, state, action)
        obs = next_state.bulbs
        done = jnp.logical_or(done, next_state.t >= params.max_steps_in_episode)
        return obs, next_state, reward, done, {}
