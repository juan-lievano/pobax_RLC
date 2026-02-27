"""
Marquee environment handler for the Bayesian belief probe pipeline.

The belief state is a probability distribution over goals (shape [n_goals]),
stored as a flat vector of length n_goals.

Belief computation is analytical (online Bayesian filtering):
  - Prior: uniform over goals
  - Update at each step: use (obs_t, action_t, obs_{t+1}) to determine which
    bulbs the human flipped, then rule out goals incompatible with that flip.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from . import EnvHandler


class MarqueeHandler(EnvHandler):
    def __init__(self, n_bulbs: int, n_goals: int):
        self.n_bulbs = n_bulbs
        self.n_goals = n_goals
        # Load goals and prior from the config file at build time (numpy only)
        self._goals_np, self._prior_np = self._load_config()

    def _config_path(self) -> Path:
        from pobax.definitions import ROOT_DIR
        return (
            Path(ROOT_DIR) / "envs" / "configs"
            / f"marquee_{self.n_bulbs}_{self.n_goals}_config.json"
        )

    def _load_config(self):
        import json
        with open(self._config_path()) as f:
            data = json.load(f)
        goals = np.asarray(data["goals"], dtype=np.float32)  # [n_goals, n_bulbs]
        dist  = np.asarray(data["distribution"], dtype=np.float64)
        dist  = dist / dist.sum()
        return goals, dist

    @property
    def env_name(self) -> str:
        return f"marquee_{self.n_bulbs}_{self.n_goals}"

    def make_raw_env(self):
        from pobax.envs.jax.marquee import Marquee
        env = Marquee(config_path=str(self._config_path()))
        return env, env.default_params

    def obs_dim(self) -> int:
        return self.n_bulbs

    def action_dim(self) -> int:
        return self.n_bulbs + 1  # +1 for robot no-op

    def belief_dim(self) -> int:
        return self.n_goals

    def belief_shape(self) -> tuple:
        return (self.n_goals,)

    def extras_spec(self) -> dict:
        return {"goal_idx": ()}

    # ------------------------------------------------------------------
    # Generic JAX extras: record goal index at each step
    # ------------------------------------------------------------------

    def extras_flat_dim(self) -> int:
        return 1  # [goal_idx]

    def get_jax_extras_fn(self):
        import jax.numpy as jnp
        import numpy as np
        # Close over the goals array so we can identify which goal is active
        goals_jax = jnp.asarray(self._goals_np, dtype=jnp.int32)  # [n_goals, n_bulbs]

        def _fn(state):
            # state.goal: [n_bulbs]
            matches = jnp.all(
                state.goal[None, :].astype(jnp.int32) == goals_jax, axis=1
            )  # [n_goals]
            return jnp.argmax(matches).reshape(1).astype(jnp.float32)  # [1]

        return _fn

    def unpack_extras(self, extras_np: np.ndarray, payload: dict) -> None:
        # extras_np: [n_seeds, n_traj, max_len, 1]
        payload["extras_goal_idx"] = extras_np[..., 0].astype(np.int32)

    # ------------------------------------------------------------------
    # Belief computation (analytical Bayesian filtering in numpy)
    # ------------------------------------------------------------------

    def compute_beliefs(
        self,
        obs_seq: np.ndarray,
        action_seq: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """
        Compute the analytical posterior over goals for one trajectory.

        The belief at step t is computed AFTER using the transition
        (obs_t → obs_{t+1}) with action_t to observe what the human did.
        This matches the online posterior recorded during trajectory sampling.

        Args:
            obs_seq:    [max_len, n_bulbs]  bulb array at each step
            action_seq: [max_len]           robot action at each step
            length:     actual episode length

        Returns:
            [max_len, n_goals]  (zeros beyond `length`)
        """
        max_len = obs_seq.shape[0]
        beliefs = np.zeros((max_len, self.n_goals), dtype=np.float32)

        if length <= 0:
            return beliefs

        post = self._prior_np.copy()

        for t in range(length):
            if t + 1 < max_len:
                # Use the transition obs[t] → obs[t+1] with action[t]
                post = _update_posterior_np(
                    self._goals_np,
                    post,
                    obs_seq[t].astype(np.float64),
                    obs_seq[t + 1].astype(np.float64),
                    int(action_seq[t]),
                )
            # At t == length-1 with length == max_len: no next obs available,
            # keep current posterior (minor approximation for truncated episodes).
            beliefs[t] = post.astype(np.float32)

        return beliefs

    # ------------------------------------------------------------------
    # Visualization: bar chart of mean predicted belief over goals
    # ------------------------------------------------------------------

    def visualize_beliefs(
        self,
        flat_beliefs: np.ndarray,
        ax,
        title: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Draw a bar chart of a belief vector over goals."""
        n = len(flat_beliefs)
        ax.bar(np.arange(n), flat_beliefs, color="#56B4E9", edgecolor="none")
        if vmin is not None or vmax is not None:
            ax.set_ylim(
                vmin if vmin is not None else 0.0,
                vmax if vmax is not None else 1.0,
            )
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("Goal index", fontsize=8)
        ax.set_ylabel("Probability", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Belief update helper (numpy analog of update_posterior from notes/)
# ---------------------------------------------------------------------------

def _update_posterior_np(
    goals: np.ndarray,
    prev_posterior: np.ndarray,
    curr_obs: np.ndarray,
    next_obs: np.ndarray,
    robot_action: int,
) -> np.ndarray:
    """
    Bayesian posterior update over goals given observed transition.

    Args:
        goals:          [n_goals, n_bulbs]  possible goal arrays
        prev_posterior: [n_goals]
        curr_obs:       [n_bulbs]  bulb state at start of step
        next_obs:       [n_bulbs]  bulb state after robot + human moves
        robot_action:   int  (n_bulbs = no-op)

    Returns:
        [n_goals] updated posterior
    """
    delta = (next_obs - curr_obs).copy()
    # Ignore the bulb the robot toggled (its change is known, not informative)
    if 0 <= robot_action < len(delta):
        delta[robot_action] = 0.0

    # For each goal, check if the human's flip is consistent with that goal:
    #   - A positive delta means that bulb was flipped from 0→1, so the goal
    #     must have 1 at that position.
    #   - A negative delta means that bulb was flipped from 1→0, so the goal
    #     must have 0 at that position.
    pos_ok = np.all(np.where(delta > 0, goals == 1, True), axis=1)
    neg_ok = np.all(np.where(delta < 0, goals == 0, True), axis=1)
    compat = pos_ok & neg_ok  # [n_goals]

    new = prev_posterior * compat.astype(np.float64)
    z = new.sum()

    if z > 0:
        return new / z

    # Fallback: uniform over compatible goals (handles numerical edge cases)
    n_compat = int(compat.sum())
    if n_compat > 0:
        return compat.astype(np.float64) / n_compat
    return prev_posterior.copy()
