"""
RockSample environment handler for the Bayesian belief probe pipeline.

Belief state: K-dimensional factored belief [p_0, ..., p_{K-1}]
where p_i = P(rock i is good | observation history).

Rocks are conditionally independent given the observation history, so this
factored representation is the *exact* posterior (not an approximation).

Belief update rules at step t (using action[t-1] and obs[t]):
  - Check rock i  → Bayesian update on p_i using noisy signal in obs[t][2N+i]
  - Sample         → p_i = 0 for the rock at the sampled position (deterministic depletion)
  - Move           → no update

Rock positions are random but fixed at environment construction time.  They are
reproduced exactly from the training seed by replicating the key-derivation in
ppo.py:main() + make_train():

    rng           = PRNGKey(seed)
    make_train_rng, _ = split(rng)        # ppo.py:473
    env_key, _        = split(make_train_rng)  # ppo.py:167
    env               = RockSample(env_key, ...)

The seed is stored in run_args["seed"] inside every Orbax checkpoint, so the
handler can always reconstruct the exact rock layout used during training.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from . import EnvHandler


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _p_correct(dist: float, half_efficiency_distance: float) -> float:
    """Probability of receiving the *correct* check signal at a given distance.

    Mirrors half_dist_prob() in rocksample.py:
        prob = (1 + 2^(-dist / max_dist)) * 0.5
    """
    return (1.0 + np.power(2.0, -dist / half_efficiency_distance)) * 0.5


def _decode_position(obs: np.ndarray, size: int) -> np.ndarray:
    """Return (row, col) from a one-hot observation vector of length 2*size+K."""
    row = int(np.argmax(obs[:size]))
    col = int(np.argmax(obs[size: 2 * size]))
    return np.array([row, col], dtype=np.int32)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class RockSampleHandler(EnvHandler):
    def __init__(self, size: int, n_rocks: int, seed: int = 0):
        self.size = size
        self.n_rocks = n_rocks
        self.seed = seed
        # Populated in make_raw_env(); must be called before compute_beliefs()
        self._rock_positions: Optional[np.ndarray] = None        # [n_rocks, 2]
        self._half_efficiency_distance: Optional[float] = None

    def _config_path(self) -> Path:
        from pobax.definitions import ROOT_DIR
        return (
            Path(ROOT_DIR) / "envs" / "configs"
            / f"rocksample_{self.size}_{self.n_rocks}_config.json"
        )

    @property
    def env_name(self) -> str:
        return f"rocksample_{self.size}_{self.n_rocks}"

    def make_raw_env(self):
        import jax
        from pobax.envs.jax.rocksample import RockSample

        # Reproduce the exact key derivation from ppo.py:
        #   rng              = PRNGKey(seed)
        #   make_train_rng,_ = split(rng)          [ppo.py:473]
        #   env_key,_        = split(make_train_rng)[ppo.py:167]
        key0 = jax.random.PRNGKey(self.seed)
        make_train_rng, _ = jax.random.split(key0)
        env_key, _ = jax.random.split(make_train_rng)

        env = RockSample(env_key, config_path=self._config_path())
        self._rock_positions = np.array(env.rock_positions)          # [n_rocks, 2]
        self._half_efficiency_distance = float(env.half_efficiency_distance)
        return env, env.default_params

    def obs_dim(self) -> int:
        return 2 * self.size + self.n_rocks

    def action_dim(self) -> int:
        return self.n_rocks + 5

    def belief_dim(self) -> int:
        return self.n_rocks

    def belief_shape(self) -> tuple:
        return (self.n_rocks,)

    # ------------------------------------------------------------------
    # JAX extras: position (2,) + rock_morality (n_rocks,) packed flat
    # ------------------------------------------------------------------

    def extras_flat_dim(self) -> int:
        return 2 + self.n_rocks

    def get_jax_extras_fn(self):
        import jax.numpy as jnp

        def _fn(state):
            return jnp.concatenate([
                state.position.astype(jnp.float32),       # [2]
                state.rock_morality.astype(jnp.float32),  # [n_rocks]
            ])

        return _fn

    def unpack_extras(self, extras_np: np.ndarray, payload: dict) -> None:
        # extras_np: [n_seeds, n_traj, max_len, 2 + n_rocks]
        payload["extras_position"]      = extras_np[..., :2].astype(np.int32)
        payload["extras_rock_morality"] = extras_np[..., 2:].astype(np.int32)

    # ------------------------------------------------------------------
    # Analytical belief computation (numpy, one trajectory at a time)
    # ------------------------------------------------------------------

    def compute_beliefs(
        self,
        obs_seq: np.ndarray,    # [max_len, 2*size + n_rocks]
        action_seq: np.ndarray, # [max_len]
        length: int,
    ) -> np.ndarray:
        """
        Factored Bayesian belief update for one trajectory.

        obs_seq[t]    = observation the agent *saw* at step t (before action[t]).
        obs_seq[t+1]  = observation after taking action[t]; carries the check
                        signal for rock i if action[t] = 5+i.

        The belief at step t is the posterior after incorporating all of
        obs[0..t] and action[0..t-1], matching what hidden_seq[t] encodes.

        Returns
        -------
        [max_len, n_rocks]  float32  (zeros beyond `length` steps)
        """
        assert self._rock_positions is not None and self._half_efficiency_distance is not None, (
            "make_raw_env() must be called before compute_beliefs()"
        )
        rock_positions = self._rock_positions
        half_eff = self._half_efficiency_distance

        max_len = obs_seq.shape[0]
        beliefs = np.zeros((max_len, self.n_rocks), dtype=np.float32)

        if length <= 0:
            return beliefs

        p = np.full(self.n_rocks, 0.5, dtype=np.float64)   # uniform prior

        for t in range(length):
            # ----------------------------------------------------------
            # Incorporate the effect of action[t-1] into the belief.
            # The resulting signal (if any) is visible in obs[t].
            # ----------------------------------------------------------
            if t > 0:
                prev_action = int(action_seq[t - 1])

                if prev_action > 4:
                    # Check rock i: obs[t][2N+i] holds the noisy signal (±1).
                    # Check doesn't move the agent, so obs[t] position == obs[t-1] position.
                    i = prev_action - 5
                    signal = float(obs_seq[t][2 * self.size + i])   # +1 or -1

                    pos  = _decode_position(obs_seq[t], self.size)
                    dist = float(np.linalg.norm(pos - rock_positions[i]))
                    pc   = _p_correct(dist, half_eff)

                    # P(signal | good) and P(signal | bad):
                    #   good → +1 w.p. pc,   −1 w.p. (1-pc)
                    #   bad  → −1 w.p. pc,   +1 w.p. (1-pc)
                    if signal > 0:
                        num = pc * p[i]
                        den = num + (1.0 - pc) * (1.0 - p[i])
                    else:
                        num = (1.0 - pc) * p[i]
                        den = num + pc * (1.0 - p[i])

                    if den > 1e-12:
                        p[i] = num / den

                elif prev_action == 4:
                    # Sample: the rock at the agent's position becomes depleted.
                    # Use obs[t-1] for the position where the sample was taken.
                    pos = _decode_position(obs_seq[t - 1], self.size)
                    for i in range(self.n_rocks):
                        if np.array_equal(pos, rock_positions[i]):
                            p[i] = 0.0

            beliefs[t] = p.astype(np.float32)

        return beliefs
