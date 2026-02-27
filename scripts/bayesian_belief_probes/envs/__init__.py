"""
Environment handler registry for the Bayesian belief probe pipeline.

Each handler wraps one environment family and provides:
  - Raw env construction
  - Belief computation (obs/action → flat belief vector)
  - Extras specification (env-specific state fields to record)
  - Belief visualization (optional, env-specific)

To add a new env: subclass EnvHandler, implement all abstract methods,
and register it in get_env_handler().
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class EnvHandler(ABC):
    """Base class for environment-specific belief computation and visualization."""

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Full env name string, e.g. 'compass_world_8'."""
        ...

    @abstractmethod
    def make_raw_env(self):
        """Return (env, env_params) for the unwrapped gymnax-style env."""
        ...

    @abstractmethod
    def obs_dim(self) -> int:
        ...

    @abstractmethod
    def action_dim(self) -> int:
        ...

    @abstractmethod
    def belief_dim(self) -> int:
        """Flat belief vector size, e.g. G*G*4 for CompassWorld."""
        ...

    @abstractmethod
    def belief_shape(self) -> tuple:
        """Structured belief tensor shape, e.g. (G, G, 4) for CompassWorld."""
        ...

    def extras_spec(self) -> dict:
        """
        Dict of {name: shape_tuple} for env-specific extras to record per step.
        Kept for reference; the sampling pipeline uses get_jax_extras_fn /
        extras_flat_dim / unpack_extras instead.
        """
        return {}

    # ------------------------------------------------------------------
    # Generic JAX extras API used by sample_trajectories.py
    # ------------------------------------------------------------------

    def extras_flat_dim(self) -> int:
        """
        Size of the flat extras vector returned by get_jax_extras_fn().
        Override in subclasses that record per-step state extras.
        """
        return 0

    def get_jax_extras_fn(self) -> Callable:
        """
        Returns a JAX-traceable function  state -> jnp.ndarray [extras_flat_dim]
        that extracts env-specific extras from an env state at each step.
        Called inside lax.scan so must be pure and JAX-compatible.
        Default: returns an empty array (shape [0]).
        """
        import jax.numpy as jnp
        return lambda state: jnp.zeros((0,), dtype=jnp.float32)

    def unpack_extras(self, extras_np: np.ndarray, payload: dict) -> None:
        """
        Split the raw extras array  [n_seeds, n_traj, max_len, extras_flat_dim]
        into named keys and add them to the NPZ payload dict.
        Default: do nothing (no extras to save).
        """
        pass

    @abstractmethod
    def compute_beliefs(
        self,
        obs_seq: np.ndarray,
        action_seq: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """
        Compute analytical belief state for a single trajectory.

        Args:
            obs_seq:    [max_len, obs_dim]  (only first `length` steps are valid)
            action_seq: [max_len]
            length:     actual episode length (int)

        Returns:
            [max_len, belief_dim]  (zeros beyond `length` steps)
        """
        ...

    def visualize_beliefs(
        self,
        flat_beliefs: np.ndarray,
        ax,
        title: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Optional env-specific belief visualization drawn into a matplotlib Axes.
        Raise NotImplementedError to signal that this env has no custom grid viz.
        """
        raise NotImplementedError(
            f"visualize_beliefs not implemented for {self.env_name}"
        )


def get_env_handler(env_name: str) -> EnvHandler:
    """Return the appropriate EnvHandler for the given env_name."""
    if env_name.startswith("compass_world_"):
        from .compass_world import CompassWorldHandler
        grid_size = int(env_name.split("_")[-1])
        return CompassWorldHandler(grid_size)

    elif env_name.startswith("marquee_"):
        # env_name format: "marquee_<n_bulbs>_<n_goals>", e.g. "marquee_40_16"
        from .marquee import MarqueeHandler
        parts = env_name.split("_")
        n_bulbs = int(parts[1])
        n_goals = int(parts[2])
        return MarqueeHandler(n_bulbs, n_goals)

    raise ValueError(
        f"No EnvHandler registered for env_name='{env_name}'. "
        f"Add it to scripts/bayesian_belief_probes/envs/__init__.py."
    )
