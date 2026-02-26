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
from typing import Optional

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

    @abstractmethod
    def extras_spec(self) -> dict:
        """
        Dict of {name: shape_tuple} for env-specific extras to record per step.
        E.g. {"positions": (2,), "directions": ()} for CompassWorld.
        Keys must be valid numpy array names (no spaces).
        """
        ...

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

    # Add more envs here:
    # elif env_name.startswith("marquee_"):
    #     from .marquee import MarqueeHandler
    #     return MarqueeHandler(env_name)

    raise ValueError(
        f"No EnvHandler registered for env_name='{env_name}'. "
        f"Add it to scripts/bayesian_belief_probes/envs/__init__.py."
    )
