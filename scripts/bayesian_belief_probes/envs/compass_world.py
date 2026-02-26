"""
CompassWorld environment handler for the Bayesian belief probe pipeline.

Belief computation is analytical (Bayesian filtering with known transition/
observation models). The belief state is a distribution over (y, x, direction)
triples, represented as a G×G×4 tensor and stored flat as [G*G*4].
"""
from typing import Optional, Tuple

import numpy as np

from . import EnvHandler


class CompassWorldHandler(EnvHandler):
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self._gs = grid_size

    @property
    def env_name(self) -> str:
        return f"compass_world_{self.grid_size}"

    def make_raw_env(self):
        from pobax.envs.jax.compass_world import CompassWorld
        env = CompassWorld(size=self.grid_size)
        return env, env.default_params

    def obs_dim(self) -> int:
        return 5

    def action_dim(self) -> int:
        return 3

    def belief_dim(self) -> int:
        return self.grid_size * self.grid_size * 4

    def belief_shape(self) -> tuple:
        return (self.grid_size, self.grid_size, 4)

    def extras_spec(self) -> dict:
        return {
            "positions": (2,),
            "directions": (),
        }

    def compute_beliefs(
        self,
        obs_seq: np.ndarray,
        action_seq: np.ndarray,
        length: int,
    ) -> np.ndarray:
        gs = self.grid_size
        max_len = obs_seq.shape[0]
        beliefs = np.zeros((max_len, gs * gs * 4), dtype=np.float32)

        if length <= 0:
            return beliefs

        prior_mask = _initial_belief_mask(gs)
        for t in range(length):
            posterior_mask = _apply_observation_filter(prior_mask, obs_seq[t], gs)
            beliefs[t] = _normalize_mask(posterior_mask).reshape(-1)
            if t < length - 1:
                prior_mask = _transition_mask_precise(
                    posterior_mask, int(action_seq[t]), gs
                )

        return beliefs

    def visualize_beliefs(
        self,
        flat_beliefs: np.ndarray,
        ax,
        title: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Draw a triangle-grid visualization of a flat belief vector."""
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import Normalize

        gs = self.grid_size
        tensor = flat_beliefs.reshape(gs, gs, 4)

        polys, vals = [], []
        for y in range(gs):
            for x in range(gs):
                tris = _triangles_for_cell(x, y)
                for d in range(4):
                    polys.append(tris[d])
                    vals.append(float(tensor[y, x, d]))

        vals_arr = np.asarray(vals)
        if vmin is None:
            vmin = float(vals_arr.min())
        if vmax is None:
            vmax = float(vals_arr.max())

        pc = PolyCollection(polys, edgecolors="k", linewidths=0.2)
        pc.set_array(vals_arr)
        pc.set_norm(Normalize(vmin=vmin, vmax=vmax))
        ax.add_collection(pc)
        ax.set_aspect("equal")
        ax.set_xlim(0, gs)
        ax.set_ylim(0, gs)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# Belief computation helpers (ported from example scripts)
# ---------------------------------------------------------------------------

def _initial_belief_mask(grid_size: int) -> np.ndarray:
    """Uniform prior over all non-goal interior cells × 4 directions."""
    mask = np.zeros((grid_size, grid_size, 4), dtype=bool)
    mask[1:grid_size - 1, 1:grid_size - 1, :] = True
    goal_y = (grid_size - 1) // 2
    mask[goal_y, 1, 3] = False  # remove goal state from prior
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    count = int(mask.sum())
    if count > 0:
        out[mask] = 1.0 / float(count)
    return out


def _state_emits_index(y: int, x: int, d: int, grid_size: int) -> Optional[int]:
    """Return the observation index emitted by state (y, x, d), or None."""
    y_goal = (grid_size - 1) // 2
    if d == 0 and y == 1:
        return 0  # N-wall
    if d == 1 and x == grid_size - 2:
        return 1  # E-wall
    if d == 2 and y == grid_size - 2:
        return 2  # S-wall
    if d == 3 and x == 1:
        if y == y_goal:
            return 4  # goal
        return 3  # W-wall (non-goal)
    return None


def _apply_observation_filter(
    mask: np.ndarray, obs_vec: np.ndarray, grid_size: int
) -> np.ndarray:
    gs = grid_size
    if float(np.max(obs_vec)) > 0.0:
        idx = int(np.argmax(obs_vec))
        keep = np.zeros_like(mask, dtype=bool)
        for y in range(1, gs - 1):
            for x in range(1, gs - 1):
                for d in range(4):
                    if _state_emits_index(y, x, d, gs) == idx:
                        keep[y, x, d] = True
    else:
        # null observation: keep states that emit nothing
        keep = np.zeros_like(mask, dtype=bool)
        for y in range(1, gs - 1):
            for x in range(1, gs - 1):
                for d in range(4):
                    if _state_emits_index(y, x, d, gs) is None:
                        keep[y, x, d] = True
    return np.logical_and(mask, keep)


def _transition_mask_precise(
    mask: np.ndarray, action: int, grid_size: int
) -> np.ndarray:
    H = W = grid_size
    dest = np.zeros_like(mask, dtype=bool)

    if action == 1:  # turn right
        for d in range(4):
            dest[:, :, (d + 1) % 4] |= mask[:, :, d]
        return dest

    if action == 2:  # turn left
        for d in range(4):
            dest[:, :, (d + 3) % 4] |= mask[:, :, d]
        return dest

    # action == 0: move forward
    y_min, y_max = 1, H - 2
    x_min, x_max = 1, W - 2

    src = mask[:, :, 0]  # facing North → y decreases
    if np.any(src):
        dest[y_min:y_max, x_min:x_max + 1, 0] |= src[y_min + 1:y_max + 1, x_min:x_max + 1]
        dest[y_min, x_min:x_max + 1, 0] |= src[y_min, x_min:x_max + 1]

    src = mask[:, :, 1]  # facing East → x increases
    if np.any(src):
        dest[y_min:y_max + 1, x_min + 1:x_max + 1, 1] |= src[y_min:y_max + 1, x_min:x_max]
        dest[y_min:y_max + 1, x_max, 1] |= src[y_min:y_max + 1, x_max]

    src = mask[:, :, 2]  # facing South → y increases
    if np.any(src):
        dest[y_min + 1:y_max + 1, x_min:x_max + 1, 2] |= src[y_min:y_max, x_min:x_max + 1]
        dest[y_max, x_min:x_max + 1, 2] |= src[y_max, x_min:x_max + 1]

    src = mask[:, :, 3]  # facing West → x decreases
    if np.any(src):
        dest[y_min:y_max + 1, x_min:x_max, 3] |= src[y_min:y_max + 1, x_min + 1:x_max + 1]
        dest[y_min:y_max + 1, x_min, 3] |= src[y_min:y_max + 1, x_min]

    return dest


def _triangles_for_cell(x: int, y: int):
    """Return 4 triangles (N, E, S, W) for grid cell (x, y) in plot coords."""
    bl = (x, y);     br = (x + 1, y)
    tl = (x, y + 1); tr = (x + 1, y + 1)
    c  = (x + 0.5, y + 0.5)
    return [
        [tl, tr, c],  # N
        [tr, br, c],  # E
        [br, bl, c],  # S
        [bl, tl, c],  # W
    ]
