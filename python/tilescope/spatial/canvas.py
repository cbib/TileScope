"""
Canvas coordinate transformation for spatial visualization.

World-to-pixel mapping for both integer grid and float coordinate data.
"""
from tilescope.spatial.spacing import median_nn_distance


import numpy as np
from dataclasses import dataclass
from typing import Tuple

from tilescope.cols import COORD_X, COORD_Y
from tilescope.defaults import (
    DEFAULT_TARGET_PX_PER_NN,
    DEFAULT_TARGET_PX_PER_STEP,
    DEFAULT_CANVAS_MARGIN_PX
)


def _is_integer_or_shifted_grid(
    x: np.ndarray, y: np.ndarray, tol: float = 1e-6,
) -> Tuple[bool, float, float]:
    """Detect integer or shifted-integer coordinate grids (per-axis offset)."""
    def dominant_offset(vals: np.ndarray) -> float:
        frac = np.mod(vals, 1.0)
        near_zero = np.sum(frac < tol)
        near_one = np.sum(frac > (1.0 - tol))
        if near_zero + near_one > 0.9 * len(vals):
            return 0.0
        hist, edges = np.histogram(frac, bins=100, range=(0, 1))
        peak_idx = np.argmax(hist)
        o = (edges[peak_idx] + edges[peak_idx + 1]) / 2
        if o < tol or abs(o - 1.0) < tol:
            return 0.0
        return o

    ox = dominant_offset(x)
    oy = dominant_offset(y)
    is_grid = (np.allclose(x - ox, np.round(x - ox), atol=tol) and
               np.allclose(y - oy, np.round(y - oy), atol=tol))
    return is_grid, ox, oy


@dataclass
class Canvas:
    """World-to-pixel coordinate transformation.

    Attributes
    ----------
    x0, y0 : float
        World coordinates of pixel (0, 0)
    sx, sy : float
        Pixels per world unit in x and y directions
    W, H : int
        Canvas dimensions in pixels (width, height)
    """
    x0: float
    y0: float
    sx: float  # pixels per world x unit
    sy: float  # pixels per world y unit
    W: int     # width in pixels
    H: int     # height in pixels

    def to_px(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        j = int(round((x - self.x0) * self.sx))
        i = int(round((y - self.y0) * self.sy))
        # Clip to canvas bounds
        i = max(0, min(self.H - 1, i))
        j = max(0, min(self.W - 1, j))
        return i, j

    def to_world(self, i: int, j: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        x = j / self.sx + self.x0
        y = i / self.sy + self.y0
        return x, y


def make_canvas(df, target_px_per_step=DEFAULT_TARGET_PX_PER_STEP,
                target_px_per_nn=DEFAULT_TARGET_PX_PER_NN,
                margin_px=DEFAULT_CANVAS_MARGIN_PX):
    """Create Canvas transform from coordinate data (auto-detects grid vs float)."""
    x = df[COORD_X].to_numpy(float)
    y = df[COORD_Y].to_numpy(float)
    is_grid, ox, oy = _is_integer_or_shifted_grid(x, y)

    if is_grid:
        sx = sy = float(target_px_per_step)
        x_min, x_max = np.floor(x.min() - ox), np.ceil(x.max() - ox)
        y_min, y_max = np.floor(y.min() - oy), np.ceil(y.max() - oy)
        W = int((x_max - x_min) * sx) + 1 + 2 * margin_px
        H = int((y_max - y_min) * sy) + 1 + 2 * margin_px
        x0 = x_min + ox - margin_px / sx
        y0 = y_min + oy - margin_px / sy
    else:
        pts = np.column_stack([x, y])
        pitch = median_nn_distance(pts)
        sx = sy = target_px_per_nn / max(pitch, 1e-9)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        W = int(round((x_max - x_min) * sx)) + 1 + 2 * margin_px
        H = int(round((y_max - y_min) * sy)) + 1 + 2 * margin_px
        x0 = x_min - margin_px / sx
        y0 = y_min - margin_px / sy

    return Canvas(x0=x0, y0=y0, sx=sx, sy=sy, W=W, H=H)
