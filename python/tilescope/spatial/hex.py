"""
Hexagon geometry utilities.
"""

import math

import numpy as np
from shapely.geometry import Polygon


def hex_radius_from_area(area: float) -> float:
    """Convert target hexagon area to radius (center-to-vertex).

    Args:
        area: Hexagon area in world units²

    Returns:
        Radius (center-to-vertex distance) in world units

    Note:
        Regular hexagon area = (3√3/2) × r²
        Solving for r: r = √(2A / 3√3)
    """
    return math.sqrt((2.0 * area) / (3.0 * math.sqrt(3.0)))


# Flat-top hexagon vertex angles (degrees): a vertex at 0° (rightmost)
# and edges horizontal at top/bottom. Used to derive the shared vertex
# ring for every hex constructed anywhere in TileScope.
_FLAT_TOP_VERTEX_ANGLES_RAD = np.deg2rad(
    np.array([0, 60, 120, 180, 240, 300], dtype=float)
)


def make_flat_top_hexagon(
    center_x: float, center_y: float, radius: float,
) -> Polygon:
    """Return a regular flat-top hexagon polygon centred at
    `(center_x, center_y)` with circumradius `radius`.

    Vertices are at angles `[0°, 60°, 120°, 180°, 240°, 300°]` from
    the centre: the flat-top convention (horizontal edges at top
    and bottom). Every hex geometry in TileScope goes through this
    single constructor so vertex ordering and orientation cannot
    drift between the lattice generator and the on-demand geometry
    provider.
    """
    xs = center_x + radius * np.cos(_FLAT_TOP_VERTEX_ANGLES_RAD)
    ys = center_y + radius * np.sin(_FLAT_TOP_VERTEX_ANGLES_RAD)
    return Polygon(np.column_stack([xs, ys]))


__all__ = ["hex_radius_from_area", "make_flat_top_hexagon"]
