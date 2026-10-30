"""Canvas-derived precision grid for tessellation engine `set_precision`.

Consumed by the tessellation engine + run-metadata provenance. Not used
for post-hoc area recomputation: geometry areas are read from their
canonical records (`Tile.area`, `region_polygon.area`).
"""

from tilescope.spatial.canvas import Canvas


def precision_grid_size(canvas: Canvas) -> float:
    """1/10000 of typical world-unit spacing: the tessellation engine's
    snap grid, exposed for provenance."""
    typical_spacing = 1.0 / canvas.sx
    return typical_spacing / 1e4


__all__ = ["precision_grid_size"]
