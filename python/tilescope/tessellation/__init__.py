"""Voronoi tessellation within ROI boundaries: GEOS engine + repair + merge.

Consensus hex geometry lives elsewhere: `spatial/hex.py` for the area↔radius
conversion, `analyzers/consensus_grid.py` for lattice + clipping + sizing.
"""

from tilescope.tessellation.voronoi import (
    select_seeds,
    rng_jitter
)

from tilescope.tessellation.builder import (
    compute_tessellation,
)

from tilescope.tessellation.repair import (
    fix_degenerate_tiles,
    merge_small_tiles_geos
)

__all__ = [
    # Geometry
    # Voronoi
    'select_seeds',
    'rng_jitter',
    # Repair & Merge (GEOS-only)
    'fix_degenerate_tiles',
    'merge_small_tiles_geos',
    # High-level API
    'compute_tessellation',
]
