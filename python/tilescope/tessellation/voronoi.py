"""
Seed selection for Voronoi tessellation.
Pure computational functions (no scipy dependency).
"""

import numpy as np
from typing import Optional, Final

from tilescope.defaults import DEFAULT_VORONOI_FRACTION

# Algorithm constant for jitter robustness (avoids degenerate Voronoi vertices)
JITTER_SCALE: Final[float] = 0.25


def rng_jitter(points: np.ndarray,
              scale: float = JITTER_SCALE,
              rng: np.random.Generator = None) -> np.ndarray:
    """Add small random jitter to points to avoid degenerate Voronoi vertices."""
    if rng is None:
        rng = np.random.default_rng()
    jitter = rng.uniform(-scale, scale, size=points.shape)
    return points.astype(float) + jitter


def select_seeds(roi_coords: np.ndarray,
                fraction: float = DEFAULT_VORONOI_FRACTION,
                rng: Optional[np.random.Generator] = None,
                *,
                precomputed_unique_coords: Optional[np.ndarray] = None,
                ) -> np.ndarray:
    """Select seed points from ROI coordinates for Voronoi tessellation.

    Scale-faithful seed count: ``n_seeds = max(1, round(fraction * n_cells))``.
    This makes seed count track region size directly rather than being
    pinned to a legacy floor for small regions.

    Deterministic given a seeded ``rng``. The caller is responsible
    for ordering ``roi_coords`` reproducibly (e.g. by SOURCE_ROW_ID)
    so that repeated runs pass an identical input array.

    ``precomputed_unique_coords``: sorted unique-tuple array cached
    by ``PreparedRegion`` across K replicates on the same region. When
    supplied the ``sorted(set(...))`` reduction is skipped.
    """
    if rng is None:
        rng = np.random.default_rng()

    if precomputed_unique_coords is not None:
        unique_coords = precomputed_unique_coords
    else:
        unique_coords = np.array(sorted(set(map(tuple, roi_coords))), dtype=float)
    if unique_coords.size == 0:
        return unique_coords.reshape(0, 2)

    n_cells = unique_coords.shape[0]
    n_seeds = max(1, int(round(fraction * n_cells)))

    idx = rng.choice(n_cells, size=min(n_seeds, n_cells), replace=False)
    seeds = unique_coords[idx]

    return rng_jitter(seeds, scale=JITTER_SCALE, rng=rng)
