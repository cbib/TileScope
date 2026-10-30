"""Focused unit test for `fix_degenerate_tiles`.

Guards the sliver-merge contract of `tessellation.repair.fix_degenerate_tiles`:
a Voronoi cell clipped to a concave region boundary can produce a
tile whose area is a fraction of the median tile area (a "sliver").
The repair function merges each sliver into its best neighbour so
that (a) no tile with area below the combined `max(tau_d1*d1²,
tau_med*A_med)` threshold remains, and (b) every spot is preserved
under the union (no dropped source rows).
"""

import numpy as np
from shapely.geometry import box

from tilescope.tessellation.repair import fix_degenerate_tiles


def test_sliver_merges_into_adjacent_tile_and_conserves_spots():
    big = box(0, 0, 1, 1)
    sliver = box(1, 0.49, 1.001, 0.51)

    tiles = [
        {
            "polygon": big,
            "seed": (0.5, 0.5),
            "point_indices": np.array([0]),
            "n_spots": 1,
        },
        {
            "polygon": sliver,
            "seed": (1.0, 0.5),
            "point_indices": np.array([1]),
            "n_spots": 1,
        },
    ]

    seeds = np.array([t["seed"] for t in tiles])
    n_spots_before = sum(t["n_spots"] for t in tiles)

    repaired = fix_degenerate_tiles(tiles, seeds, d1=0.5, edge_tol=1e-9)

    assert len(repaired) == 1, "sliver should merge into the adjacent tile"
    assert repaired[0]["polygon"].area > big.area * 0.99
    assert sum(t["n_spots"] for t in repaired) == n_spots_before
    np.testing.assert_array_equal(repaired[0]["point_indices"], [0, 1])
