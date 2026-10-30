"""``VoronoiBuildResult.assign_points_geos`` is a thin wrapper over
the shared canonical primitive ``assign_points_to_first_covering_polygon``
— its outputs must match the primitive for covered points, and must
apply nearest-seed fallback ONLY for genuinely uncovered points.

Four invariants:

- shared-boundary spots resolve to the first canonical tile;
- overlapping polygons resolve to the first canonical tile;
- uncovered points fall back to nearest seed (indexed via
  ``seed_index``, not tile index);
- ``seed_index`` may permute the seeds relative to tile order — the
  fallback still selects by seed distance, not by tile distance.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box

from tilescope.tessellation.engine_geos import VoronoiBuildResult


def _square(x0: float, y0: float, side: float = 1.0) -> Polygon:
    return box(x0, y0, x0 + side, y0 + side)


def _result(tiles, seed_index=None) -> VoronoiBuildResult:
    n = len(tiles)
    if seed_index is None:
        seed_index = list(range(n))
    return VoronoiBuildResult(
        tiles=list(tiles),
        seed_index=list(seed_index),
        precision_grid=1e-9,
    )


def test_shared_boundary_resolves_to_first_canonical_tile():
    """Two adjacent unit squares share x=1. A spot at (1, 0.5) is
    covered by both under boundary-inclusive covers(). Assignment
    must go to tile 0 (first canonical), not tile 1."""
    tiles = [_square(0.0, 0.0), _square(1.0, 0.0)]
    seeds = np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=float)
    result = _result(tiles)
    points = np.asarray([[1.0, 0.5]], dtype=float)
    assignment = result.assign_points_geos(points, seeds)
    assert assignment.tolist() == [0]


def test_overlapping_polygons_resolve_to_first_canonical_tile():
    """Two rectangles that overlap in a strip [1, 2] x [0, 1]. A
    spot at (1.5, 0.5) lies inside BOTH. Assignment must go to
    tile 0."""
    left  = box(0.0, 0.0, 2.0, 1.0)  # covers x in [0, 2]
    right = box(1.0, 0.0, 3.0, 1.0)  # covers x in [1, 3]  (overlap)
    seeds = np.asarray([[1.0, 0.5], [2.0, 0.5]], dtype=float)
    result = _result([left, right])
    points = np.asarray([[1.5, 0.5]], dtype=float)
    assert result.assign_points_geos(points, seeds).tolist() == [0]


def test_uncovered_point_falls_back_to_nearest_seed():
    """A spot outside every tile takes the KDTree nearest-seed
    fallback. Two disjoint tiles; a spot far to the right must
    assign to the right seed."""
    tiles = [_square(0.0, 0.0), _square(10.0, 0.0)]
    seeds = np.asarray([[0.5, 0.5], [10.5, 0.5]], dtype=float)
    result = _result(tiles)
    # (100, 0.5) is uncovered — nearest seed is index 1 (10.5, 0.5).
    points = np.asarray([[100.0, 0.5]], dtype=float)
    assert result.assign_points_geos(points, seeds).tolist() == [1]


def test_nontrivial_seed_index_ordering_uses_seed_positions_not_tile_positions():
    """Fallback indexes seeds via ``seed_index`` — a non-identity
    permutation must reach the correct seed. Build tiles = [A, B]
    but seed_index = [1, 0] so tile A's seed is at seeds[1] and
    tile B's seed is at seeds[0]. An uncovered spot near
    seeds[0]=(10, 0.5) must fall back to tile 0 (seed_index[0]=1
    -> use seeds[1] ... wait, semantics: KDTree is built over
    seeds_xy[seed_index], so the tree points are, in order,
    seeds[seed_index[0]], seeds[seed_index[1]], .... query returns
    the position in that array, which IS the tile index. So an
    uncovered spot close to seeds[seed_index[0]] wins tile 0)."""
    A = _square(0.0, 0.0)
    B = _square(10.0, 0.0)
    # seeds_xy[seed_index] should be tile A's seed then tile B's seed.
    # Choose seeds so seeds_xy[1] = tile A's seed, seeds_xy[0] = B's.
    seeds = np.asarray([[10.5, 0.5], [0.5, 0.5]], dtype=float)
    seed_index = [1, 0]
    result = _result([A, B], seed_index=seed_index)
    # Uncovered spot at (100, 0.5) is nearest to seeds_xy[0]=(10.5,0.5)
    # which corresponds to tree entry index 1 (tile B).
    points = np.asarray([[100.0, 0.5]], dtype=float)
    assignment = result.assign_points_geos(points, seeds)
    assert assignment.tolist() == [1], (
        "fallback should have selected tile 1 (B) whose seed is "
        f"nearest to the query; got {assignment.tolist()}"
    )
    # And a spot near the OTHER seed (near 0.5, 0.5) chooses tile 0.
    points2 = np.asarray([[-50.0, 0.5]], dtype=float)
    assignment2 = result.assign_points_geos(points2, seeds)
    assert assignment2.tolist() == [0]


def test_covered_points_bypass_fallback():
    """A point that lies clearly inside a tile must be assigned by
    the geometric path, not the fallback — even when seeds are
    positioned so nearest-seed would name a DIFFERENT tile."""
    left = _square(0.0, 0.0)
    right = _square(10.0, 0.0)
    # Seeds placed so the LEFT tile's spot (0.5, 0.5) is nearer to
    # the RIGHT seed — if the code short-circuited to nearest-seed
    # it would mis-assign to tile 1.
    seeds = np.asarray([[100.0, 0.5], [0.6, 0.5]], dtype=float)
    result = _result([left, right])
    points = np.asarray([[0.5, 0.5]], dtype=float)
    assert result.assign_points_geos(points, seeds).tolist() == [0]
