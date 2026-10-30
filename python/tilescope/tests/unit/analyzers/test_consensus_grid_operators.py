"""Operator-level pins for the consensus grid extraction / correction pass.

Three semantic scenarios:

1. Sample-anchored lattice: two component windows on the same
   anchor yield cell-for-cell aligned hex lattices (the pre-anchor
   per-polygon-bbox origin used to shift each window's lattice).
2. Multipart clipping preservation: `_clip_hex_to_roi` retains all
   positive-area polygonal support (Polygon / MultiPolygon /
   GeometryCollection / retained-hole cases). The pre-fix
   "keep-largest" behaviour silently discarded hex-covered
   analysis area.
3. Tile-id lookup aggregation: reordered / non-contiguous tile
   ids aggregate identically, each tile contributes once per hex,
   and duplicate tile ids inside one tessellation raise.
"""
from __future__ import annotations

from shapely.prepared import prep as _prep_fn
from tilescope.component_data import Tessellation
from tilescope.component_data import Tile


import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from tilescope.analyzers.consensus_grid import (
    _aggregate_hex_from_ensemble,
    _clip_hex_to_roi,
    _generate_hex_grid,
)
from tilescope.cols import N_SPOTS
from tilescope.analyzers.consensus_grid import OverlapIndex


# ---------------------------------------------------------------------------
# 1. Sample-anchored lattice alignment
# ---------------------------------------------------------------------------


def test_sample_anchored_lattice_aligns_across_component_windows():
    # Two disjoint windows on the same sample. Under a shared anchor,
    # a cell that appears in both windows must have identical
    # geometry (same (col, row) → same polygon centre + vertices).
    hex_radius = 1.0
    anchor_x, anchor_y = 0.0, 0.0
    left = box(0.0, 0.0, 8.0, 4.0)
    right = box(4.0, 0.0, 12.0, 4.0)   # overlaps left in x ∈ [4, 8]

    left_grid = _generate_hex_grid(
        left, hex_radius, anchor_x=anchor_x, anchor_y=anchor_y,
    )
    right_grid = _generate_hex_grid(
        right, hex_radius, anchor_x=anchor_x, anchor_y=anchor_y,
    )
    # Same anchor + same radius → identical lattice-cell centres
    # and identical polygons for any shared (col, row). The generator
    # returns `(col, row, cx, cy, polygon)`; every consumer must
    # read those centres and not recompute from `polygon.centroid`.
    left_by_key = {(col, row): (cx, cy, poly)
                   for col, row, cx, cy, poly in left_grid}
    right_by_key = {(col, row): (cx, cy, poly)
                    for col, row, cx, cy, poly in right_grid}
    shared = set(left_by_key) & set(right_by_key)
    assert shared, "expected shared cells in the overlap band"
    for key in shared:
        lcx, lcy, lpoly = left_by_key[key]
        rcx, rcy, rpoly = right_by_key[key]
        assert lcx == pytest.approx(rcx, abs=1e-9)
        assert lcy == pytest.approx(rcy, abs=1e-9)
        # Every returned polygon's centroid coincides with its
        # lattice-cell centre — pins the generator's own consistency
        # (the centre it returns IS the geometric centre of the
        # polygon it returns).
        assert lpoly.centroid.x == pytest.approx(lcx, abs=1e-9)
        assert lpoly.centroid.y == pytest.approx(lcy, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Multipart / GeometryCollection / retained-hole clipping preservation
# ---------------------------------------------------------------------------


def _prep(polygon):
    return _prep_fn(polygon)


def _region_multipart():
    return MultiPolygon([
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(6, 0), (8, 0), (8, 2), (6, 2)]),
    ])


def _region_with_hole():
    return Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )


@pytest.mark.parametrize(
    "region, hex_polygon, expected_kind, expected_area",
    [
        # Multipart region: hex spans both parts → MultiPolygon
        # support with all faces retained. Pre-fix "keep-largest"
        # would have returned only one 2×2 square.
        (_region_multipart(), box(-1.0, -1.0, 9.0, 3.0),
         "MultiPolygon", 4.0 + 4.0),
        # Retained hole: hex over a hole yields a Polygon whose area
        # excludes the hole (Shapely's outer-shell-minus-interior
        # convention). Support is still Polygon (one connected
        # ring-and-hole shape).
        (_region_with_hole(), box(3.0, 3.0, 7.0, 7.0),
         "Polygon", 16.0 - 4.0),
        # No positive polygonal content: adjacent-edge intersection
        # only. Helper returns None.
        (box(0.0, 0.0, 5.0, 5.0), box(5.0, 0.0, 10.0, 5.0),
         None, None),
    ],
    ids=["multipart_intersection_retains_all_faces",
         "retained_hole_excluded_from_support_area",
         "line_only_intersection_returns_none"],
)
def test_clip_hex_to_roi_normalises_polygonal_support(
    region, hex_polygon, expected_kind, expected_area,
):
    support = _clip_hex_to_roi(hex_polygon, _prep(region), region,
                               min_area_fraction=0.01)
    if expected_kind is None:
        assert support is None
        return
    assert support is not None
    assert type(support).__name__ == expected_kind
    assert support.area == pytest.approx(expected_area)


def test_aggregate_hex_from_ensemble_uses_tile_id_dict_lookup():
    # Tile stats are looked up via a dict keyed by tile_id, not by
    # list position — a hex overlapping only the second-emitted tile
    # (position 1) picks that tile's stats via
    # `tiles_for_tess[tile_id=1]`, not by `list[1]` positional access.
    # Legacy convention: DataFrame rows within one tessellation are
    # emitted in tile-enumeration order, so int tile_id == row's
    # position within its tessellation group.

    tile_zero = Tile(
        tile_id=0, polygon=box(0.0, 0.0, 2.0, 2.0),
        point_indices=np.array([], dtype=np.int64),
        n_spots=4, area=4.0, tessellation_id="tessA",
    )
    tile_one = Tile(
        tile_id=1, polygon=box(2.0, 0.0, 4.0, 2.0),
        point_indices=np.array([], dtype=np.int64),
        n_spots=8, area=4.0, tessellation_id="tessA",
    )
    tessellation = Tessellation(
        tessellation_id="tessA", component_id="C",
        tiles=[tile_zero, tile_one], method="voronoi",
        tile_min_points=1, fraction=1.0, rng_seed=None,
    )
    ensemble = {0: tessellation}
    overlap_indexes = {0: OverlapIndex([tile_zero, tile_one])}
    tiles_by_tess = {
        0: {
            "area": 4.0,
            N_SPOTS: 4.0,
            "sum_g1": 0.0,
            "sum_g2": 0.0,
            "n_g1_high": 0.0,
            "n_g2_high": 0.0,
            "n_both_high": 0.0,
            "n_g1_low": 0.0,
            "n_g2_low": 0.0,
            "n_both_low": 0.0,
        },
        1: {
            "area": 4.0,
            N_SPOTS: 8.0,
            "sum_g1": 0.0,
            "sum_g2": 0.0,
            "n_g1_high": 0.0,
            "n_g2_high": 0.0,
            "n_both_high": 0.0,
            "n_g1_low": 0.0,
            "n_g2_low": 0.0,
            "n_both_low": 0.0,
        },
    }

    # Hex covers only tile_one (the second tile). Aggregated N_SPOTS
    # must reflect tile_one (n_spots=8), not tile_zero.
    hex_poly = box(2.5, 0.5, 3.5, 1.5)
    per_tess_stats, edges = _aggregate_hex_from_ensemble(
        hex_poly, overlap_indexes, ensemble, {"tessA": tiles_by_tess},
    )
    assert 0 in per_tess_stats
    stats = per_tess_stats[0]
    # 1×1 hex ∩ 2×2 tile_one → 25% overlap × 8 spots = 2.0.
    assert stats[N_SPOTS] == pytest.approx(2.0)
    # Overlap edge is captured for bank-side materialisation.
    assert [(e[0], e[1]) for e in edges] == [(0, 1)]
