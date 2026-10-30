"""Contract tests for the region tessellation runner.

Load-bearing invariant: after tessellating a region, the returned
`Tessellation.tiles` must be the canonical hole-aware set whose
component-local `point_indices` correctly partition the region's
source rows. If the runner ever left region-local indices in place
or exposed the pre-reclip engine tiles, downstream consumers
(correlation, consensus) would read the wrong authority.

One two-region regression fixture pins the assignment invariant on
a real Voronoi run (via the GEOS engine). The other scenarios
cover the outcome branches without repeating the assertion.
"""
from __future__ import annotations

from tilescope.analyzers import region_tessellation as rt_mod
from tilescope.tessellation.repair import NoPolygonalContentError


from typing import List

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from tilescope.analyzers.region_tessellation import (
    RegionTessellationOutcome,
    _remap_point_indices,
    prepare_region,
    run_region_tessellation,
)
from tilescope.analyzers.tessellation_outcomes import RegionExecutionStatus
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, SOURCE_ROW_ID
from tilescope.component_data import ComponentData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_region_component(
    n_left: int = 25,
    n_right: int = 25,
    right_offset: float = 8.0,
) -> ComponentData:
    """Component with two clearly-separated dense regions.

    Left rows come first in `expression_data`; right rows come
    second. `SOURCE_ROW_ID` values are interleaved (odd = right,
    even = left) so a naive "region-local == component-local"
    assumption would be visibly wrong on the SOURCE_ROW_ID assertion.
    """
    rng = np.random.default_rng(0)
    left_xy = np.column_stack([
        rng.uniform(0.1, 0.9, n_left),
        rng.uniform(0.1, 0.9, n_left),
    ])
    right_xy = np.column_stack([
        right_offset + rng.uniform(0.1, 0.9, n_right),
        rng.uniform(0.1, 0.9, n_right),
    ])
    left_sids = np.arange(n_left) * 2                # even
    right_sids = np.arange(n_right) * 2 + 1          # odd
    # Component-canonical order: left region first, then right.
    coords = np.vstack([left_xy, right_xy])
    sids = np.concatenate([left_sids, right_sids])
    expr = pd.DataFrame({
        SOURCE_ROW_ID: sids,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.0, 1.0, n_left + n_right),
        GENE2: rng.uniform(0.0, 1.0, n_left + n_right),
    })
    geometry = MultiPolygon([
        box(0.0, 0.0, 1.0, 1.0),
        box(right_offset, 0.0, right_offset + 1.0, 1.0),
    ])
    return ComponentData(
        component_id="A",
        points=coords,
        canvas=None,
        polygon=geometry,
        expression_data=expr,
    )


# ---------------------------------------------------------------------------
# Regression: SOURCE_ROW_ID assignment across two regions
# ---------------------------------------------------------------------------


def _run_both_regions(
    component: ComponentData, right_offset: float = 8.0,
) -> List[RegionTessellationOutcome]:
    left_ids = component.expression_data[
        component.expression_data[COORD_X] < right_offset
    ][SOURCE_ROW_ID].tolist()
    right_ids = component.expression_data[
        component.expression_data[COORD_X] >= right_offset
    ][SOURCE_ROW_ID].tolist()
    left_polygon, right_polygon = list(component.polygon.geoms)
    outcomes = [
        run_region_tessellation(
            prepared=prepare_region(
                component_id="A", domain_region_id=0,
                region_polygon=left_polygon, component_data=component,
                source_row_ids_in_region=sorted(left_ids),
            ),
            replicate_index=0,
            tile_min_points=3, voronoi_fraction=0.3, rng_seed=42,
        ),
        run_region_tessellation(
            prepared=prepare_region(
                component_id="A", domain_region_id=1,
                region_polygon=right_polygon, component_data=component,
                source_row_ids_in_region=sorted(right_ids),
            ),
            replicate_index=0,
            tile_min_points=3, voronoi_fraction=0.3, rng_seed=43,
        ),
    ]
    return outcomes


def _tile_positions(tile) -> np.ndarray:
    return tile.point_indices


def test_two_region_source_row_id_assignment_is_exact():
    # Four-way invariant on the canonical hole-aware tile authority
    # (returned via `outcome.tessellation.tiles`):
    #   1. Per region, the union of tile-assigned SOURCE_ROW_IDs
    #      equals the region's expected source-id set exactly.
    #   2. Tile assignments within a region are pairwise disjoint.
    #   3. Every tile's `n_spots` matches its remapped point count.
    #   4. Every tile carries the correct `domain_region_id`.
    component = _two_region_component()
    outcomes = _run_both_regions(component)
    expr = component.expression_data
    src_ids = expr[SOURCE_ROW_ID].to_numpy()

    left_ids = set(expr.loc[expr[COORD_X] < 8.0, SOURCE_ROW_ID].tolist())
    right_ids = set(expr.loc[expr[COORD_X] >= 8.0, SOURCE_ROW_ID].tolist())
    assert left_ids.isdisjoint(right_ids)

    for outcome, expected_ids in zip(outcomes, (left_ids, right_ids)):
        assert outcome.diagnostic.region_status is RegionExecutionStatus.CONTRIBUTED
        assert outcome.tessellation is not None
        tiles = outcome.tessellation.tiles

        per_tile_sets = []
        for tile in tiles:
            positions = _tile_positions(tile)
            claimed_ids = set(src_ids[positions].tolist())
            per_tile_sets.append(claimed_ids)

            # (3) Reported spot count matches the remapped assignment.
            assert tile.n_spots == len(claimed_ids), (
                f"Tile {tile.tile_id}: n_spots={tile.n_spots} "
                f"disagrees with remapped assignment size {len(claimed_ids)}."
            )
            # (4) Tile carries the region id it came from.
            assert tile.domain_region_id == outcome.diagnostic.domain_region_id

        # (1) Union equals expected set exactly.
        union = set().union(*per_tile_sets) if per_tile_sets else set()
        assert union == expected_ids, (
            f"Region {outcome.diagnostic.domain_region_id}: tile-assigned "
            f"source_row_ids union {sorted(union)} does not equal "
            f"expected {sorted(expected_ids)} — under-coverage or leak."
        )
        # (2) Pairwise disjoint within region.
        total = sum(len(s) for s in per_tile_sets)
        assert total == len(union), (
            f"Region {outcome.diagnostic.domain_region_id}: tile "
            f"assignments overlap — sum-of-sizes {total} != union size "
            f"{len(union)}."
        )


# ---------------------------------------------------------------------------
# Region-execution branches
# ---------------------------------------------------------------------------


def test_one_or_two_tile_region_is_contributed_not_insufficient():
    # Small region with only ~4 spots. The engine may produce 1-3
    # tiles here; whatever it produces, the status must be
    # CONTRIBUTED (the ≥3-tile rule applies after pooling, not per
    # region).
    rng = np.random.default_rng(1)
    n = 4
    coords = np.column_stack([
        rng.uniform(0.2, 0.8, n),
        rng.uniform(0.2, 0.8, n),
    ])
    sids = np.arange(n, dtype=np.int64)
    expr = pd.DataFrame({
        SOURCE_ROW_ID: sids,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.0, 1.0, n),
        GENE2: rng.uniform(0.0, 1.0, n),
    })
    region = box(0.0, 0.0, 1.0, 1.0)
    component = ComponentData(
        component_id="A", points=coords, canvas=None, polygon=region,
        expression_data=expr,
    )
    outcome = run_region_tessellation(
        prepared=prepare_region(
            component_id="A", domain_region_id=0,
            region_polygon=region, component_data=component,
            source_row_ids_in_region=sids.tolist(),
        ),
        replicate_index=0,
        tile_min_points=1,        # let single-spot tiles survive
        voronoi_fraction=0.75,    # request many seeds
        rng_seed=17,
    )
    assert outcome.diagnostic.region_status is RegionExecutionStatus.CONTRIBUTED
    assert outcome.diagnostic.n_tiles >= 1
    assert outcome.tessellation is not None
    assert len(outcome.tessellation.tiles) == outcome.diagnostic.n_tiles


def _hole_straddling_component(n_spots: int = 12) -> ComponentData:
    """Component whose canonical region has an interior hole large
    enough that at least one Voronoi cell straddles it.

    Outer 0..10 square with a 3..7 x 3..7 hole. Spots along y=5 push
    a seed near the middle whose Voronoi cell must span the hole,
    producing a MultiPolygon after clip."""
    outer = box(0.0, 0.0, 10.0, 10.0)
    hole = Polygon([(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
    region_with_hole = outer.difference(hole)
    rng = np.random.default_rng(1)
    xs = np.linspace(0.5, 9.5, n_spots)
    ys = np.full(n_spots, 5.0)
    coords = np.column_stack([xs, ys])
    sids = np.arange(n_spots, dtype=np.int64)
    expr = pd.DataFrame({
        SOURCE_ROW_ID: sids,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.0, 1.0, n_spots),
        GENE2: rng.uniform(0.0, 1.0, n_spots),
    })
    return ComponentData(
        component_id="H", points=coords, canvas=None,
        polygon=region_with_hole, expression_data=expr,
    ), region_with_hole


def test_multipart_tile_survives_as_one_statistical_tile():
    # A Voronoi cell clipped through a retained hole
    # becomes MultiPolygon. That is one statistical tile with one
    # identity, one point assignment, and area = sum of face areas.
    component, region_polygon = _hole_straddling_component()
    outcome = run_region_tessellation(
        prepared=prepare_region(
            component_id="H", domain_region_id=7,
            region_polygon=region_polygon, component_data=component,
            source_row_ids_in_region=list(range(12)),
        ),
        replicate_index=0,
        tile_min_points=1, voronoi_fraction=0.5, rng_seed=42,
    )
    assert outcome.diagnostic.region_status is RegionExecutionStatus.CONTRIBUTED
    assert outcome.tessellation is not None
    tiles = outcome.tessellation.tiles
    assert len(tiles) >= 1
    multipart = [t for t in tiles if isinstance(t.polygon, MultiPolygon)]
    assert len(multipart) >= 1, (
        f"Expected at least one MultiPolygon tile from the hole-straddling "
        f"fixture, got types "
        f"{[type(t.polygon).__name__ for t in tiles]}"
    )
    for t in multipart:
        # (a) identity preserved
        assert isinstance(t.tile_id, int)
        # (b) point assignment preserved (n_spots derived from remapped
        #     indices; should equal the integer array length)
        assert t.n_spots == t.point_indices.size
        # (c) area equals total face area
        face_area_sum = sum(f.area for f in t.polygon.geoms)
        assert t.area == pytest.approx(face_area_sum)
        # (d) domain_region_id survives the runner
        assert t.domain_region_id == 7


def test_replicate_with_multipart_tiles_remains_contributed():
    # A region that produces at least one multipart tile still
    # reports CONTRIBUTED because multipart is not a
    # failure condition.
    component, region_polygon = _hole_straddling_component()
    outcome = run_region_tessellation(
        prepared=prepare_region(
            component_id="H", domain_region_id=0,
            region_polygon=region_polygon, component_data=component,
            source_row_ids_in_region=list(range(12)),
        ),
        replicate_index=0,
        tile_min_points=1, voronoi_fraction=0.5, rng_seed=42,
    )
    assert outcome.diagnostic.region_status is RegionExecutionStatus.CONTRIBUTED
    assert outcome.diagnostic.n_tiles == len(outcome.tessellation.tiles)
    assert outcome.diagnostic.n_tiles >= 1


def test_no_polygonal_content_produces_correct_status_and_zero_tiles(monkeypatch):
    # If the reclip helper raises NoPolygonalContentError for any
    # tile, the runner invalidates the whole region-replicate and
    # emits TILE_NO_POLYGONAL_CONTENT with n_tiles=0. Integrity
    # failure: an engine-assigned tile has no spatial support.

    def _always_fail(**_kwargs):
        raise NoPolygonalContentError(tile_id="X", residue_summary="empty")

    monkeypatch.setattr(rt_mod, "clip_to_region_and_refresh", _always_fail)

    component, region_polygon = _hole_straddling_component()
    outcome = run_region_tessellation(
        prepared=prepare_region(
            component_id="H", domain_region_id=3,
            region_polygon=region_polygon, component_data=component,
            source_row_ids_in_region=list(range(12)),
        ),
        replicate_index=1,
        tile_min_points=1, voronoi_fraction=0.5, rng_seed=42,
    )
    assert outcome.tessellation is None
    assert outcome.diagnostic.region_status is (
        RegionExecutionStatus.TILE_NO_POLYGONAL_CONTENT
    )
    assert outcome.diagnostic.n_tiles == 0
    assert outcome.diagnostic.domain_region_id == 3
    assert outcome.diagnostic.replicate_index == 1


def test_missing_source_row_id_raises():
    # Spatial attachment rejects missing SOURCE_ROW_ID at construction.
    coords = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="missing required 'source_row_id'"):
        ComponentData(
            component_id="A", points=coords, canvas=None,
            polygon=box(0, 0, 1, 1),
            expression_data=pd.DataFrame({
                COORD_X: [0.5], COORD_Y: [0.5],
                GENE1: [1.0], GENE2: [1.0],
            }),
        )


def test_empty_region_source_ids_raises():
    # Runner is not the place to skip empty regions — the stage's
    # eligibility gate is.
    component = _two_region_component()
    with pytest.raises(ValueError, match="no source_row_ids"):
        prepare_region(
            component_id="A", domain_region_id=0,
            region_polygon=list(component.polygon.geoms)[0],
            component_data=component, source_row_ids_in_region=[],
        )


def test_source_id_not_in_component_raises():
    component = _two_region_component()
    with pytest.raises(ValueError, match="not present in component_data"):
        prepare_region(
            component_id="A", domain_region_id=0,
            region_polygon=list(component.polygon.geoms)[0],
            component_data=component,
            source_row_ids_in_region=[10_000_000],  # not a legit id
        )


# ---------------------------------------------------------------------------
# _remap_point_indices — integer membership and rejected legacy inputs
# ---------------------------------------------------------------------------


def test_remap_integer_indices():
    # Region positions 0,1,2 → component positions 10,20,30.
    mapping = np.array([10, 20, 30], dtype=np.int64)
    out = _remap_point_indices(
        np.array([0, 2, 1]), mapping, component_row_count=50,
    )
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, [10, 20, 30])


def test_remap_boolean_mask_rejected():
    mapping = np.array([10, 20, 30], dtype=np.int64)
    mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="not Boolean masks"):
        _remap_point_indices(mask, mapping, component_row_count=50)


def test_remap_out_of_range_integer_raises():
    mapping = np.array([10, 20], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        _remap_point_indices(
            np.array([0, 5]), mapping, component_row_count=50,
        )


def test_remap_boolean_length_does_not_enable_coercion():
    mapping = np.array([10, 20], dtype=np.int64)
    with pytest.raises(ValueError, match="not Boolean masks"):
        _remap_point_indices(
            np.array([True, False, True]), mapping, component_row_count=50,
        )
