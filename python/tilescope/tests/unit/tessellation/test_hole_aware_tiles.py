"""Per-tile invariants under the polygonal-support normalization contract.

`clip_to_region_and_refresh` is a pure normalization step. Its
contract:

- Polygon in                        → Polygon out (may carry interior rings)
- MultiPolygon in                   → MultiPolygon out (caller policy decides)
- GeometryCollection in             → union of positive-area polygonal
                                      members; non-areal residue recorded
                                      in `tile["non_areal_residue"]`
- no positive-area polygonal content → `NoPolygonalContentError`

The connected-tile requirement (rejecting MultiPolygon) lives at the
CALLER (`region_tessellation.run_region_tessellation`), not in the
helper. This suite exercises the helper only.
"""
from __future__ import annotations

from tilescope.tessellation.repair import extract_polygonal_support


import pytest
from shapely.geometry import (
    GeometryCollection, LineString, MultiPolygon, Point, Polygon, box,
)

from tilescope.tessellation.repair import (
    NoPolygonalContentError,
    clip_to_region_and_refresh,
)


def _tile(polygon, area: float = 1.0, tile_id=0) -> dict:
    return {"tile_id": tile_id, "polygon": polygon, "area": area}


# ---------------------------------------------------------------------------
# Baseline reclip behavior (unchanged by the normalization refactor)
# ---------------------------------------------------------------------------


def test_clip_refreshes_area_from_polygon():
    tile_polygon = box(0.5, 0.5, 1.5, 1.5)
    region = box(0.0, 0.0, 1.0, 1.0)
    tile = _tile(tile_polygon, area=1.0)

    clipped = clip_to_region_and_refresh(
        tile=tile, snapped_region=region, precision_grid=1e-6,
    )

    assert clipped["polygon"].bounds == (0.5, 0.5, 1.0, 1.0)
    assert clipped["area"] == pytest.approx(0.25)
    assert clipped["area"] == pytest.approx(clipped["polygon"].area)
    assert clipped["non_areal_residue"] == ""


def test_clip_is_idempotent_when_tile_already_inside_region():
    region = box(0.0, 0.0, 1.0, 1.0)
    tile_polygon = box(0.1, 0.1, 0.9, 0.9)
    tile = _tile(tile_polygon, area=tile_polygon.area)

    clipped = clip_to_region_and_refresh(
        tile=tile, snapped_region=region, precision_grid=1e-6,
    )

    assert clipped["polygon"].equals(tile_polygon)
    assert clipped["area"] == pytest.approx(0.64)


def test_clip_removes_retained_hole_area_from_tile():
    # Region has an interior hole; tile that covers the whole region
    # loses the hole area but stays a single connected Polygon (with an
    # interior ring). Interior rings are legitimate tile support.
    outer = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    hole = [(0.8, 0.8), (1.2, 0.8), (1.2, 1.2), (0.8, 1.2)]
    region = Polygon(outer, [hole])
    tile_polygon = box(0.0, 0.0, 2.0, 2.0)
    tile = _tile(tile_polygon, area=4.0)

    clipped = clip_to_region_and_refresh(
        tile=tile, snapped_region=region, precision_grid=1e-6,
    )

    # Polygon.area = outer − holes = 4 − 0.16 = 3.84.
    assert isinstance(clipped["polygon"], Polygon)
    assert clipped["area"] == pytest.approx(3.84)
    # Hole geometry excluded → the center is not covered by the tile.
    assert not clipped["polygon"].covers(Point(1.0, 1.0))
    # Interior ring is retained on the tile (not filled).
    assert len(list(clipped["polygon"].interiors)) == 1


# ---------------------------------------------------------------------------
# Polygonal-content normalization semantics.
# ---------------------------------------------------------------------------


def test_polygon_with_interior_hole_survives_normalization():
    # A tile whose own geometry is Polygon-with-hole (e.g. it wraps
    # around an interior ring of the region) must survive the helper
    # unchanged as a Polygon — no MultiPolygon promotion, no hole
    # filling, no loss of the ring.
    outer = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
    hole = [(1.2, 1.2), (1.8, 1.2), (1.8, 1.8), (1.2, 1.8)]
    tile_polygon = Polygon(outer, [hole])
    region = box(0.0, 0.0, 3.0, 3.0)
    tile = _tile(tile_polygon, area=tile_polygon.area, tile_id="ringed")

    clipped = clip_to_region_and_refresh(
        tile=tile, snapped_region=region, precision_grid=1e-6,
    )

    assert isinstance(clipped["polygon"], Polygon)
    assert len(list(clipped["polygon"].interiors)) == 1
    # Area equals the polygon's own outer − holes.
    assert clipped["area"] == pytest.approx(9.0 - 0.36)
    assert clipped["non_areal_residue"] == ""


def test_multipolygon_is_preserved_by_normalization():
    # A tile that, after clipping against a hole-bearing region,
    # becomes a MultiPolygon must be returned as a MultiPolygon.
    # (The runner separately decides its own connected-tile policy.)
    outer = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    hole = [(0.7, 0.3), (1.3, 0.3), (1.3, 0.7), (0.7, 0.7)]
    region = Polygon(outer, [hole])
    tile_polygon = box(0.2, 0.35, 1.8, 0.65)  # straddles the hole
    tile = _tile(tile_polygon, area=tile_polygon.area, tile_id="cut")

    clipped = clip_to_region_and_refresh(
        tile=tile, snapped_region=region, precision_grid=1e-6,
    )

    assert isinstance(clipped["polygon"], MultiPolygon)
    assert len(clipped["polygon"].geoms) == 2
    # Total area equals the sum of both faces.
    assert clipped["area"] == pytest.approx(
        sum(face.area for face in clipped["polygon"].geoms)
    )
    assert clipped["non_areal_residue"] == ""


def test_geometry_collection_extracts_polygonal_members():
    # Hand-built GeometryCollection of a polygon + line + point.
    # The extractor keeps the polygon and records the non-areal
    # geometry types in `residue`.

    coll = GeometryCollection([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        LineString([(2, 2), (3, 3)]),
        Point(5, 5),
    ])
    support, residue = extract_polygonal_support(coll)
    assert isinstance(support, Polygon)
    assert support.area == pytest.approx(1.0)
    assert "LineString" in residue and "Point" in residue


def test_nested_geometry_collection_extracts_polygonal_members():
    # Nested collection: outer wraps an inner collection that itself
    # carries the Polygon + LineString. The old implementation stopped
    # at the outer boundary and returned `None`; the corrected one must
    # recurse and accumulate residue through arbitrary depth.

    inner = GeometryCollection([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        LineString([(2, 2), (3, 3)]),
    ])
    outer = GeometryCollection([inner, Point(5, 5)])
    support, residue = extract_polygonal_support(outer)
    assert isinstance(support, Polygon)
    assert support.area == pytest.approx(1.0)
    assert "LineString" in residue
    assert "Point" in residue


def test_pure_line_or_point_collection_raises_no_polygonal_content():
    # A collection containing only lines/points has no positive-area
    # polygonal support; the assigned spots cannot be placed against
    # empty geometry. Must fail loud, never silently drop.

    only_non_areal = GeometryCollection([
        LineString([(0, 0), (1, 1)]),
        Point(0.5, 0.5),
    ])
    support, residue = extract_polygonal_support(only_non_areal)
    assert support is None
    assert "non_areal" in residue

    # And end-to-end via the public helper (using a region that will
    # yield an empty intersection — same fail-loud path).
    disjoint_region = box(100.0, 100.0, 101.0, 101.0)
    tile_polygon = box(0.0, 0.0, 1.0, 1.0)
    tile = _tile(tile_polygon, area=tile_polygon.area, tile_id="orphan")

    with pytest.raises(NoPolygonalContentError) as exc:
        clip_to_region_and_refresh(
            tile=tile, snapped_region=disjoint_region, precision_grid=1e-6,
        )
    assert exc.value.tile_id == "orphan"
    assert exc.value.residue_summary  # non-empty description
