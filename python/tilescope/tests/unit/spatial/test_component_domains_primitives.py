"""Primitive-helper tests for `spatial/component_domains.py`.

These are operator checks for properties guaranteed by construction
(Voronoi disjointness, union-of-disks topology, GeometryCollection
filtering). They verify implementation correctness, not scientific
fitness of the geometry contract.

Composition through `build_component_polygons` is tested separately.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union

from tilescope.cols import COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.component_domains import (
    _buffered_point_support,
    _build_parent_ownership,
    _compose_component_masks,
    _extract_polygonal_geometry,
    _topology_aware_point_support,
    _validate_source_coordinate_contract,
)


# ---------------------------------------------------------------------------
# _compose_component_masks
# ---------------------------------------------------------------------------


def test_mask_composition_general_path_matches_set_definitions():
    component = box(0.0, 0.0, 4.0, 4.0)
    ownership = box(1.0, -1.0, 3.0, 5.0)
    acquisition = box(-1.0, 1.0, 5.0, 3.0)

    composed = _compose_component_masks(
        component,
        ownership,
        acquisition,
        component_is_within_acquisition=False,
    )
    expected = component.intersection(ownership).intersection(acquisition)
    expected_ownership_excluded = (
        component.difference(ownership).intersection(acquisition).area
    )
    expected_acquisition_excluded = (
        component.difference(acquisition).intersection(ownership).area
    )

    assert composed.raw_geometry.equals(expected)
    assert composed.ownership_excluded_area == pytest.approx(
        expected_ownership_excluded,
    )
    assert composed.acquisition_excluded_area == pytest.approx(
        expected_acquisition_excluded,
    )


def test_mask_composition_containment_fast_path_matches_set_definitions():
    component = box(0.0, 0.0, 4.0, 4.0)
    ownership = box(1.0, -1.0, 3.0, 5.0)
    acquisition = box(-1.0, -1.0, 5.0, 5.0)

    composed = _compose_component_masks(
        component,
        ownership,
        acquisition,
        component_is_within_acquisition=True,
    )
    expected = component.intersection(ownership).intersection(acquisition)

    assert composed.raw_geometry.equals(expected)
    assert composed.ownership_excluded_area == pytest.approx(
        component.difference(ownership).intersection(acquisition).area,
    )
    assert composed.acquisition_excluded_area == pytest.approx(0.0)


def _source_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[SOURCE_ROW_ID] = np.arange(len(df), dtype=np.int64)
    return df


def _edges_within(coords: np.ndarray, distance: float) -> set[tuple[int, int]]:
    """Canonical test edges whose Euclidean length is within a cutoff."""
    return {
        (left, right)
        for left in range(len(coords))
        for right in range(left + 1, len(coords))
        if np.linalg.norm(coords[left] - coords[right]) <= distance
    }


# ---------------------------------------------------------------------------
# _validate_source_coordinate_contract
# ---------------------------------------------------------------------------


def test_validate_accepts_valid_frame():
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 5.0, COORD_Y: 5.0, ROI_ID: "B"},
    ])
    # Must not raise
    _validate_source_coordinate_contract(df)


def test_validate_raises_on_non_finite():
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: float("nan"), COORD_Y: 0.0, ROI_ID: "A"},
    ])
    with pytest.raises(ValueError, match=r"non-finite"):
        _validate_source_coordinate_contract(df)


def test_validate_raises_on_missing_roi_id():
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: None},
    ])
    with pytest.raises(ValueError, match=r"missing 'roi_id'"):
        _validate_source_coordinate_contract(df)


def test_validate_raises_on_duplicate_xy_same_label():
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
    ])
    with pytest.raises(ValueError, match=r"duplicated \(x, y\)"):
        _validate_source_coordinate_contract(df)


def test_validate_raises_on_conflicting_roi_id():
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "B"},
    ])
    with pytest.raises(ValueError, match=r"conflicting 'roi_id'"):
        _validate_source_coordinate_contract(df)


def test_validate_raises_on_missing_column():
    df = pd.DataFrame({COORD_X: [0.0], COORD_Y: [0.0]})  # no ROI_ID
    df[SOURCE_ROW_ID] = np.arange(len(df), dtype=np.int64)
    with pytest.raises(ValueError, match=r"missing required column"):
        _validate_source_coordinate_contract(df)


# ---------------------------------------------------------------------------
# _buffered_point_support: whole-sample footprints
# ---------------------------------------------------------------------------


def test_acquisition_area_matches_sum_of_disjoint_disks():
    # Three disks at (0, 0), (10, 0), (20, 0) with radius 1 — well-
    # separated; union area ≈ 3 · π · r² up to Shapely's buffer
    # discretization (~0.16% low at the default quad_segs=16).
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 20.0, COORD_Y: 0.0, ROI_ID: "A"},
    ])
    A = _buffered_point_support(df[[COORD_X, COORD_Y]].to_numpy(), radius=1.0)
    assert A.area == pytest.approx(3 * math.pi, rel=5e-3)


def test_acquisition_support_hex_ring_creates_interior_hole():
    # Six hex-neighbours of the origin at distance d=1.0. With
    # r_A = 0.60·d, adjacent neighbours (chord distance d apart)
    # overlap (sum of radii = 1.2 > 1.0), so the union is CONNECTED;
    # the origin (distance 1 from every neighbour) is uncovered
    # because r_A < d. The result is a single Polygon with one
    # interior ring — a genuine acquisition-hole topology, not
    # exterior space between islands.
    d = 1.0
    r_A = 0.60 * d
    s = math.sqrt(3.0) / 2.0
    neighbours = [
        (d, 0.0),
        (0.5 * d, s * d),
        (-0.5 * d, s * d),
        (-d, 0.0),
        (-0.5 * d, -s * d),
        (0.5 * d, -s * d),
    ]
    df = _source_frame([
        {COORD_X: x, COORD_Y: y, ROI_ID: "A"} for (x, y) in neighbours
    ])
    A = _buffered_point_support(df[[COORD_X, COORD_Y]].to_numpy(), radius=r_A)
    # Connected: a single Polygon, not a MultiPolygon of six islands.
    assert isinstance(A, Polygon)
    # Interior ring exists → topological hole (not exterior space).
    assert len(A.interiors) == 1
    # The missing centre is not covered.
    assert not A.covers(Point(0.0, 0.0))


def test_acquisition_support_disconnected_islands_produce_multipolygon():
    # Two clusters far apart — union is a MultiPolygon.
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "B"},
    ])
    A = _buffered_point_support(df[[COORD_X, COORD_Y]].to_numpy(), radius=1.0)
    assert isinstance(A, MultiPolygon)
    assert len(A.geoms) == 2


def test_topology_support_fills_ordinary_square_sampling_cell():
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 1.0],
    ])
    support = _topology_aware_point_support(
        coords, radius=0.60, allowed_edges=_edges_within(coords, 1.50),
    )
    assert support.covers(Point(0.5, 0.5))
    assert isinstance(support, Polygon)
    assert len(support.interiors) == 0


def test_topology_support_preserves_missing_hex_centre():
    s = math.sqrt(3.0) / 2.0
    coords = np.array([
        [1.0, 0.0], [0.5, s], [-0.5, s],
        [-1.0, 0.0], [-0.5, -s], [0.5, -s],
    ])
    support = _topology_aware_point_support(
        coords, radius=0.60, allowed_edges=_edges_within(coords, 1.50),
    )
    assert not support.covers(Point(0.0, 0.0))
    assert isinstance(support, Polygon)
    assert len(support.interiors) == 1


@pytest.mark.parametrize("max_edge", [None, 3.5])
def test_topology_face_boundary_matches_legacy_face_overlay(max_edge):
    """The faster planar-face union preserves the former support set."""
    rng = np.random.default_rng(4201)
    coords = rng.uniform(0.0, 10.0, size=(40, 2))
    simplices = Delaunay(coords).simplices
    candidate_edges = {
        tuple(sorted((int(left), int(right))))
        for simplex in simplices
        for left, right in (
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[0], simplex[2]),
        )
    }
    allowed_edges = {
        edge for edge in candidate_edges
        if max_edge is None
        or np.linalg.norm(coords[edge[0]] - coords[edge[1]]) <= max_edge
    }
    base = unary_union([
        Point(float(x), float(y)).buffer(0.2) for x, y in coords
    ])
    retained_faces = []
    for simplex in simplices:
        simplex_edges = {
            tuple(sorted((int(simplex[0]), int(simplex[1])))),
            tuple(sorted((int(simplex[1]), int(simplex[2])))),
            tuple(sorted((int(simplex[0]), int(simplex[2])))),
        }
        if simplex_edges.issubset(allowed_edges):
            retained_faces.append(Polygon(coords[simplex]))
    legacy = unary_union([base, *retained_faces])

    optimized = _topology_aware_point_support(
        coords, radius=0.2, allowed_edges=allowed_edges,
        optimize_planar_face_union=True,
    )

    assert optimized.symmetric_difference(legacy).area == pytest.approx(
        0.0, abs=1e-9,
    )


# ---------------------------------------------------------------------------
# _buffered_point_support: component footprints
# ---------------------------------------------------------------------------


def test_component_support_single_spot_area_is_disk():
    coords = np.array([[0.0, 0.0]])
    C = _buffered_point_support(coords, radius=1.0)
    # Shapely buffer discretization is ~0.16% short of π at default quad_segs.
    assert C.area == pytest.approx(math.pi, rel=5e-3)


def test_component_support_three_close_spots_form_single_polygon():
    # Spots at (0, 0), (1.5, 0), (3.0, 0) with radius 1 — overlapping
    # disks; union is one connected Polygon.
    coords = np.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]])
    C = _buffered_point_support(coords, radius=1.0)
    assert isinstance(C, Polygon)


def test_component_support_empty_returns_empty_polygon():
    coords = np.zeros((0, 2))
    C = _buffered_point_support(coords, radius=1.0)
    assert C.is_empty


def test_component_support_covers_every_supplied_spot():
    # Every supplied component spot must be inside its own disk union.
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    C = _buffered_point_support(coords, radius=0.75)
    for x, y in coords:
        assert C.covers(Point(x, y)), f"seed ({x}, {y}) not covered by C_c"


def test_component_support_separated_spots_produce_multipolygon():
    # Component identity does NOT imply connected final support.
    # Two spots far apart with r=0.75 disks give a MultiPolygon.
    coords = np.array([[0.0, 0.0], [100.0, 0.0]])
    C = _buffered_point_support(coords, radius=0.75)
    assert isinstance(C, MultiPolygon)
    assert len(C.geoms) == 2


# ---------------------------------------------------------------------------
# _build_parent_ownership  (OPERATOR CHECKS — Voronoi properties)
# ---------------------------------------------------------------------------


def _envelope(minx=-100.0, miny=-100.0, maxx=100.0, maxy=100.0):
    return box(minx, miny, maxx, maxy)


def test_ownership_every_supported_seed_is_in_its_parent_domain():
    # Two parents, three seeds each, well separated.
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.5, COORD_Y: 1.0, ROI_ID: "A"},
        {COORD_X: 20.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 21.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 20.5, COORD_Y: 1.0, ROI_ID: "B"},
    ])
    spatial = df.copy()
    owned = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=[],
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    assert set(owned.keys()) == {"A", "B"}
    for _, row in spatial.iterrows():
        parent = str(row[ROI_ID])
        pt = Point(float(row[COORD_X]), float(row[COORD_Y]))
        assert owned[parent].contains(pt) or owned[parent].intersects(pt), (
            f"seed ({row[COORD_X]}, {row[COORD_Y]}) with parent {parent!r} "
            f"is not inside its owned domain"
        )


def test_ownership_cross_parent_domains_are_materially_disjoint():
    # OPERATOR CHECK — Voronoi partitions the plane, so two parents'
    # owned unions can overlap only on a set of measure zero
    # (shared boundary segments).
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 11.0, COORD_Y: 0.0, ROI_ID: "B"},
    ])
    spatial = df.copy()
    owned = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=[],
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    overlap = owned["A"].intersection(owned["B"]).area
    assert overlap == pytest.approx(0.0, abs=1e-9)


def test_ownership_partitions_envelope_when_no_exclusions():
    # OPERATOR CHECK — disjointness + exhaustiveness together prove
    # that ownership is a true partition of the envelope. A faulty
    # implementation could satisfy disjointness while leaving large
    # areas unassigned; the symmetric difference test catches that.
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 11.0, COORD_Y: 0.0, ROI_ID: "B"},
    ])
    spatial = df.copy()
    envelope = _envelope()
    owned = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=[],
        envelope=envelope,
        exclusion_radius=1.0,
    )
    partition = unary_union(list(owned.values()))
    residual = partition.symmetric_difference(envelope).area
    # Envelope area is 200 × 200 = 40 000; tolerance ~1e-6 easily
    # catches missing cells while allowing snap-boundary residue.
    assert residual == pytest.approx(0.0, abs=1e-6)


def test_ownership_exclusion_removes_material_area_from_both_straddled_parents():
    # The ineligible spot or cell at (5, 5) sits exactly on the
    # perpendicular bisector between A(0, 0) and B(10, 0). Its r=1
    # exclusion disk straddles the boundary, so BOTH parents must
    # lose material area. The ineligible spot itself is ROI_ID "A" —
    # this doubles as a check that an ineligible spot blocks its own
    # parent as well as foreign parents, including supported siblings.
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 5.0, COORD_Y: 5.0, ROI_ID: "A"},   # ineligible below
    ])
    spatial = df.iloc[[0, 1]].copy()
    ineligible = [int(df.iloc[2][SOURCE_ROW_ID])]

    owned_no = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=[],
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    owned_yes = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=ineligible,
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    delta_A = owned_no["A"].area - owned_yes["A"].area
    delta_B = owned_no["B"].area - owned_yes["B"].area

    # By construction, the disk is symmetric about x=5, so each side
    # is ~π/2. Both losses must be substantial — this rules out a
    # faulty implementation that removes the whole disk from one
    # parent and nothing from the other.
    exclusion_area = math.pi  # up to Shapely buffer discretization
    assert delta_A > 0.4 * exclusion_area
    assert delta_B > 0.4 * exclusion_area
    # Total loss ≈ full disk area.
    assert delta_A + delta_B == pytest.approx(exclusion_area, rel=5e-3)
    # The ineligible point is not inside either parent's owned domain.
    pt = Point(5.0, 5.0)
    assert not owned_yes["A"].contains(pt)
    assert not owned_yes["B"].contains(pt)


def test_ownership_exclusion_indexed_by_source_row_id_not_row_position():
    # Non-default index + SOURCE_ROW_ID shuffled so position ≠ src_id.
    # If the builder ever regresses to positional or index-based
    # lookup, the exclusion lands at the wrong coordinate and the
    # assertions fail.
    #
    # Row layout (position → src_id → coord):
    #   pos 0 → src 5  → (30, 30) — B, supported
    #   pos 1 → src 3  → (5, 5)   — A, INELIGIBLE
    #   pos 2 → src 8  → (50, 50) — B, supported
    #   pos 3 → src 0  → (0, 0)   — A, supported
    #   pos 4 → src 10 → (20, 0)  — B, supported
    #
    # Under src_id lookup for ineligible=[3]: exclusion at (5, 5).
    # Under positional lookup df.iloc[3]: coord (0, 0) — A's own
    # supported seed; excluding it would leave A's seed uncovered.
    rows = [
        {SOURCE_ROW_ID: 5, COORD_X: 30.0, COORD_Y: 30.0, ROI_ID: "B"},
        {SOURCE_ROW_ID: 3, COORD_X: 5.0, COORD_Y: 5.0, ROI_ID: "A"},
        {SOURCE_ROW_ID: 8, COORD_X: 50.0, COORD_Y: 50.0, ROI_ID: "B"},
        {SOURCE_ROW_ID: 0, COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {SOURCE_ROW_ID: 10, COORD_X: 20.0, COORD_Y: 0.0, ROI_ID: "B"},
    ]
    df = pd.DataFrame(rows, index=[100, 200, 300, 400, 500])
    df[SOURCE_ROW_ID] = df[SOURCE_ROW_ID].astype(np.int64)
    spatial = df[df[SOURCE_ROW_ID] != 3].copy()

    owned = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=[3],
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    # Exclusion is centred at src_id=3 → (5, 5).
    assert not owned["A"].contains(Point(5.0, 5.0))
    assert not owned["B"].contains(Point(5.0, 5.0))
    # A's own supported seed at (0, 0) is inside A's domain — which
    # rules out a positional-lookup regression (that would have
    # excluded (0, 0) instead).
    assert owned["A"].contains(Point(0.0, 0.0))


def test_ownership_parent_with_zero_supported_spots_absent_from_result():
    # ROI_ID "B" has one source row that is NOT in spatial — its
    # ownership entry should simply be absent.
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 50.0, COORD_Y: 50.0, ROI_ID: "B"},  # ineligible → not in spatial
    ])
    spatial = df.iloc[[0, 1]].copy()
    ineligible = [int(df.iloc[2][SOURCE_ROW_ID])]
    owned = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=ineligible,
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    assert set(owned.keys()) == {"A"}


def test_ownership_empty_spatial_returns_empty_dict():
    df = _source_frame([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
    ])
    spatial = df.iloc[[]].copy()
    owned = _build_parent_ownership(
        source_expression_df=df,
        spatial_component_expression_df=spatial,
        ineligible_src_row_ids=[int(df.iloc[0][SOURCE_ROW_ID])],
        envelope=_envelope(),
        exclusion_radius=1.0,
    )
    assert owned == {}


# ---------------------------------------------------------------------------
# _extract_polygonal_geometry
# ---------------------------------------------------------------------------


def test_extract_returns_polygon_unchanged():
    p = box(0, 0, 1, 1)
    assert _extract_polygonal_geometry(p) is p


def test_extract_returns_multipolygon_unchanged():
    mp = MultiPolygon([box(0, 0, 1, 1), box(10, 10, 11, 11)])
    assert _extract_polygonal_geometry(mp) is mp


def test_extract_strips_points_and_lines_from_geometrycollection():
    gc = GeometryCollection([
        box(0, 0, 1, 1),
        Point(5, 5),
        LineString([(10, 10), (11, 11)]),
    ])
    out = _extract_polygonal_geometry(gc)
    assert isinstance(out, Polygon)
    assert out.area == pytest.approx(1.0)


def test_extract_returns_empty_polygon_for_geometrycollection_with_no_polygons():
    gc = GeometryCollection([Point(0, 0), LineString([(1, 1), (2, 2)])])
    out = _extract_polygonal_geometry(gc)
    assert out.is_empty
    assert isinstance(out, Polygon)


def test_extract_returns_empty_polygon_for_empty_input():
    out = _extract_polygonal_geometry(Polygon())
    assert out.is_empty
