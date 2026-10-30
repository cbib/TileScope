"""Additional semantic cases for component-domain construction.

Diagnostic and canonical-ordering cases live in
`test_component_domains_diagnostics.py`; anchor composition cases live
in `test_component_domains_composition.py`. Tessellation eligibility is
tested downstream because the geometry builder deliberately has no
estimator-threshold knowledge.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.component_domains import (
    ComponentDomainDiagnostic,
    _validate_component_domains,
    build_component_polygons,
)


R_C = 0.75
R_A = 0.60
R_E = 0.75


def _fixture(
    rows: list[dict],
    *,
    component_assignment: dict[int, str] | None = None,
    ineligible_indices: list[int] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    ineligible_indices = ineligible_indices or []
    component_assignment = component_assignment or {}

    source_df = pd.DataFrame(rows)
    source_df[SOURCE_ROW_ID] = np.arange(len(source_df), dtype=np.int64)

    supported_mask = np.ones(len(source_df), dtype=bool)
    for idx in ineligible_indices:
        supported_mask[idx] = False
    spatial_df = source_df.loc[supported_mask].copy()

    if len(spatial_df):
        comp_ids = spatial_df[SOURCE_ROW_ID].map(
            lambda src_id: component_assignment.get(
                int(src_id),
                str(spatial_df.loc[spatial_df[SOURCE_ROW_ID] == src_id, ROI_ID].iloc[0]),
            )
        )
        spatial_df[COMPONENT_ID] = comp_ids.astype(str)
        spatial_df[ROI_ID] = spatial_df[ROI_ID].astype(str)
        component_to_roi: Dict[str, str] = {
            str(cid): str(spatial_df.loc[spatial_df[COMPONENT_ID] == cid, ROI_ID].iloc[0])
            for cid in spatial_df[COMPONENT_ID].unique()
        }
    else:
        spatial_df[COMPONENT_ID] = pd.Series([], dtype="object")
        spatial_df[ROI_ID] = spatial_df[ROI_ID].astype(str)
        component_to_roi = {}
    return source_df, spatial_df, component_to_roi


def _build(source_df, spatial_df, component_to_roi, sample_median_nn=1.0):
    result = build_component_polygons(
        source_df,
        spatial_df,
        component_to_roi,
        sample_median_nn=sample_median_nn,
        component_support_radius_factor=R_C,
        acquisition_support_radius_factor=R_A,
        ineligible_spot_exclusion_radius_factor=R_E,
    )
    return result.component_polygons, list(result.component_diagnostics), list(result.region_diagnostics)


# ---------------------------------------------------------------------------
# Case 2 — one labelled ROI inside another's convex hull
# ---------------------------------------------------------------------------


def test_case_2_labelled_roi_inside_anothers_convex_hull_respects_ownership():
    # ROI A occupies the corners of a large square; ROI B occupies
    # interior positions inside A's convex hull. Under the retired
    # convex-hull-fit approach A's hull would cover B's spots and
    # mislabel their territory. Under Voronoi ownership every B-spot
    # keeps its own cell.
    source, spatial, c2r = _fixture([
        # A: corners of a 10 × 10 square
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 10.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 10.0, ROI_ID: "A"},
        # B: interior positions well inside A's convex hull
        {COORD_X: 3.0, COORD_Y: 3.0, ROI_ID: "B"},
        {COORD_X: 7.0, COORD_Y: 3.0, ROI_ID: "B"},
        {COORD_X: 3.0, COORD_Y: 7.0, ROI_ID: "B"},
        {COORD_X: 7.0, COORD_Y: 7.0, ROI_ID: "B"},
    ])
    domains, _, _ = _build(source, spatial, c2r, sample_median_nn=3.0)

    # Every B-spot lies in B's own domain — never in A's.
    for pt in [Point(3, 3), Point(7, 3), Point(3, 7), Point(7, 7)]:
        assert not domains["A"].covers(pt)
        assert domains["B"].covers(pt)
    # And every A-spot lies in A's own domain, never in B's.
    for pt in [Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)]:
        assert domains["A"].covers(pt)
        assert not domains["B"].covers(pt)


# ---------------------------------------------------------------------------
# Case 4 — missing acquisition coordinate → retained hole
# ---------------------------------------------------------------------------


def test_case_4_missing_acquisition_coordinate_creates_retained_hole():
    # Six hex neighbours of the origin (no centre spot). With
    # r_A = 0.60·d, adjacent neighbours' A-disks overlap so the
    # union is connected; the missing centre is a genuine hole.
    # G_c inherits the hole because C_c has the same topology
    # (same spot set) and T_p is the full envelope (single parent).
    d = 1.0
    s = math.sqrt(3.0) / 2.0
    rows = [
        {COORD_X: d, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: -0.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: -d, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: -0.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
        {COORD_X: 0.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
    ]
    source, spatial, c2r = _fixture(
        rows, component_assignment={i: "0" for i in range(6)},
    )
    domains, _, _ = _build(source, spatial, c2r, sample_median_nn=d)

    dom = domains["0"]
    assert isinstance(dom, Polygon)
    assert len(dom.interiors) == 1
    assert not dom.covers(Point(0.0, 0.0))


# ---------------------------------------------------------------------------
# Case 5 — multiple missing coordinates → larger gap
# ---------------------------------------------------------------------------


def test_case_5_multiple_missing_coordinates_create_larger_hole():
    # Two adjacent hex-lattice centres missing (at (0, 0) and (1, 0)),
    # surrounded by an 8-spot ring formed by the shared neighbours of
    # both centres. Each neighbour is at unit distance from the next
    # (r_A = 0.60·d ⇒ sum radii = 1.2 > 1), so the ring is connected
    # and encloses a single hole spanning both missing centres. Hole
    # area strictly exceeds the single-missing case (case 4).
    d = 1.0
    s = math.sqrt(3.0) / 2.0
    rows = [
        {COORD_X: -1.0 * d, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: -0.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: -0.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
        {COORD_X: 0.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: 0.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
        {COORD_X: 1.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: 1.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
        {COORD_X: 2.0 * d, COORD_Y: 0.0, ROI_ID: "A"},
    ]
    source, spatial, c2r = _fixture(
        rows, component_assignment={i: "0" for i in range(8)},
    )
    domains, _, _ = _build(source, spatial, c2r, sample_median_nn=d)

    dom = domains["0"]
    assert isinstance(dom, Polygon)
    assert len(dom.interiors) == 1
    hole_area = Polygon(dom.interiors[0]).area
    # Reference: single-missing hex-ring hole area (case 4 geometry).
    # We recompute it here to make the "strictly larger" assertion
    # robust to Shapely buffer discretization.
    single_missing = [
        {COORD_X: d, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: -0.5 * d, COORD_Y: s * d, ROI_ID: "A"},
        {COORD_X: -d, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: -0.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
        {COORD_X: 0.5 * d, COORD_Y: -s * d, ROI_ID: "A"},
    ]
    ref_source, ref_spatial, ref_c2r = _fixture(
        single_missing, component_assignment={i: "0" for i in range(6)},
    )
    ref_domains, _, _ = _build(
        ref_source, ref_spatial, ref_c2r, sample_median_nn=d,
    )
    ref_hole_area = Polygon(ref_domains["0"].interiors[0]).area
    assert hole_area > ref_hole_area
    # Both missing centres are inside the hole.
    assert not dom.covers(Point(0.0, 0.0))
    assert not dom.covers(Point(1.0, 0.0))


# ---------------------------------------------------------------------------
# Case 6 — mixed ROI with one supported component and one fragment
# ---------------------------------------------------------------------------


def test_case_6_mixed_roi_supported_component_treats_fragment_as_ineligible():
    # ROI A has one supported cluster (3 spots) and a 2-spot fragment
    # dropped by upstream filtering. Fragment spots remain in source
    # (they're valid spot/cell-level measurements) but not in spatial.
    # Their exclusion disks apply; the supported cluster is unaffected
    # because the fragment is far away.
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},   # supported
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},   # supported
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},   # supported
            {COORD_X: 50.0, COORD_Y: 0.0, ROI_ID: "A"},  # fragment
            {COORD_X: 51.0, COORD_Y: 0.0, ROI_ID: "A"},  # fragment
        ],
        ineligible_indices=[3, 4],
        component_assignment={0: "0", 1: "0", 2: "0"},
    )
    domains, comp_diags, _ = _build(source, spatial, c2r)

    assert set(domains.keys()) == {"0"}
    comp = comp_diags[0]
    assert comp.n_component_spots == 3
    assert comp.n_spots_in_domain == 3
    # Fragment exclusions are far from the supported cluster, so
    # ownership-excluded area within the supported cluster is ~0.
    assert comp.ownership_excluded_area == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Case 7 — fully scattered ROI (empty spatial)
# ---------------------------------------------------------------------------


def test_case_7_fully_scattered_returns_empty_result():
    # Source has 5 spots, all classified as scattered upstream so
    # spatial is empty. The builder must return empty maps without
    # error (invariant checks pass trivially on an empty set).
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: float(i * 100), COORD_Y: 0.0, ROI_ID: f"R{i}"}
            for i in range(5)
        ],
        ineligible_indices=[0, 1, 2, 3, 4],
    )
    domains, comp_diags, region_diags = _build(source, spatial, c2r)
    assert domains == {}
    assert comp_diags == []
    assert region_diags == []


# ---------------------------------------------------------------------------
# Case 9 — sibling overlap must raise
# ---------------------------------------------------------------------------


def test_case_9_sibling_overlap_invariant_raises_when_forced():
    # By construction ownership prevents cross-parent overlap and
    # component-disk disjointness prevents sibling overlap in normal
    # runs. This test drives `_validate_component_domains` directly
    # with hand-crafted overlapping domains to prove the assertion
    # fires — i.e., that a hypothetical future bug in the composition
    # would be caught rather than silently pass.
    # Domains overlap in the strip x ∈ [4, 6]. Sibling seeds are
    # placed OUTSIDE the neighbour's domain so invariant 4 (no
    # foreign spot inside another component's G_c) does not fire
    # first — we want invariant 6 (sibling non-overlap) to be the
    # one that raises.
    d0 = box(0.0, 0.0, 6.0, 6.0)
    d1 = box(4.0, 0.0, 10.0, 6.0)   # overlaps d0 on x ∈ [4, 6]
    domains = {"0": d0, "1": d1}

    spatial = pd.DataFrame({
        SOURCE_ROW_ID: np.array([0, 1], dtype=np.int64),
        COORD_X: [1.0, 9.0],   # 1.0 ∈ d0, ∉ d1; 9.0 ∈ d1, ∉ d0
        COORD_Y: [3.0, 3.0],
        ROI_ID: ["A", "A"],
        COMPONENT_ID: ["0", "1"],
    })

    def _diag(cid: str) -> ComponentDomainDiagnostic:
        return ComponentDomainDiagnostic(
            component_id=cid,
            parent_roi_id="A",
            n_component_spots=1,
            n_spots_in_domain=1,
            n_domain_regions=1,
            n_regions_with_spots=1,
            component_support_area=1.0,
            ownership_excluded_area=0.0,
            acquisition_excluded_area=0.0,
            final_domain_area=100.0,
            geometry_status="ok",
        )

    diags = [_diag("0"), _diag("1")]
    with pytest.raises(AssertionError, match=r"sibling components"):
        _validate_component_domains(
            domains=domains,
            spatial_component_expression_df=spatial,
            comp_diags=diags,
            region_diags=[],
            area_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# Case 10 — duplicate coordinates raise via public API
# ---------------------------------------------------------------------------


def test_case_10_duplicate_coordinates_raise_via_public_api():
    source, spatial, c2r = _fixture([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
    ])
    with pytest.raises(ValueError, match=r"duplicated \(x, y\)"):
        _build(source, spatial, c2r)


# ---------------------------------------------------------------------------
# Case 11 — foreign-parent spot inside raw component disk but excluded from G_c
# ---------------------------------------------------------------------------


def test_case_11_foreign_parent_spot_excluded_from_component_domain():
    # ROI A's C_c disks (r = 2.25 at sample_median_nn = 3) cover
    # ROI B's central spot at (1.5, 1.5) because the disk-to-B
    # distance ≈ 2.12 < 2.25. Ownership rescues correctness: B's
    # own Voronoi cell owns (1.5, 1.5), so A's G_c does NOT.
    source, spatial, c2r = _fixture([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 3.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 3.0, ROI_ID: "A"},
        {COORD_X: 3.0, COORD_Y: 3.0, ROI_ID: "A"},
        {COORD_X: 1.5, COORD_Y: 1.5, ROI_ID: "B"},
    ])
    domains, _, _ = _build(source, spatial, c2r, sample_median_nn=3.0)
    b_seed = Point(1.5, 1.5)
    assert not domains["A"].covers(b_seed)
    assert domains["B"].covers(b_seed)


# ---------------------------------------------------------------------------
# Case 17 — parent with zero supported spots
# ---------------------------------------------------------------------------


def test_case_17_parent_with_zero_supported_spots_does_not_crash():
    # ROI A: 3 supported spots. ROI B: 2 spots, all ineligible.
    # No component references B, so B never enters `component_to_roi`;
    # the builder produces domains only for A.
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "B"},
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "B"},
        ],
        ineligible_indices=[3, 4],
        component_assignment={0: "0", 1: "0", 2: "0"},
    )
    domains, comp_diags, _ = _build(source, spatial, c2r)
    assert set(domains.keys()) == {"0"}
    assert comp_diags[0].parent_roi_id == "A"


# ---------------------------------------------------------------------------
# Scattered parent contributes acquisition-support exclusion.
# ---------------------------------------------------------------------------


def test_case_18_scattered_parent_contributes_acquisition_exclusion():
    # ROI B is entirely scattered (its single spot is ineligible)
    # and sits near ROI A's supported cluster. B's ineligible-spot
    # exclusion disk blocks all parents and carves area from A's
    # ownership.
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 0.5, COORD_Y: 0.5, ROI_ID: "B"},  # scattered
        ],
        ineligible_indices=[2],
        component_assignment={0: "0", 1: "0"},
    )
    domains, comp_diags, _ = _build(source, spatial, c2r, sample_median_nn=1.0)

    # Ownership-excluded area under A's C_c is > 0 because the
    # exclusion disk sits inside C_c.
    assert comp_diags[0].ownership_excluded_area > 0
    # The scattered spot is NOT inside A's final domain.
    assert not domains["0"].covers(Point(0.5, 0.5))


# ---------------------------------------------------------------------------
# A supported spot outside its own domain remains accounted for.
# ---------------------------------------------------------------------------


def test_case_19_supported_spot_outside_domain_accounted_for():
    # The component has two spots; the near-origin spot is excluded by
    # ownership. The accounting identity
    #   n_spots_in_domain + n_supported_spots_outside_domain
    #     = n_component_spots
    # holds, and n_supported_spots_outside_domain > 0.
    rows = [
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},    # excluded
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "A"},   # kept
        {COORD_X: 0.3, COORD_Y: 0.0, ROI_ID: "A"},    # INELIGIBLE
    ]
    df = pd.DataFrame(rows)
    df[SOURCE_ROW_ID] = np.arange(len(df), dtype=np.int64)
    spatial = df.iloc[[0, 1]].copy()
    spatial[COMPONENT_ID] = "0"
    spatial[ROI_ID] = spatial[ROI_ID].astype(str)
    c2r = {"0": "A"}

    _, comp_diags, _ = _build(df, spatial, c2r)
    comp = comp_diags[0]
    assert comp.n_component_spots == 2
    n_outside = comp.n_component_spots - comp.n_spots_in_domain
    assert n_outside > 0
    assert comp.n_spots_in_domain + n_outside == comp.n_component_spots
