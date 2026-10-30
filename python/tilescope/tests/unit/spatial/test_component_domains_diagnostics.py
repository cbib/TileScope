"""Per-region diagnostics and canonical region-ordering tests.

The geometry builder retains every polygonal region of `G_c`
regardless of spot count, keeping geometry construction separate from
estimator eligibility. The word "sub-threshold" below refers only to
the fixture spot count; the builder itself has no estimator-threshold
knowledge.

Ordering is pinned across all five keys:
  1. descending spot count
  2. ascending min(source_row_id) — infinity for empty regions
  3. descending area
  4. ascending centroid.x
  5. ascending centroid.y

NOT rotation / reflection invariant; metamorphic tests belong to a
later suite.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Point, Polygon, box

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.component_domains import (
    DomainRegionDiagnostic,
    _canonical_region_order,
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


def _regions_of_component(
    region_diags: List[DomainRegionDiagnostic], cid: str,
) -> List[DomainRegionDiagnostic]:
    return sorted(
        [d for d in region_diags if d.component_id == cid],
        key=lambda d: d.domain_region_id,
    )


# ---------------------------------------------------------------------------
# Case 12 — geometry that becomes a valid MultiPolygon
# ---------------------------------------------------------------------------


def test_case_12_disconnected_component_yields_multipolygon_and_two_region_diagnostics():
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 102.0, COORD_Y: 0.0, ROI_ID: "A"},
        ],
        component_assignment={i: "0" for i in range(6)},
    )
    domains, comp_diags, region_diags = _build(source, spatial, c2r)

    assert isinstance(domains["0"], MultiPolygon)
    assert len(domains["0"].geoms) == 2

    comp = comp_diags[0]
    assert comp.n_component_spots == 6
    assert comp.n_spots_in_domain == 6
    assert comp.n_domain_regions == 2
    assert comp.n_regions_with_spots == 2

    regs = _regions_of_component(region_diags, "0")
    assert len(regs) == 2
    # Both regions have 3 spots; sort falls to min_source_row_id.
    # Cluster with src_ids 0..2 (min=0) precedes cluster with 3..5.
    assert regs[0].n_component_spots == 3
    assert regs[1].n_component_spots == 3


# ---------------------------------------------------------------------------
# Case 15 — low-spot region retained and reported
# ---------------------------------------------------------------------------


def test_case_15_low_spot_region_retained_with_diagnostic():
    # Fixture has one 5-spot cluster + one isolated 1-spot cluster,
    # same COMPONENT_ID. The builder retains the 1-spot region and
    # emits its diagnostic. Whether downstream tessellation deems it
    # eligible is NOT this module's concern.
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 3.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 4.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 50.0, COORD_Y: 0.0, ROI_ID: "A"},  # isolated
        ],
        component_assignment={i: "0" for i in range(6)},
    )
    _, comp_diags, region_diags = _build(source, spatial, c2r)

    comp = comp_diags[0]
    assert comp.n_domain_regions == 2
    assert comp.n_regions_with_spots == 2
    assert comp.n_spots_in_domain == 6

    regs = _regions_of_component(region_diags, "0")
    assert [r.n_component_spots for r in regs] == [5, 1]
    assert regs[1].area > 0  # low-spot region has strictly positive area


# ---------------------------------------------------------------------------
# Case 16 — two multi-spot regions get independent diagnostics
# ---------------------------------------------------------------------------


def test_case_16_two_multi_spot_regions_get_independent_diagnostics():
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 3.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 102.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 103.0, COORD_Y: 0.0, ROI_ID: "A"},
        ],
        component_assignment={i: "0" for i in range(8)},
    )
    _, comp_diags, region_diags = _build(source, spatial, c2r)

    assert comp_diags[0].n_domain_regions == 2
    assert comp_diags[0].n_spots_in_domain == 8

    regs = _regions_of_component(region_diags, "0")
    assert [r.n_component_spots for r in regs] == [4, 4]
    for r in regs:
        assert r.area > 0


# ---------------------------------------------------------------------------
# Region ordering (composition-level, via build_component_polygons)
# ---------------------------------------------------------------------------


def test_ordering_primary_key_is_descending_spot_count():
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 3.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 4.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 102.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 200.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 201.0, COORD_Y: 0.0, ROI_ID: "A"},
        ],
        component_assignment={i: "0" for i in range(10)},
    )
    _, _, region_diags = _build(source, spatial, c2r)
    regs = _regions_of_component(region_diags, "0")
    assert [r.n_component_spots for r in regs] == [5, 3, 2]


def test_ordering_ties_broken_by_min_source_row_id_ascending():
    # Two clusters, 2 spots each — primary key ties. Cluster placed
    # first in the rows list (src_ids 0, 1) must sort before the
    # second (src_ids 2, 3) even though the second sits at smaller
    # centroid.x, because min_src_id (key 2) is checked before
    # centroid.x (key 4).
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "A"},  # src 0
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "A"},  # src 1
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},    # src 2
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},    # src 3
        ],
        component_assignment={i: "0" for i in range(4)},
    )
    domains, _, region_diags = _build(source, spatial, c2r)
    regs = _regions_of_component(region_diags, "0")
    assert [r.n_component_spots for r in regs] == [2, 2]
    # Assert region id 0 is the src_ids-0/1 cluster: its centroid must
    # sit at x ≈ 100.5, NOT at x ≈ 0.5 (that would signal centroid.x
    # tie-break firing first, which is wrong).
    assert isinstance(domains["0"], MultiPolygon)
    first_region_cx = domains["0"].geoms[0].centroid.x
    assert 90 < first_region_cx < 110


def test_ordering_empty_region_sorts_after_spotted_via_infinity_min_src_id():
    # Component has 2 spots at (0, 0) and (10, 0). An ineligible
    # spot at (0.3, 0) triggers an exclusion disk that covers (0, 0),
    # so the spot at (0, 0) is excluded from G_c. A thin crescent
    # remains (no spot centre inside it) on the left of the exclusion,
    # sliced by acquisition support; the intact disk around (10, 0)
    # forms the spotted region. The crescent has min_src_id = ∞ by
    # construction (empty) and must sort AFTER the spotted region.
    rows = [
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},    # src 0 — excluded
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "A"},   # src 1 — kept
        {COORD_X: 0.3, COORD_Y: 0.0, ROI_ID: "A"},    # src 2 — INELIGIBLE
    ]
    df = pd.DataFrame(rows)
    df[SOURCE_ROW_ID] = np.arange(len(df), dtype=np.int64)
    spatial = df.iloc[[0, 1]].copy()
    spatial[COMPONENT_ID] = "0"
    spatial[ROI_ID] = spatial[ROI_ID].astype(str)
    c2r = {"0": "A"}

    domains, comp_diags, region_diags = _build(df, spatial, c2r)

    regs = _regions_of_component(region_diags, "0")
    # If the geometry didn't happen to split — e.g., if the crescent
    # had zero area under this exclusion configuration — this test
    # itself would need to be rethought. Assert the fixture produced
    # what we designed for before pinning ordering.
    assert len(regs) == 2, (
        f"fixture produced {len(regs)} region(s), expected 2 (spotted + empty)"
    )
    # Spotted region first (id 0), empty second (id 1).
    assert regs[0].n_component_spots == 1
    assert regs[1].n_component_spots == 0

    comp = comp_diags[0]
    assert comp.n_domain_regions == 2
    assert comp.n_regions_with_spots == 1
    assert comp.n_spots_in_domain == 1
    # The (0, 0) spot is excluded — accounted for as
    # "outside_domain" (n_component_spots - n_spots_in_domain = 1).
    assert comp.n_component_spots - comp.n_spots_in_domain == 1


# ---------------------------------------------------------------------------
# All five sort keys pinned via the sort helper directly
# ---------------------------------------------------------------------------


def test_canonical_region_order_pins_all_five_sort_keys():
    # Craft seven regions that individually stress each sort key.
    # Hand-tuned Polygon geometry via `box` + `translate` keeps
    # area / centroid deterministic (no dependence on Voronoi /
    # buffer discretization).
    #
    # Expected canonical order:
    #   R0: 5 spots {10..14}                       (key 1 wins)
    #   R1: 5 spots {20..24}                       (key 1 ties, key 2 later)
    #   R2: 1 spot  {0}                            (key 1)
    #   R3: 0 spots, area 100, centroid.x = 10     (keys 1+2 tie; area wins over R6)
    #   R4: 0 spots, area 100, centroid.x = 105    (keys 1+2+3 tie with R3; cx later)
    #   R5: 0 spots, area 100, centroid.x = 105,
    #                          centroid.y = 105    (keys 1+2+3+4 tie with R4; cy later)
    #   R6: 0 spots, area 25                        (loses to R3-R5 on area)
    ten_box = box(0.0, 0.0, 10.0, 10.0)       # area=100, centroid=(5, 5)
    twenty_by_five = box(0.0, 0.0, 20.0, 5.0)  # area=100, centroid=(10, 2.5)
    five_box = box(0.0, 0.0, 5.0, 5.0)         # area=25,  centroid=(2.5, 2.5)

    r0 = ten_box
    r1 = translate(ten_box, xoff=1000.0)
    r2 = translate(ten_box, xoff=2000.0)
    r3 = twenty_by_five                        # centroid.x = 10
    r4 = translate(ten_box, xoff=100.0)        # centroid = (105, 5), area 100
    r5 = translate(ten_box, xoff=100.0, yoff=100.0)  # centroid = (105, 105), area 100
    r6 = five_box                              # area 25, centroid (2.5, 2.5)

    regions = [r0, r1, r2, r3, r4, r5, r6]
    spots = [
        [10, 11, 12, 13, 14],   # r0
        [20, 21, 22, 23, 24],   # r1
        [0],                     # r2
        [],                      # r3
        [],                      # r4
        [],                      # r5
        [],                      # r6
    ]

    order = _canonical_region_order(regions, spots)
    assert order == [0, 1, 2, 3, 4, 5, 6]


def test_canonical_region_order_single_region_returns_identity():
    only = [box(0.0, 0.0, 1.0, 1.0)]
    assert _canonical_region_order(only, [[]]) == [0]


def test_canonical_region_order_empty_input_returns_empty():
    assert _canonical_region_order([], []) == []


# ---------------------------------------------------------------------------
# Diagnostic invariants — one row per polygon, counts agree
# ---------------------------------------------------------------------------


def test_every_polygonal_region_gets_exactly_one_diagnostic():
    # Assemble a fixture that yields multiple components AND multiple
    # regions per component (case 12 style, times two components).
    source, spatial, c2r = _fixture(
        rows=[
            # Component 0: 2 clusters
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "A"},
            # Component 1: 1 cluster
            {COORD_X: 500.0, COORD_Y: 0.0, ROI_ID: "B"},
            {COORD_X: 501.0, COORD_Y: 0.0, ROI_ID: "B"},
            {COORD_X: 502.0, COORD_Y: 0.0, ROI_ID: "B"},
        ],
        component_assignment={0: "0", 1: "0", 2: "0", 3: "0", 4: "1", 5: "1", 6: "1"},
    )
    domains, comp_diags, region_diags = _build(source, spatial, c2r)

    for cid, geom in domains.items():
        n_polys = 1 if isinstance(geom, Polygon) and not geom.is_empty else (
            len(geom.geoms) if hasattr(geom, "geoms") else (0 if geom.is_empty else 1)
        )
        diags = _regions_of_component(region_diags, cid)
        assert len(diags) == n_polys, (
            f"component {cid!r}: {n_polys} polygon(s) but "
            f"{len(diags)} diagnostic row(s)"
        )
        # domain_region_ids are contiguous 0..k-1
        assert [d.domain_region_id for d in diags] == list(range(len(diags)))


def test_component_diagnostic_region_counts_agree_with_region_diagnostics():
    # Same fixture as above; assert the two grains of accounting agree.
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 500.0, COORD_Y: 0.0, ROI_ID: "B"},
            {COORD_X: 501.0, COORD_Y: 0.0, ROI_ID: "B"},
            {COORD_X: 502.0, COORD_Y: 0.0, ROI_ID: "B"},
        ],
        component_assignment={0: "0", 1: "0", 2: "0", 3: "0", 4: "1", 5: "1", 6: "1"},
    )
    _, comp_diags, region_diags = _build(source, spatial, c2r)

    for comp in comp_diags:
        diags = _regions_of_component(region_diags, comp.component_id)
        assert comp.n_domain_regions == len(diags), (
            f"{comp.component_id}: n_domain_regions={comp.n_domain_regions} "
            f"but len(region_diags)={len(diags)}"
        )
        n_spotted = sum(1 for d in diags if d.n_component_spots > 0)
        assert comp.n_regions_with_spots == n_spotted
        total_in_regions = sum(d.n_component_spots for d in diags)
        assert comp.n_spots_in_domain == total_in_regions
