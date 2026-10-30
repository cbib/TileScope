"""Composition tests for `build_component_polygons`.

The anchor scenarios are:
  1. Clean disjoint ROIs
  3. Buffer-induced overlap regression
  8. Two sibling components under one parent

Additional geometry, diagnostic, and ordering scenarios live in the
companion component-domain test modules.

Radius factors used throughout match the production defaults:
`r_C = 0.75 · d`, `r_A = 0.60 · d`, `r_E = 0.75 · d`.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.component_domains import (
    ComponentDomainDiagnostic,
    DomainRegionDiagnostic,
    build_component_polygons,
)


# Production radius factors.
R_C = 0.75
R_A = 0.60
R_E = 0.75


def _fixture(
    rows: list[dict],
    *,
    ineligible_indices: list[int] | None = None,
    component_assignment: dict[int, str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Assemble `(source_df, spatial_df, component_to_roi)` from rows.

    `rows` — dicts of {COORD_X, COORD_Y, ROI_ID}. Row index → SOURCE_ROW_ID.
    `ineligible_indices` — row positions to drop from `spatial_df`.
    `component_assignment` — row_index → COMPONENT_ID for supported
        rows. Rows not listed inherit COMPONENT_ID = str(ROI_ID).
    """
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
            int(src_id), str(spatial_df.loc[spatial_df[SOURCE_ROW_ID] == src_id, ROI_ID].iloc[0])
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


# ---------------------------------------------------------------------------
# Case 1 — clean disjoint ROIs
# ---------------------------------------------------------------------------


def test_case_1_clean_disjoint_rois():
    source, spatial, c2r = _fixture([
        # ROI A: 4 spots on a tight 1-D lattice
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 3.0, COORD_Y: 0.0, ROI_ID: "A"},
        # ROI B: 4 spots far away
        {COORD_X: 100.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 101.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 102.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 103.0, COORD_Y: 0.0, ROI_ID: "B"},
    ])
    domains, comp_diags, region_diags = _build(source, spatial, c2r)

    assert set(domains.keys()) == {"A", "B"}
    # Every spot lies inside its own component's domain
    for _, row in spatial.iterrows():
        cid = str(row[COMPONENT_ID])
        pt = Point(float(row[COORD_X]), float(row[COORD_Y]))
        assert domains[cid].covers(pt)

    # No overlap between the two domains
    assert domains["A"].intersection(domains["B"]).area == pytest.approx(0.0, abs=1e-9)

    # Component-level diagnostics
    by_comp = {d.component_id: d for d in comp_diags}
    for cid in ("A", "B"):
        d = by_comp[cid]
        assert d.parent_roi_id == cid
        assert d.n_component_spots == 4
        assert d.n_spots_in_domain == 4
        assert d.geometry_status == "ok"
        assert d.final_domain_area > 0
        # No ineligible spots, no acquisition holes far from the source ⇒
        # ownership_excluded ≈ 0.
        assert d.ownership_excluded_area == pytest.approx(0.0, abs=1e-9)

    # One region per component, each with all 4 spots
    by_cid_region = {(d.component_id, d.domain_region_id): d for d in region_diags}
    for cid in ("A", "B"):
        assert (cid, 0) in by_cid_region
        assert by_cid_region[(cid, 0)].n_component_spots == 4


# ---------------------------------------------------------------------------
# Case 3 — buffer-induced overlap regression
# ---------------------------------------------------------------------------


def test_case_3_buffer_induced_overlap_defended_by_ownership():
    # Two parents with spots so close that raw component-support disks
    # (radius 0.75) would overlap the neighbouring parent's spots.
    # Voronoi ownership carves them apart, and no A-spot lands in B's
    # domain (or vice versa).
    source, spatial, c2r = _fixture([
        # ROI A: 3 spots
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
        # ROI B: 3 spots, first only 0.5 units from A's last
        {COORD_X: 2.5, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 3.5, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 4.5, COORD_Y: 0.0, ROI_ID: "B"},
    ])
    domains, comp_diags, _ = _build(source, spatial, c2r)

    # Domains materially disjoint (Voronoi property + intersection with A)
    overlap_area = domains["A"].intersection(domains["B"]).area
    assert overlap_area == pytest.approx(0.0, abs=1e-9)

    # No foreign spot inside a neighbour's domain
    for _, row in spatial.iterrows():
        own = str(row[COMPONENT_ID])
        foreign = "B" if own == "A" else "A"
        pt = Point(float(row[COORD_X]), float(row[COORD_Y]))
        assert not domains[foreign].covers(pt), (
            f"foreign leakage: spot at ({row[COORD_X]}, {row[COORD_Y]}) "
            f"owned by {own!r} lies inside {foreign!r}'s domain"
        )

    # All own spots retained in domain (no ownership-driven exclusion here)
    by_comp = {d.component_id: d for d in comp_diags}
    for cid in ("A", "B"):
        assert by_comp[cid].n_spots_in_domain == 3


# ---------------------------------------------------------------------------
# Case 8 — two sibling components under one parent
# ---------------------------------------------------------------------------


def test_case_8_two_sibling_components_under_one_parent():
    # ROI "A" is split into two spatial components at very different
    # locations. Both share parent_roi_id = "A"; sibling non-overlap
    # is not guaranteed by Voronoi (they share the parent) and must
    # be verified by the builder's invariant 6.
    source, spatial, c2r = _fixture(
        rows=[
            {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 2.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 20.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 21.0, COORD_Y: 0.0, ROI_ID: "A"},
            {COORD_X: 22.0, COORD_Y: 0.0, ROI_ID: "A"},
        ],
        component_assignment={0: "0", 1: "0", 2: "0", 3: "1", 4: "1", 5: "1"},
    )

    domains, comp_diags, region_diags = _build(source, spatial, c2r)

    assert set(domains.keys()) == {"0", "1"}
    # Both parents map to "A"
    by_comp = {d.component_id: d for d in comp_diags}
    assert by_comp["0"].parent_roi_id == "A"
    assert by_comp["1"].parent_roi_id == "A"

    # Sibling non-overlap (invariant 6 already asserted during build;
    # verify explicitly)
    assert domains["0"].intersection(domains["1"]).area == pytest.approx(
        0.0, abs=1e-9,
    )

    # Each spot lives inside its own component's domain
    for _, row in spatial.iterrows():
        cid = str(row[COMPONENT_ID])
        pt = Point(float(row[COORD_X]), float(row[COORD_Y]))
        assert domains[cid].covers(pt)

    # Region ordering: both components have one region each,
    # domain_region_id = 0
    by_region = {(d.component_id, d.domain_region_id): d for d in region_diags}
    for cid in ("0", "1"):
        assert (cid, 0) in by_region
        assert by_region[(cid, 0)].n_component_spots == 3


# ---------------------------------------------------------------------------
# Contract: builder validation
# ---------------------------------------------------------------------------


def test_builder_rejects_non_finite_sample_median_nn():
    source, spatial, c2r = _fixture([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
    ])
    with pytest.raises(ValueError, match=r"sample_median_nn"):
        _build(source, spatial, c2r, sample_median_nn=0.0)
    with pytest.raises(ValueError, match=r"sample_median_nn"):
        _build(source, spatial, c2r, sample_median_nn=float("nan"))


def test_builder_rejects_missing_component_to_roi_key():
    source, spatial, c2r = _fixture([
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
    ])
    # Break the mapping intentionally
    del c2r["A"]
    with pytest.raises(ValueError, match=r"component_to_roi missing key"):
        _build(source, spatial, c2r)
