"""Input-contract guards and construction invariants for the component-domain builder.

Covers the four input-contract guards (SOURCE_ROW_ID hygiene,
supported-frame agreement with the source frame, radius-factor
positivity, and per-component parent consistency) plus negative-path
coverage for the cross-parent non-overlap invariant. The happy-path
composition tests exercise invariants 1, 2, 3, 4, 6, 8 on realistic
fixtures.
"""
from __future__ import annotations

from shapely.geometry import Polygon


from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.component_domains import (
    ComponentDomainDiagnostic,
    DomainRegionDiagnostic,
    _validate_component_domains,
    _validate_component_parent_consistency,
    _validate_radius_factors,
    _validate_source_row_ids,
    _validate_supported_frame_agreement,
    build_component_polygons,
)


R_C = 0.75
R_A = 0.60
R_E = 0.75


# ---------------------------------------------------------------------------
# Well-formed baseline (deliberately mutated per test)
# ---------------------------------------------------------------------------


def _baseline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    rows = [
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 1.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 1.0, ROI_ID: "A"},
    ]
    source = pd.DataFrame(rows)
    source[SOURCE_ROW_ID] = np.arange(len(source), dtype=np.int64)
    spatial = source.copy()
    spatial[COMPONENT_ID] = "A"
    spatial[ROI_ID] = spatial[ROI_ID].astype(str)
    return source, spatial, {"A": "A"}


def _build(source, spatial, c2r):
    result = build_component_polygons(
        source, spatial, c2r,
        sample_median_nn=1.0,
        component_support_radius_factor=R_C,
        acquisition_support_radius_factor=R_A,
        ineligible_spot_exclusion_radius_factor=R_E,
    )
    return result.component_polygons, list(result.component_diagnostics), list(result.region_diagnostics)


# ---------------------------------------------------------------------------
# Input contract — SOURCE_ROW_ID hygiene
# ---------------------------------------------------------------------------


def test_missing_source_row_id_column_raises():
    source, _, _ = _baseline()
    bad = source.drop(columns=[SOURCE_ROW_ID])
    with pytest.raises(ValueError, match=SOURCE_ROW_ID):
        _validate_source_row_ids(bad, "source frame")


def test_null_source_row_id_raises():
    source, _, _ = _baseline()
    bad = source.copy()
    bad.loc[0, SOURCE_ROW_ID] = None
    with pytest.raises(ValueError, match="null"):
        _validate_source_row_ids(bad, "source frame")


def test_fractional_float_source_row_id_raises():
    source, _, _ = _baseline()
    bad = source.copy()
    bad[SOURCE_ROW_ID] = bad[SOURCE_ROW_ID].astype(float) + 0.5
    with pytest.raises(ValueError, match="not integer-typed"):
        _validate_source_row_ids(bad, "source frame")


def test_whole_valued_float_source_row_id_raises():
    """A `[0.0, 1.0]` column looks integer-valued but is not integer-typed;
    accepting it would let a downstream join silently pair by float
    equality across the frames."""
    source, _, _ = _baseline()
    bad = source.copy()
    bad[SOURCE_ROW_ID] = bad[SOURCE_ROW_ID].astype(float)
    with pytest.raises(ValueError, match="not integer-typed"):
        _validate_source_row_ids(bad, "source frame")


def test_string_source_row_id_raises():
    source, _, _ = _baseline()
    bad = source.copy()
    bad[SOURCE_ROW_ID] = bad[SOURCE_ROW_ID].astype(str)
    with pytest.raises(ValueError, match="not integer-typed"):
        _validate_source_row_ids(bad, "source frame")


def test_bool_source_row_id_raises():
    """Booleans coerce to 0/1 under many int checks; the SOURCE_ROW_ID
    contract must reject them explicitly so a two-row misdesign cannot
    slip through as `[True, False]`."""
    source, _, _ = _baseline()
    bad = source.iloc[:2].copy()
    bad[SOURCE_ROW_ID] = pd.array([True, False], dtype="bool")
    with pytest.raises(ValueError, match="bool"):
        _validate_source_row_ids(bad, "source frame")


def test_duplicate_source_row_id_raises():
    source, _, _ = _baseline()
    bad = source.copy()
    bad.loc[3, SOURCE_ROW_ID] = bad.loc[0, SOURCE_ROW_ID]
    with pytest.raises(ValueError, match="not unique"):
        _validate_source_row_ids(bad, "source frame")


# ---------------------------------------------------------------------------
# Input contract — supported ⊂ source; coords + ROI_ID agree
# ---------------------------------------------------------------------------


def test_supported_id_absent_from_source_raises():
    source, spatial, _ = _baseline()
    bad_spatial = spatial.copy()
    bad_spatial.loc[0, SOURCE_ROW_ID] = 999  # not present in source
    with pytest.raises(ValueError, match="absent from source frame"):
        _validate_supported_frame_agreement(source, bad_spatial)


def test_supported_coordinate_disagreement_raises():
    source, spatial, _ = _baseline()
    bad_spatial = spatial.copy()
    bad_spatial.loc[0, COORD_X] = 42.0  # differs from source row 0
    with pytest.raises(ValueError, match="disagree"):
        _validate_supported_frame_agreement(source, bad_spatial)


def test_supported_roi_id_disagreement_raises():
    source, spatial, _ = _baseline()
    bad_spatial = spatial.copy()
    bad_spatial.loc[0, ROI_ID] = "X"  # differs from source row 0
    with pytest.raises(ValueError, match=f"{ROI_ID!r}"):
        _validate_supported_frame_agreement(source, bad_spatial)


def test_supported_roi_id_agreement_survives_wide_mixed_dtype_source():
    """Regression: whole-row access via `src_indexed.loc[i]` used to
    upcast an integer ROI_ID to `float64` whenever the source frame
    carried any float-typed column, so `str(np.float64(1.0))` = `'1.0'`
    stopped matching a spatial `'1'`. Per-column `.at[i]` lookup
    preserves each column's dtype, and this pair must validate cleanly.
    """
    source, spatial, _ = _baseline()
    # Force int ROI_ID on source, object ('1') on spatial. Add a float
    # column so the row-Series upcast path (the old bug) would trigger.
    source = source.copy()
    source[ROI_ID] = source[ROI_ID].map({"A": 1}).astype(np.int64)
    source["gene1"] = np.arange(len(source), dtype=float) * 0.5
    spatial = spatial.copy()
    spatial[ROI_ID] = spatial[ROI_ID].map({"A": "1"}).astype(object)
    _validate_supported_frame_agreement(source, spatial)  # must not raise


def test_supported_roi_id_string_zero_prefix_still_distinct():
    """Guard: the fix must not silently equate `'01'` and `'1'`. Any
    future "helpful" numeric normalization would break identifier
    semantics for zero-padded / large-integer / textual IDs; keep the
    literal comparison and let this test fail loudly if that ever
    changes.
    """
    source, spatial, _ = _baseline()
    source = source.copy()
    source[ROI_ID] = source[ROI_ID].map({"A": "01"}).astype(object)
    spatial = spatial.copy()
    spatial[ROI_ID] = spatial[ROI_ID].map({"A": "1"}).astype(object)
    with pytest.raises(ValueError, match=f"{ROI_ID!r}"):
        _validate_supported_frame_agreement(source, spatial)


# ---------------------------------------------------------------------------
# Input contract — radius factors finite and positive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_or_non_finite_radius_factor_raises(bad_value):
    with pytest.raises(ValueError, match="finite and > 0"):
        _validate_radius_factors(
            component_support_radius_factor=bad_value,
            acquisition_support_radius_factor=0.6,
            ineligible_spot_exclusion_radius_factor=0.75,
        )


def test_build_rejects_zero_radius_factor():
    source, spatial, c2r = _baseline()
    with pytest.raises(ValueError, match="finite and > 0"):
        build_component_polygons(
            source, spatial, c2r,
            sample_median_nn=1.0,
            component_support_radius_factor=0.0,
            acquisition_support_radius_factor=R_A,
            ineligible_spot_exclusion_radius_factor=R_E,
        )


# ---------------------------------------------------------------------------
# Invariant 7 — component → single non-null parent, agrees with map
# ---------------------------------------------------------------------------


def test_component_with_multiple_parents_raises():
    _, spatial, c2r = _baseline()
    bad_spatial = spatial.copy()
    bad_spatial.loc[0, ROI_ID] = "B"  # component 'A' now spans two parents
    with pytest.raises(ValueError, match="multiple parent ROI_IDs"):
        _validate_component_parent_consistency(bad_spatial, c2r)


def test_component_parent_disagrees_with_map_raises():
    _, spatial, _ = _baseline()
    with pytest.raises(ValueError, match="disagrees with component_to_roi"):
        _validate_component_parent_consistency(spatial, {"A": "WRONG"})


def test_component_missing_from_map_raises():
    _, spatial, _ = _baseline()
    with pytest.raises(ValueError, match="missing key"):
        _validate_component_parent_consistency(spatial, {})


# ---------------------------------------------------------------------------
# Construction invariants — direct negative-path drives of `_validate_component_domains`
#
# Voronoi ownership plus geometry construction prevent these outcomes
# in production, so happy-path fixtures cannot exercise the assertions.
# To prove each guard raises correctly, we hand-craft a `domains` map
# (and the paired diagnostic rows) that violates the invariant and
# call the validator directly.
# ---------------------------------------------------------------------------


def _diag(
    component_id: str,
    parent_roi_id: str,
    *,
    n_component_spots: int = 0,
    n_spots_in_domain: int = 0,
    final_domain_area: float = 0.0,
) -> ComponentDomainDiagnostic:
    """Minimal diagnostic; consulted fields default to spot-free values."""
    return ComponentDomainDiagnostic(
        component_id=component_id,
        parent_roi_id=parent_roi_id,
        n_component_spots=n_component_spots,
        n_spots_in_domain=n_spots_in_domain,
        n_domain_regions=0,
        n_regions_with_spots=0,
        component_support_area=0.0,
        ownership_excluded_area=0.0,
        acquisition_excluded_area=0.0,
        final_domain_area=final_domain_area,
        geometry_status="ok",
    )


def _empty_spatial() -> pd.DataFrame:
    return pd.DataFrame(
        {COORD_X: [], COORD_Y: [], ROI_ID: [], COMPONENT_ID: [], SOURCE_ROW_ID: []},
    ).astype({SOURCE_ROW_ID: np.int64})


def _spatial_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[SOURCE_ROW_ID] = df[SOURCE_ROW_ID].astype(np.int64)
    df[COMPONENT_ID] = df[COMPONENT_ID].astype(str)
    df[ROI_ID] = df[ROI_ID].astype(str)
    return df


# --- Invariant 1: final component geometry must be valid Shapely ---


def test_invalid_final_geometry_raises_invariant_1():
    # Bowtie: two triangles sharing a single vertex — self-intersecting,
    # hence `is_valid == False`. `make_valid` would rescue this in
    # production; the direct-drive test bypasses that on purpose.
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not bowtie.is_valid
    with pytest.raises(AssertionError, match=r"invariant 1.*invalid final geometry"):
        _validate_component_domains(
            domains={"A": bowtie},
            spatial_component_expression_df=_empty_spatial(),
            comp_diags=[_diag("A", "parent_A")],
            region_diags=[],
            area_tol=1e-9,
        )


# --- Invariant 2: retained regions holding spots have finite, positive area ---


def test_zero_area_spotted_region_raises_invariant_2():
    region_diag = DomainRegionDiagnostic(
        component_id="A",
        domain_region_id=0,
        n_component_spots=1,   # region carries a spot
        area=0.0,              # but area is 0.0 — construction bug
        n_holes=0,
        geometry_status="ok",
        source_row_fingerprint="",
    )
    with pytest.raises(AssertionError, match=r"invariant 2.*expected finite and > 0"):
        _validate_component_domains(
            domains={"A": box(0, 0, 1, 1)},
            spatial_component_expression_df=_empty_spatial(),
            comp_diags=[_diag("A", "parent_A")],
            region_diags=[region_diag],
            area_tol=1e-9,
        )


def test_non_finite_area_spotted_region_raises_invariant_2():
    region_diag = DomainRegionDiagnostic(
        component_id="A",
        domain_region_id=0,
        n_component_spots=2,
        area=float("nan"),
        n_holes=0,
        geometry_status="ok",
        source_row_fingerprint="",
    )
    with pytest.raises(AssertionError, match=r"invariant 2"):
        _validate_component_domains(
            domains={"A": box(0, 0, 1, 1)},
            spatial_component_expression_df=_empty_spatial(),
            comp_diags=[_diag("A", "parent_A")],
            region_diags=[region_diag],
            area_tol=1e-9,
        )


def test_zero_area_empty_region_is_permitted_by_invariant_2():
    # A spot-less region legitimately has zero area residues from
    # `unary_union` snap boundaries; invariant 2 only fires when the
    # region carries at least one spot.
    region_diag = DomainRegionDiagnostic(
        component_id="A",
        domain_region_id=0,
        n_component_spots=0,
        area=0.0,
        n_holes=0,
        geometry_status="ok",
        source_row_fingerprint="",
    )
    # No assertion should fire.
    _validate_component_domains(
        domains={"A": box(0, 0, 1, 1)},
        spatial_component_expression_df=_empty_spatial(),
        comp_diags=[_diag("A", "parent_A")],
        region_diags=[region_diag],
        area_tol=1e-9,
    )


# --- Invariant 3: assignment count matches geometric coverage; missing diag fails loud ---


def test_assignment_count_disagrees_with_geometric_coverage_raises_invariant_3():
    # A unit-square domain that covers three own spots at (0.5, 0.5),
    # (0.25, 0.25), (0.75, 0.75). The diagnostic falsely reports
    # `n_spots_in_domain=1`, which the validator must catch.
    spatial = _spatial_rows([
        {COORD_X: 0.5, COORD_Y: 0.5, ROI_ID: "A", COMPONENT_ID: "A", SOURCE_ROW_ID: 0},
        {COORD_X: 0.25, COORD_Y: 0.25, ROI_ID: "A", COMPONENT_ID: "A", SOURCE_ROW_ID: 1},
        {COORD_X: 0.75, COORD_Y: 0.75, ROI_ID: "A", COMPONENT_ID: "A", SOURCE_ROW_ID: 2},
    ])
    diag = _diag("A", "A", n_component_spots=3, n_spots_in_domain=1)
    with pytest.raises(AssertionError, match=r"invariant 3.*assignment count"):
        _validate_component_domains(
            domains={"A": box(0, 0, 1, 1)},
            spatial_component_expression_df=spatial,
            comp_diags=[diag],
            region_diags=[],
            area_tol=1e-9,
        )


def test_missing_diagnostic_for_non_empty_component_raises_invariant_3():
    # Domain is non-empty and covers an own spot, but no ComponentDomainDiagnostic
    # entry exists for it — the validator must fail loudly rather than
    # silently skipping the coverage check for that component.
    spatial = _spatial_rows([
        {COORD_X: 0.5, COORD_Y: 0.5, ROI_ID: "A", COMPONENT_ID: "A", SOURCE_ROW_ID: 0},
    ])
    with pytest.raises(AssertionError, match=r"invariant 3.*no diagnostic row"):
        _validate_component_domains(
            domains={"A": box(0, 0, 1, 1)},
            spatial_component_expression_df=spatial,
            comp_diags=[],  # diag missing
            region_diags=[],
            area_tol=1e-9,
        )


# --- Invariant 5 & 6: pairwise material non-overlap ---


def test_cross_parent_overlap_raises_invariant_5():
    # Two overlapping unit squares under DIFFERENT parents. Empty
    # spatial frame keeps invariants 3, 4, 8 quiet; the only firing
    # assertion is 5.
    domains = {"A": box(0, 0, 1, 1), "B": box(0.5, 0, 1.5, 1)}
    comp_diags: List[ComponentDomainDiagnostic] = [
        _diag("A", "parent_A"),
        _diag("B", "parent_B"),
    ]
    with pytest.raises(AssertionError, match=r"invariant 5.*cross-parent"):
        _validate_component_domains(
            domains=domains,
            spatial_component_expression_df=_empty_spatial(),
            comp_diags=comp_diags,
            region_diags=[],
            area_tol=1e-9,
        )


def test_sibling_overlap_still_raises_invariant_6():
    # Same shape as above but siblings under one parent — the same
    # sweep must now tag the violation as invariant 6.
    domains = {"A_c1": box(0, 0, 1, 1), "A_c2": box(0.5, 0, 1.5, 1)}
    comp_diags: List[ComponentDomainDiagnostic] = [
        _diag("A_c1", "A"),
        _diag("A_c2", "A"),
    ]
    with pytest.raises(AssertionError, match=r"invariant 6.*sibling"):
        _validate_component_domains(
            domains=domains,
            spatial_component_expression_df=_empty_spatial(),
            comp_diags=comp_diags,
            region_diags=[],
            area_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# Sanity: the baseline still builds after all the guards are in place.
# ---------------------------------------------------------------------------


def test_baseline_still_builds_cleanly():
    source, spatial, c2r = _baseline()
    domains, comp_diags, region_diags = _build(source, spatial, c2r)
    assert set(domains.keys()) == {"A"}
    assert comp_diags and comp_diags[0].component_id == "A"
    assert all(rd.component_id == "A" for rd in region_diags)
