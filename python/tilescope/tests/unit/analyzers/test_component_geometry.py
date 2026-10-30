"""Focused tests for the canonical component_geometry.csv builder.

Contracts:

1. Column set matches the canonical CSV contract (identity
   with ``sample_id`` + geometry + contract provenance +
   region/spot counts + exclusion areas + tessellation eligibility).
   ``geometry_method`` and ``alpha_mult`` are absent.
2. WKT round-trips through shapely; ``component_domain_area`` = the
   authoritative ``ComponentDomainDiagnostic.final_domain_area``.
3. ``geometry_contract_parameter_fingerprint`` stable + sensitive
   to radius-factor changes.
4. Tessellation-eligibility columns merge in from the typed frame
   under a strict one-to-one join; components absent from the
   frame (component with no tessellation stage output) keep the
   eligibility fields null.
5. Coherence invariants raise:
   - parent-ROI identity drift between diagnostic and
     ``component_to_roi``.
   - ``n_spots_in_domain > n_component_spots``.
   - ``polygon.area`` vs ``diagnostic.final_domain_area`` mismatch
     beyond tolerance on ``geometry_status == "ok"`` rows.
   - Duplicate eligibility row for one component.
   - Extra eligibility row (component in frame but not in
     ``component_polygons``).
6. Identity contract raises on missing ``component_to_roi`` OR
   missing ``ComponentDomainDiagnostic``.
7. Empty inputs yield zero-row schema-stable frames.
"""
from __future__ import annotations

from tilescope.analyzers.sample_support_frames import _COMPONENT_GEOMETRY_DTYPES


import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import Polygon

from tilescope.spatial.component_domains import ComponentDomainDiagnostic
from tilescope.analyzers.sample_support_frames import (
    GEOMETRY_CONTRACT_NAME,
    GEOMETRY_CONTRACT_VERSION,
    build_component_geometry,
    compute_geometry_contract_parameter_fingerprint,
)


_EXPECTED_COLUMNS = [
    "sample_id", "roi_id", "component_id",
    "geometry_wkt", "component_domain_area", "geometry_status",
    "geometry_contract_name", "geometry_contract_version",
    "geometry_contract_parameter_fingerprint",
    "sample_median_nn",
    "component_support_radius_factor",
    "acquisition_support_radius_factor",
    "ineligible_spot_exclusion_radius_factor",
    "n_all_domain_regions", "n_regions_with_spots",
    "n_component_spots", "n_spots_in_domain",
    "n_supported_spots_outside_domain",
    "ownership_excluded_area", "acquisition_excluded_area",
    "n_tessellation_eligible_regions",
    "n_spots_in_tessellation_eligible_regions",
    "n_spots_excluded_by_tessellation_eligibility",
    "tessellated_domain_area",
]


def _square(cx: float, cy: float, side: float = 1.0) -> Polygon:
    h = side / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h),
                    (cx + h, cy + h), (cx - h, cy + h)])


def _diag(cid: str, parent: str, *,
          area: float = 1.0, n_spots: int = 10,
          n_in_domain: int = 10, n_regions: int = 1,
          n_regions_with_spots: int = 1,
          support_area: float = 1.2,
          ownership_excluded: float = 0.1,
          acquisition_excluded: float = 0.05,
          status: str = "ok") -> ComponentDomainDiagnostic:
    return ComponentDomainDiagnostic(
        component_id=cid, parent_roi_id=parent,
        n_component_spots=n_spots, n_spots_in_domain=n_in_domain,
        n_domain_regions=n_regions,
        n_regions_with_spots=n_regions_with_spots,
        component_support_area=support_area,
        ownership_excluded_area=ownership_excluded,
        acquisition_excluded_area=acquisition_excluded,
        final_domain_area=area,
        geometry_status=status,
    )


def _empty_eligibility_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "roi_id", "component_id",
        "n_tessellation_eligible_regions",
        "n_spots_in_tessellation_eligible_regions",
        "n_spots_excluded_by_tessellation_eligibility",
        "tessellated_domain_area",
    ])


def _eligibility_frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _run(
    component_polygons, component_to_roi, diags,
    tess_elig_frame=None, sample_id="S1",
    component_support_radius_factor=0.60,
    acquisition_support_radius_factor=0.60,
    ineligible_spot_exclusion_radius_factor=0.60,
    sample_median_nn=1.5,
):
    return build_component_geometry(
        sample_id=sample_id,
        component_polygons=component_polygons,
        component_to_roi=component_to_roi,
        component_domain_diagnostics=diags,
        sample_median_nn=sample_median_nn,
        component_support_radius_factor=component_support_radius_factor,
        acquisition_support_radius_factor=acquisition_support_radius_factor,
        ineligible_spot_exclusion_radius_factor=(
            ineligible_spot_exclusion_radius_factor
        ),
        tessellation_eligibility_frame=(
            tess_elig_frame if tess_elig_frame is not None
            else _empty_eligibility_frame()
        ),
    )


# ---------------------------------------------------------------------------


def test_column_set_matches_migration_doc_lock():
    poly = _square(0.0, 0.0)
    df = _run(
        component_polygons={"1": poly},
        component_to_roi={"1": "A"},
        diags=[_diag("1", "A")],
    )
    assert list(df.columns) == _EXPECTED_COLUMNS
    # sample_id populated on every row.
    assert (df["sample_id"] == "S1").all()


def test_wkt_round_trips_and_domain_area_from_diagnostic():
    poly = _square(1.5, 2.5, side=3.0)
    df = _run(
        component_polygons={"7": poly},
        component_to_roi={"7": "R7"},
        diags=[_diag("7", "R7", area=poly.area, status="ok")],
    )
    reparsed = wkt.loads(df.loc[0, "geometry_wkt"])
    assert reparsed.geom_type == "Polygon"
    assert reparsed.area == pytest.approx(poly.area)
    assert df.loc[0, "component_domain_area"] == pytest.approx(poly.area)
    assert df.loc[0, "geometry_status"] == "ok"
    assert df.loc[0, "geometry_contract_name"] == GEOMETRY_CONTRACT_NAME
    assert df.loc[0, "geometry_contract_version"] == GEOMETRY_CONTRACT_VERSION


def test_parameter_fingerprint_stable_and_sensitive_to_radius_change():
    a = compute_geometry_contract_parameter_fingerprint(
        component_support_radius_factor=0.60,
        acquisition_support_radius_factor=0.60,
        ineligible_spot_exclusion_radius_factor=0.60,
    )
    b = compute_geometry_contract_parameter_fingerprint(
        component_support_radius_factor=0.60,
        acquisition_support_radius_factor=0.60,
        ineligible_spot_exclusion_radius_factor=0.60,
    )
    c = compute_geometry_contract_parameter_fingerprint(
        component_support_radius_factor=0.75,
        acquisition_support_radius_factor=0.60,
        ineligible_spot_exclusion_radius_factor=0.60,
    )
    assert a == b
    assert a != c
    assert len(a) == 16


def test_tessellation_eligibility_fields_merge_when_present():
    df = _run(
        component_polygons={"1": _square(0.0, 0.0)},
        component_to_roi={"1": "A"},
        diags=[_diag("1", "A")],
        tess_elig_frame=_eligibility_frame([{
            "sample_id": "S1", "roi_id": "A", "component_id": "1",
            "n_tessellation_eligible_regions": 3,
            "n_spots_in_tessellation_eligible_regions": 25,
            "n_spots_excluded_by_tessellation_eligibility": 4,
            "tessellated_domain_area": 0.85,
        }]),
    )
    row = df.iloc[0]
    assert int(row["n_tessellation_eligible_regions"]) == 3
    assert int(row["n_spots_in_tessellation_eligible_regions"]) == 25
    assert int(row["n_spots_excluded_by_tessellation_eligibility"]) == 4
    assert float(row["tessellated_domain_area"]) == pytest.approx(0.85)


def test_empty_inputs_yield_zero_row_schema_stable_frame():
    df = _run(
        component_polygons={},
        component_to_roi={},
        diags=[],
    )
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert len(df) == 0
    # Dtypes still enforced on empty frame.
    for c in _EXPECTED_COLUMNS:
        assert str(df[c].dtype) == _COMPONENT_GEOMETRY_DTYPES[c], c


# ---------------------------------------------------------------------------
# Coherence invariants
# ---------------------------------------------------------------------------


def test_parent_roi_identity_drift_raises():
    """Diagnostic says parent='B'; component_to_roi says parent='A'."""
    with pytest.raises(ValueError, match="identity drift"):
        _run(
            component_polygons={"1": _square(0.0, 0.0)},
            component_to_roi={"1": "A"},
            diags=[_diag("1", "B")],
        )


def test_impossible_spot_accounting_raises():
    """n_spots_in_domain > n_component_spots is an impossible
    accounting state — subtract-with-max would previously mask it."""
    with pytest.raises(ValueError, match="impossible accounting"):
        _run(
            component_polygons={"1": _square(0.0, 0.0)},
            component_to_roi={"1": "A"},
            diags=[_diag("1", "A", n_spots=5, n_in_domain=8)],
        )


def test_polygon_area_disagreement_raises_on_ok_geometry():
    poly = _square(0.0, 0.0)  # area = 1.0
    with pytest.raises(ValueError, match="area drift"):
        _run(
            component_polygons={"1": poly},
            component_to_roi={"1": "A"},
            diags=[_diag("1", "A", area=999.0, status="ok")],
        )


def test_polygon_area_disagreement_ignored_on_non_ok_geometry():
    """Divergent area on `geometry_status != "ok"` is legitimate —
    the diagnostic tracks the intended domain area even when the
    downstream polygon is missing / empty / invalid."""
    poly = _square(0.0, 0.0)  # area = 1.0
    df = _run(
        component_polygons={"1": poly},
        component_to_roi={"1": "A"},
        diags=[_diag("1", "A", area=999.0, status="invalid")],
    )
    assert df.loc[0, "component_domain_area"] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Eligibility join contract
# ---------------------------------------------------------------------------


def test_duplicate_eligibility_row_raises():
    rows = [{
        "sample_id": "S1", "roi_id": "A", "component_id": "1",
        "n_tessellation_eligible_regions": 1,
        "n_spots_in_tessellation_eligible_regions": 5,
        "n_spots_excluded_by_tessellation_eligibility": 0,
        "tessellated_domain_area": 1.0,
    }] * 2
    with pytest.raises(ValueError, match="duplicate row"):
        _run(
            component_polygons={"1": _square(0.0, 0.0)},
            component_to_roi={"1": "A"},
            diags=[_diag("1", "A")],
            tess_elig_frame=_eligibility_frame(rows),
        )


def test_extra_eligibility_row_absent_from_component_polygons_raises():
    """A component present in the eligibility frame but absent from
    component_polygons is a coherence failure — the eligibility frame
    should not carry rows the geometry builder never emitted."""
    rows = [{
        "sample_id": "S1", "roi_id": "A", "component_id": "1",
        "n_tessellation_eligible_regions": 1,
        "n_spots_in_tessellation_eligible_regions": 5,
        "n_spots_excluded_by_tessellation_eligibility": 0,
        "tessellated_domain_area": 1.0,
    }, {
        "sample_id": "S1", "roi_id": "B", "component_id": "99",
        "n_tessellation_eligible_regions": 1,
        "n_spots_in_tessellation_eligible_regions": 5,
        "n_spots_excluded_by_tessellation_eligibility": 0,
        "tessellated_domain_area": 1.0,
    }]
    with pytest.raises(ValueError, match="absent from component_polygons"):
        _run(
            component_polygons={"1": _square(0.0, 0.0)},
            component_to_roi={"1": "A"},
            diags=[_diag("1", "A")],
            tess_elig_frame=_eligibility_frame(rows),
        )


# ---------------------------------------------------------------------------
# Identity contract
# ---------------------------------------------------------------------------


def test_missing_component_to_roi_raises():
    with pytest.raises(KeyError, match="component_to_roi missing entry"):
        _run(
            component_polygons={"1": _square(0.0, 0.0)},
            component_to_roi={},
            diags=[_diag("1", "A")],
        )


def test_missing_component_domain_diagnostic_raises():
    with pytest.raises(KeyError, match="component_domain_diagnostics missing"):
        _run(
            component_polygons={"1": _square(0.0, 0.0)},
            component_to_roi={"1": "A"},
            diags=[],
        )
