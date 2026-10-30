"""Focused tests for the canonical component_domain_regions.csv builder.

Contracts:

1. Column set (11 columns): identity with ``sample_id`` + region
   geometry + tessellation eligibility.
2. Total table over ``region_diagnostics``.
3. Tessellation eligibility merges from the typed frame under a
   strict one-to-one join by
   ``(sample_id, component_id, domain_region_id)``. Regions absent
   from the frame keep the eligibility fields null.
4. Coherence / identity contract raises on:
   - Missing ``component_to_roi`` entry.
   - Duplicate eligibility row.
   - Extra eligibility row (present in frame, absent from
     ``region_diagnostics``).
5. Empty inputs yield zero-row schema-stable frame.
6. ``source_row_fingerprint`` preserved verbatim.
"""
from __future__ import annotations

from tilescope.analyzers.sample_support_frames import _COMPONENT_DOMAIN_REGIONS_DTYPES


import pandas as pd
import pytest

from tilescope.spatial.component_domains import DomainRegionDiagnostic
from tilescope.analyzers.sample_support_frames import build_component_domain_regions


_EXPECTED_COLUMNS = [
    "sample_id", "roi_id", "component_id", "domain_region_id",
    "area", "n_component_spots", "n_holes",
    "geometry_status", "source_row_fingerprint",
    "tessellation_eligible", "tessellation_ineligibility_reason",
]


def _region(cid: str, rid: int, *,
            area: float = 1.0, n_spots: int = 5, n_holes: int = 0,
            status: str = "ok", fp: str = "stub_fp") -> DomainRegionDiagnostic:
    return DomainRegionDiagnostic(
        component_id=cid, domain_region_id=rid,
        n_component_spots=n_spots, area=area, n_holes=n_holes,
        geometry_status=status, source_row_fingerprint=fp,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "sample_id", "roi_id", "component_id", "domain_region_id",
        "tessellation_eligible", "tessellation_ineligibility_reason",
    ])


def _frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _run(diags, component_to_roi, elig=None, sample_id="S1"):
    return build_component_domain_regions(
        sample_id=sample_id,
        region_diagnostics=diags,
        component_to_roi=component_to_roi,
        tessellation_domain_regions_frame=(
            elig if elig is not None else _empty_frame()
        ),
    )


# ---------------------------------------------------------------------------


def test_column_set_locked():
    df = _run(diags=[_region("1", 0)], component_to_roi={"1": "A"})
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert (df["sample_id"] == "S1").all()


def test_total_over_region_diagnostics():
    diags = [
        _region("1", 0, area=1.0, n_spots=5),
        _region("1", 1, area=2.0, n_spots=8),
        _region("2", 0, area=3.0, n_spots=12),
    ]
    df = _run(diags=diags, component_to_roi={"1": "A", "2": "B"})
    assert len(df) == 3
    triples = list(zip(df["component_id"], df["domain_region_id"], df["roi_id"]))
    assert set(triples) == {("1", 0, "A"), ("1", 1, "A"), ("2", 0, "B")}


def test_tessellation_eligibility_merges_when_present():
    df = _run(
        diags=[
            _region("1", 0, n_spots=5),
            _region("1", 1, n_spots=2),
        ],
        component_to_roi={"1": "A"},
        elig=_frame([
            {"sample_id": "S1", "roi_id": "A", "component_id": "1",
             "domain_region_id": 0,
             "tessellation_eligible": True,
             "tessellation_ineligibility_reason": ""},
            {"sample_id": "S1", "roi_id": "A", "component_id": "1",
             "domain_region_id": 1,
             "tessellation_eligible": False,
             "tessellation_ineligibility_reason":
                 "n_spots_below_min_spots_per_tile"},
        ]),
    )
    row0 = df[df["domain_region_id"] == 0].iloc[0]
    row1 = df[df["domain_region_id"] == 1].iloc[0]
    assert bool(row0["tessellation_eligible"]) is True
    assert pd.isna(row0["tessellation_ineligibility_reason"])
    assert bool(row1["tessellation_eligible"]) is False
    assert row1["tessellation_ineligibility_reason"] == (
        "n_spots_below_min_spots_per_tile"
    )


def test_empty_inputs_yield_zero_row_schema_stable_frame():
    df = _run(diags=[], component_to_roi={})
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert len(df) == 0
    for c in _EXPECTED_COLUMNS:
        assert str(df[c].dtype) == _COMPONENT_DOMAIN_REGIONS_DTYPES[c], c


def test_source_row_fingerprint_preserved_verbatim():
    df = _run(
        diags=[
            _region("1", 0, fp="abc123"),
            _region("1", 1, fp="def456"),
        ],
        component_to_roi={"1": "A"},
    )
    fps = dict(zip(df["domain_region_id"], df["source_row_fingerprint"]))
    assert fps == {0: "abc123", 1: "def456"}


# ---------------------------------------------------------------------------
# Coherence / identity invariants
# ---------------------------------------------------------------------------


def test_missing_component_to_roi_raises():
    with pytest.raises(KeyError, match="component_to_roi missing entry"):
        _run(diags=[_region("1", 0)], component_to_roi={})


def test_duplicate_eligibility_row_raises():
    dup_rows = [{
        "sample_id": "S1", "roi_id": "A", "component_id": "1",
        "domain_region_id": 0,
        "tessellation_eligible": True,
        "tessellation_ineligibility_reason": "",
    }] * 2
    with pytest.raises(ValueError, match="duplicate row"):
        _run(
            diags=[_region("1", 0)],
            component_to_roi={"1": "A"},
            elig=_frame(dup_rows),
        )


def test_extra_eligibility_row_absent_from_region_diagnostics_raises():
    """Region present in eligibility frame but absent from
    region_diagnostics is a coherence failure — the tessellation
    stage should not carry eligibility for a region the domain
    builder never emitted."""
    rows = [{
        "sample_id": "S1", "roi_id": "A", "component_id": "1",
        "domain_region_id": 999,  # not in region_diagnostics
        "tessellation_eligible": True,
        "tessellation_ineligibility_reason": "",
    }]
    with pytest.raises(ValueError, match="absent from region_diagnostics"):
        _run(
            diags=[_region("1", 0)],
            component_to_roi={"1": "A"},
            elig=_frame(rows),
        )
