"""Column, dtype, ordering, and key-uniqueness contract tests for
`TessellationOutcomeTables`.

Semantic completeness of the underlying projection is covered by
`test_tessellation_analysis.py`; this suite verifies the public-surface
materialiser correctly composes the projection outputs, applies the
frozen dtype map, and enforces determinism (sorted by composite key,
key columns non-null and unique).
"""
from __future__ import annotations

import numpy as _np
import pandas as _pd
from tilescope.component_data import ComponentData
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, ROI_ID, SOURCE_ROW_ID
from tilescope.component_data import Tessellation
from tilescope.component_data import Tile


from types import MappingProxyType

import pandas as pd
import pytest

from tilescope import cols
from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult,
    SampleTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus,
    RegionIneligibilityReason,
    RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.analyzers.tessellation_outcome_frames import (
    COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS,
    COMPONENT_REGION_REPLICATE_STATUS_COLUMNS,
    COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS,
    DTYPE_MAP,
    build_tessellation_outcome_tables,
)
from shapely.geometry import box

from tilescope.spatial.component_domains import ComponentDomainPartition


def _partition_matching(region_diagnostics) -> ComponentDomainPartition:
    """Minimal ComponentDomainPartition consistent with the supplied
    region_diagnostics. Region polygons are unit boxes offset by
    region_id — they exist only so the diagnostic invariant on
    ComponentTessellationResult passes; nothing in these frame tests
    inspects the polygon itself.
    """
    if not region_diagnostics:
        return ComponentDomainPartition(
            canonical_regions=(),
            canonical_ids_by_source_row=MappingProxyType({}),
            source_row_ids_by_canonical_id=MappingProxyType({}),
            source_row_fingerprint_by_canonical_id=MappingProxyType({}),
        )
    rids = sorted({d.domain_region_id for d in region_diagnostics})
    if rids != list(range(len(rids))):
        raise ValueError(
            f"_partition_matching: region_diagnostics IDs must be a "
            f"contiguous 0..N-1 sequence; got {rids}."
        )
    n_by_rid = {d.domain_region_id: d.n_component_spots for d in region_diagnostics}
    canonical_regions = tuple(
        box(float(rid), 0.0, float(rid) + 1.0, 1.0) for rid in rids
    )
    ids_by_rid: dict = {}
    fp_by_rid: dict = {}
    canonical_ids_by_src: dict = {}
    next_src = 0
    for rid in rids:
        n = n_by_rid[rid]
        assigned = tuple(range(next_src, next_src + n))
        ids_by_rid[rid] = assigned
        fp_by_rid[rid] = f"stub_fp_r{rid}_n{n}"
        for sid in assigned:
            canonical_ids_by_src[sid] = rid
        next_src += n
    return ComponentDomainPartition(
        canonical_regions=canonical_regions,
        canonical_ids_by_source_row=MappingProxyType(canonical_ids_by_src),
        source_row_ids_by_canonical_id=MappingProxyType(ids_by_rid),
        source_row_fingerprint_by_canonical_id=MappingProxyType(fp_by_rid),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stub_component_data(component_id: str, n_spots: int = 1):
    _expr = _pd.DataFrame({
        SOURCE_ROW_ID: _np.arange(n_spots, dtype=_np.int64),
        COORD_X: [0.0] * n_spots, COORD_Y: [0.0] * n_spots,
        GENE1: [0.0] * n_spots, GENE2: [0.0] * n_spots,
        ROI_ID: ["R"] * n_spots,
    })
    return ComponentData(
        component_id=component_id,
        points=_np.zeros((n_spots, 2)),
        canvas=None, polygon=box(0.0, 0.0, 1.0, 1.0),
        expression_data=_expr,
    )


def _region(rid: int, n_spots: int, area: float, eligible: bool = True,
            component_id: str = "C"):
    return RegionTessellationEligibilityDiagnostic(
        component_id=component_id, domain_region_id=rid,
        n_component_spots=n_spots, region_area=area,
        ineligibility_reason=(
            None if eligible else RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN
        ),
    )


def _region_replicate(rid: int, replicate: int, n_tiles: int, status,
                      component_id: str = "C"):
    return RegionReplicateTessellationDiagnostic(
        component_id=component_id, domain_region_id=rid,
        replicate_index=replicate,
        n_tiles=n_tiles, region_status=status,
    )


def _sample(
    *,
    components,
    sample_id: str = "S", n_replicates_planned: int = 2,
) -> SampleTessellationResult:
    return SampleTessellationResult.from_component_results(
        sample_id=sample_id,
        components=MappingProxyType(components),
        n_replicates_planned=n_replicates_planned,
    )


def _component_result(
    cid: str = "C",
    parent_roi_id: str = "R",
    region_diagnostics=(),
    region_replicate_diagnostics=(),
):
    # Build minimal Tessellation stubs for every CONTRIBUTED
    # region-replicate diagnostic so the completeness invariant holds.
    # Skip duplicates so the invariant surfaces the duplicate itself,
    # not a downstream tile-count mismatch.
    stub_polygon = box(0.0, 0.0, 1.0, 1.0)
    tessellations: dict = {}
    seen_keys: set = set()
    for rr in region_replicate_diagnostics:
        if rr.region_status is not RegionExecutionStatus.CONTRIBUTED:
            continue
        key = (rr.domain_region_id, rr.replicate_index)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        tiles = [
            Tile(tile_id=i, polygon=stub_polygon,
                 point_indices=_np.array([], dtype=int),
                 n_spots=0, area=1.0,
                 tessellation_id=f"{cid}_r{rr.domain_region_id}_t{rr.replicate_index}",
                 domain_region_id=rr.domain_region_id)
            for i in range(rr.n_tiles)
        ]
        tess = Tessellation(
            tessellation_id=f"{cid}_r{rr.domain_region_id}_t{rr.replicate_index}",
            component_id=cid, tiles=tiles,
            method="voronoi", tile_min_points=1, fraction=1.0,
        )
        tessellations.setdefault(rr.domain_region_id, {})[rr.replicate_index] = tess
    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=_stub_component_data(cid),
        tessellations=MappingProxyType({
            rid: MappingProxyType(dict(rmap))
            for rid, rmap in tessellations.items()
        }),
        region_diagnostics=tuple(region_diagnostics),
        region_replicate_diagnostics=tuple(region_replicate_diagnostics),
        domain_partition=_partition_matching(region_diagnostics),
    )


# ---------------------------------------------------------------------------
# Column contract locks
# ---------------------------------------------------------------------------


def test_column_contracts_are_exact_and_ordered():
    result = _sample(components={})
    tables = build_tessellation_outcome_tables(result)

    assert tuple(tables.component_tessellation_eligibility_fields.columns) == \
        COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS
    assert tuple(tables.component_domain_regions_fields.columns) == \
        COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS
    assert tuple(tables.component_region_replicate_status.columns) == \
        COMPONENT_REGION_REPLICATE_STATUS_COLUMNS


# ---------------------------------------------------------------------------
# Dtype contract — stable across empty / populated / all-null
# ---------------------------------------------------------------------------


def _assert_dtypes(frame: pd.DataFrame, columns: tuple) -> None:
    for c in columns:
        expected = DTYPE_MAP[c]
        actual = frame[c].dtype
        # `pandas.api.types.pandas_dtype` normalises "Int64"/"Float64"
        # /"string"/"bool" strings to their canonical dtypes.
        expected_dtype = pd.api.types.pandas_dtype(expected)
        assert actual == expected_dtype, (
            f"column {c!r}: expected dtype {expected!r} ({expected_dtype}), "
            f"got {actual}"
        )


def test_dtypes_are_stable_on_empty_frames():
    tables = build_tessellation_outcome_tables(_sample(components={}))
    _assert_dtypes(
        tables.component_tessellation_eligibility_fields,
        COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS,
    )
    _assert_dtypes(
        tables.component_domain_regions_fields,
        COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS,
    )
    _assert_dtypes(
        tables.component_region_replicate_status,
        COMPONENT_REGION_REPLICATE_STATUS_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Ordering — materialiser sorts each frame; upstream construction order
# does not leak
# ---------------------------------------------------------------------------


def test_shuffled_upstream_component_order_yields_canonical_sorted_frames():
    # Construct the components mapping in reverse-alphabetical order.
    # The sorted composite key must land the rows in canonical order
    # on every frame.
    comps = {}
    for cid in ("Z_c1", "A_c1", "M_c1"):
        comps[cid] = _component_result(
            cid=cid, parent_roi_id=cid.lower(),
            region_diagnostics=[_region(0, 40, 80.0, component_id=cid)],
            region_replicate_diagnostics=[
                _region_replicate(
                    0, 0, n_tiles=4,
                    status=RegionExecutionStatus.CONTRIBUTED,
                    component_id=cid,
                ),
            ],
        )
    sample = _sample(components=comps, sample_id="S", n_replicates_planned=1)
    tables = build_tessellation_outcome_tables(sample)
    expected = ["A_c1", "M_c1", "Z_c1"]
    for frame in (
        tables.component_tessellation_eligibility_fields,
        tables.component_domain_regions_fields,
        tables.component_region_replicate_status,
    ):
        assert list(frame[cols.COMPONENT_ID]) == expected


def test_within_component_region_and_replicate_ordering_is_stable():
    # Insert region/replicate rows in reverse order upstream; materialiser
    # must sort them by (domain_region_id, replicate_index).
    comp = _component_result(
        region_diagnostics=[
            _region(1, 50, 100.0), _region(0, 50, 100.0),
        ],
        region_replicate_diagnostics=[
            _region_replicate(1, 1, 4, RegionExecutionStatus.CONTRIBUTED),
            _region_replicate(1, 0, 4, RegionExecutionStatus.CONTRIBUTED),
            _region_replicate(0, 1, 4, RegionExecutionStatus.CONTRIBUTED),
            _region_replicate(0, 0, 4, RegionExecutionStatus.CONTRIBUTED),
        ],
    )
    tables = build_tessellation_outcome_tables(
        _sample(components={"C": comp}, n_replicates_planned=2),
    )
    dr = tables.component_domain_regions_fields
    assert list(dr[cols.DOMAIN_REGION_ID]) == [0, 1]
    st = tables.component_region_replicate_status
    assert list(zip(st[cols.DOMAIN_REGION_ID], st[cols.REPLICATE_INDEX])) == \
        [(0, 0), (0, 1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# Uniqueness + null-key rejection
# ---------------------------------------------------------------------------


def test_duplicate_composite_key_on_region_replicate_status_raises():
    # Duplicate (domain_region_id, replicate_index) is rejected at
    # ComponentTessellationResult construction time — the typed
    # completeness invariant is the primary-records guard.
    with pytest.raises(
        ValueError,
        match="duplicate region-replicate diagnostic key",
    ):
        _component_result(
            region_diagnostics=[_region(0, 50, 100.0)],
            region_replicate_diagnostics=[
                _region_replicate(0, 0, 4, RegionExecutionStatus.CONTRIBUTED),
                _region_replicate(0, 0, 5, RegionExecutionStatus.CONTRIBUTED),
            ],
        )


# ---------------------------------------------------------------------------
# End-to-end composition sanity (identity prefix + a few values)
# ---------------------------------------------------------------------------


def test_ineligibility_reason_encoded_as_string():
    comp = _component_result(
        region_diagnostics=[_region(0, 4, 50.0, eligible=False)],
    )
    tables = build_tessellation_outcome_tables(
        _sample(components={"C": comp}, n_replicates_planned=1),
    )
    dr = tables.component_domain_regions_fields
    assert bool(dr.iloc[0][cols.TESSELLATION_ELIGIBLE]) is False
    assert dr.iloc[0][cols.TESSELLATION_INELIGIBILITY_REASON] == \
        "n_spots_below_min_spots_per_tile"
