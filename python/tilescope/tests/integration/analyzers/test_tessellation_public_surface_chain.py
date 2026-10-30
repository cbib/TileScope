"""End-to-end integration: build_component_polygons → run_tessellation_stage
→ build_tessellation_outcome_tables.

Exercises the three-stage chain that produces the public tessellation
export surface. A synthetic multi-region / multipart fixture is used
so the test can pin: schema, identities, accounting.

The fixture is deliberately compact and hand-built. Two components
under two ROIs; one component has two spatially disconnected clusters
(builder produces a two-region domain → stage tessellates both →
the frame materialiser emits two region-eligibility rows + two region-replicate
rows per replicate). Multipart tile support is exercised in unit
tests (`test_region_tessellation.py`); this test focuses on the
public-surface composition contract.

Deliberate boundary: this test does NOT go through the KNN splitter
or scattered-ROI classifier. It hand-builds `source_df` and
`spatial_df` because those pre-analysis steps are covered by their
own tests, and covering them here would obscure what this test
verifies (the public tessellation surface).
"""
from __future__ import annotations

from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
from tilescope.settings import load_run_config


from types import MappingProxyType
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from tilescope import cols
from tilescope.analyzers.tessellation_analysis import (
    run_tessellation_stage,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus,
)
from tilescope.component_data import ComponentData
from tilescope.analyzers.tessellation_outcome_frames import (
    COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS,
    COMPONENT_REGION_REPLICATE_STATUS_COLUMNS,
    COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS,
    build_tessellation_outcome_tables,
)
from tilescope.spatial.component_domains import build_component_polygons


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


_SAMPLE_MEDIAN_NN = 1.0
_RADIUS_FACTORS = {
    "component_support_radius_factor": 0.60,
    "acquisition_support_radius_factor": 0.60,
    "ineligible_spot_exclusion_radius_factor": 0.60,
}
_N_REPLICATES = 2
_TILE_MIN_POINTS = 1
_VORONOI_FRACTION = 0.6
_BASE_SEED = 4242


def _grid_cluster(x0: float, y0: float, n_side: int, spacing: float = 0.8):
    """Compact `n_side × n_side` grid of coordinates anchored at (x0, y0)."""
    xs, ys = np.meshgrid(
        x0 + np.arange(n_side) * spacing,
        y0 + np.arange(n_side) * spacing,
    )
    return np.column_stack([xs.ravel(), ys.ravel()])


def _build_frames() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Synthetic geometry:
      - Component C_multi (ROI R1): two clusters separated by 6 units;
        the 0.60·d_sample acquisition-support radius can't bridge the
        gap, so `build_component_polygons` yields a two-region domain.
      - Component C_single (ROI R2): one 6×6 cluster; single region.
    """
    left = _grid_cluster(0.0, 0.0, n_side=4)
    right = _grid_cluster(6.0, 0.0, n_side=4)
    single = _grid_cluster(20.0, 0.0, n_side=6)

    coords = np.vstack([left, right, single])
    n_left, n_right, n_single = len(left), len(right), len(single)
    rng = np.random.default_rng(0)
    gene1 = rng.uniform(0.0, 1.0, len(coords))
    gene2 = rng.uniform(0.0, 1.0, len(coords))

    roi_ids = ["R1"] * (n_left + n_right) + ["R2"] * n_single
    component_ids = ["C_multi"] * (n_left + n_right) + ["C_single"] * n_single
    source_row_ids = np.arange(len(coords), dtype=np.int64)

    source_df = pd.DataFrame({
        cols.COORD_X: coords[:, 0],
        cols.COORD_Y: coords[:, 1],
        cols.ROI_ID: roi_ids,
        cols.SOURCE_ROW_ID: source_row_ids,
        cols.GENE1: gene1,
        cols.GENE2: gene2,
    })
    spatial_df = source_df.copy()
    spatial_df[cols.COMPONENT_ID] = component_ids

    component_to_roi = {"C_multi": "R1", "C_single": "R2"}
    return source_df, spatial_df, component_to_roi


def _run_chain():
    source_df, spatial_df, component_to_roi = _build_frames()
    component_build = build_component_polygons(
        source_df, spatial_df, component_to_roi,
        sample_median_nn=_SAMPLE_MEDIAN_NN,
        component_support_radius_factor=_RADIUS_FACTORS[
            "component_support_radius_factor"],
        acquisition_support_radius_factor=_RADIUS_FACTORS[
            "acquisition_support_radius_factor"],
        ineligible_spot_exclusion_radius_factor=_RADIUS_FACTORS[
            "ineligible_spot_exclusion_radius_factor"],
    )
    domains = component_build.component_polygons

    # Build ComponentData with SOURCE_ROW_ID retained (matches
    # sample authority's canonical component-data shape).
    spatial_components: Dict[str, ComponentData] = {}
    for cid, geom in domains.items():
        if geom.is_empty:
            continue
        comp_rows = spatial_df[spatial_df[cols.COMPONENT_ID].astype(str) == cid]
        coords = comp_rows[[cols.COORD_X, cols.COORD_Y]].to_numpy(dtype=float)
        expression = comp_rows[[
            cols.SOURCE_ROW_ID, cols.GENE1, cols.GENE2,
            cols.COORD_X, cols.COORD_Y,
        ]].copy()
        spatial_components[cid] = ComponentData(
            component_id=cid, points=coords, canvas=None,
            polygon=geom, expression_data=expression,
        )

    sample_result = run_tessellation_stage(
        sample_id="S_integration",
        spatial_components=MappingProxyType(spatial_components),
        domain_partitions=component_build.partitions,
        component_to_roi=component_to_roi,
        n_replicates=_N_REPLICATES,
        tile_min_points=_TILE_MIN_POINTS,
        voronoi_fraction=_VORONOI_FRACTION,
        base_seed=_BASE_SEED,
        source_spot_adjacency=build_source_spot_adjacency(
            build_source_spot_table_from_frame(spatial_df),
        ),
        config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
        },
        ),
    )
    tables = build_tessellation_outcome_tables(sample_result)
    return sample_result, tables


@pytest.fixture(scope="module")
def chain_result():
    return _run_chain()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_public_frames_carry_locked_columns_and_order(chain_result):
    _, tables = chain_result
    assert tuple(tables.component_tessellation_eligibility_fields.columns) == \
        COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS
    assert tuple(tables.component_domain_regions_fields.columns) == \
        COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS
    assert tuple(tables.component_region_replicate_status.columns) == \
        COMPONENT_REGION_REPLICATE_STATUS_COLUMNS


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_prefix_and_foreign_key_roi_id_populated_correctly(chain_result):
    _, tables = chain_result
    expected_roi_by_component = {"C_multi": "R1", "C_single": "R2"}
    for frame in (
        tables.component_tessellation_eligibility_fields,
        tables.component_domain_regions_fields,
        tables.component_region_replicate_status,
    ):
        assert (frame[cols.SAMPLE_ID] == "S_integration").all()
        for cid, expected_roi in expected_roi_by_component.items():
            rows = frame[frame[cols.COMPONENT_ID] == cid]
            if rows.empty:
                continue
            assert (rows[cols.ROI_ID] == expected_roi).all(), (
                f"component {cid!r} rows carry wrong roi_id"
            )


def test_component_order_canonical_across_all_frames(chain_result):
    _, tables = chain_result
    for frame in (
        tables.component_tessellation_eligibility_fields,
        tables.component_domain_regions_fields,
        tables.component_region_replicate_status,
    ):
        # Composite key is sorted; component_id sequence must be
        # non-decreasing on every frame.
        comp_seq = list(frame[cols.COMPONENT_ID])
        assert comp_seq == sorted(comp_seq), (
            f"component_id column not sorted: {comp_seq}"
        )


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_multi_region_component_produces_multiple_regions(chain_result):
    _, tables = chain_result
    # `C_multi` should have >= 2 domain regions (the two clusters).
    dr = tables.component_domain_regions_fields
    c_multi_regions = dr[dr[cols.COMPONENT_ID] == "C_multi"]
    assert len(c_multi_regions) >= 2, (
        f"C_multi expected >=2 regions, got {len(c_multi_regions)}"
    )
    # `C_single` should have exactly 1 region.
    c_single_regions = dr[dr[cols.COMPONENT_ID] == "C_single"]
    assert len(c_single_regions) == 1


def test_region_replicate_accounting_matches_eligibility_and_replicates(
    chain_result,
):
    _, tables = chain_result
    dr = tables.component_domain_regions_fields
    st = tables.component_region_replicate_status

    # For every eligible region, we expect exactly `n_replicates` rows
    # in the region-replicate status frame. Ineligible regions have
    # no replicate rows.
    for cid in dr[cols.COMPONENT_ID].unique():
        eligible_ids = set(
            dr[
                (dr[cols.COMPONENT_ID] == cid)
                & (dr[cols.TESSELLATION_ELIGIBLE].astype(bool))
            ][cols.DOMAIN_REGION_ID].tolist()
        )
        status_ids = set(
            st[st[cols.COMPONENT_ID] == cid][cols.DOMAIN_REGION_ID].tolist()
        )
        assert status_ids == eligible_ids, (
            f"component {cid!r}: status rows cover {status_ids}, "
            f"expected {eligible_ids} (eligible regions)"
        )
        # Each eligible region appears in exactly n_replicates rows.
        n_rows = st[st[cols.COMPONENT_ID] == cid].groupby(
            cols.DOMAIN_REGION_ID
        ).size()
        assert (n_rows == _N_REPLICATES).all(), (
            f"component {cid!r}: region_replicate row counts != "
            f"{_N_REPLICATES} → {n_rows.to_dict()}"
        )


# ---------------------------------------------------------------------------
# Region-execution status matches per-tile geometry (no silent drops)
# ---------------------------------------------------------------------------


def test_region_status_values_are_in_the_enum_wire_format_and_all_slots_covered(
    chain_result,
):
    _, tables = chain_result
    st = tables.component_region_replicate_status
    valid = {s.value for s in RegionExecutionStatus}
    got = set(st[cols.REGION_STATUS])
    assert got.issubset(valid), (
        f"region_status values {got - valid} not in enum wire format"
    )
    # Composite key uniqueness (defence-in-depth; reporter also
    # enforces this).
    dup = st.duplicated(subset=[
        cols.COMPONENT_ID, cols.DOMAIN_REGION_ID, cols.REPLICATE_INDEX,
    ]).any()
    assert not dup
