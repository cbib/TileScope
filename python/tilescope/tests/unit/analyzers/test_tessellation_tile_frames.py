"""Contract tests for the canonical tessellation-tile builder.

Covers the four semantic responsibilities of
:func:`~tilescope.analyzers.tessellation_tile_frames.build_tessellation_tiles_df`:

1. Empty-frame dtypes — a component with no tessellations still
   returns a Pandera-validated frame with the locked column dtypes
   (not object dtypes).
2. Sample-ID conflict — a caller-supplied ``sample_id`` that
   disagrees with the typed record raises rather than silently
   preferring one authority.
3. Component -> ROI mismatch — the ``roi_id`` convenience column
   cannot drift from the typed
   ``SampleTessellationResult.component_to_roi`` authority.
4. Known numeric tile — per-tile marginals from the projection
   stubs + joint counts from raw values match the expected
   sufficient-stat counts on a fixture with hand-picked thresholds
   and known point membership.

Broader contract properties (composite-key uniqueness, functional
dependency + integer ``tile_id`` on a real corpus) still ride the
sister real-data E2E in ``tests/e2e/test_roi_surface_greenwald.py``.
"""
from __future__ import annotations

from tilescope.component_data import ComponentData
from tilescope.cols import COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from shapely import wkt as _wkt


from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult, SampleTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus, RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.component_data import Tessellation
from tilescope.cols import (
    COMPONENT_ID,
    DOMAIN_REGION_ID,
    GENE1,
    GENE1_DENSITY,
    GENE2,
    GENE2_DENSITY,
    N_BOTH_HIGH,
    N_BOTH_LOW,
    N_G1_HIGH,
    N_G1_LOW,
    N_G2_HIGH,
    N_G2_LOW,
    N_SPOTS,
    REPLICATE_INDEX,
    SAMPLE_ID,
    SUM_G1,
    SUM_G2,
    TESSELLATION_ID,
    TILE_AREA,
    TILE_GEOMETRY_WKT,
    TILE_ID,
)
from tilescope.metrics.enrichment_thresholds import PercentileThresholds
from tilescope.analyzers.tessellation_tile_frames import (
    TESSELLATION_TILES_COLUMNS, build_tessellation_tile_support,
    build_tessellation_tiles_df as _build_tessellation_tiles_df,
    _assert_roi_id_functional_dependency,
)
from tilescope.spatial.component_domains import ComponentDomainPartition
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds, TileMarginalSums,
)
from tilescope.spatial.sample_support_bank import SampleSupportFingerprint
from tilescope.component_data import Tile


_STUB_FINGERPRINT = SampleSupportFingerprint(
    sample_id="S", source_frame_hash="stub-hash",
    support_policy_version="standard-v1", source_adjacency_config_hash="stub-graph",
    scattered_support_policy_hash="stub-scat",
    component_geometry_contract_hash="stub-geom",
    n_replicates_planned=1, base_seed=0,
    tessellation_config_hash="stub-tess",
)


def build_tessellation_tiles_df(
    *, sample_tess_result, thresholds, projection_g1, projection_g2,
    sample_id="",
):
    """Test adapter exercising the two-stage support/value contract."""
    support = build_tessellation_tile_support(
        sample_tess_result=sample_tess_result, sample_id=sample_id,
    )
    return _build_tessellation_tiles_df(
        tile_support=support, thresholds=thresholds,
        projection_g1=projection_g1, projection_g2=projection_g2,
    )


def _projection_stub(
    feature_name: str, values, thresholds: PercentileThresholds,
    *, is_g1: bool, tile_marginals=None,
) -> FeatureSpatialProjection:
    """Minimum-surface projection stub for the tile-builder tests: only
    tile_marginal_sums + source_values + bank_fingerprint are read."""
    q_hi = thresholds.q_g1_threshold if is_g1 else thresholds.q_g2_threshold
    q_lo = thresholds.q_g1_low_threshold if is_g1 else thresholds.q_g2_low_threshold
    return FeatureSpatialProjection(
        feature_name=feature_name,
        bank_fingerprint=_STUB_FINGERPRINT,
        source_values=np.asarray(values, dtype=float),
        thresholds=FeatureThresholds(
            q_high=thresholds.q_percentile, q_low=thresholds.q_low_percentile,
            resolved_high=q_hi, resolved_low=q_lo,
            n_source_values_used=len(values),
        ),
        tile_marginal_sums=tile_marginals or {},
        hex_ensemble_projection={},
        hex_direct_projection={},
    )


# --------------------------------------------------------------------------- #
# Small fixture helpers
# --------------------------------------------------------------------------- #


def _empty_component_stub(
    component_id: str, parent_roi_id: str,
) -> ComponentTessellationResult:
    """No tessellations — the empty-input path."""
    component = ComponentData(
        component_id=component_id, points=np.zeros((0, 2)),
        canvas=None, polygon=box(0.0, 0.0, 1.0, 1.0),
        expression_data=pd.DataFrame({
            SOURCE_ROW_ID: np.asarray([], dtype=np.int64),
            COORD_X: [], COORD_Y: [],
            GENE1: [], GENE2: [], ROI_ID: [],
        }),
    )
    partition = ComponentDomainPartition(
        canonical_regions=(),
        canonical_ids_by_source_row=MappingProxyType({}),
        source_row_ids_by_canonical_id=MappingProxyType({}),
        source_row_fingerprint_by_canonical_id=MappingProxyType({}),
    )
    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=component,
        tessellations=MappingProxyType({}),
        region_diagnostics=(),
        region_replicate_diagnostics=(),
        domain_partition=partition,
    )


def _one_tile_component(
    component_id: str, parent_roi_id: str,
    *,
    gene1_values: list, gene2_values: list,
    region_id: int = 0, replicate_index: int = 0,
    tile_id: int = 0,
) -> ComponentTessellationResult:
    """Component whose single Tessellation has one Tile owning the
    supplied gene values (indexed 0..N-1).
    """
    n = len(gene1_values)
    assert n == len(gene2_values)
    expression = pd.DataFrame({
        GENE1: gene1_values, GENE2: gene2_values,
    })
    tile = Tile(
        tile_id=tile_id,
        polygon=box(0.0, 0.0, 1.0, 1.0),
        point_indices=np.arange(n, dtype=np.int64),
        n_spots=n,
        area=1.0,
        tessellation_id="tess-0",
        domain_region_id=region_id,
    )
    tess = Tessellation(
        tessellation_id="tess-0",
        component_id=component_id,
        tiles=[tile],
        method="voronoi",
        tile_min_points=1,
        fraction=1.0,
        rng_seed=None,
    )
    partition = ComponentDomainPartition(
        canonical_regions=(box(0.0, 0.0, 1.0, 1.0),),
        canonical_ids_by_source_row=MappingProxyType(
            {sid: int(region_id) for sid in range(n)}
        ),
        source_row_ids_by_canonical_id=MappingProxyType(
            {int(region_id): tuple(range(n))}
        ),
        source_row_fingerprint_by_canonical_id=MappingProxyType(
            {int(region_id): "stub_fp"}
        ),
    )
    expr_full = expression.copy()
    if SOURCE_ROW_ID not in expr_full.columns:
        expr_full[SOURCE_ROW_ID] = np.arange(n, dtype=np.int64)
    if COORD_X not in expr_full.columns:
        expr_full[COORD_X] = 0.0
    if COORD_Y not in expr_full.columns:
        expr_full[COORD_Y] = 0.0
    if ROI_ID not in expr_full.columns:
        expr_full[ROI_ID] = parent_roi_id
    component = ComponentData(
        component_id=component_id, points=np.zeros((n, 2)),
        canvas=None, polygon=box(0.0, 0.0, 1.0, 1.0),
        expression_data=expr_full,
    )
    region_diag = RegionTessellationEligibilityDiagnostic(
        component_id=component_id, domain_region_id=int(region_id),
        n_component_spots=n, region_area=1.0, ineligibility_reason=None,
    )
    rr_diag = RegionReplicateTessellationDiagnostic(
        component_id=component_id, domain_region_id=int(region_id),
        replicate_index=int(replicate_index), n_tiles=1,
        region_status=RegionExecutionStatus.CONTRIBUTED,
    )
    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=component,
        tessellations=MappingProxyType({
            int(region_id): MappingProxyType({int(replicate_index): tess}),
        }),
        region_diagnostics=(region_diag,),
        region_replicate_diagnostics=(rr_diag,),
        domain_partition=partition,
    )


def _sample(
    sample_id: str,
    components: dict,
    n_replicates_planned: int = 1,
) -> SampleTessellationResult:
    return SampleTessellationResult.from_component_results(
        sample_id=sample_id,
        components=MappingProxyType(components),
        n_replicates_planned=n_replicates_planned,
    )


_THRESHOLDS = PercentileThresholds(
    q_g1_threshold=10.0, q_g2_threshold=10.0, q_percentile=0.75,
    q_g1_low_threshold=2.0, q_g2_low_threshold=2.0, q_low_percentile=0.25,
    min_points_threshold=1,
)


# --------------------------------------------------------------------------- #
# 1. Empty-frame dtypes
# --------------------------------------------------------------------------- #


def test_empty_input_returns_typed_empty_frame():
    """Zero components -> zero rows, but column dtypes are still the
    locked Int64 / Float64 / string dtypes (not `object`)."""
    result = _sample("S", {"c1": _empty_component_stub("c1", "roi-1")})
    proj_g1 = _projection_stub("gene1", [], _THRESHOLDS, is_g1=True)
    proj_g2 = _projection_stub("gene2", [], _THRESHOLDS, is_g1=False)
    df = build_tessellation_tiles_df(
        sample_tess_result=result, thresholds=_THRESHOLDS,
        projection_g1=proj_g1, projection_g2=proj_g2,
    )
    assert list(df.columns) == list(TESSELLATION_TILES_COLUMNS)
    assert len(df) == 0
    assert all(str(dtype) != "object" for dtype in df.dtypes)


# --------------------------------------------------------------------------- #
# 2. Sample-ID conflict
# --------------------------------------------------------------------------- #


def test_sample_id_conflict_raises():
    """Caller-supplied sample_id must match the typed record — two
    identity authorities are not permitted."""
    result = _sample("typed-S", {"c1": _empty_component_stub("c1", "roi-1")})
    proj_g1 = _projection_stub("gene1", [], _THRESHOLDS, is_g1=True)
    proj_g2 = _projection_stub("gene2", [], _THRESHOLDS, is_g1=False)
    with pytest.raises(ValueError, match="sample_id identity conflict"):
        build_tessellation_tiles_df(
            sample_tess_result=result, thresholds=_THRESHOLDS,
            projection_g1=proj_g1, projection_g2=proj_g2,
            sample_id="caller-S",
        )


# --------------------------------------------------------------------------- #
# 3. Component -> ROI mismatch
# --------------------------------------------------------------------------- #


def test_functional_dependency_multiple_roi_ids_per_component_raises():
    """`(sample_id, component_id) -> one roi_id` is enforced. A
    hand-built frame that violates it fails loud."""
    df = pd.DataFrame({
        SAMPLE_ID: ["S", "S"],
        COMPONENT_ID: ["c1", "c1"],
        ROI_ID: ["roi-1", "roi-2"],
    })
    with pytest.raises(ValueError, match="multiple roi_id values"):
        _assert_roi_id_functional_dependency(df, {"c1": "roi-1"})


def test_functional_dependency_emitted_roi_disagrees_with_authority_raises():
    """Even a single roi_id per component must equal the typed
    `component_to_roi` authority."""
    df = pd.DataFrame({
        SAMPLE_ID: ["S"], COMPONENT_ID: ["c1"], ROI_ID: ["roi-wrong"],
    })
    with pytest.raises(ValueError, match="emitted roi_id"):
        _assert_roi_id_functional_dependency(df, {"c1": "roi-right"})


def test_functional_dependency_unknown_component_raises():
    df = pd.DataFrame({
        SAMPLE_ID: ["S"], COMPONENT_ID: ["c-unknown"], ROI_ID: ["roi-1"],
    })
    with pytest.raises(ValueError, match="not in "):
        _assert_roi_id_functional_dependency(df, {"c1": "roi-1"})


# --------------------------------------------------------------------------- #
# 4. Known numeric tile
# --------------------------------------------------------------------------- #


def test_known_tile_sufficient_counts_and_density_match_thresholds():
    """One tile, five spots, hand-picked thresholds -> emitted
    sufficient counts, sums, and densities match projection marginals
    + strict ``>`` / ``<`` joint counts."""
    # Spot indices           0     1     2    3    4
    gene1 = [   1.0,  8.0, 12.0, 15.0, 20.0]
    gene2 = [   0.5,  1.0,  9.0, 11.0, 25.0]
    # Under q_g1 = q_g2 = 10.0 (strict >) and q_g1_low = q_g2_low = 2.0
    # (strict <):
    #   g1_high: indices {2, 3, 4}     -> n_g1_high = 3
    #   g2_high: indices {3, 4}        -> n_g2_high = 2
    #   both_high: {3, 4}              -> n_both_high = 2
    #   g1_low: indices {0}            -> n_g1_low = 1
    #   g2_low: indices {0, 1}         -> n_g2_low = 2
    #   both_low: {0}                  -> n_both_low = 1
    result = _sample(
        "S",
        {"c1": _one_tile_component(
            "c1", "roi-A",
            gene1_values=gene1, gene2_values=gene2,
        )},
    )
    tk = ("c1", 0, 0, 0)
    tms_g1 = TileMarginalSums(
        sum_g=float(sum(gene1)),
        n_high=float(sum(v > _THRESHOLDS.q_g1_threshold for v in gene1)),
        n_low=float(sum(v < _THRESHOLDS.q_g1_low_threshold for v in gene1)),
    )
    tms_g2 = TileMarginalSums(
        sum_g=float(sum(gene2)),
        n_high=float(sum(v > _THRESHOLDS.q_g2_threshold for v in gene2)),
        n_low=float(sum(v < _THRESHOLDS.q_g2_low_threshold for v in gene2)),
    )
    proj_g1 = _projection_stub(
        "gene1", gene1, _THRESHOLDS, is_g1=True, tile_marginals={tk: tms_g1},
    )
    proj_g2 = _projection_stub(
        "gene2", gene2, _THRESHOLDS, is_g1=False, tile_marginals={tk: tms_g2},
    )
    df = build_tessellation_tiles_df(
        sample_tess_result=result, thresholds=_THRESHOLDS,
        projection_g1=proj_g1, projection_g2=proj_g2,
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row[SAMPLE_ID] == "S"
    assert row[ROI_ID] == "roi-A"
    assert row[COMPONENT_ID] == "c1"
    assert int(row[DOMAIN_REGION_ID]) == 0
    assert int(row[REPLICATE_INDEX]) == 0
    assert row[TESSELLATION_ID] == "tess-0"
    assert int(row[TILE_ID]) == 0
    assert int(row[N_SPOTS]) == 5
    assert float(row[SUM_G1]) == pytest.approx(sum(gene1))
    assert float(row[SUM_G2]) == pytest.approx(sum(gene2))
    assert int(row[N_G1_HIGH]) == 3
    assert int(row[N_G2_HIGH]) == 2
    assert int(row[N_BOTH_HIGH]) == 2
    assert int(row[N_G1_LOW]) == 1
    assert int(row[N_G2_LOW]) == 2
    assert int(row[N_BOTH_LOW]) == 1
    assert float(row[GENE1_DENSITY]) == pytest.approx(sum(gene1) / 5)
    assert float(row[GENE2_DENSITY]) == pytest.approx(sum(gene2) / 5)
    assert float(row[TILE_AREA]) == pytest.approx(1.0)
    # WKT is round-trippable — construction always emits geometry.
    poly = _wkt.loads(row[TILE_GEOMETRY_WKT])
    assert poly.is_valid and poly.area == pytest.approx(1.0)
