"""Contract tests for the per-feature spatial projection layer.

Six semantic scenarios:

1. Marginal-value identity: hex ensemble sum_g and n_high / n_low match
   a hand-recomputed reference from the bank's overlap edges + tile
   marginals.
2. Recursive immutability: mutating `source_values`, `tile_marginal_sums`,
   `hex_ensemble_projection`, or `hex_direct_projection` raises.
3. Source-order alignment: `source_values` must have `SourceSpotTable.n_spots`
   length; wrong length raises.
4. Exactly one projection per feature: two calls with distinct
   `feature_name` are independent records; each stores its own
   `feature_name` and thresholds.
5. Direct-surface marginals aggregate to canonical hex bank membership
   only (UNASSIGNED excluded).
6. Density stability is defined per hex from per-tess densities and
   returns NaN if fewer than two contributing tessellations.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.analyzers.consensus_stage import run_consensus_stage
from tilescope.analyzers.feature_projection_stage import (
    build_feature_spatial_projection,
)
from tilescope.analyzers.sample_support_stage import (
    sample_support_bank_from_component_results,
)
from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult, SampleTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus, RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.cols import GENE1, GENE2, SOURCE_ROW_ID
from tilescope.component_data import ComponentData, Tessellation, Tile
from tilescope.spatial.canvas import make_canvas
from tilescope.spatial.component_domains import partition_component_domain
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
)
from tilescope.settings import load_run_config


def _build_sample():
    region_polygon = box(0.0, 0.0, 4.0, 4.0)
    coords = np.array([
        [1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0],
    ], dtype=float)
    n = len(coords)
    source_row_ids = np.arange(n, dtype=np.int64)
    canvas = make_canvas(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}))
    g1_vals = np.array([1.0, 2.0, 3.0, 4.0])
    g2_vals = np.array([0.5, 1.5, 2.5, 3.5])
    expr = pd.DataFrame({
        GENE1: g1_vals, GENE2: g2_vals,
        SOURCE_ROW_ID: source_row_ids,
        "x": coords[:, 0], "y": coords[:, 1],
    })
    comp_data = ComponentData(
        component_id="C", points=coords.astype(float),
        canvas=canvas, polygon=region_polygon, expression_data=expr,
    )
    tid = "C_r0_rep0"
    tess = Tessellation(
        tessellation_id=tid, component_id="C",
        tiles=[Tile(
            tile_id=0, polygon=region_polygon, point_indices=np.arange(n, dtype=np.int64),
            n_spots=n, area=float(region_polygon.area),
            tessellation_id=tid, domain_region_id=0,
        )],
        method="voronoi", tile_min_points=1, fraction=1.0, rng_seed=42,
    )
    partition = partition_component_domain(
        comp_data.polygon, coords, source_row_ids,
    )
    comp_result = ComponentTessellationResult(
        parent_roi_id="R", component_data=comp_data,
        tessellations=MappingProxyType({0: {0: tess}}),
        region_diagnostics=(RegionTessellationEligibilityDiagnostic(
            component_id="C", domain_region_id=0,
            n_component_spots=n, region_area=float(region_polygon.area),
            ineligibility_reason=None,
        ),),
        region_replicate_diagnostics=(RegionReplicateTessellationDiagnostic(
            component_id="C", domain_region_id=0, replicate_index=0,
            n_tiles=1, region_status=RegionExecutionStatus.CONTRIBUTED,
        ),),
        domain_partition=partition,
    )
    sample_tess = SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({"C": comp_result}),
        n_replicates_planned=1,
    )
    sample_consensus = run_consensus_stage(
        sample_tessellation_result=sample_tess, canvas=canvas,
        sample_anchor=(0.0, 0.0),
        consensus_config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
            'consensus': {
                'min_tess_coverage_fraction': 0.1,
            },
        },
        ),
    )
    return sample_tess, sample_consensus, g1_vals, g2_vals


def _thresholds(high: float, low: float, n_used: int) -> FeatureThresholds:
    return FeatureThresholds(
        q_high=0.9, q_low=0.1,
        resolved_high=high, resolved_low=low, n_source_values_used=n_used,
    )


# ---------------------------------------------------------------------------
# 1. Marginal-value identity
# ---------------------------------------------------------------------------


def test_hex_ensemble_marginals_match_bank_edge_recomputation():
    _, sample_consensus, g1_vals, _ = _build_sample()
    thresholds = _thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size))
    proj = build_feature_spatial_projection(
        feature_name="gene1", source_values=g1_vals,
        thresholds=thresholds,
        spatial_bank=sample_consensus.spatial_bank,
    )
    support = sample_consensus.spatial_bank.consensus_bank.components["C"]
    # Recompute per-hex sum_g from bank edges + per-tile marginals.
    tile_sum_g = {}
    for tk, tms in proj.tile_marginal_sums.items():
        tile_sum_g[tk] = tms.sum_g
    for hx in support.hexes:
        expected = 0.0
        for edge in support.tile_hex_overlaps:
            if edge.hex_id != hx.hex_id:
                continue
            tk = (
                edge.component_id, edge.domain_region_id,
                edge.replicate_index, edge.tile_id,
            )
            expected += (edge.overlap_area / edge.tile_area) * tile_sum_g[tk]
        got = proj.hex_ensemble_projection[
            (hx.component_id, int(hx.hex_id))
        ].sum_g_ensemble_weighted
        assert got == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Recursive immutability
# ---------------------------------------------------------------------------


def test_projection_is_recursively_immutable():
    _, sample_consensus, g1_vals, _ = _build_sample()
    proj = build_feature_spatial_projection(
        feature_name="gene1", source_values=g1_vals,
        thresholds=_thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size)),
        spatial_bank=sample_consensus.spatial_bank,
    )
    with pytest.raises(ValueError):
        proj.source_values[0] = 999.0
    for name in (
        "tile_marginal_sums", "hex_ensemble_projection", "hex_direct_projection",
    ):
        with pytest.raises(TypeError):
            getattr(proj, name)["fake"] = None  # type: ignore[index]


# ---------------------------------------------------------------------------
# 3. Source-order alignment
# ---------------------------------------------------------------------------


def test_source_values_length_mismatch_rejected():
    _, sample_consensus, g1_vals, _ = _build_sample()
    thresholds = _thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size))
    with pytest.raises(ValueError, match="SourceSpotTable"):
        build_feature_spatial_projection(
            feature_name="gene1",
            source_values=np.array([1.0, 2.0]),
            thresholds=thresholds,
            spatial_bank=sample_consensus.spatial_bank,
        )


# ---------------------------------------------------------------------------
# 4. Exactly one projection per feature
# ---------------------------------------------------------------------------


def test_two_features_produce_independent_projections():
    _, sample_consensus, g1_vals, g2_vals = _build_sample()
    p_g1 = build_feature_spatial_projection(
        feature_name="gene1", source_values=g1_vals,
        thresholds=_thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size)),
        spatial_bank=sample_consensus.spatial_bank,
    )
    p_g2 = build_feature_spatial_projection(
        feature_name="gene2", source_values=g2_vals,
        thresholds=_thresholds(high=2.0, low=1.0, n_used=int(g2_vals.size)),
        spatial_bank=sample_consensus.spatial_bank,
    )
    assert p_g1.feature_name == "gene1"
    assert p_g2.feature_name == "gene2"
    assert p_g1.thresholds is not p_g2.thresholds
    # Distinct source arrays are stored per record.
    assert not np.array_equal(p_g1.source_values, p_g2.source_values)


# ---------------------------------------------------------------------------
# 5. Direct surface excludes UNASSIGNED spots
# ---------------------------------------------------------------------------


def test_direct_hex_marginals_exclude_unassigned_spots():
    _, sample_consensus, g1_vals, _ = _build_sample()
    proj = build_feature_spatial_projection(
        feature_name="gene1", source_values=g1_vals,
        thresholds=_thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size)),
        spatial_bank=sample_consensus.spatial_bank,
    )
    support = sample_consensus.spatial_bank.consensus_bank.components["C"]
    assigned_sum = sum(
        float(g1_vals[m.source_row_id])
        for m in support.direct_hex_membership
        if m.hex_id != -1
    )
    projection_sum = sum(
        r.sum_g_direct for r in proj.hex_direct_projection.values()
    )
    assert projection_sum == pytest.approx(assigned_sum, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Density stability semantics
# ---------------------------------------------------------------------------


def test_single_replicate_density_stability_is_nan():
    _, sample_consensus, g1_vals, _ = _build_sample()
    proj = build_feature_spatial_projection(
        feature_name="gene1", source_values=g1_vals,
        thresholds=_thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size)),
        spatial_bank=sample_consensus.spatial_bank,
    )
    # Single-replicate sample -> every hex has fewer than two contributing
    # tess densities -> stability is NaN.
    for hep in proj.hex_ensemble_projection.values():
        assert np.isnan(hep.density_ensemble_stability)


# ---------------------------------------------------------------------------
# 7. FeatureSpatialProjection rejects non-1D source_values
# ---------------------------------------------------------------------------


def test_source_values_must_be_one_dimensional():
    _, sample_consensus, g1_vals, _ = _build_sample()
    thresholds = _thresholds(high=2.5, low=1.5, n_used=int(g1_vals.size))
    with pytest.raises(ValueError, match="1-D"):
        build_feature_spatial_projection(
            feature_name="gene1",
            source_values=np.zeros((2, 2)),
            thresholds=thresholds,
            spatial_bank=sample_consensus.spatial_bank,
        )
