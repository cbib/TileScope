"""Contract tests for the parent-ROI tessellation correlation surface.

Covers ``build_roi_replicate_correlations_df`` and
``compute_roi_tessellation_correlations`` — total-table shape, status
precedence, finite-row population, conditional analytic p + BH q,
summary reduction, identity guards, replicate BH families, and the
CORE-INF-01b synchronized-median reduced-estimator null end-to-end.
"""
from __future__ import annotations

from tilescope.metrics.enrichment_thresholds import PercentileThresholds
from tilescope.analyzers.tessellation_tile_frames import build_tessellation_tile_support, build_tessellation_tiles_df
from tilescope.spatial.feature_projection import FeatureSpatialProjection, FeatureThresholds, TileMarginalSums


import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.parent_roi_outcomes import (
    ParentReplicateCorrelationStatus,
    ParentTessellationCorrelationStatus,
)
from tilescope.analyzers.parent_roi_tessellation_correlation import (
    build_roi_replicate_correlations_df as _build_replicate_correlations,
    compute_roi_tessellation_correlations as _compute_tessellation_correlations,
)
from tilescope.analyzers.parent_roi_tessellation_common import (
    build_tessellation_pooling_index,
)
from tilescope.analyzers.tessellation_analysis import SampleTessellationResult
from tilescope.cols import (
    CORRELATION_VALUE,
    ELIGIBLE_SPOT_COVERAGE_FRACTION,
    N_CONTRIBUTED_SPOTS, N_ELIGIBLE_SPOTS, N_TILES_CONTRIBUTED,
    PARENT_REPLICATE_CORRELATION_STATUS,
    PARENT_TESSELLATION_CORRELATION_STATUS,
    REPLICATE_INDEX, ROI_ID, SAMPLE_ID, SAMPLE_SIZE,
    SOURCE_ROI_SPOT_COVERAGE_FRACTION,
)
from tilescope.tests.unit.analyzers.parent_roi_tessellation_fixtures import (
    component_result,
    contrib,
    real_component_result,
    region_diag,
    support_frame,
    tile_df,
    tiles_from_dict,
)


def build_roi_replicate_correlations_df(
    *, sample_tess_result, tessellation_tiles_df, **kwargs,
):
    index = build_tessellation_pooling_index(
        sample_tess_result=sample_tess_result,
        tessellation_tiles_df=tessellation_tiles_df,
        support_status_df=kwargs["support_status_df"],
    )
    return _build_replicate_correlations(
        pooling_index=index, g1_id="Neuron", g2_id="AC", **kwargs,
    )


def compute_roi_tessellation_correlations(**kwargs):
    return _compute_tessellation_correlations(
        g1_id="Neuron", g2_id="AC", **kwargs,
    )


def _summary_row(*, rep: int, status: str, value: float, roi: str = "R"):
    """Minimal per-replicate row shape the reducer reads."""
    return {
        SAMPLE_ID: "S", ROI_ID: roi, REPLICATE_INDEX: rep,
        PARENT_REPLICATE_CORRELATION_STATUS: status,
        CORRELATION_VALUE: value,
        N_TILES_CONTRIBUTED: 5,
        ELIGIBLE_SPOT_COVERAGE_FRACTION: 1.0,
        SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.9,
    }


# ---------------------------------------------------------------------------
# Total-table shape + status precedence
# ---------------------------------------------------------------------------


def test_total_table_emits_row_per_source_roi_and_planned_replicate():
    """Every (source ROI, planned replicate) cell gets a row, including
    all-scattered samples and unfinished replicates."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 5)],
        contributed_regions=[
            contrib(t, n_eligible=1, n_contributing=1, n_tiles=0)
            for t in range(3)
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=3,
    )
    support = support_frame([
        {"roi_id": "R", "source_spots": 5, "spatial_support_class": "supported"},
        {"roi_id": "Scattered", "source_spots": 3, "spatial_support_class": "scattered"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict({}),
        support_status_df=support, sample_id="S",
    )
    assert len(df) == 6
    scattered_rows = df[df[ROI_ID] == "Scattered"]
    assert len(scattered_rows) == 3
    assert (
        scattered_rows[PARENT_REPLICATE_CORRELATION_STATUS]
        == ParentReplicateCorrelationStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    ).all()


def test_no_eligible_support_row_marks_no_tessellation_eligible_component():
    """No eligible support → NO_TESSELLATION_ELIGIBLE_COMPONENT + null
    eligible coverage + 0.0 source coverage."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 3, eligible=False)],
        contributed_regions=(),
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=2,
    )
    support = support_frame([
        {"roi_id": "R", "source_spots": 3, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict({}),
        support_status_df=support, sample_id="S",
    )
    assert len(df) == 2
    assert (
        df[PARENT_REPLICATE_CORRELATION_STATUS]
        == ParentReplicateCorrelationStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    ).all()
    assert df[ELIGIBLE_SPOT_COVERAGE_FRACTION].isna().all()
    assert (df[SOURCE_ROI_SPOT_COVERAGE_FRACTION] == 0.0).all()


def test_failed_eligible_region_marks_incomplete_support_coverage():
    """Region-shortfall variant: eligible regions present but one failed
    to contribute → INCOMPLETE_SUPPORT_COVERAGE."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 10), region_diag(1, 10)],
        contributed_regions=[
            contrib(0, n_eligible=2, n_contributing=1, n_tiles=3),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=1,
    )
    tiles = {"C": {0: tile_df([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2, 2, 2])}}
    support = support_frame([
        {"roi_id": "R", "source_spots": 20, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support,
    )
    assert df.iloc[0][PARENT_REPLICATE_CORRELATION_STATUS] == (
        ParentReplicateCorrelationStatus.INCOMPLETE_SUPPORT_COVERAGE.value
    )


def test_spot_shortfall_marks_incomplete_support_coverage():
    """Spot-shortfall variant: every region contributed but contributed
    < eligible → INCOMPLETE_SUPPORT_COVERAGE."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 20)],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=3),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=1,
    )
    tiles = {"C": {0: tile_df([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5, 5, 5])}}
    support = support_frame([
        {"roi_id": "R", "source_spots": 25, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support, sample_id="S",
    )
    row = df.iloc[0]
    assert row[PARENT_REPLICATE_CORRELATION_STATUS] == (
        ParentReplicateCorrelationStatus.INCOMPLETE_SUPPORT_COVERAGE.value
    )
    assert row[N_CONTRIBUTED_SPOTS] == 15
    assert row[N_ELIGIBLE_SPOTS] == 20
    assert pd.isna(row[CORRELATION_VALUE])


def test_contributed_exceeds_eligible_raises_invariant_violation():
    """Construction invariant: contributed_spots > eligible_spots →
    raise (not a status row)."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 10)],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=3),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=1,
    )
    tiles = {"C": {0: tile_df([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [10, 10, 10])}}
    support = support_frame([
        {"roi_id": "R", "source_spots": 40, "spatial_support_class": "supported"},
    ])
    with pytest.raises(ValueError, match="overshot the eligible source-row set"):
        build_roi_replicate_correlations_df(
            sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
            support_status_df=support, sample_id="S",
        )


# ---------------------------------------------------------------------------
# Finite path + conditional analytic p / BH q columns
# ---------------------------------------------------------------------------


def test_finite_row_populates_corr_value_and_sample_size():
    """FINITE row emits the effect + sample size; no permutation fields
    on this surface."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 15)],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=5),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=1,
    )
    tiles = {"C": {0: tile_df(
        [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], [3]*5,
    )}}
    support = support_frame([
        {"roi_id": "R", "source_spots": 15, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support, sample_id="S",
    )
    row = df.iloc[0]
    assert row[PARENT_REPLICATE_CORRELATION_STATUS] == (
        ParentReplicateCorrelationStatus.FINITE.value
    )
    assert row[CORRELATION_VALUE] == pytest.approx(1.0)
    assert row[SAMPLE_SIZE] == 5
    assert row[N_TILES_CONTRIBUTED] == 5
    assert row[N_CONTRIBUTED_SPOTS] == 15
    assert row[ELIGIBLE_SPOT_COVERAGE_FRACTION] == pytest.approx(1.0)
    assert row[SOURCE_ROI_SPOT_COVERAGE_FRACTION] == pytest.approx(1.0)


def test_replicate_correlation_emits_conditional_p_and_q_only():
    """Replicate surface carries conditional analytic p + BH q; never
    permutation p, never marginal q, never FDR across replicates."""
    comp_r = component_result(
        region_diagnostics=[region_diag(0, 15)],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=5),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp_r}, n_replicates_planned=1,
    )
    tiles = {"C": {0: tile_df(
        [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 2.5, 5.0, 5.5], [3]*5,
    )}}
    support = support_frame([
        {"roi_id": "R", "source_spots": 15,
         "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support,
    )
    required = (
        "corr_p_analytic_conditional",
        "corr_q_analytic_conditional",
        "fdr_family_size",
    )
    forbidden = (
        "corr_p", "corr_p_perm", "corr_perm_mean", "corr_perm_sd",
        "n_permutations", "corr_q", "corr_q_perm",
    )
    for c in required:
        assert c in df.columns, f"{c!r} must appear on roi_replicate_correlations"
    for c in forbidden:
        assert c not in df.columns, f"{c!r} must NOT appear on roi_replicate_correlations"


def test_replicate_bh_family_independent_per_replicate():
    """Replicate BH families are keyed by replicate_index. Two replicates
    with different eligible ROI subsets produce different family sizes."""
    r1 = component_result(
        cid="C1", parent_roi_id="R1",
        region_diagnostics=[region_diag(0, 15, cid="C1")],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=5),
            contrib(1, n_eligible=1, n_contributing=1, n_tiles=5),
        ],
    )
    r2 = component_result(
        cid="C2", parent_roi_id="R2",
        region_diagnostics=[region_diag(0, 15, cid="C2")],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=5),
            contrib(1, n_eligible=1, n_contributing=0, n_tiles=0),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C1": r1, "C2": r2}, n_replicates_planned=2,
    )
    tiles = {
        "C1": {
            0: tile_df([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 2.5, 5.0, 5.5], [3]*5),
            1: tile_df([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], [3]*5),
        },
        "C2": {
            0: tile_df([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 5.0], [3]*5),
        },
    }
    support = support_frame([
        {"roi_id": "R1", "source_spots": 15, "spatial_support_class": "supported"},
        {"roi_id": "R2", "source_spots": 15, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support,
    )
    assert len(df) == 4
    t0 = df[df["replicate_index"] == 0]
    t1 = df[df["replicate_index"] == 1]
    assert (t0["fdr_family_size"] == 2).all()
    assert (t1["fdr_family_size"] == 1).all()
    r1_t1 = df[(df["roi_id"] == "R1") & (df["replicate_index"] == 1)].iloc[0]
    assert r1_t1[PARENT_REPLICATE_CORRELATION_STATUS] == "finite"
    assert not pd.isna(r1_t1["corr_p_analytic_conditional"])
    assert r1_t1["corr_q_analytic_conditional"] == pytest.approx(
        r1_t1["corr_p_analytic_conditional"],
    )


# ---------------------------------------------------------------------------
# Summary reduction + identity guards
# ---------------------------------------------------------------------------


def test_summary_correlation_quantiles_over_finite_only_coverage_over_all_planned():
    """Correlation quantiles reduce over FINITE rows only; tile /
    coverage medians reduce over ALL planned rows including
    zero-contribution ones."""
    replicate_rows = pd.DataFrame([
        {SAMPLE_ID: "S", ROI_ID: "R", REPLICATE_INDEX: 0,
         PARENT_REPLICATE_CORRELATION_STATUS: (
             ParentReplicateCorrelationStatus.FINITE.value),
         CORRELATION_VALUE: 0.3, N_TILES_CONTRIBUTED: 5,
         ELIGIBLE_SPOT_COVERAGE_FRACTION: 1.0,
         SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.8},
        {SAMPLE_ID: "S", ROI_ID: "R", REPLICATE_INDEX: 1,
         PARENT_REPLICATE_CORRELATION_STATUS: (
             ParentReplicateCorrelationStatus.NO_CONTRIBUTING_COMPONENTS.value),
         CORRELATION_VALUE: float("nan"), N_TILES_CONTRIBUTED: 0,
         ELIGIBLE_SPOT_COVERAGE_FRACTION: 0.0,
         SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.0},
        {SAMPLE_ID: "S", ROI_ID: "R", REPLICATE_INDEX: 2,
         PARENT_REPLICATE_CORRELATION_STATUS: (
             ParentReplicateCorrelationStatus.FINITE.value),
         CORRELATION_VALUE: 0.5, N_TILES_CONTRIBUTED: 7,
         ELIGIBLE_SPOT_COVERAGE_FRACTION: 1.0,
         SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.9},
    ])
    support = support_frame([
        {"roi_id": "R", "source_spots": 25, "spatial_support_class": "supported"},
    ])
    summary = compute_roi_tessellation_correlations(
        roi_replicate_correlations_df=replicate_rows,
        support_status_df=support, sample_id="S",
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row[PARENT_TESSELLATION_CORRELATION_STATUS] == (
        ParentTessellationCorrelationStatus.FINITE.value
    )
    assert row["n_valid_replicates"] == 2
    assert row["n_planned_replicates"] == 3
    assert row["corr_median"] == pytest.approx(0.4)
    assert row["n_tiles_contributed_median"] == pytest.approx(5.0)
    assert row["source_roi_spot_coverage_fraction_median"] == pytest.approx(0.8)


def test_summary_carries_no_inference_fields_on_correlation():
    """Tessellation-summary correlation output must not carry p / q /
    permutation-mean / permutation-sd / confidence fields. Reduced-null
    columns (corr_p_perm_reduced etc.) are allowed — they belong to
    the summary surface — but bare-name resurrections are forbidden."""
    replicate_rows = pd.DataFrame([{
        SAMPLE_ID: "S", ROI_ID: "R", REPLICATE_INDEX: 0,
        PARENT_REPLICATE_CORRELATION_STATUS: (
            ParentReplicateCorrelationStatus.FINITE.value),
        CORRELATION_VALUE: 0.4, N_TILES_CONTRIBUTED: 5,
        ELIGIBLE_SPOT_COVERAGE_FRACTION: 1.0,
        SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.9,
    }])
    support = support_frame([
        {"roi_id": "R", "source_spots": 25, "spatial_support_class": "supported"},
    ])
    summary = compute_roi_tessellation_correlations(
        roi_replicate_correlations_df=replicate_rows,
        support_status_df=support,
    )
    forbidden = {
        "corr_p", "corr_p_perm", "corr_perm_mean", "corr_perm_sd",
        "n_permutations", "corr_q",
        "joint_confidence", "joint_confidence_adj",
    }
    assert forbidden.isdisjoint(set(summary.columns))


def test_sample_id_disagreement_between_caller_and_typed_record_raises():
    """Caller-side sample_id disagreeing with the typed record must
    fail loud — one identity authority."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 5)],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=0),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S_typed", components={"C": comp}, n_replicates_planned=1,
    )
    support = support_frame([
        {"roi_id": "R", "source_spots": 5, "spatial_support_class": "supported"},
    ])
    with pytest.raises(ValueError, match="sample_id identity conflict"):
        build_roi_replicate_correlations_df(
            sample_tess_result=sample,
            tessellation_tiles_df=tiles_from_dict({}),
            support_status_df=support,
            sample_id="S_caller",
        )


def test_reducer_rejects_duplicate_replicate_index_and_unknown_roi():
    """Reducer inherits the total-table contract: duplicate replicate
    index within a ROI, or a ROI absent from support_status → raise."""
    dup_rows = pd.DataFrame([
        _summary_row(rep=0, status=ParentReplicateCorrelationStatus.FINITE.value, value=0.4),
        _summary_row(rep=0, status=ParentReplicateCorrelationStatus.FINITE.value, value=0.5),
    ])
    support = support_frame([
        {"roi_id": "R", "source_spots": 5, "spatial_support_class": "supported"},
    ])
    with pytest.raises(ValueError, match="replicate_index set"):
        compute_roi_tessellation_correlations(
            roi_replicate_correlations_df=dup_rows,
            support_status_df=support,
        )
    unknown_rows = pd.DataFrame([
        _summary_row(rep=0, status=ParentReplicateCorrelationStatus.FINITE.value, value=0.4),
    ])
    empty_support = support_frame([])
    with pytest.raises(ValueError, match="identity drift"):
        compute_roi_tessellation_correlations(
            roi_replicate_correlations_df=unknown_rows,
            support_status_df=empty_support,
        )


# ---------------------------------------------------------------------------
# Reduced-estimator permutation null (CORE-INF-01b) end-to-end
# ---------------------------------------------------------------------------


def test_summary_emits_reduced_null_p_q_seed_and_family_size_end_to_end():
    """Two ROIs × three replicates, B=19. FINITE rows carry
    corr_p_perm_reduced + corr_q_perm_reduced + perm_seed +
    n_valid_permutations. Every row carries n_permutations_reduced ==
    B; fdr_family_size counts ROIs with a finite p in the family.
    """
    n = 30
    g1_r1 = np.linspace(0.1, 1.0, n)
    g2_r1 = g1_r1 + 0.05 * np.arange(n) / n
    g1_r2 = np.linspace(0.2, 1.2, n)
    g2_r2 = np.flip(g1_r2)
    tile_slices = [
        [np.arange(0, 5), np.arange(5, 10), np.arange(10, 15),
         np.arange(15, 20), np.arange(20, 25), np.arange(25, 30)],
        [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)],
        [np.arange(0, 6), np.arange(6, 12), np.arange(12, 18),
         np.arange(18, 24), np.arange(24, 30)],
    ]
    r1_comp = real_component_result(
        cid="C1", parent_roi_id="R1",
        g1_values=g1_r1, g2_values=g2_r1,
        tile_membership_per_replicate=tile_slices,
        source_row_offset=0,
    )
    r2_comp = real_component_result(
        cid="C2", parent_roi_id="R2",
        g1_values=g1_r2, g2_values=g2_r2,
        tile_membership_per_replicate=tile_slices,
        source_row_offset=n,
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C1": r1_comp, "C2": r2_comp},
        n_replicates_planned=3,
    )
    thresholds = PercentileThresholds(
        q_g1_threshold=2.5, q_g2_threshold=2.5, q_percentile=0.75,
        q_g1_low_threshold=1.5, q_g2_low_threshold=1.5, q_low_percentile=0.25,
    )
    # Build per-tile marginals from real component values + tile membership.
    def _tile_marginals(vals, q_hi, q_lo, per_replicate):
        out = {}
        for cid, values in vals.items():
            for rep, tile_positions in enumerate(per_replicate):
                for tid, positions in enumerate(tile_positions):
                    v = values[np.asarray(positions, dtype=np.int64)]
                    out[(cid, 0, rep, tid)] = TileMarginalSums(
                        sum_g=float(v.sum()),
                        n_high=float((v > q_hi).sum()),
                        n_low=float((v < q_lo).sum()),
                    )
        return out

    def _stub_proj(
        feature_name, source_values, tile_marginals, resolved_hi, resolved_lo,
    ):
        return FeatureSpatialProjection(
            feature_name=feature_name,
            bank_fingerprint=sample.support_bank.fingerprint,
            source_values=np.asarray(source_values, dtype=float),
            thresholds=FeatureThresholds(
                q_high=0.75, q_low=0.25,
                resolved_high=resolved_hi, resolved_low=resolved_lo,
                n_source_values_used=2 * n,
            ),
            tile_marginal_sums=tile_marginals,
            hex_ensemble_projection={}, hex_direct_projection={},
        )

    tms_g1 = _tile_marginals(
        {"C1": g1_r1, "C2": g1_r2}, 2.5, 1.5, tile_slices,
    )
    tms_g2 = _tile_marginals(
        {"C1": g2_r1, "C2": g2_r2}, 2.5, 1.5, tile_slices,
    )
    proj_g1 = _stub_proj(
        "gene1", np.concatenate([g1_r1, g1_r2]), tms_g1, 2.5, 1.5,
    )
    proj_g2 = _stub_proj(
        "gene2", np.concatenate([g2_r1, g2_r2]), tms_g2, 2.5, 1.5,
    )
    tile_support = build_tessellation_tile_support(
        sample_tess_result=sample, sample_id="S",
    )
    tiles_df = build_tessellation_tiles_df(
        tile_support=tile_support, thresholds=thresholds,
        projection_g1=proj_g1, projection_g2=proj_g2,
    )
    support = support_frame([
        {"roi_id": "R1", "source_spots": n, "spatial_support_class": "supported"},
        {"roi_id": "R2", "source_spots": n, "spatial_support_class": "supported"},
    ])
    replicate_df = build_roi_replicate_correlations_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_df,
        support_status_df=support, sample_id="S",
    )
    B = 19
    with pytest.raises(ValueError, match="require explicit projection_g1"):
        compute_roi_tessellation_correlations(
            roi_replicate_correlations_df=replicate_df,
            support_status_df=support,
            sample_tess_result=sample,
            n_permutations=B, seed=17, sample_id="S",
        )
    summary = compute_roi_tessellation_correlations(
        roi_replicate_correlations_df=replicate_df,
        support_status_df=support,
        sample_tess_result=sample,
        projection_g1=proj_g1, projection_g2=proj_g2,
        n_permutations=B, seed=17,
        sample_id="S",
    )
    assert (summary["n_permutations_reduced"] == B).all()
    assert (summary["fdr_family_size"] == 2).all()
    for _, row in summary.iterrows():
        if row[PARENT_TESSELLATION_CORRELATION_STATUS] == "finite":
            assert not pd.isna(row["corr_p_perm_reduced"])
            assert not pd.isna(row["corr_q_perm_reduced"])
            assert not pd.isna(row["perm_seed"])
            assert row["n_valid_permutations"] == B
            assert row["corr_q_perm_reduced"] >= row["corr_p_perm_reduced"]
        else:
            assert pd.isna(row["corr_p_perm_reduced"])
            assert pd.isna(row["corr_q_perm_reduced"])
            assert pd.isna(row["perm_seed"])
            assert pd.isna(row["n_valid_permutations"])
