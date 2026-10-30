"""Adapter tests: PairStatisticResult -> CrossGrainSensitivityInputs.

Fixtures use canonical column constants from ``tilescope.cols`` so any
production rename fails these tests rather than silently changing what
the adapter reports.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.cross_grain_adapter import (
    CrossGrainPairFrames,
    build_cross_grain_inputs,
)
from tilescope.analyzers.cross_grain_types import (
    Channel,
    ComparisonStatus,
    Statistic,
    Surface,
    UnitKind,
)
from tilescope.analyzers.parent_roi_moran_common import MORAN_DIRECTIONS
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.analyzers.roi_cross_grain_sensitivity import (
    CANONICAL_COLUMNS,
    build_roi_cross_grain_sensitivity_df,
)
from tilescope.cols import (
    CORR_MEDIAN, CORR_OBS, CORR_Q25, CORR_Q75,
    CORRELATION_VALUE,
    GENE1_ENRICHMENT, GENE2_ENRICHMENT,
    GENE1_ENRICHMENT_STATUS, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
    JOINT_HIGH_ENRICHMENT_STATUS, JOINT_LOW_ENRICHMENT_STATUS,
    MORAN_DIRECTION, MORAN_VALUE,
    MORAN_VALUE_P25_ACROSS_REPLICATES, MORAN_VALUE_P75_ACROSS_REPLICATES,
    N_CONTEXT_HEXES_POOLED,
    N_CONTEXT_TILES_CONTRIBUTED,
    N_CTX,
    N_FINITE_REPLICATES,
    N_HEXES_POOLED, N_HEXES_WITH_FINITE_DENSITY_POOLED,
    N_PLANNED_REPLICATES,
    N_REPLICATES,
    N_ROI,
    N_TILES_CONTRIBUTED, N_TILES_CONTRIBUTED_MEDIAN,
    N_UNITS,
    PARENT_CONSENSUS_CORRELATION_STATUS,
    PARENT_CONSENSUS_ENRICHMENT_STATUS,
    PARENT_CORRELATION_STATUS,
    PARENT_ENRICHMENT_STATUS,
    PARENT_MORAN_STATUS,
    PARENT_REPLICATE_CORRELATION_STATUS,
    PARENT_REPLICATE_ENRICHMENT_STATUS,
    PARENT_TESSELLATION_CORRELATION_STATUS,
    REPLICATE_INDEX,
    ROI_ID,
    VALUE_SURFACE,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus

FINITE = EnrichmentRatioStatus.FINITE.value
SATURATION = EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value
ZERO_OVER_ZERO = EnrichmentRatioStatus.ZERO_OVER_ZERO_CONTEXT.value
CONSENSUS_ENSEMBLE = PartitionValueSurface.CONSENSUS_ENSEMBLE.value
CONSENSUS_DIRECT = PartitionValueSurface.CONSENSUS_DIRECT.value


# ---------------------------------------------------------------------
# Fixture builders (canonical column names only)
# ---------------------------------------------------------------------

def _corr_row(roi_id: str, value: float = 0.3, status: str = FINITE, n_spots: int = 100):
    """Source-native roi_correlations_df: raw r lives under CORR_OBS."""
    return {
        ROI_ID: roi_id, CORR_OBS: value,
        PARENT_CORRELATION_STATUS: status, N_ROI: n_spots,
    }


def _enr_row(
    roi_id: str,
    g1: float = 1.5, g2: float = 0.8, jh: float = 1.2, jl: float = 1.1,
    per_ratio_statuses: Optional[dict] = None,
    parent_status: str = FINITE,
    n_roi: int = 100, n_ctx: int = 400,
):
    st = {"g1": FINITE, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE}
    if per_ratio_statuses:
        st.update(per_ratio_statuses)
    return {
        ROI_ID: roi_id,
        GENE1_ENRICHMENT: g1, GENE2_ENRICHMENT: g2,
        JOINT_HIGH_ENRICHMENT_FC: jh, JOINT_LOW_ENRICHMENT_FC: jl,
        GENE1_ENRICHMENT_STATUS: st["g1"], GENE2_ENRICHMENT_STATUS: st["g2"],
        JOINT_HIGH_ENRICHMENT_STATUS: st["joint_high"],
        JOINT_LOW_ENRICHMENT_STATUS: st["joint_low"],
        PARENT_ENRICHMENT_STATUS: parent_status,
        N_ROI: n_roi, N_CTX: n_ctx,
    }


def _moran_rows(roi_id: str, value_by_dir=None, status: str = FINITE, n_units: int = 90):
    values = {MORAN_DIRECTIONS[0]: 0.2, MORAN_DIRECTIONS[1]: 0.15}
    if value_by_dir:
        values.update(value_by_dir)
    return [
        {
            ROI_ID: roi_id, MORAN_DIRECTION: direction,
            MORAN_VALUE: values[direction], PARENT_MORAN_STATUS: status,
            N_UNITS: n_units,
        }
        for direction in MORAN_DIRECTIONS
    ]


def _replicate_corr_rows(roi_id: str, values, statuses=None):
    if statuses is None:
        statuses = [FINITE] * len(values)
    return [
        {
            ROI_ID: roi_id, REPLICATE_INDEX: i,
            CORRELATION_VALUE: v,
            PARENT_REPLICATE_CORRELATION_STATUS: s,
        }
        for i, (v, s) in enumerate(zip(values, statuses))
    ]


def _tess_corr_summary_row(
    roi_id: str, median: float = 0.2, q25: float = 0.15, q75: float = 0.25,
    status: str = FINITE, n_planned: int = 5,
    n_tiles_median: float = 42.0,
):
    return {
        ROI_ID: roi_id, CORR_MEDIAN: median,
        CORR_Q25: q25, CORR_Q75: q75,
        PARENT_TESSELLATION_CORRELATION_STATUS: status,
        N_PLANNED_REPLICATES: n_planned,
        N_TILES_CONTRIBUTED_MEDIAN: n_tiles_median,
    }


def _replicate_enrichment_rows(
    roi_id: str,
    replicate_values,
    replicate_statuses=None,
    parent_statuses=None,
    n_tiles: float = 40.0, n_context_tiles: float = 200.0,
):
    if replicate_statuses is None:
        replicate_statuses = [
            {"g1": FINITE, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE}
        ] * len(replicate_values)
    if parent_statuses is None:
        parent_statuses = [FINITE] * len(replicate_values)
    rows = []
    for i, (raw_map, st_map, parent_st) in enumerate(
        zip(replicate_values, replicate_statuses, parent_statuses)
    ):
        rows.append({
            ROI_ID: roi_id, REPLICATE_INDEX: i,
            GENE1_ENRICHMENT: raw_map["g1"],
            GENE2_ENRICHMENT: raw_map["g2"],
            JOINT_HIGH_ENRICHMENT_FC: raw_map["joint_high"],
            JOINT_LOW_ENRICHMENT_FC: raw_map["joint_low"],
            GENE1_ENRICHMENT_STATUS: st_map["g1"],
            GENE2_ENRICHMENT_STATUS: st_map["g2"],
            JOINT_HIGH_ENRICHMENT_STATUS: st_map["joint_high"],
            JOINT_LOW_ENRICHMENT_STATUS: st_map["joint_low"],
            PARENT_REPLICATE_ENRICHMENT_STATUS: parent_st,
            N_TILES_CONTRIBUTED: n_tiles,
            N_CONTEXT_TILES_CONTRIBUTED: n_context_tiles,
        })
    return rows


def _tess_moran_rows(
    roi_id: str, median=0.15, p25=0.1, p75=0.2,
    status: str = FINITE, n_finite: int = 5, n_planned: int = 5,
    n_units: float = 40.0,
):
    return [
        {
            ROI_ID: roi_id, MORAN_DIRECTION: direction,
            MORAN_VALUE: median,
            MORAN_VALUE_P25_ACROSS_REPLICATES: p25,
            MORAN_VALUE_P75_ACROSS_REPLICATES: p75,
            PARENT_MORAN_STATUS: status,
            N_FINITE_REPLICATES: n_finite,
            N_REPLICATES: n_planned,
            N_UNITS: n_units,
        }
        for direction in MORAN_DIRECTIONS
    ]


def _consensus_corr_rows(
    roi_id: str, ensemble_value=0.25, direct_value=0.30,
    ensemble_status: str = FINITE, direct_status: str = FINITE,
    n_finite_hexes: float = 55.0,
):
    return [
        {
            ROI_ID: roi_id, VALUE_SURFACE: CONSENSUS_ENSEMBLE,
            CORRELATION_VALUE: ensemble_value,
            PARENT_CONSENSUS_CORRELATION_STATUS: ensemble_status,
            N_HEXES_WITH_FINITE_DENSITY_POOLED: n_finite_hexes,
        },
        {
            ROI_ID: roi_id, VALUE_SURFACE: CONSENSUS_DIRECT,
            CORRELATION_VALUE: direct_value,
            PARENT_CONSENSUS_CORRELATION_STATUS: direct_status,
            N_HEXES_WITH_FINITE_DENSITY_POOLED: n_finite_hexes,
        },
    ]


def _consensus_enr_rows(
    roi_id: str, g1=1.3, g2=0.9, jh=1.1, jl=1.05,
    parent_status: str = FINITE,
    per_ratio_statuses: Optional[dict] = None,
    n_hexes: float = 60.0, n_context_hexes: float = 200.0,
):
    st = {"g1": FINITE, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE}
    if per_ratio_statuses:
        st.update(per_ratio_statuses)
    rows = []
    for surface in (CONSENSUS_ENSEMBLE, CONSENSUS_DIRECT):
        rows.append({
            ROI_ID: roi_id, VALUE_SURFACE: surface,
            GENE1_ENRICHMENT: g1, GENE2_ENRICHMENT: g2,
            JOINT_HIGH_ENRICHMENT_FC: jh, JOINT_LOW_ENRICHMENT_FC: jl,
            GENE1_ENRICHMENT_STATUS: st["g1"], GENE2_ENRICHMENT_STATUS: st["g2"],
            JOINT_HIGH_ENRICHMENT_STATUS: st["joint_high"],
            JOINT_LOW_ENRICHMENT_STATUS: st["joint_low"],
            PARENT_CONSENSUS_ENRICHMENT_STATUS: parent_status,
            N_HEXES_POOLED: n_hexes,
            N_CONTEXT_HEXES_POOLED: n_context_hexes,
        })
    return rows


def _consensus_moran_rows(
    roi_id: str, value: float = 0.18, status: str = FINITE, n_units: float = 55.0,
):
    rows = []
    for surface in (CONSENSUS_ENSEMBLE, CONSENSUS_DIRECT):
        for direction in MORAN_DIRECTIONS:
            rows.append({
                ROI_ID: roi_id, VALUE_SURFACE: surface,
                MORAN_DIRECTION: direction, MORAN_VALUE: value,
                PARENT_MORAN_STATUS: status, N_UNITS: n_units,
            })
    return rows


def _pair_result(roi_ids=("R1",), **overrides):
    corr = pd.DataFrame([_corr_row(r) for r in roi_ids])
    enr = pd.DataFrame([_enr_row(r) for r in roi_ids])
    src_moran = pd.DataFrame(sum(([_moran_rows(r) for r in roi_ids]), []))
    rep_corr = pd.DataFrame(sum(
        [_replicate_corr_rows(r, [0.1, 0.2, 0.3, 0.4, 0.5]) for r in roi_ids], []
    ))
    tess_corr_summary = pd.DataFrame([_tess_corr_summary_row(r) for r in roi_ids])
    rep_enr = pd.DataFrame(sum(
        [
            _replicate_enrichment_rows(
                r,
                replicate_values=[
                    {"g1": 1.4, "g2": 0.9, "joint_high": 1.2, "joint_low": 1.0},
                    {"g1": 1.5, "g2": 1.0, "joint_high": 1.1, "joint_low": 1.1},
                    {"g1": 1.6, "g2": 0.8, "joint_high": 1.3, "joint_low": 0.9},
                ],
            )
            for r in roi_ids
        ], []
    ))
    tess_moran = pd.DataFrame(sum(([_tess_moran_rows(r) for r in roi_ids]), []))
    cons_corr = pd.DataFrame(sum(([_consensus_corr_rows(r) for r in roi_ids]), []))
    cons_enr = pd.DataFrame(sum(([_consensus_enr_rows(r) for r in roi_ids]), []))
    cons_moran = pd.DataFrame(sum(([_consensus_moran_rows(r) for r in roi_ids]), []))

    frames = dict(
        feature_g1_id="Neuron", feature_g2_id="AC",
        roi_correlations_df=corr,
        roi_enrichment_df=enr,
        roi_source_moran_df=src_moran,
        roi_replicate_correlations_df=rep_corr,
        roi_tessellation_correlations_df=tess_corr_summary,
        roi_replicate_enrichment_df=rep_enr,
        roi_tessellation_moran_df=tess_moran,
        roi_consensus_correlations_df=cons_corr,
        roi_consensus_enrichment_df=cons_enr,
        roi_consensus_moran_df=cons_moran,
    )
    frames.update(overrides)
    return CrossGrainPairFrames(**frames)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_golden_path_end_to_end_two_rois():
    pair = _pair_result(roi_ids=("R1", "R2"))
    inputs = build_cross_grain_inputs("S1", pair)
    df = build_roi_cross_grain_sensitivity_df(inputs)
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert len(df) == 14 * 2
    assert (df["sample_id"] == "S1").all()
    assert set(df["roi_id"]) == {"R1", "R2"}
    assert (df["g1_id"] == "Neuron").all()
    assert (df["g2_id"] == "AC").all()
    assert (df["comparison_status"] == ComparisonStatus.BOTH_COMPARABLE.value).all()


def test_structural_parent_status_gates_all_enrichment_channels():
    """When the parent status is structural (INSUFFICIENT_ROI_SPOTS)
    and the per-ratio columns are unpopulated, every channel takes the
    parent status as its unavailability reason."""
    pair = _pair_result()
    pair = replace(pair, roi_enrichment_df=pd.DataFrame([{
        ROI_ID: "R1",
        GENE1_ENRICHMENT: None, GENE2_ENRICHMENT: None,
        JOINT_HIGH_ENRICHMENT_FC: None, JOINT_LOW_ENRICHMENT_FC: None,
        GENE1_ENRICHMENT_STATUS: "", GENE2_ENRICHMENT_STATUS: "",
        JOINT_HIGH_ENRICHMENT_STATUS: "", JOINT_LOW_ENRICHMENT_STATUS: "",
        PARENT_ENRICHMENT_STATUS: "insufficient_roi_spots",
        N_ROI: 3, N_CTX: 400,
    }]))
    inputs = build_cross_grain_inputs("S1", pair)
    enrichments = [
        obs for obs in inputs.source_observations
        if obs.statistic == Statistic.ENRICHMENT
    ]
    assert len(enrichments) == 4
    for obs in enrichments:
        assert obs.raw_status == "insufficient_roi_spots"
        assert obs.raw_value is None


def test_per_ratio_status_wins_over_parent_when_populated():
    """context_proportion_zero at parent coexists with valid per-ratio
    statuses. Per-ratio precedence keeps individual channels typed
    correctly instead of blanket-marking all four unavailable."""
    pair = _pair_result()
    pair = replace(pair, roi_enrichment_df=pd.DataFrame([_enr_row(
        "R1",
        parent_status="context_proportion_zero",
        per_ratio_statuses={
            "g1": SATURATION, "g2": FINITE,
            "joint_high": FINITE, "joint_low": ZERO_OVER_ZERO,
        },
    )]))
    inputs = build_cross_grain_inputs("S1", pair)
    per_channel = {
        obs.channel: obs for obs in inputs.source_observations
        if obs.statistic == Statistic.ENRICHMENT
    }
    assert per_channel[Channel.G1].raw_status == SATURATION
    assert per_channel[Channel.G1].raw_value is None  # saturation has no raw
    assert per_channel[Channel.G2].raw_status == FINITE
    assert per_channel[Channel.G2].raw_value is not None
    assert per_channel[Channel.JOINT_HIGH].raw_status == FINITE
    assert per_channel[Channel.JOINT_LOW].raw_status == ZERO_OVER_ZERO


def test_tessellation_correlation_effect_from_finite_replicates():
    pair = _pair_result()
    pair = replace(pair, roi_replicate_correlations_df=pd.DataFrame(_replicate_corr_rows(
        "R1", values=[0.1, 0.3, 0.5, None],
        statuses=[FINITE, FINITE, FINITE, "insufficient_tiles"],
    )))
    inputs = build_cross_grain_inputs("S1", pair)
    tess = next(
        obs for obs in inputs.tessellation_observations
        if obs.statistic == Statistic.CORRELATION
    )
    expected_median = float(np.median([0.1, 0.3, 0.5]))
    expected_iqr = float(np.quantile([0.1, 0.3, 0.5], 0.75)
                         - np.quantile([0.1, 0.3, 0.5], 0.25))
    assert tess.effect_median == pytest.approx(expected_median)
    assert tess.effect_iqr == pytest.approx(expected_iqr)
    assert tess.n_comparable_replicates == 3


def test_tessellation_enrichment_includes_saturation_in_effect_population():
    """POSITIVE_OVER_ZERO_CONTEXT contributes effect +0.5 at
    tessellation grain. Effect_median, effect_iqr, and
    n_comparable_replicates describe the same population: finite +
    saturated."""
    pair = _pair_result()
    pair = replace(pair, roi_replicate_enrichment_df=pd.DataFrame(_replicate_enrichment_rows(
        "R1",
        replicate_values=[
            {"g1": 1.0, "g2": 1.0, "joint_high": 1.0, "joint_low": 1.0},  # E(g1)=0
            {"g1": 3.0, "g2": 1.0, "joint_high": 1.0, "joint_low": 1.0},  # E(g1)=0.25
            {"g1": None, "g2": 1.0, "joint_high": 1.0, "joint_low": 1.0},  # E(g1)=+0.5
        ],
        replicate_statuses=[
            {"g1": FINITE, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE},
            {"g1": FINITE, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE},
            {"g1": SATURATION, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE},
        ],
    )))
    inputs = build_cross_grain_inputs("S1", pair)
    g1_tess = next(
        obs for obs in inputs.tessellation_observations
        if obs.statistic == Statistic.ENRICHMENT and obs.channel == Channel.G1
    )
    # Saturation included: n_comparable == 3, effects = [0, 0.25, 0.5]
    assert g1_tess.n_comparable_replicates == 3
    expected_effects = [0.0, 0.25, 0.5]
    assert g1_tess.effect_median == pytest.approx(float(np.median(expected_effects)))
    expected_iqr = float(np.quantile(expected_effects, 0.75)
                         - np.quantile(expected_effects, 0.25))
    assert g1_tess.effect_iqr == pytest.approx(expected_iqr)
    # raw_summary is over finite raws only (saturation has no raw)
    assert g1_tess.raw_summary == pytest.approx(float(np.median([1.0, 3.0])))


def test_tessellation_enrichment_all_saturation_reports_saturation_status():
    """When every comparable replicate is POSITIVE_OVER_ZERO_CONTEXT and
    no finite raws exist, raw_summary is null and raw_status must be
    POSITIVE_OVER_ZERO_CONTEXT (not FINITE)."""
    pair = _pair_result()
    pair = replace(pair, roi_replicate_enrichment_df=pd.DataFrame(_replicate_enrichment_rows(
        "R1",
        replicate_values=[{"g1": None, "g2": 1.0, "joint_high": 1.0, "joint_low": 1.0}] * 3,
        replicate_statuses=[
            {"g1": SATURATION, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE}
        ] * 3,
    )))
    inputs = build_cross_grain_inputs("S1", pair)
    g1 = next(
        obs for obs in inputs.tessellation_observations
        if obs.statistic == Statistic.ENRICHMENT and obs.channel == Channel.G1
    )
    assert g1.n_comparable_replicates == 3
    assert g1.raw_summary is None  # no finite raws to median over
    assert g1.raw_status == SATURATION
    assert g1.effect_median == pytest.approx(0.5)


def test_no_comparable_replicates_status_from_ratio_statuses():
    """When no replicate is comparable AND all per-ratio statuses are a
    single non-comparable value, aggregate to that value. When they are
    mixed, fall back to no_comparable_replicates."""
    pair = _pair_result()
    # All three replicates report ZERO_OVER_ZERO for g1 -> unanimous
    pair = replace(pair, roi_replicate_enrichment_df=pd.DataFrame(_replicate_enrichment_rows(
        "R1",
        replicate_values=[{"g1": None, "g2": 1.0, "joint_high": 1.0, "joint_low": 1.0}] * 3,
        replicate_statuses=[
            {"g1": ZERO_OVER_ZERO, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE}
        ] * 3,
    )))
    inputs = build_cross_grain_inputs("S1", pair)
    g1 = next(
        obs for obs in inputs.tessellation_observations
        if obs.statistic == Statistic.ENRICHMENT and obs.channel == Channel.G1
    )
    assert g1.n_comparable_replicates == 0
    assert g1.raw_status == ZERO_OVER_ZERO

    # Mixed non-comparable per-ratio statuses -> no_comparable_replicates
    pair = replace(pair, roi_replicate_enrichment_df=pd.DataFrame(_replicate_enrichment_rows(
        "R1",
        replicate_values=[{"g1": None, "g2": 1.0, "joint_high": 1.0, "joint_low": 1.0}] * 2,
        replicate_statuses=[
            {"g1": ZERO_OVER_ZERO, "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE},
            {"g1": "some_other_status", "g2": FINITE, "joint_high": FINITE, "joint_low": FINITE},
        ],
    )))
    inputs = build_cross_grain_inputs("S1", pair)
    g1 = next(
        obs for obs in inputs.tessellation_observations
        if obs.statistic == Statistic.ENRICHMENT and obs.channel == Channel.G1
    )
    assert g1.raw_status == "no_comparable_replicates"


def test_tessellation_moran_uses_summary_quantiles():
    pair = _pair_result()
    pair = replace(pair, roi_tessellation_moran_df=pd.DataFrame(_tess_moran_rows(
        "R1", median=0.42, p25=0.30, p75=0.55, n_finite=7, n_planned=10,
    )))
    inputs = build_cross_grain_inputs("S1", pair)
    moran = [
        obs for obs in inputs.tessellation_observations
        if obs.statistic == Statistic.MORAN
    ]
    assert len(moran) == 2
    for obs in moran:
        assert obs.effect_median == pytest.approx(0.42)
        assert obs.effect_iqr == pytest.approx(0.55 - 0.30)
        assert obs.n_comparable_replicates == 7
        assert obs.n_planned_replicates == 10


def test_consensus_ensemble_filter_drops_direct_rows():
    pair = _pair_result()
    inputs = build_cross_grain_inputs("S1", pair)
    assert len(inputs.consensus_observations) == 7
    keys = {
        (obs.roi_id, obs.statistic, obs.channel, obs.moran_direction)
        for obs in inputs.consensus_observations
    }
    assert len(keys) == 7
    corr = next(
        obs for obs in inputs.consensus_observations
        if obs.statistic == Statistic.CORRELATION
    )
    assert corr.raw_value == pytest.approx(0.25)
    assert corr.target_unit_kind == UnitKind.HEX
