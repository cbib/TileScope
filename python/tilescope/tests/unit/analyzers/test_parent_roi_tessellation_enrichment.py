"""Contract tests for the parent-ROI tessellation enrichment surface.

Covers ``build_roi_replicate_enrichment_df`` and
``compute_roi_tessellation_enrichment`` — total-table shape, status
precedence (target then context), two independent typed statuses
(enrichment vs effect-size), summary reduction over independent
valid sets, and descriptive direction from summary medians.
"""
from __future__ import annotations

from tilescope.cols import GENE1_ENRICHMENT_STATUS as _G1S, GENE2_ENRICHMENT_STATUS as _G2S, JOINT_HIGH_ENRICHMENT_STATUS as _JHS, JOINT_LOW_ENRICHMENT_STATUS as _JLS


import pandas as pd
import pytest

from tilescope.analyzers.parent_roi_outcomes import (
    ParentReplicateEnrichmentStatus,
    ParentTessellationEffectSizeStatus,
    ParentTessellationEnrichmentStatus,
)
from tilescope.analyzers.parent_roi_tessellation_enrichment import (
    build_roi_replicate_enrichment_df as _build_replicate_enrichment,
    compute_roi_tessellation_enrichment as _compute_tessellation_enrichment,
)
from tilescope.analyzers.parent_roi_tessellation_common import (
    build_tessellation_pooling_index,
)
from tilescope.analyzers.tessellation_analysis import SampleTessellationResult
from tilescope.cols import (
    CLES_G1, CLES_G2,
    CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION,
    CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION,
    ELIGIBLE_SPOT_COVERAGE_FRACTION,
    GENE1_ENRICHMENT, GENE2_ENRICHMENT,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
    LOG2FC_G1, LOG2FC_G2,
    N_BOTH_HIGH,
    N_CONTEXT_CONTRIBUTED_SPOTS, N_CONTEXT_CONTRIBUTING_COMPONENTS,
    N_CONTEXT_ELIGIBLE_COMPONENTS, N_CONTEXT_ELIGIBLE_SPOTS,
    N_CONTEXT_TILES_CONTRIBUTED,
    N_CONTRIBUTED_SPOTS, N_G1_HIGH, N_SPOTS, N_TILES_CONTRIBUTED,
    PARENT_REPLICATE_EFFECT_SIZE_STATUS,
    PARENT_REPLICATE_ENRICHMENT_STATUS,
    PARENT_TESSELLATION_EFFECT_SIZE_STATUS,
    PARENT_TESSELLATION_ENRICHMENT_STATUS,
    REPLICATE_INDEX, ROI_ID, SAMPLE_ID,
    SOURCE_ROI_SPOT_COVERAGE_FRACTION,
)
from tilescope.metrics.enrichment_thresholds import PercentileThresholds
from tilescope.metrics.normalization import ScaleNormalizationParams
from tilescope.tests.unit.analyzers.parent_roi_tessellation_fixtures import (
    component_result,
    contrib,
    region_diag,
    support_frame,
    tile_df,
    tiles_from_dict,
)


def build_roi_replicate_enrichment_df(
    *, sample_tess_result, tessellation_tiles_df, **kwargs,
):
    index = build_tessellation_pooling_index(
        sample_tess_result=sample_tess_result,
        tessellation_tiles_df=tessellation_tiles_df,
        support_status_df=kwargs["support_status_df"],
    )
    return _build_replicate_enrichment(
        pooling_index=index, g1_id="Neuron", g2_id="AC", **kwargs,
    )


def compute_roi_tessellation_enrichment(**kwargs):
    return _compute_tessellation_enrichment(
        g1_id="Neuron", g2_id="AC", **kwargs,
    )


def _thresholds(q_high: float = 0.75, q_low: float = 0.25):
    return PercentileThresholds(
        q_g1_threshold=2.5, q_g2_threshold=2.5,
        q_percentile=q_high,
        q_g1_low_threshold=1.5, q_g2_low_threshold=1.5,
        q_low_percentile=q_low,
    )


def _norm_params():
    return ScaleNormalizationParams(
        gene1_min_spot=0.0, gene2_min_spot=0.0,
        gene1_min_density=0.0, gene2_min_density=0.0,
    )


def _enrichment_kwargs(**overrides):
    kw = {"thresholds": _thresholds(), "normalization_params": _norm_params()}
    kw.update(overrides)
    return kw


def _two_roi_finite_scenario():
    """Two ROIs, one component each, both eligible + contributing at
    replicate 0. Each side carries one joint-high and one joint-low
    tile plus filler under the (1.5, 2.5) thresholds."""
    comp_a = component_result(
        cid="Ca", parent_roi_id="A",
        region_diagnostics=[region_diag(0, 8, cid="Ca")],
        contributed_regions=[contrib(0, n_eligible=1, n_contributing=1, n_tiles=4)],
    )
    comp_b = component_result(
        cid="Cb", parent_roi_id="B",
        region_diagnostics=[region_diag(0, 8, cid="Cb")],
        contributed_regions=[contrib(0, n_eligible=1, n_contributing=1, n_tiles=4)],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"Ca": comp_a, "Cb": comp_b},
        n_replicates_planned=1,
    )
    tiles = {
        "Ca": {0: tile_df(
            [3.0, 4.0, 1.0, 0.5], [3.0, 4.0, 1.0, 0.5], [2, 2, 2, 2],
        )},
        "Cb": {0: tile_df(
            [4.0, 3.5, 1.0, 0.5], [3.5, 4.0, 0.5, 1.0], [2, 2, 2, 2],
        )},
    }
    support = support_frame([
        {"roi_id": "A", "source_spots": 8, "spatial_support_class": "supported"},
        {"roi_id": "B", "source_spots": 8, "spatial_support_class": "supported"},
    ])
    return sample, tiles, support


_AGGREGATE_STATUSES_WITH_PER_RATIO = {
    "finite", "context_proportion_zero",
}


def _replicate_row(*, rep, enrichment_status, effect_status, roi="R", **fields):
    """Minimal per-replicate row shape the summary reducer reads."""
    if enrichment_status == "finite":
        per_ratio_default = "finite"
    elif enrichment_status in _AGGREGATE_STATUSES_WITH_PER_RATIO:
        per_ratio_default = "positive_over_zero_context"
    else:
        per_ratio_default = pd.NA
    row = {
        SAMPLE_ID: "S", ROI_ID: roi, REPLICATE_INDEX: rep,
        PARENT_REPLICATE_ENRICHMENT_STATUS: enrichment_status,
        PARENT_REPLICATE_EFFECT_SIZE_STATUS: effect_status,
        GENE1_ENRICHMENT: float("nan"),
        GENE2_ENRICHMENT: float("nan"),
        JOINT_HIGH_ENRICHMENT_FC: float("nan"),
        JOINT_LOW_ENRICHMENT_FC: float("nan"),
        _G1S: per_ratio_default,
        _G2S: per_ratio_default,
        _JHS: per_ratio_default,
        _JLS: per_ratio_default,
        LOG2FC_G1: float("nan"),
        LOG2FC_G2: float("nan"),
        CLES_G1: float("nan"),
        CLES_G2: float("nan"),
        N_TILES_CONTRIBUTED: 5,
        N_CONTEXT_TILES_CONTRIBUTED: 5,
        ELIGIBLE_SPOT_COVERAGE_FRACTION: 1.0,
        SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.8,
        CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION: 1.0,
        CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION: 0.9,
    }
    row.update(fields)
    return row


# ---------------------------------------------------------------------------
# ROI × replicate enrichment builder — status precedence + identity
# ---------------------------------------------------------------------------


def test_enrichment_total_table_two_rois_one_replicate_each():
    """Two ROIs, one replicate each: both rows FINITE with populated
    numeric fields. Classification / mode columns are forbidden on
    this surface."""
    sample, tiles, support = _two_roi_finite_scenario()
    df = build_roi_replicate_enrichment_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support, **_enrichment_kwargs(),
    )
    assert len(df) == 2
    assert set(df[ROI_ID]) == {"A", "B"}
    assert set(df[REPLICATE_INDEX]) == {0}
    assert (
        df[PARENT_REPLICATE_ENRICHMENT_STATUS]
        == ParentReplicateEnrichmentStatus.FINITE.value
    ).all()
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert int(row_a[N_CONTEXT_ELIGIBLE_COMPONENTS]) == 1
    assert int(row_a[N_CONTEXT_CONTRIBUTING_COMPONENTS]) == 1
    assert int(row_a[N_CONTEXT_ELIGIBLE_SPOTS]) == 8
    assert int(row_a[N_CONTEXT_CONTRIBUTED_SPOTS]) == 8
    assert row_a[CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION] == pytest.approx(1.0)
    for column in (
        GENE1_ENRICHMENT, GENE2_ENRICHMENT,
        JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
        LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2,
    ):
        assert pd.notna(row_a[column]), f"FINITE row missing {column}"
    for column in (
        "joint_confidence", "joint_confidence_adj",
        "enrichment_direction", "mode",
    ):
        assert column not in df.columns, (
            f"{column!r} must NOT appear on roi_replicate_enrichment_df"
        )
    assert row_a[N_SPOTS] == 4
    assert row_a[N_G1_HIGH] >= 0
    assert row_a[N_BOTH_HIGH] >= 0


def test_enrichment_scattered_parent_gets_no_tessellation_eligible_component():
    """Scattered / no-eligible ROIs emit rows with typed status +
    NA numeric fields — no silent absence."""
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
    df = build_roi_replicate_enrichment_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict({}),
        support_status_df=support, **_enrichment_kwargs(),
    )
    assert len(df) == 2
    assert (
        df[PARENT_REPLICATE_ENRICHMENT_STATUS]
        == ParentReplicateEnrichmentStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    ).all()
    assert df[GENE1_ENRICHMENT].isna().all()
    assert df[N_SPOTS].isna().all()


def test_enrichment_single_roi_sample_marks_no_eligible_context():
    """Single-ROI sample: target eligible + contributing, but no OTHER
    ROI to draw context from → NO_ELIGIBLE_CONTEXT."""
    comp = component_result(
        region_diagnostics=[region_diag(0, 8)],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=4),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"C": comp}, n_replicates_planned=1,
    )
    tiles = {"C": {0: tile_df(
        [3.0, 4.0, 3.5, 4.5], [1.0, 0.5, 1.2, 0.8], [2, 2, 2, 2],
    )}}
    support = support_frame([
        {"roi_id": "R", "source_spots": 8, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_enrichment_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support, **_enrichment_kwargs(),
    )
    assert len(df) == 1
    assert df.iloc[0][PARENT_REPLICATE_ENRICHMENT_STATUS] == (
        ParentReplicateEnrichmentStatus.NO_ELIGIBLE_CONTEXT.value
    )
    assert int(df.iloc[0][N_CONTRIBUTED_SPOTS]) == 8


def test_enrichment_context_shortfall_marks_incomplete_context_coverage():
    """Target contributes fully but context ROI's component contributes
    only partially at this replicate → INCOMPLETE_CONTEXT_COVERAGE."""
    comp_a = component_result(
        cid="Ca", parent_roi_id="A",
        region_diagnostics=[region_diag(0, 8, cid="Ca")],
        contributed_regions=[
            contrib(0, n_eligible=1, n_contributing=1, n_tiles=4),
        ],
    )
    comp_b = component_result(
        cid="Cb", parent_roi_id="B",
        region_diagnostics=[
            region_diag(0, 8, cid="Cb"),
            region_diag(1, 8, cid="Cb"),
        ],
        contributed_regions=[
            contrib(0, n_eligible=2, n_contributing=1, n_tiles=4),
        ],
    )
    sample = SampleTessellationResult.from_component_results(
        sample_id="S", components={"Ca": comp_a, "Cb": comp_b},
        n_replicates_planned=1,
    )
    tiles = {
        "Ca": {0: tile_df(
            [3.0, 4.0, 3.5, 4.5], [1.0, 0.5, 1.2, 0.8], [2, 2, 2, 2],
        )},
        "Cb": {0: tile_df(
            [1.0, 0.5, 1.2, 0.8], [3.0, 4.0, 3.5, 4.5], [2, 2, 2, 2],
        )},
    }
    support = support_frame([
        {"roi_id": "A", "source_spots": 8, "spatial_support_class": "supported"},
        {"roi_id": "B", "source_spots": 16, "spatial_support_class": "supported"},
    ])
    df = build_roi_replicate_enrichment_df(
        sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict(tiles),
        support_status_df=support, **_enrichment_kwargs(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert row_a[PARENT_REPLICATE_ENRICHMENT_STATUS] == (
        ParentReplicateEnrichmentStatus.INCOMPLETE_CONTEXT_COVERAGE.value
    )
    assert pd.isna(row_a[GENE1_ENRICHMENT])


def test_enrichment_sample_id_disagreement_raises():
    """Caller-side sample_id disagreeing with the typed record → raise."""
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
        build_roi_replicate_enrichment_df(
            sample_tess_result=sample, tessellation_tiles_df=tiles_from_dict({}),
            support_status_df=support, sample_id="S_caller",
            **_enrichment_kwargs(),
        )


# ---------------------------------------------------------------------------
# Summary reducer — independent valid sets, no-eligible rollup, identity
# ---------------------------------------------------------------------------


def test_summary_enrichment_and_effect_size_reduce_over_independent_valid_sets():
    """Enrichment ratios reduce over enrichment-FINITE rows; log2FC /
    CLES reduce independently over effect-size-FINITE rows. The two
    valid sets need NOT coincide — effect sizes survive
    ``context_proportion_zero``."""
    rows = [
        _replicate_row(
            rep=0, enrichment_status="finite", effect_status="finite",
            **{
                GENE1_ENRICHMENT: 1.0, GENE2_ENRICHMENT: 1.4,
                JOINT_HIGH_ENRICHMENT_FC: 1.2, JOINT_LOW_ENRICHMENT_FC: 0.8,
                LOG2FC_G1: 0.5, LOG2FC_G2: -0.2, CLES_G1: 0.6, CLES_G2: 0.45,
            },
        ),
        _replicate_row(
            rep=1, enrichment_status="context_proportion_zero",
            effect_status="finite",
            **{LOG2FC_G1: 0.9, LOG2FC_G2: -0.4, CLES_G1: 0.7, CLES_G2: 0.4},
        ),
        _replicate_row(
            rep=2, enrichment_status="incomplete_support_coverage",
            effect_status="not_computed",
        ),
    ]
    replicate_df = pd.DataFrame(rows)
    support = support_frame([
        {"roi_id": "R", "source_spots": 25, "spatial_support_class": "supported"},
    ])
    summary = compute_roi_tessellation_enrichment(
        roi_replicate_enrichment_df=replicate_df,
        support_status_df=support,
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row[PARENT_TESSELLATION_ENRICHMENT_STATUS] == (
        ParentTessellationEnrichmentStatus.FINITE.value
    )
    assert row[PARENT_TESSELLATION_EFFECT_SIZE_STATUS] == (
        ParentTessellationEffectSizeStatus.FINITE.value
    )
    assert row["n_valid_g1_enrichment_replicates"] == 1
    assert row["n_valid_g2_enrichment_replicates"] == 1
    assert row["n_valid_joint_high_enrichment_replicates"] == 1
    assert row["n_valid_joint_low_enrichment_replicates"] == 1
    assert row["n_valid_effect_size_replicates"] == 2
    assert row["n_planned_replicates"] == 3
    assert row["g1_enrichment_median"] == pytest.approx(1.0)
    assert row["log2fc_g1_median"] == pytest.approx(0.7)
    assert row["enrichment_direction"] is not None
    assert "g1:" in row["enrichment_direction"]
    forbidden = {
        "joint_confidence", "joint_confidence_adj", "corr_p", "corr_q",
        "corr_p_perm", "corr_q_perm",
    }
    assert forbidden.isdisjoint(set(summary.columns))


def test_summary_partial_ratio_availability_when_only_some_ratios_have_finite_replicates():
    """Asymmetric per-ratio availability across replicates: g1 has one
    finite replicate; g2, joint-high, joint-low have none. The
    aggregate summary status must be PARTIAL_RATIO_AVAILABILITY (not
    FINITE, not NO_VALID_REPLICATES). Per-ratio counts + medians
    reflect each ratio independently.
    """
    rows = [
        _replicate_row(
            rep=0, enrichment_status="context_proportion_zero",
            effect_status="finite",
            **{
                GENE1_ENRICHMENT: 1.7,
                _G1S: "finite",
                _G2S: "positive_over_zero_context",
                _JHS: "zero_over_zero_context",
                _JLS: "positive_over_zero_context",
                LOG2FC_G1: 0.3, LOG2FC_G2: -0.1,
                CLES_G1: 0.55, CLES_G2: 0.48,
            },
        ),
        _replicate_row(
            rep=1, enrichment_status="context_proportion_zero",
            effect_status="finite",
            **{
                GENE1_ENRICHMENT: 2.1,
                _G1S: "finite",
                _G2S: "positive_over_zero_context",
                _JHS: "zero_over_zero_context",
                _JLS: "positive_over_zero_context",
                LOG2FC_G1: 0.4, LOG2FC_G2: -0.2,
                CLES_G1: 0.60, CLES_G2: 0.45,
            },
        ),
    ]
    replicate_df = pd.DataFrame(rows)
    support = support_frame([
        {"roi_id": "R", "source_spots": 25, "spatial_support_class": "supported"},
    ])
    summary = compute_roi_tessellation_enrichment(
        roi_replicate_enrichment_df=replicate_df,
        support_status_df=support,
    )
    row = summary.iloc[0]
    assert row[PARENT_TESSELLATION_ENRICHMENT_STATUS] == (
        ParentTessellationEnrichmentStatus.PARTIAL_RATIO_AVAILABILITY.value
    )
    assert row["n_valid_g1_enrichment_replicates"] == 2
    assert row["n_valid_g2_enrichment_replicates"] == 0
    assert row["n_valid_joint_high_enrichment_replicates"] == 0
    assert row["n_valid_joint_low_enrichment_replicates"] == 0
    assert row["g1_enrichment_median"] == pytest.approx(1.9)
    for col in (
        "g2_enrichment_median",
        "joint_high_enrichment_fc_median",
        "joint_low_enrichment_fc_median",
    ):
        assert pd.isna(row[col])
    # Direction only populated when both marginals have finite median.
    assert row["enrichment_direction"] is None


def test_summary_no_eligible_component_rolls_up_to_typed_no_component_status():
    """A parent with every replicate on NO_TESSELLATION_ELIGIBLE_COMPONENT
    rolls up to the same typed state on both status columns — never
    NO_VALID_REPLICATES."""
    rows = [
        _replicate_row(
            rep=0,
            enrichment_status="no_tessellation_eligible_component",
            effect_status="not_computed",
        ),
        _replicate_row(
            rep=1,
            enrichment_status="no_tessellation_eligible_component",
            effect_status="not_computed",
        ),
    ]
    replicate_df = pd.DataFrame(rows)
    support = support_frame([
        {"roi_id": "R", "source_spots": 3, "spatial_support_class": "scattered"},
    ])
    summary = compute_roi_tessellation_enrichment(
        roi_replicate_enrichment_df=replicate_df,
        support_status_df=support,
    )
    row = summary.iloc[0]
    assert row[PARENT_TESSELLATION_ENRICHMENT_STATUS] == (
        ParentTessellationEnrichmentStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    )
    assert row[PARENT_TESSELLATION_EFFECT_SIZE_STATUS] == (
        ParentTessellationEffectSizeStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    )
    assert row["n_valid_g1_enrichment_replicates"] == 0
    assert row["n_valid_g2_enrichment_replicates"] == 0
    assert row["n_valid_joint_high_enrichment_replicates"] == 0
    assert row["n_valid_joint_low_enrichment_replicates"] == 0
    assert row["n_valid_effect_size_replicates"] == 0
    for col in ("g1_enrichment_median", "log2fc_g1_median", "CLES_g1_median"):
        assert pd.isna(row[col])
    assert row["enrichment_direction"] is None


def test_summary_rejects_duplicate_replicate_and_unknown_roi():
    """Duplicate replicate index within a parent → raise; ROIs absent
    from support_status → raise."""
    dup_rows = pd.DataFrame([
        _replicate_row(rep=0, enrichment_status="finite", effect_status="finite"),
        _replicate_row(rep=0, enrichment_status="finite", effect_status="finite"),
    ])
    support = support_frame([
        {"roi_id": "R", "source_spots": 5, "spatial_support_class": "supported"},
    ])
    with pytest.raises(ValueError, match="replicate_index set"):
        compute_roi_tessellation_enrichment(
            roi_replicate_enrichment_df=dup_rows,
            support_status_df=support,
        )
    unknown_rows = pd.DataFrame([
        _replicate_row(
            rep=0, enrichment_status="finite", effect_status="finite",
            roi="R",
        ),
    ])
    empty_support = support_frame([])
    with pytest.raises(ValueError, match="identity drift"):
        compute_roi_tessellation_enrichment(
            roi_replicate_enrichment_df=unknown_rows,
            support_status_df=empty_support,
        )
