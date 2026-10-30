"""Contract tests for the ROI-grain consensus correlation + enrichment surfaces."""
from __future__ import annotations

from tilescope.tests.helpers.fixture_projections import register_hex_features
from dataclasses import replace as _replace
from tilescope.tests.helpers.fixture_projections import _HEX_FEATURES
from tilescope.analyzers.sample_support_stage import sample_support_bank_from_component_results
from tilescope.tests.helpers.fixture_projections import projection_from_hex_features
from tilescope.stats.fdr import benjamini_hochberg


from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.analyzers.consensus_stage import (
    ComponentConsensusResult, ConsensusHex,
    SampleConsensusResult,
)
from tilescope.analyzers.parent_roi_consensus_correlation import (
    build_roi_consensus_correlations_df as _build_correlations,
)
from tilescope.analyzers.parent_roi_consensus_enrichment import (
    build_roi_consensus_enrichment_df as _build_enrichment,
)
from tilescope.analyzers.consensus_frames import build_consensus_analysis_frames
from tilescope.analyzers.consensus_hex_surface import (
    build_consensus_hexes_df as _build_consensus_hexes,
)
from tilescope.analyzers.parent_roi_outcomes import (
    ParentConsensusCorrelationStatus,
    ParentConsensusEffectSizeStatus,
    ParentConsensusEnrichmentStatus,
)
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.metrics.enrichment import EnrichmentRatioStatus
from tilescope.cols import (
    CLES_G1,
    CLES_G2,
    LOG2FC_G1,
    LOG2FC_G2,
    CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS,
    CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_HIGH_MASS,
    CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
    CONTEXT_OBS_EQUIV_G1_LOW_MASS,
    CONTEXT_OBS_EQUIV_G2_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_SUM_MASS,
    CONTEXT_OBS_EQUIV_G2_SUM_MASS,
    CONTEXT_OBS_EQUIV_SPOT_MASS,
    GENE1_ENRICHMENT,
    GENE1_ENRICHMENT_STATUS,
    GENE2_ENRICHMENT,
    GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC,
    JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC,
    JOINT_LOW_ENRICHMENT_STATUS,
    OBS_EQUIV_BOTH_HIGH_MASS,
    OBS_EQUIV_BOTH_LOW_MASS,
    OBS_EQUIV_G1_HIGH_MASS,
    OBS_EQUIV_G2_HIGH_MASS,
    OBS_EQUIV_G1_LOW_MASS,
    OBS_EQUIV_G2_LOW_MASS,
    OBS_EQUIV_G1_SUM_MASS,
    OBS_EQUIV_G2_SUM_MASS,
    OBS_EQUIV_SPOT_MASS,
    PARENT_CONSENSUS_CORRELATION_STATUS,
    PARENT_CONSENSUS_EFFECT_SIZE_STATUS,
    PARENT_CONSENSUS_ENRICHMENT_STATUS,
    ROI_ID,
    VALUE_SURFACE,
)
from tilescope.metrics.normalization import ScaleNormalizationParams


def _hex(
    hid: int, cx: float, cy: float, g1_sum: float, g2_sum: float,
    n_spots: float = 4.0, n_cov: int = 2, region_id: int = 0,
    high_g1: float = 0.0, high_g2: float = 0.0, both_high: float = 0.0,
    low_g1: float = 0.0, low_g2: float = 0.0, both_low: float = 0.0,
    direct_g1_sum: float = None, direct_g2_sum: float = None,
    n_direct: int = None, component_id: str = "C",
) -> ConsensusHex:
    if direct_g1_sum is None:
        direct_g1_sum = g1_sum
    if direct_g2_sum is None:
        direct_g2_sum = g2_sum
    if n_direct is None:
        n_direct = int(n_spots)
    hx = ConsensusHex.build(
        component_id=component_id,
        hex_id=hid, domain_region_id=region_id,
        polygon=box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5),
        lattice_col=hid, lattice_row=0,
        center_x=cx, center_y=cy,
        n_tessellations_covered=n_cov,
        n_spots_ensemble_weighted=n_spots,
        n_spots_direct=n_direct,
        n_both_high_ensemble_weighted=both_high,
        n_both_low_ensemble_weighted=both_low,
    )
    register_hex_features(
        component_id=component_id, hex_id=hid,
        sum_g1_ens=g1_sum, sum_g2_ens=g2_sum,
        g1_high=high_g1, g2_high=high_g2,
        g1_low=low_g1, g2_low=low_g2,
        sum_g1_direct=direct_g1_sum, sum_g2_direct=direct_g2_sum,
    )
    return hx


def _component(
    cid: str, parent_roi_id: str, hexes: Sequence[ConsensusHex],
    *, n_eligible_regions: int = 1,
) -> ComponentConsensusResult:
    rebound = []
    for h in hexes:
        orig_cid = h.geometry.component_id
        rebound_hex = _replace(h, geometry=_replace(h.geometry, component_id=cid))
        rebound.append(rebound_hex)
        data = _HEX_FEATURES.pop((str(orig_cid), int(h.hex_id)), None)
        if data is not None:
            register_hex_features(
                component_id=cid, hex_id=int(h.hex_id),
                sum_g1_ens=data["sum_g1_ens"], sum_g2_ens=data["sum_g2_ens"],
                g1_high=data["g1_high"], g2_high=data["g2_high"],
                g1_low=data["g1_low"], g2_low=data["g2_low"],
                sum_g1_direct=data["sum_g1_direct"],
                sum_g2_direct=data["sum_g2_direct"],
            )
    return ComponentConsensusResult.from_hexes(
        parent_roi_id=parent_roi_id, component_id=cid,
        hex_radius=1.0, hexes=tuple(rebound),
        n_tessellation_eligible_regions=n_eligible_regions,
    )


def _sample(
    components: Sequence[ComponentConsensusResult], sample_id: str = "S",
) -> SampleConsensusResult:
    support_bank = sample_support_bank_from_component_results(
        sample_id=sample_id, component_results={},
        n_replicates=2, base_seed=0,
    )
    return SampleConsensusResult.from_components(
        sample_id=sample_id,
        components={c.component_id: c for c in components},
        n_replicates_planned=2,
        sample_anchor_x=0.0, sample_anchor_y=0.0,
        support_bank=support_bank,
    )


def _support(rows):
    return pd.DataFrame(
        rows, columns=["roi_id", "source_spots", "spatial_support_class"],
    )


def _norm():
    return ScaleNormalizationParams(
        gene1_min_spot=0.1, gene2_min_spot=0.1,
        gene1_min_density=0.1, gene2_min_density=0.1,
    )


def _frames(sample, support=None):
    roi_ids = [] if support is None else support[ROI_ID].astype(str).tolist()
    p_g1, p_g2 = projection_from_hex_features(sample_consensus_result=sample)
    return build_consensus_analysis_frames(
        sample_consensus_result=sample,
        roi_ids=roi_ids,
        feature_projection_g1=p_g1,
        feature_projection_g2=p_g2,
    )


def build_roi_consensus_correlations_df(
    *, sample_consensus_result, component_to_roi=None, **kwargs,
):
    kwargs.setdefault("g1_id", "Neuron")
    kwargs.setdefault("g2_id", "AC")
    return _build_correlations(
        consensus_frames=_frames(
            sample_consensus_result, kwargs.get("support_status_df"),
        ), **kwargs,
    )


def build_roi_consensus_enrichment_df(
    *, sample_consensus_result, component_to_roi=None, **kwargs,
):
    kwargs.setdefault("g1_id", "Neuron")
    kwargs.setdefault("g2_id", "AC")
    return _build_enrichment(
        consensus_frames=_frames(
            sample_consensus_result, kwargs.get("support_status_df"),
        ), **kwargs,
    )


def build_consensus_hexes_df(
    *, sample_consensus_result, **kwargs,
):
    return _build_consensus_hexes(
        consensus_frames=_frames(sample_consensus_result), **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Pooled-sibling correlation differs from mean-of-component correlations
# ---------------------------------------------------------------------------


def test_pooled_correlation_differs_from_mean_of_component_correlations():
    """Two sibling components with locally-negative correlations pool
    into a positively-correlated ROI-level vector — the estimand is
    the pooled Pearson, not the mean of per-component Pearsons."""
    # Component A: (g1, g2) = (1,4), (2,3) — locally r = -1
    # Component B: (g1, g2) = (5,8), (6,7) — locally r = -1
    # Pooled: [(1,4),(2,3),(5,8),(6,7)] — cross-shift creates r > 0
    a = _component("Ca", "R", [
        _hex(0, 0.0, 0.0, 1.0, 4.0),
        _hex(1, 1.0, 0.0, 2.0, 3.0),
    ])
    b = _component("Cb", "R", [
        _hex(0, 5.0, 0.0, 5.0, 8.0),
        _hex(1, 6.0, 0.0, 6.0, 7.0),
    ])
    df = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "R", "source_spots": 10,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "R", "Cb": "R"},
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    )
    row = df.iloc[0]
    assert row[PARENT_CONSENSUS_CORRELATION_STATUS] == (
        ParentConsensusCorrelationStatus.FINITE.value
    )
    # Pooled r is strongly positive (~+0.88) — never negative like
    # either component's local Pearson (which is -1 on each side).
    assert row["correlation_value"] > 0.5


# ---------------------------------------------------------------------------
# 2. Ensemble AND direct correlations both emitted
# ---------------------------------------------------------------------------


def test_ensemble_and_direct_correlations_both_emitted():
    a = _component("Ca", "R", [
        _hex(0, 0.0, 0.0, 1.0, 1.0, n_spots=2.0),
        _hex(1, 1.0, 0.0, 2.0, 2.0, n_spots=2.0),
        _hex(2, 2.0, 0.0, 3.0, 3.0, n_spots=2.0),
    ])
    df = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a]),
        support_status_df=_support([
            {"roi_id": "R", "source_spots": 10,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "R"},
    )
    surfaces = set(df[VALUE_SURFACE])
    assert surfaces == {"consensus_ensemble", "consensus_direct"}


# ---------------------------------------------------------------------------
# 3. Zero / incomplete sibling support remains visible
# ---------------------------------------------------------------------------


def test_incomplete_sibling_support_marks_incomplete_support_coverage():
    """One sibling contributes finite-density hexes; the other has
    hexes but all non-finite on the ensemble surface. Result must be
    INCOMPLETE_SUPPORT_COVERAGE — NOT silently computed from the
    contributing sibling alone."""
    a = _component("Ca", "R", [
        _hex(0, 0.0, 0.0, 1.0, 2.0, n_spots=2.0),
        _hex(1, 1.0, 0.0, 2.0, 4.0, n_spots=2.0),
        _hex(2, 2.0, 0.0, 3.0, 6.0, n_spots=2.0),
    ])
    # Component B has hexes but zero exposure → density NaN on this
    # surface. Present but non-contributing.
    b = _component("Cb", "R", [
        _hex(0, 10.0, 0.0, 0.0, 0.0, n_spots=0.0, n_cov=1),
    ])
    df = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "R", "source_spots": 10,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "R", "Cb": "R"},
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    )
    row = df.iloc[0]
    assert row[PARENT_CONSENSUS_CORRELATION_STATUS] == (
        ParentConsensusCorrelationStatus.INCOMPLETE_SUPPORT_COVERAGE.value
    )
    assert pd.isna(row["correlation_value"])


def test_zero_hex_sibling_marks_incomplete_support_coverage():
    contributing = _component("Ca", "R", [
        _hex(0, 0.0, 0.0, 1.0, 2.0),
        _hex(1, 1.0, 0.0, 2.0, 3.0),
        _hex(2, 2.0, 0.0, 3.0, 5.0),
    ])
    empty = _component("Cb", "R", [])
    row = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([contributing, empty]),
        support_status_df=_support([{
            "roi_id": "R", "source_spots": 10,
            "spatial_support_class": "supported",
        }]),
        component_to_roi={"Ca": "R", "Cb": "R"},
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    ).iloc[0]
    assert row[PARENT_CONSENSUS_CORRELATION_STATUS] == (
        ParentConsensusCorrelationStatus.INCOMPLETE_SUPPORT_COVERAGE.value
    )
    assert row["n_eligible_consensus_components"] == 2
    assert row["n_contributing_components_consensus"] == 1


def test_component_without_tessellation_eligible_region_is_not_eligible():
    empty = _component("Ca", "R", [], n_eligible_regions=0)
    row = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([empty]),
        support_status_df=_support([{
            "roi_id": "R", "source_spots": 10,
            "spatial_support_class": "supported",
        }]),
        component_to_roi={"Ca": "R"},
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    ).iloc[0]
    assert row[PARENT_CONSENSUS_CORRELATION_STATUS] == (
        ParentConsensusCorrelationStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    )
    assert row["n_eligible_consensus_components"] == 0


def test_scattered_parent_emits_no_tessellation_eligible_component_row():
    """Every declared surface still emits a typed row for a scattered
    parent — no row absence."""
    a = _component("Ca", "R", [
        _hex(0, 0.0, 0.0, 1.0, 2.0), _hex(1, 1.0, 0.0, 2.0, 4.0),
        _hex(2, 2.0, 0.0, 3.0, 6.0),
    ])
    df = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a]),
        support_status_df=_support([
            {"roi_id": "R", "source_spots": 10,
             "spatial_support_class": "supported"},
            {"roi_id": "S", "source_spots": 3,
             "spatial_support_class": "scattered"},
        ]),
        component_to_roi={"Ca": "R"},
    )
    scattered_rows = df[df[ROI_ID] == "S"]
    # One per surface.
    assert len(scattered_rows) == 2
    assert (
        scattered_rows[PARENT_CONSENSUS_CORRELATION_STATUS]
        == ParentConsensusCorrelationStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    ).all()


# ---------------------------------------------------------------------------
# 4. Enrichment invariance to duplicated identical tessellation exposure
# ---------------------------------------------------------------------------


def test_enrichment_invariant_under_duplicated_tessellation_exposure():
    """Doubling n_tessellations_covered AND the accumulated weighted
    counts on every hex must not change the pooled enrichment ratios
    (each hex's counts are divided by its own exposure before
    pooling)."""
    def _make_hex(hid, cx, high_g1, high_g2, both_high, low_g1, low_g2,
                  both_low, n_spots, n_cov):
        return _hex(
            hid, cx, 0.0, g1_sum=n_spots * 1.0, g2_sum=n_spots * 1.0,
            n_spots=n_spots, n_cov=n_cov,
            high_g1=high_g1, high_g2=high_g2, both_high=both_high,
            low_g1=low_g1, low_g2=low_g2, both_low=both_low,
        )

    def _build(scale: int):
        # Two ROIs so context is non-empty. Both have 3 hexes; scale
        # scales both counts and exposure uniformly.
        a = _component("Ca", "A", [
            _make_hex(hid, cx, 2.0*scale, 1.0*scale, 1.0*scale,
                      1.0*scale, 2.0*scale, 1.0*scale,
                      n_spots=4.0*scale, n_cov=2*scale)
            for hid, cx in enumerate((0.0, 1.0, 2.0))
        ])
        b = _component("Cb", "B", [
            _make_hex(hid, cx, 1.0*scale, 2.0*scale, 1.0*scale,
                      2.0*scale, 1.0*scale, 1.0*scale,
                      n_spots=4.0*scale, n_cov=2*scale)
            for hid, cx in enumerate((5.0, 6.0, 7.0))
        ])
        return build_roi_consensus_enrichment_df(
            sample_consensus_result=_sample([a, b]),
            support_status_df=_support([
                {"roi_id": "A", "source_spots": 20,
                 "spatial_support_class": "supported"},
                {"roi_id": "B", "source_spots": 20,
                 "spatial_support_class": "supported"},
            ]),
            component_to_roi={"Ca": "A", "Cb": "B"},
            normalization_params=_norm(),
        )

    df1 = _build(scale=1)
    df2 = _build(scale=2)
    for column in ("g1_enrichment", "g2_enrichment",
                   "joint_high_enrichment_fc", "joint_low_enrichment_fc"):
        v1 = df1[df1[ROI_ID] == "A"].iloc[0][column]
        v2 = df2[df2[ROI_ID] == "A"].iloc[0][column]
        assert v1 == pytest.approx(v2, rel=1e-9), (
            f"Duplicated tessellation exposure changed pooled "
            f"{column!r}: {v1} vs {v2}"
        )


# ---------------------------------------------------------------------------
# 5. Context excludes every sibling and includes other ROIs only
# ---------------------------------------------------------------------------


def test_context_excludes_siblings_and_includes_other_rois_only():
    """ROI A has 2 sibling components (Ca1, Ca2); ROI B has 1 (Cb).
    The context counters on A's row must NOT include Ca1 or Ca2 —
    only Cb's hexes."""
    ca1 = _component("Ca1", "A", [
        _hex(0, 0.0, 0.0, 1.0, 1.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(1, 1.0, 0.0, 2.0, 2.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(2, 2.0, 0.0, 3.0, 3.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
    ])
    ca2 = _component("Ca2", "A", [
        _hex(0, 5.0, 0.0, 4.0, 4.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(1, 6.0, 0.0, 5.0, 5.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
    ])
    cb = _component("Cb", "B", [
        _hex(0, 10.0, 0.0, 6.0, 6.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(1, 11.0, 0.0, 7.0, 7.0, high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([ca1, ca2, cb]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 25,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 25,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca1": "A", "Ca2": "A", "Cb": "B"},
        normalization_params=_norm(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    # A's context = only B's hexes (2 finite hexes), not Ca1 or Ca2.
    assert int(row_a["n_context_hexes_pooled"]) == 2
    assert int(row_a["n_context_hexes_with_finite_density_pooled"]) == 2
    assert int(row_a["n_context_eligible_consensus_components"]) == 1
    # A's target = 5 finite hexes (3 from Ca1 + 2 from Ca2).
    assert int(row_a["n_hexes_with_finite_density_pooled"]) == 5


# ---------------------------------------------------------------------------
# 6. context_proportion_zero preserves log2FC / CLES
# ---------------------------------------------------------------------------


def test_context_proportion_zero_preserves_log2fc_and_cles():
    """When every context hex has zero joint-high count, enrichment
    ratios are undefined (``context_proportion_zero``) but log2FC and
    CLES remain FINITE because they only require both sides to have
    ≥ 1 finite hex density."""
    # Target: hexes with plenty of high counts.
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 3.0, 3.0, n_spots=4.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(1, 1.0, 0.0, 4.0, 4.0, n_spots=4.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(2, 2.0, 0.0, 5.0, 5.0, n_spots=4.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
    ])
    # Context: every high count is zero; g1_high total = 0 → ratio undefined.
    b = _component("Cb", "B", [
        _hex(0, 10.0, 0.0, 1.0, 1.0, n_spots=4.0,
             high_g1=0.0, high_g2=0.0, both_high=0.0,
             low_g1=1.0, low_g2=1.0, both_low=1.0),
        _hex(1, 11.0, 0.0, 1.0, 1.0, n_spots=4.0,
             high_g1=0.0, high_g2=0.0, both_high=0.0,
             low_g1=1.0, low_g2=1.0, both_low=1.0),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 15,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 15,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A", "Cb": "B"},
        normalization_params=_norm(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert row_a[PARENT_CONSENSUS_ENRICHMENT_STATUS] == (
        ParentConsensusEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value
    )
    assert row_a[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] == (
        ParentConsensusEffectSizeStatus.FINITE.value
    )
    assert pd.notna(row_a[LOG2FC_G1])
    assert pd.notna(row_a[CLES_G1])
    # Sufficient stats are self-contained: both target and context
    # mass sets are populated on CONTEXT_PROPORTION_ZERO rows.
    for col in (OBS_EQUIV_SPOT_MASS, CONTEXT_OBS_EQUIV_SPOT_MASS):
        assert pd.notna(row_a[col])
    # Per-ratio breakdown: context has zero HIGH counts (all three
    # HIGH-related ratios non-finite) but non-zero both_low, so
    # joint_low remains FINITE with numeric value 0.
    finite = EnrichmentRatioStatus.FINITE.value
    assert row_a[GENE1_ENRICHMENT_STATUS] != finite
    assert row_a[GENE2_ENRICHMENT_STATUS] != finite
    assert row_a[JOINT_HIGH_ENRICHMENT_STATUS] != finite
    assert row_a[JOINT_LOW_ENRICHMENT_STATUS] == finite
    assert pd.isna(row_a[GENE1_ENRICHMENT])
    assert pd.isna(row_a[GENE2_ENRICHMENT])
    assert pd.isna(row_a[JOINT_HIGH_ENRICHMENT_FC])
    assert pd.notna(row_a[JOINT_LOW_ENRICHMENT_FC])


# ---------------------------------------------------------------------------
# 6b. Target masses independent of context — four coupling cases.
#     Each test asserts the "self-contained ratio-reconstruction"
#     invariant for one enrichment status.
# ---------------------------------------------------------------------------


_TARGET_MASS_COLS = (
    OBS_EQUIV_SPOT_MASS,
    OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G2_HIGH_MASS,
    OBS_EQUIV_BOTH_HIGH_MASS,
    OBS_EQUIV_G1_LOW_MASS, OBS_EQUIV_G2_LOW_MASS,
    OBS_EQUIV_BOTH_LOW_MASS,
    OBS_EQUIV_G1_SUM_MASS, OBS_EQUIV_G2_SUM_MASS,
)
_CONTEXT_MASS_COLS = (
    CONTEXT_OBS_EQUIV_SPOT_MASS,
    CONTEXT_OBS_EQUIV_G1_HIGH_MASS, CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
    CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS,
    CONTEXT_OBS_EQUIV_G1_LOW_MASS, CONTEXT_OBS_EQUIV_G2_LOW_MASS,
    CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_SUM_MASS, CONTEXT_OBS_EQUIV_G2_SUM_MASS,
)


def test_incomplete_context_populates_target_and_partial_context_masses():
    """A context ROI with an eligible-but-non-contributing sibling
    triggers ``INCOMPLETE_CONTEXT_COVERAGE``. Target masses populate
    (target itself is valid); the partial context pool (from
    contributing siblings) still emits its masses as provenance.
    Enrichment ratios and effect sizes are both non-finite —
    canonical log2FC / CLES must not be published on a partial-
    population context.
    """
    # ROI A: valid single-component target.
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 1.0, 2.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(1, 1.0, 0.0, 2.0, 4.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(2, 2.0, 0.0, 3.0, 6.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
    ])
    # ROI B: two siblings. Cb2 has hexes with n_spots=0 → zero finite
    # hexes → non-contributing while eligible. Cb1 contributes finite
    # hexes normally. Together this drives ROI B into the "incomplete
    # context coverage" branch when B acts as A's context.
    b1 = _component("Cb1", "B", [
        _hex(0, 10.0, 0.0, 1.0, 1.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(1, 11.0, 0.0, 2.0, 2.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
    ])
    b2 = _component("Cb2", "B", [
        _hex(0, 20.0, 0.0, 0.0, 0.0, n_spots=0.0, n_cov=1),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([a, b1, b2]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 15,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 15,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A", "Cb1": "B", "Cb2": "B"},
        normalization_params=_norm(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert row_a[PARENT_CONSENSUS_ENRICHMENT_STATUS] == (
        ParentConsensusEnrichmentStatus.INCOMPLETE_CONTEXT_COVERAGE.value
    )
    assert row_a[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] == (
        ParentConsensusEffectSizeStatus.INCOMPLETE_CONTEXT_COVERAGE.value
    )
    for col in _TARGET_MASS_COLS:
        assert pd.notna(row_a[col]), f"target mass {col} unexpectedly NA"
    for col in _CONTEXT_MASS_COLS:
        assert pd.notna(row_a[col]), (
            f"context mass {col} unexpectedly NA — partial context "
            "must still emit masses as provenance"
        )
    for col in (GENE1_ENRICHMENT, GENE2_ENRICHMENT,
                JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
                LOG2FC_G1, CLES_G1):
        assert pd.isna(row_a[col]), (
            f"{col} unexpectedly finite under INCOMPLETE_CONTEXT_COVERAGE"
        )


def test_no_eligible_context_populates_target_masses_only():
    """When A is the only ROI with eligible components, its context
    pool is empty. Target masses populate (target itself is valid);
    context masses stay null; enrichment ratios non-finite; effect
    sizes carry ``INSUFFICIENT_CONTEXT_HEXES`` (zero context finite
    hexes falls through the natural count-based branch — distinct
    from partial-context coverage).
    """
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 1.0, 2.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(1, 1.0, 0.0, 2.0, 4.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(2, 2.0, 0.0, 3.0, 6.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([a]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 15,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A"},
        normalization_params=_norm(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert row_a[PARENT_CONSENSUS_ENRICHMENT_STATUS] == (
        ParentConsensusEnrichmentStatus.NO_ELIGIBLE_CONTEXT.value
    )
    assert row_a[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] == (
        ParentConsensusEffectSizeStatus.INSUFFICIENT_CONTEXT_HEXES.value
    )
    for col in _TARGET_MASS_COLS:
        assert pd.notna(row_a[col]), f"target mass {col} unexpectedly NA"
    for col in _CONTEXT_MASS_COLS:
        assert pd.isna(row_a[col]), (
            f"context mass {col} unexpectedly populated with no context"
        )
    for col in (GENE1_ENRICHMENT, LOG2FC_G1, CLES_G1):
        assert pd.isna(row_a[col])


def test_incomplete_support_coverage_emits_partial_target_masses():
    """When ROI A has one contributing sibling AND one eligible-but-
    non-contributing sibling, target-side hard gate fires as
    ``INCOMPLETE_SUPPORT_COVERAGE``. Partial target masses from the
    contributing sibling emit as provenance (mirroring the partial-
    context treatment); ratios / effect sizes remain non-finite.
    Effect-size status is ``INCOMPLETE_SUPPORT_COVERAGE`` — target
    pool cannot support a canonical estimate even though the
    contributing subset has finite hexes.
    """
    ca1 = _component("Ca1", "A", [
        _hex(0, 0.0, 0.0, 1.0, 2.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(1, 1.0, 0.0, 2.0, 4.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(2, 2.0, 0.0, 3.0, 6.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
    ])
    # Ca2 has a hex but zero exposure content → non-contributing
    # while still eligible.
    ca2 = _component("Ca2", "A", [
        _hex(0, 10.0, 0.0, 0.0, 0.0, n_spots=0.0, n_cov=1),
    ])
    # Provide a distinct context ROI so any context-side code that
    # runs sees a populated pool.
    cb = _component("Cb", "B", [
        _hex(0, 20.0, 0.0, 1.0, 1.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
        _hex(1, 21.0, 0.0, 2.0, 2.0, high_g1=1.0, high_g2=1.0, both_high=1.0),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([ca1, ca2, cb]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 15,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 15,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca1": "A", "Ca2": "A", "Cb": "B"},
        normalization_params=_norm(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert row_a[PARENT_CONSENSUS_ENRICHMENT_STATUS] == (
        ParentConsensusEnrichmentStatus.INCOMPLETE_SUPPORT_COVERAGE.value
    )
    assert row_a[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] == (
        ParentConsensusEffectSizeStatus.INCOMPLETE_SUPPORT_COVERAGE.value
    )
    for col in _TARGET_MASS_COLS:
        assert pd.notna(row_a[col]), (
            f"target mass {col} unexpectedly NA under "
            "INCOMPLETE_SUPPORT_COVERAGE — partial target pool "
            "should emit as provenance"
        )
    # Context masses populate too (context is independent of target
    # completeness; other ROIs' contributing hexes still exist).
    for col in _CONTEXT_MASS_COLS:
        assert pd.notna(row_a[col]), (
            f"context mass {col} unexpectedly NA — context is "
            "independent of target-side completeness and other ROIs "
            "contributed finite hexes"
        )
    # Ratios and effect sizes stay null.
    for col in (GENE1_ENRICHMENT, GENE2_ENRICHMENT,
                JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
                LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2):
        assert pd.isna(row_a[col])


def test_finite_row_ratios_reconstruct_from_target_and_context_masses():
    """FINITE row: both target and context mass sets populated; the
    consumer reconstructs each emitted enrichment ratio to floating-
    point tolerance as
    ``(target_mass_num / target_spot_mass) /
      (context_mass_num / context_spot_mass)``.
    """
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 3.0, 3.0, n_spots=4.0,
             high_g1=2.0, high_g2=1.0, both_high=1.0,
             low_g1=1.0, low_g2=2.0, both_low=1.0),
        _hex(1, 1.0, 0.0, 4.0, 4.0, n_spots=4.0,
             high_g1=2.0, high_g2=1.0, both_high=1.0,
             low_g1=1.0, low_g2=2.0, both_low=1.0),
        _hex(2, 2.0, 0.0, 5.0, 5.0, n_spots=4.0,
             high_g1=2.0, high_g2=1.0, both_high=1.0,
             low_g1=1.0, low_g2=2.0, both_low=1.0),
    ])
    b = _component("Cb", "B", [
        _hex(0, 10.0, 0.0, 1.0, 1.0, n_spots=4.0,
             high_g1=1.0, high_g2=2.0, both_high=1.0,
             low_g1=2.0, low_g2=1.0, both_low=1.0),
        _hex(1, 11.0, 0.0, 2.0, 2.0, n_spots=4.0,
             high_g1=1.0, high_g2=2.0, both_high=1.0,
             low_g1=2.0, low_g2=1.0, both_low=1.0),
        _hex(2, 12.0, 0.0, 3.0, 3.0, n_spots=4.0,
             high_g1=1.0, high_g2=2.0, both_high=1.0,
             low_g1=2.0, low_g2=1.0, both_low=1.0),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 15,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 15,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A", "Cb": "B"},
        normalization_params=_norm(),
    )
    row_a = df[df[ROI_ID] == "A"].iloc[0]
    assert row_a[PARENT_CONSENSUS_ENRICHMENT_STATUS] == (
        ParentConsensusEnrichmentStatus.FINITE.value
    )
    for col in _TARGET_MASS_COLS + _CONTEXT_MASS_COLS:
        assert pd.notna(row_a[col])

    # Reconstruct each ratio from the emitted mass pairs.
    def _rate(num, den):
        return float(num) / float(den)

    for emitted_col, target_num, context_num in (
        (GENE1_ENRICHMENT, OBS_EQUIV_G1_HIGH_MASS, CONTEXT_OBS_EQUIV_G1_HIGH_MASS),
        (GENE2_ENRICHMENT, OBS_EQUIV_G2_HIGH_MASS, CONTEXT_OBS_EQUIV_G2_HIGH_MASS),
        (JOINT_HIGH_ENRICHMENT_FC,
         OBS_EQUIV_BOTH_HIGH_MASS, CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS),
        (JOINT_LOW_ENRICHMENT_FC,
         OBS_EQUIV_BOTH_LOW_MASS, CONTEXT_OBS_EQUIV_BOTH_LOW_MASS),
    ):
        target_rate = _rate(row_a[target_num], row_a[OBS_EQUIV_SPOT_MASS])
        context_rate = _rate(
            row_a[context_num], row_a[CONTEXT_OBS_EQUIV_SPOT_MASS],
        )
        recomputed = target_rate / context_rate
        assert abs(recomputed - float(row_a[emitted_col])) < 1e-9, (
            f"{emitted_col}: emitted={row_a[emitted_col]} vs "
            f"recomputed={recomputed}"
        )


# ---------------------------------------------------------------------------
# 7. No direct-surface enrichment row emitted
# ---------------------------------------------------------------------------


def test_no_direct_surface_enrichment_row_emitted():
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 1.0, 1.0), _hex(1, 1.0, 0.0, 2.0, 2.0),
    ])
    b = _component("Cb", "B", [
        _hex(0, 5.0, 0.0, 3.0, 3.0), _hex(1, 6.0, 0.0, 4.0, 4.0),
    ])
    df = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 10,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 10,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A", "Cb": "B"},
        normalization_params=_norm(),
    )
    assert set(df[VALUE_SURFACE]) == {"consensus_ensemble"}


def test_no_inference_fields_on_any_roi_consensus_surface():
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 1.0, 1.0, n_spots=2.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(1, 1.0, 0.0, 2.0, 2.0, n_spots=2.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(2, 2.0, 0.0, 3.0, 3.0, n_spots=2.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
    ])
    b = _component("Cb", "B", [
        _hex(0, 10.0, 0.0, 1.0, 2.0, n_spots=2.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
        _hex(1, 11.0, 0.0, 2.0, 3.0, n_spots=2.0,
             high_g1=1.0, high_g2=1.0, both_high=1.0,
             low_g1=0.0, low_g2=0.0, both_low=0.0),
    ])
    support = _support([
        {"roi_id": "A", "source_spots": 10,
         "spatial_support_class": "supported"},
        {"roi_id": "B", "source_spots": 10,
         "spatial_support_class": "supported"},
    ])
    corr = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=support,
        component_to_roi={"Ca": "A", "Cb": "B"},
    )
    enr = build_roi_consensus_enrichment_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=support,
        component_to_roi={"Ca": "A", "Cb": "B"},
        normalization_params=_norm(),
    )
    # Bare-name resurrections + non-declared inference fields.
    forbidden_shared = {
        "corr_p", "corr_p_perm", "corr_perm_mean", "corr_perm_sd",
        "n_permutations", "corr_q", "corr_q_perm",
        "joint_confidence", "joint_confidence_adj",
        "seed",
    }
    for label, df in (("correlations", corr), ("enrichment", enr)):
        overlap = forbidden_shared.intersection(df.columns)
        assert not overlap, (
            f"roi_consensus {label}: unexpected inference fields "
            f"{sorted(overlap)!r}"
        )
    # CORE-INF-01a: consensus correlations now carry analytic p + BH q +
    # family_size (per (sample_id, value_surface, g1_id, g2_id, mode)).
    for column in ("corr_p_analytic", "corr_q_analytic", "fdr_family_size"):
        assert column in corr.columns, (
            f"roi_consensus correlations missing {column!r}"
        )


def test_consensus_fdr_isolates_ensemble_from_direct_surface():
    """BH families are per value_surface: consensus_ensemble and
    consensus_direct are independent FDR groups. Two-ROI fixture where
    each ROI has different ensemble vs direct hex densities produces
    different fdr_family_size (or family membership) per surface.
    """
    # Two ROIs with distinct ensemble vs direct hex densities so both
    # surfaces have a chance to yield differing finite-row memberships.
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 1.0, 4.0, direct_g1_sum=2.0, direct_g2_sum=3.0),
        _hex(1, 1.0, 0.0, 2.0, 3.0, direct_g1_sum=1.0, direct_g2_sum=4.0),
    ])
    b = _component("Cb", "B", [
        _hex(0, 5.0, 0.0, 5.0, 8.0, direct_g1_sum=4.0, direct_g2_sum=9.0),
        _hex(1, 6.0, 0.0, 6.0, 7.0, direct_g1_sum=7.0, direct_g2_sum=6.0),
    ])
    corr = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 10,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 10,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A", "Cb": "B"},
        surfaces=(
            PartitionValueSurface.CONSENSUS_ENSEMBLE,
            PartitionValueSurface.CONSENSUS_DIRECT,
        ),
    )
    ens = corr[corr[VALUE_SURFACE] == "consensus_ensemble"]
    dir_ = corr[corr[VALUE_SURFACE] == "consensus_direct"]

    # Both surfaces present, one row per (ROI × surface) = 2 × 2 = 4 rows.
    assert len(ens) == 2 and len(dir_) == 2

    # Each surface's fdr_family_size is uniform within the surface
    # (single family per surface × sample); the two surfaces are
    # independent BH groups, so their p / q values are computed
    # separately even when family sizes numerically match.
    assert ens["fdr_family_size"].nunique() == 1
    assert dir_["fdr_family_size"].nunique() == 1

    # Isolation check: mutate only ensemble p's in the frame and rerun
    # BH on ensemble ⇒ direct q's must not change. Encoded here by
    # asserting that swapping one surface's finite membership can't
    # move the other surface's q values (BH derived per grouping key).
    # Concrete assertion: q values are functions ONLY of same-surface
    # p values.
    for surface_df in (ens, dir_):
        p = surface_df["corr_p_analytic"].to_numpy(dtype=float)
        expected_q = benjamini_hochberg(p)
        actual_q = surface_df["corr_q_analytic"].to_numpy(dtype=float)
        # Compare non-NaN entries; NaN mask must match.
        assert (np.isnan(expected_q) == np.isnan(actual_q)).all()
        mask = ~np.isnan(expected_q)
        assert np.allclose(expected_q[mask], actual_q[mask])


def test_consensus_non_finite_row_retains_null_p_q_but_non_null_family_size():
    """A component whose consensus support cannot compute correlation
    (e.g. degenerate variance) produces a non-finite row: p / q null,
    fdr_family_size populated.
    """
    # ROI A: finite (usable variance across 4 hexes). ROI B: constant
    # gene1 on every hex — zero variance on gene1 → non-finite.
    a = _component("Ca", "A", [
        _hex(0, 0.0, 0.0, 1.0, 4.0),
        _hex(1, 1.0, 0.0, 2.0, 3.0),
        _hex(2, 2.0, 0.0, 3.0, 5.0),
        _hex(3, 3.0, 0.0, 4.0, 2.0),
    ])
    b = _component("Cb", "B", [
        _hex(0, 5.0, 0.0, 1.0, 4.0),  # gene1 sum constant across hexes
        _hex(1, 6.0, 0.0, 1.0, 7.0),
        _hex(2, 7.0, 0.0, 1.0, 3.0),
        _hex(3, 8.0, 0.0, 1.0, 5.0),
    ])
    corr = build_roi_consensus_correlations_df(
        sample_consensus_result=_sample([a, b]),
        support_status_df=_support([
            {"roi_id": "A", "source_spots": 10,
             "spatial_support_class": "supported"},
            {"roi_id": "B", "source_spots": 10,
             "spatial_support_class": "supported"},
        ]),
        component_to_roi={"Ca": "A", "Cb": "B"},
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    )
    a_row = corr[corr[ROI_ID] == "A"].iloc[0]
    b_row = corr[corr[ROI_ID] == "B"].iloc[0]
    assert a_row[PARENT_CONSENSUS_CORRELATION_STATUS] == "finite"
    assert b_row[PARENT_CONSENSUS_CORRELATION_STATUS] != "finite"

    # A finite → p + q populated. B non-finite → p + q NaN.
    assert not np.isnan(a_row["corr_p_analytic"])
    assert not np.isnan(a_row["corr_q_analytic"])
    assert np.isnan(b_row["corr_p_analytic"])
    assert np.isnan(b_row["corr_q_analytic"])

    # fdr_family_size populated on every row (non-nullable per schema).
    assert not pd.isna(a_row["fdr_family_size"])
    assert not pd.isna(b_row["fdr_family_size"])
    assert int(a_row["fdr_family_size"]) == 1  # only A finite in family
    assert int(b_row["fdr_family_size"]) == 1
