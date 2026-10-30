"""Focused unit tests for the canonical build_consensus_hexes_df builder.

Six semantic contracts:

1. Empty inputs yield a zero-row frame with the frozen column set + dtypes.
2. Missing (sample_id, roi_id, ensemble) row on roi_consensus_enrichment_df
   raises KeyError with a helpful message.
3. Threshold masses missing under a finite parent ROI raises ValueError.
4. Finite hexes emit non-null hex_* ratios; parent_consensus_enrichment_status
   is copied verbatim onto every hex row of that ROI.
5. Missingness statuses (roi_enrichment_unavailable / no_hex_observations)
   fire correctly and emit null ratios.
6. Weighted coherence identity: Σ_h (m_h · hex_FC_h) / Σ_h m_h == roi_FC
   on a hand-constructed 2-ROI × 3-hex fixture with known masses.
"""
from __future__ import annotations

from dataclasses import replace as _replace
from tilescope.tests.helpers.fixture_projections import _HEX_FEATURES, register_hex_features
from tilescope.analyzers.sample_support_stage import sample_support_bank_from_component_results


import pandas as pd
import pytest
from shapely.geometry import box

from tilescope import cols
from tilescope.analyzers.consensus_stage import (
    ComponentConsensusResult,
    ConsensusHex,
    SampleConsensusResult,
)
from tilescope.analyzers.consensus_frames import build_consensus_analysis_frames
from tilescope.analyzers.consensus_hex_surface import (
    CONSENSUS_HEXES_COLUMNS,
    DTYPE_MAP,
    build_consensus_hexes_df as _build_consensus_hexes,
)
from tilescope.tests.helpers.fixture_projections import projection_from_hex_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hex(
    hex_id: int, *, n_cov: int = 3, component_id: str = "C",
    n_ens: float = 4.0, sum_g1_ens: float = 8.0, sum_g2_ens: float = 12.0,
    thresholds: bool = True,
    g1_high: float = 2.0, g2_high: float = 1.5,
    both_high: float = 1.0,
    g1_low: float = 0.5, g2_low: float = 0.5, both_low: float = 0.25,
) -> ConsensusHex:
    poly = box(float(hex_id), 0.0, float(hex_id) + 1.0, 1.0)
    kwargs = dict(
        component_id=component_id,
        hex_id=hex_id, domain_region_id=0, polygon=poly,
        lattice_col=hex_id, lattice_row=0,
        center_x=hex_id + 0.5, center_y=0.5,
        n_tessellations_covered=n_cov,
        n_spots_ensemble_weighted=n_ens,
        n_spots_direct=int(n_ens),
    )
    if thresholds:
        kwargs.update(
            n_both_high_ensemble_weighted=both_high,
            n_both_low_ensemble_weighted=both_low,
        )
    hx = ConsensusHex.build(**kwargs)
    register_hex_features(
        component_id=component_id, hex_id=hex_id,
        sum_g1_ens=sum_g1_ens, sum_g2_ens=sum_g2_ens,
        g1_high=g1_high, g2_high=g2_high,
        g1_low=g1_low, g2_low=g2_low,
        sum_g1_direct=sum_g1_ens, sum_g2_direct=sum_g2_ens,
    )
    return hx


def _component(
    cid: str, parent_roi_id: str, hexes, hex_radius: float = 1.5,
) -> ComponentConsensusResult:
    rebound = []
    for h in hexes:
        orig_cid = h.geometry.component_id
        rebound.append(
            _replace(h, geometry=_replace(h.geometry, component_id=cid)),
        )
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
        hex_radius=hex_radius, hexes=tuple(rebound),
        n_tessellation_eligible_regions=1,
    )


def _sample(components) -> SampleConsensusResult:
    support_bank = sample_support_bank_from_component_results(
        sample_id="S", component_results={},
        n_replicates=3, base_seed=0,
    )
    return SampleConsensusResult.from_components(
        sample_id="S", components=components,
        n_replicates_planned=3,
        sample_anchor_x=0.0, sample_anchor_y=0.0,
        support_bank=support_bank,
    )


def build_consensus_hexes_df(*, sample_consensus_result, **kwargs):
    p_g1, p_g2 = projection_from_hex_features(
        sample_consensus_result=sample_consensus_result,
    )
    frames = build_consensus_analysis_frames(
        sample_consensus_result=sample_consensus_result,
        feature_projection_g1=p_g1, feature_projection_g2=p_g2,
    )
    return _build_consensus_hexes(consensus_frames=frames, **kwargs)


def _roi_enrichment_row(
    roi_id: str, *,
    status: str = "finite",
    ctx_spot: float = 100.0,
    ctx_g1_high: float = 50.0, ctx_g2_high: float = 40.0,
    ctx_both_high: float = 25.0, ctx_both_low: float = 10.0,
    ctx_g1_low: float = 15.0, ctx_g2_low: float = 12.0,
    tgt_spot: float = 10.0,
    g1_enr: float = 1.2, g2_enr: float = 1.1,
    joint_high: float = 1.5, joint_low: float = 0.8,
) -> dict:
    per_ratio_status = "finite" if status == "finite" else pd.NA
    return {
        cols.SAMPLE_ID: "S", cols.ROI_ID: roi_id,
        cols.VALUE_SURFACE: "consensus_ensemble",
        cols.PARENT_CONSENSUS_ENRICHMENT_STATUS: status,
        cols.OBS_EQUIV_SPOT_MASS: tgt_spot,
        cols.CONTEXT_OBS_EQUIV_SPOT_MASS: ctx_spot,
        cols.CONTEXT_OBS_EQUIV_G1_HIGH_MASS: ctx_g1_high,
        cols.CONTEXT_OBS_EQUIV_G2_HIGH_MASS: ctx_g2_high,
        cols.CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS: ctx_both_high,
        cols.CONTEXT_OBS_EQUIV_BOTH_LOW_MASS: ctx_both_low,
        cols.CONTEXT_OBS_EQUIV_G1_LOW_MASS: ctx_g1_low,
        cols.CONTEXT_OBS_EQUIV_G2_LOW_MASS: ctx_g2_low,
        cols.GENE1_ENRICHMENT: g1_enr,
        cols.GENE2_ENRICHMENT: g2_enr,
        cols.JOINT_HIGH_ENRICHMENT_FC: joint_high,
        cols.JOINT_LOW_ENRICHMENT_FC: joint_low,
        cols.GENE1_ENRICHMENT_STATUS: per_ratio_status,
        cols.GENE2_ENRICHMENT_STATUS: per_ratio_status,
        cols.JOINT_HIGH_ENRICHMENT_STATUS: per_ratio_status,
        cols.JOINT_LOW_ENRICHMENT_STATUS: per_ratio_status,
    }


def _roi_enrichment_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Empty inputs
# ---------------------------------------------------------------------------


def test_empty_sample_yields_zero_row_schema_stable_frame():
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({}),
        roi_consensus_enrichment_df=pd.DataFrame(),
    )
    assert len(df) == 0
    assert tuple(df.columns) == CONSENSUS_HEXES_COLUMNS
    for col_name in CONSENSUS_HEXES_COLUMNS:
        assert str(df[col_name].dtype) == DTYPE_MAP[col_name], col_name


# ---------------------------------------------------------------------------
# 2. Missing ROI enrichment row raises KeyError
# ---------------------------------------------------------------------------


def test_missing_roi_row_raises_key_error_with_helpful_message():
    comp = _component("Ca", "R", [_hex(0)])
    with pytest.raises(KeyError, match=r"roi_id='R'"):
        build_consensus_hexes_df(
            sample_consensus_result=_sample({"Ca": comp}),
            roi_consensus_enrichment_df=pd.DataFrame(),
        )


# ---------------------------------------------------------------------------
# 3. Missing threshold masses under finite parent raises ValueError
# ---------------------------------------------------------------------------


def test_missing_threshold_masses_under_finite_parent_raises():
    # Hex with no thresholds; parent status finite.
    comp = _component("Ca", "R", [_hex(0, thresholds=False)])
    with pytest.raises(ValueError, match="threshold mass fields"):
        build_consensus_hexes_df(
            sample_consensus_result=_sample({"Ca": comp}),
            roi_consensus_enrichment_df=_roi_enrichment_df(
                [_roi_enrichment_row("R")]
            ),
        )


# ---------------------------------------------------------------------------
# 4. Finite hexes → non-null ratios; parent status copied
# ---------------------------------------------------------------------------


def test_finite_hex_emits_non_null_ratios_and_copies_parent_status():
    comp = _component("Ca", "R", [_hex(0), _hex(1)])
    roi_row = _roi_enrichment_row(
        "R",
        # Choose enrichment values matching the hand-computed ratio:
        # per-hex g1_high mass = 2, spot mass = 4, tgt_rate = 0.5;
        # ctx g1_high = 50, ctx spot = 100, ctx_rate = 0.5 → ratio 1.0.
        # We also need the ROI FC to match weighted mean of these ratios,
        # which for constant per-hex ratios = ratio itself.
        g1_enr=1.0, g2_enr=(1.5 / 4) / (40 / 100),
        joint_high=(1.0 / 4) / (25 / 100),
        joint_low=(0.25 / 4) / (10 / 100),
    )
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({"Ca": comp}),
        roi_consensus_enrichment_df=_roi_enrichment_df([roi_row]),
    )
    assert len(df) == 2
    assert (df[cols.PARENT_CONSENSUS_ENRICHMENT_STATUS] == "finite").all()
    assert (df[cols.HEX_ENRICHMENT_STATUS] == "finite").all()
    for col in (
        cols.HEX_G1_ENRICHMENT, cols.HEX_G2_ENRICHMENT,
        cols.HEX_JOINT_HIGH_ENRICHMENT_FC, cols.HEX_JOINT_LOW_ENRICHMENT_FC,
    ):
        assert df[col].notna().all(), col


# ---------------------------------------------------------------------------
# 5. Missingness statuses (roi_unavailable / no_hex_observations)
# ---------------------------------------------------------------------------


def test_partial_context_proportion_zero_emits_positive_over_zero_context():
    """Parent status ``context_proportion_zero`` no longer nulls every
    hex ratio. The two marginal per-ratio statuses stay ``finite`` (their
    context proportions were > 0). The joint LOW status resolves to
    exactly ``positive_over_zero_context`` because context is zero AND
    the hex's joint LOW mass is strictly positive. Hex aggregate
    becomes ``partial_ratio_availability``.
    """
    comp = _component("Ca", "R", [_hex(0)])
    # Zero-out only the joint-LOW context; keep marginal LOW context
    # non-zero to preserve grain coherence (ctx_both_low <=
    # min(ctx_g1_low, ctx_g2_low)). The hex fixture's default
    # `both_low=0.25` mass is strictly positive → per-ratio status
    # must be positive_over_zero_context (not zero_over_zero_context).
    roi_row = _roi_enrichment_row(
        "R", status="context_proportion_zero",
        ctx_both_low=0.0,
        g1_enr=(2.0 / 4) / (50 / 100),
        g2_enr=(1.5 / 4) / (40 / 100),
        joint_high=(1.0 / 4) / (25 / 100),
        joint_low=None,
    )
    roi_row[cols.GENE1_ENRICHMENT_STATUS] = "finite"
    roi_row[cols.GENE2_ENRICHMENT_STATUS] = "finite"
    roi_row[cols.JOINT_HIGH_ENRICHMENT_STATUS] = "finite"
    roi_row[cols.JOINT_LOW_ENRICHMENT_STATUS] = "positive_over_zero_context"
    roi_row[cols.JOINT_LOW_ENRICHMENT_FC] = pd.NA
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({"Ca": comp}),
        roi_consensus_enrichment_df=_roi_enrichment_df([roi_row]),
    )
    row = df.iloc[0]
    assert row[cols.HEX_ENRICHMENT_STATUS] == "partial_ratio_availability"
    assert row[cols.HEX_G1_ENRICHMENT_STATUS] == "finite"
    assert row[cols.HEX_G2_ENRICHMENT_STATUS] == "finite"
    assert row[cols.HEX_JOINT_HIGH_ENRICHMENT_STATUS] == "finite"
    assert row[cols.HEX_JOINT_LOW_ENRICHMENT_STATUS] == "positive_over_zero_context"
    assert pd.notna(row[cols.HEX_G1_ENRICHMENT])
    assert pd.notna(row[cols.HEX_G2_ENRICHMENT])
    assert pd.notna(row[cols.HEX_JOINT_HIGH_ENRICHMENT_FC])
    assert pd.isna(row[cols.HEX_JOINT_LOW_ENRICHMENT_FC])


def test_partial_context_proportion_zero_emits_zero_over_zero_context():
    """Zero-context AND zero hex numerator ⇒ per-ratio status is
    exactly ``zero_over_zero_context`` (not ``positive_over_zero_context``).
    """
    # Hex with zero joint-LOW mass alongside three normal ratios.
    zero_low_hex = _hex(
        0, both_low=0.0,
        g1_low=0.5, g2_low=0.5,  # keep marginals non-zero for realism
    )
    comp = _component("Ca", "R", [zero_low_hex])
    roi_row = _roi_enrichment_row(
        "R", status="context_proportion_zero",
        ctx_both_low=0.0,
        g1_enr=(2.0 / 4) / (50 / 100),
        g2_enr=(1.5 / 4) / (40 / 100),
        joint_high=(1.0 / 4) / (25 / 100),
        joint_low=None,
    )
    roi_row[cols.GENE1_ENRICHMENT_STATUS] = "finite"
    roi_row[cols.GENE2_ENRICHMENT_STATUS] = "finite"
    roi_row[cols.JOINT_HIGH_ENRICHMENT_STATUS] = "finite"
    roi_row[cols.JOINT_LOW_ENRICHMENT_STATUS] = "zero_over_zero_context"
    roi_row[cols.JOINT_LOW_ENRICHMENT_FC] = pd.NA
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({"Ca": comp}),
        roi_consensus_enrichment_df=_roi_enrichment_df([roi_row]),
    )
    row = df.iloc[0]
    assert row[cols.HEX_JOINT_LOW_ENRICHMENT_STATUS] == "zero_over_zero_context"
    assert pd.isna(row[cols.HEX_JOINT_LOW_ENRICHMENT_FC])


def test_parent_non_finite_marks_hex_roi_enrichment_unavailable():
    comp = _component("Ca", "R", [_hex(0), _hex(1)])
    roi_row = _roi_enrichment_row(
        "R", status="no_eligible_context",
    )
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({"Ca": comp}),
        roi_consensus_enrichment_df=_roi_enrichment_df([roi_row]),
    )
    assert (
        df[cols.PARENT_CONSENSUS_ENRICHMENT_STATUS] == "no_eligible_context"
    ).all()
    assert (
        df[cols.HEX_ENRICHMENT_STATUS] == "roi_enrichment_unavailable"
    ).all()
    for col in (
        cols.HEX_G1_ENRICHMENT, cols.HEX_G2_ENRICHMENT,
        cols.HEX_JOINT_HIGH_ENRICHMENT_FC, cols.HEX_JOINT_LOW_ENRICHMENT_FC,
    ):
        assert df[col].isna().all(), col


def test_zero_spot_mass_under_finite_parent_marks_no_hex_observations():
    # Zero-mass retained hex + normal hex under finite parent.
    zero_hex = _hex(0, n_ens=0.0, sum_g1_ens=0.0, sum_g2_ens=0.0,
                    g1_high=0.0, g2_high=0.0, both_high=0.0,
                    g1_low=0.0, g2_low=0.0, both_low=0.0)
    good_hex = _hex(1)
    comp = _component("Ca", "R", [zero_hex, good_hex])
    # Choose ROI enrichment so weighted coherence holds trivially
    # (weighted mean over the single finite hex equals its ratio).
    roi_row = _roi_enrichment_row(
        "R",
        g1_enr=(2.0 / 4) / (50 / 100),
        g2_enr=(1.5 / 4) / (40 / 100),
        joint_high=(1.0 / 4) / (25 / 100),
        joint_low=(0.25 / 4) / (10 / 100),
    )
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({"Ca": comp}),
        roi_consensus_enrichment_df=_roi_enrichment_df([roi_row]),
    ).sort_values(cols.HEX_ID).reset_index(drop=True)
    zero_row = df.iloc[0]
    good_row = df.iloc[1]
    assert zero_row[cols.HEX_ENRICHMENT_STATUS] == "no_hex_observations"
    assert pd.isna(zero_row[cols.HEX_G1_ENRICHMENT])
    assert good_row[cols.HEX_ENRICHMENT_STATUS] == "finite"
    assert not pd.isna(good_row[cols.HEX_G1_ENRICHMENT])


# ---------------------------------------------------------------------------
# 6. Weighted coherence identity on a 2-ROI × 3-hex fixture
# ---------------------------------------------------------------------------


def test_weighted_coherence_identity_holds_across_two_rois():
    # Two ROIs, three hexes each, varied per-hex masses so weighted
    # mean is a real test (not trivially equal to constant ratios).
    def _mk_hex(cid, hex_id, n_ens, g1_high, g2_high, both_high, both_low, g1_low, g2_low):
        return _hex(
            hex_id, component_id=cid, n_ens=n_ens,
            g1_high=g1_high, g2_high=g2_high, both_high=both_high,
            g1_low=g1_low, g2_low=g2_low, both_low=both_low,
        )

    hexes_a = [
        _mk_hex("Ca", 0, n_ens=3.0, g1_high=1.5, g2_high=1.2, both_high=0.8,
                both_low=0.2, g1_low=0.4, g2_low=0.5),
        _mk_hex("Ca", 1, n_ens=6.0, g1_high=3.0, g2_high=2.4, both_high=1.6,
                both_low=0.4, g1_low=0.8, g2_low=1.0),
        _mk_hex("Ca", 2, n_ens=9.0, g1_high=4.5, g2_high=3.6, both_high=2.4,
                both_low=0.6, g1_low=1.2, g2_low=1.5),
    ]
    hexes_b = [
        _mk_hex("Cb", 0, n_ens=4.0, g1_high=1.0, g2_high=2.0, both_high=0.8,
                both_low=0.3, g1_low=0.5, g2_low=0.6),
        _mk_hex("Cb", 1, n_ens=5.0, g1_high=1.5, g2_high=2.5, both_high=1.1,
                both_low=0.4, g1_low=0.6, g2_low=0.7),
        _mk_hex("Cb", 2, n_ens=6.0, g1_high=2.0, g2_high=3.0, both_high=1.4,
                both_low=0.5, g1_low=0.7, g2_low=0.9),
    ]
    comp_a = _component("Ca", "R1", hexes_a)
    comp_b = _component("Cb", "R2", hexes_b)

    # Compute the *actual* weighted mean of per-hex ratios and use that
    # as the ROI FC. Uses the fixture's declared per-hex feature values
    # via the fixture_projections registry (post-Phase-3a marginals live
    # in FeatureSpatialProjection, not on ConsensusHex).

    def _feats(component_id, hex_id):
        return _HEX_FEATURES[(component_id, hex_id)]

    def _roi_fcs(component_id, hexes, ctx_g1_h, ctx_g2_h, ctx_both_h, ctx_both_l, ctx_spot):
        wts = [float(h.n_spots_ensemble_weighted) for h in hexes]
        wsum = sum(wts)
        r_g1, r_g2, r_jh, r_jl = [], [], [], []
        for h in hexes:
            n_spots = float(h.n_spots_ensemble_weighted)
            data = _feats(component_id, int(h.hex_id))
            r_g1.append((data["g1_high"] / n_spots) / (ctx_g1_h / ctx_spot))
            r_g2.append((data["g2_high"] / n_spots) / (ctx_g2_h / ctx_spot))
            r_jh.append(
                (float(h.n_both_high_ensemble_weighted) / n_spots)
                / (ctx_both_h / ctx_spot),
            )
            r_jl.append(
                (float(h.n_both_low_ensemble_weighted) / n_spots)
                / (ctx_both_l / ctx_spot),
            )
        return (
            sum(w * v for w, v in zip(wts, r_g1)) / wsum,
            sum(w * v for w, v in zip(wts, r_g2)) / wsum,
            sum(w * v for w, v in zip(wts, r_jh)) / wsum,
            sum(w * v for w, v in zip(wts, r_jl)) / wsum,
        )

    fcs_a = _roi_fcs("Ca", hexes_a, 50.0, 40.0, 25.0, 10.0, 100.0)
    fcs_b = _roi_fcs("Cb", hexes_b, 60.0, 50.0, 30.0, 12.0, 120.0)
    roi_rows = [
        _roi_enrichment_row(
            "R1", ctx_spot=100.0,
            ctx_g1_high=50.0, ctx_g2_high=40.0, ctx_both_high=25.0,
            ctx_both_low=10.0, ctx_g1_low=15.0, ctx_g2_low=12.0,
            g1_enr=fcs_a[0], g2_enr=fcs_a[1],
            joint_high=fcs_a[2], joint_low=fcs_a[3],
        ),
        _roi_enrichment_row(
            "R2", ctx_spot=120.0,
            ctx_g1_high=60.0, ctx_g2_high=50.0, ctx_both_high=30.0,
            # Grain coherence requires ctx_both_low <= min(ctx_g1_low,
            # ctx_g2_low); previous fixture value (15.0) exceeded
            # ctx_g2_low (14.0) — undetected before Commit B because
            # the per-hex loop bypassed grain-coherence.
            ctx_both_low=12.0, ctx_g1_low=18.0, ctx_g2_low=14.0,
            g1_enr=fcs_b[0], g2_enr=fcs_b[1],
            joint_high=fcs_b[2], joint_low=fcs_b[3],
        ),
    ]
    # Builder raises on delta > tolerance; test passes iff it does not raise.
    df = build_consensus_hexes_df(
        sample_consensus_result=_sample({"Ca": comp_a, "Cb": comp_b}),
        roi_consensus_enrichment_df=_roi_enrichment_df(roi_rows),
    )
    assert len(df) == 6
