"""Semantic contract tests for the cross-grain sensitivity producer.

Fixtures build a complete 14-row scope (source + tess + consensus
observations for every triple) so tests exercise the semantic clauses
in isolation. Fail-loud closure is exercised explicitly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.cross_grain_types import (
    Channel,
    ComparisonStatus,
    Sign,
    Statistic,
    Surface,
    UnitKind,
)
from tilescope.analyzers.parent_roi_moran_common import MORAN_DIRECTIONS
from tilescope.analyzers.roi_cross_grain_sensitivity import (
    CANONICAL_COLUMNS,
    CrossGrainSensitivityInputs,
    PointSurfaceObservation,
    TessellationSummaryObservation,
    build_roi_cross_grain_sensitivity_df,
    compute_effect,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus


SAMPLE = "S1"
G1_ID = "Neuron"
G2_ID = "AC"
ROI = "R1"

FINITE = EnrichmentRatioStatus.FINITE.value
SATURATION = EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value
UNAVAILABLE_ENRICHMENT = EnrichmentRatioStatus.ZERO_OVER_ZERO_CONTEXT.value


def _point(
    surface: Surface,
    statistic: Statistic,
    *,
    channel: Channel = Channel.NONE,
    moran_direction: str = "",
    raw_value: Optional[float] = 0.0,
    raw_status: str = FINITE,
) -> PointSurfaceObservation:
    is_enrichment = statistic == Statistic.ENRICHMENT
    ctx_units = 400.0 if is_enrichment else None
    ctx_kind: Optional[UnitKind]
    if not is_enrichment:
        ctx_kind = None
    elif surface == Surface.SOURCE_NATIVE:
        ctx_kind = UnitKind.SOURCE_SPOT
    elif surface == Surface.TESSELLATION_MEDIAN:
        ctx_kind = UnitKind.TILE
    else:
        ctx_kind = UnitKind.HEX
    if surface == Surface.SOURCE_NATIVE:
        target_kind = UnitKind.SOURCE_SPOT
    elif surface == Surface.TESSELLATION_MEDIAN:
        target_kind = UnitKind.TILE
    else:
        target_kind = UnitKind.HEX
    return PointSurfaceObservation(
        roi_id=ROI, surface=surface,
        statistic=statistic, channel=channel, moran_direction=moran_direction,
        raw_value=raw_value, raw_status=raw_status, undefined_reason="",
        n_target_units=100.0, n_context_units=ctx_units,
        target_unit_kind=target_kind, context_unit_kind=ctx_kind,
    )


def _tess(
    statistic: Statistic,
    *,
    channel: Channel = Channel.NONE,
    moran_direction: str = "",
    raw_summary: Optional[float] = 0.0,
    raw_status: str = FINITE,
    effect_median: Optional[float] = None,
    effect_iqr: Optional[float] = None,
    n_comparable_replicates: int = 10,
    n_planned_replicates: int = 10,
) -> TessellationSummaryObservation:
    is_enrichment = statistic == Statistic.ENRICHMENT
    ctx_kind = UnitKind.TILE if is_enrichment else None
    ctx_units = 400.5 if is_enrichment else None
    if effect_median is None:
        effect_median = compute_effect(raw_summary, raw_status, statistic)
    return TessellationSummaryObservation(
        roi_id=ROI,
        statistic=statistic, channel=channel, moran_direction=moran_direction,
        raw_summary=raw_summary, raw_status=raw_status, undefined_reason="",
        effect_median=effect_median, effect_iqr=effect_iqr,
        n_comparable_replicates=n_comparable_replicates,
        n_planned_replicates=n_planned_replicates,
        n_target_units=95.5, n_context_units=ctx_units,
        target_unit_kind=UnitKind.TILE, context_unit_kind=ctx_kind,
    )


def _all_triples() -> List[tuple]:
    """The seven canonical (statistic, channel, moran_direction) triples."""
    return [
        (Statistic.CORRELATION, Channel.NONE, ""),
        (Statistic.ENRICHMENT, Channel.G1, ""),
        (Statistic.ENRICHMENT, Channel.G2, ""),
        (Statistic.ENRICHMENT, Channel.JOINT_HIGH, ""),
        (Statistic.ENRICHMENT, Channel.JOINT_LOW, ""),
        (Statistic.MORAN, Channel.NONE, MORAN_DIRECTIONS[0]),
        (Statistic.MORAN, Channel.NONE, MORAN_DIRECTIONS[1]),
    ]


def _complete_inputs(
    *,
    source_overrides=None,
    tess_overrides=None,
    consensus_overrides=None,
) -> CrossGrainSensitivityInputs:
    """Build inputs with a valid finite default for every triple; caller
    supplies overrides for the rows a test cares about.

    Default raws:
    - correlation / Moran: raw=0.0 (finite)
    - enrichment: raw=1.0 (finite -> effect 0)
    """
    src_over = {k: v for k, v in (source_overrides or {}).items()}
    tess_over = {k: v for k, v in (tess_overrides or {}).items()}
    cons_over = {k: v for k, v in (consensus_overrides or {}).items()}
    sources = []
    tesses = []
    consensuses = []
    for stat, chan, direction in _all_triples():
        key = (stat, chan, direction)
        default_raw = 1.0 if stat == Statistic.ENRICHMENT else 0.0
        sources.append(src_over.pop(key, _point(
            Surface.SOURCE_NATIVE, stat, channel=chan, moran_direction=direction,
            raw_value=default_raw,
        )))
        tesses.append(tess_over.pop(key, _tess(
            stat, channel=chan, moran_direction=direction,
            raw_summary=default_raw,
        )))
        consensuses.append(cons_over.pop(key, _point(
            Surface.CONSENSUS_ENSEMBLE, stat, channel=chan, moran_direction=direction,
            raw_value=default_raw,
        )))
    assert not src_over, f"unused source overrides: {src_over!r}"
    assert not tess_over, f"unused tess overrides: {tess_over!r}"
    assert not cons_over, f"unused consensus overrides: {cons_over!r}"
    return CrossGrainSensitivityInputs(
        sample_id=SAMPLE, g1_id=G1_ID, g2_id=G2_ID, roi_ids=[ROI],
        source_observations=sources,
        tessellation_observations=tesses,
        consensus_observations=consensuses,
    )


def _row(df, target_surface: Surface, **filters):
    sub = df[df["target_surface"] == target_surface.value]
    for col, val in filters.items():
        sub = sub[sub[col] == val]
    assert len(sub) == 1, f"expected one row, got {len(sub)} for {filters!r}"
    return sub.iloc[0]


def test_identical_surfaces_yield_zero_delta_and_preserved_sign():
    key = (Statistic.CORRELATION, Channel.NONE, "")
    inputs = _complete_inputs(
        source_overrides={key: _point(Surface.SOURCE_NATIVE, Statistic.CORRELATION, raw_value=0.4)},
        tess_overrides={key: _tess(Statistic.CORRELATION, raw_summary=0.4, effect_median=0.4, effect_iqr=0.0)},
        consensus_overrides={key: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.CORRELATION, raw_value=0.4)},
    )
    df = build_roi_cross_grain_sensitivity_df(inputs)
    for surface in (Surface.TESSELLATION_MEDIAN, Surface.CONSENSUS_ENSEMBLE):
        row = _row(df, surface, statistic="correlation")
        assert row["delta_effect"] == pytest.approx(0.0)
        assert row["abs_delta_effect"] == pytest.approx(0.0)
        assert bool(row["sign_preserved"]) is True
        assert row["reference_sign"] == Sign.POSITIVE.value
        assert row["target_sign"] == Sign.POSITIVE.value
        assert row["comparison_status"] == ComparisonStatus.BOTH_COMPARABLE.value


def test_attenuation_reduces_magnitude_and_preserves_sign():
    key = (Statistic.CORRELATION, Channel.NONE, "")
    inputs = _complete_inputs(
        source_overrides={key: _point(Surface.SOURCE_NATIVE, Statistic.CORRELATION, raw_value=0.6)},
        tess_overrides={key: _tess(Statistic.CORRELATION, raw_summary=0.3, effect_median=0.3, effect_iqr=0.05)},
        consensus_overrides={key: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.CORRELATION, raw_value=0.5)},
    )
    df = build_roi_cross_grain_sensitivity_df(inputs)
    tess = _row(df, Surface.TESSELLATION_MEDIAN, statistic="correlation")
    cons = _row(df, Surface.CONSENSUS_ENSEMBLE, statistic="correlation")
    assert tess["delta_effect"] == pytest.approx(-0.3)
    assert tess["abs_delta_effect"] == pytest.approx(0.3)
    assert bool(tess["sign_preserved"]) is True
    assert cons["delta_effect"] == pytest.approx(-0.1)
    assert cons["abs_delta_effect"] == pytest.approx(0.1)
    assert bool(cons["sign_preserved"]) is True


def test_sign_reversal_with_both_effects_material():
    key = (Statistic.CORRELATION, Channel.NONE, "")
    inputs = _complete_inputs(
        source_overrides={key: _point(Surface.SOURCE_NATIVE, Statistic.CORRELATION, raw_value=0.5)},
        consensus_overrides={key: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.CORRELATION, raw_value=-0.4)},
    )
    df = build_roi_cross_grain_sensitivity_df(inputs)
    row = _row(df, Surface.CONSENSUS_ENSEMBLE, statistic="correlation")
    assert row["reference_sign"] == Sign.POSITIVE.value
    assert row["target_sign"] == Sign.NEGATIVE.value
    assert row["delta_effect"] == pytest.approx(-0.9)
    assert row["abs_delta_effect"] == pytest.approx(0.9)
    assert bool(row["sign_preserved"]) is False


def test_enrichment_null_centring_and_saturation():
    g1 = (Statistic.ENRICHMENT, Channel.G1, "")
    g2 = (Statistic.ENRICHMENT, Channel.G2, "")
    jh = (Statistic.ENRICHMENT, Channel.JOINT_HIGH, "")
    jl = (Statistic.ENRICHMENT, Channel.JOINT_LOW, "")
    src_over = {
        g1: _point(Surface.SOURCE_NATIVE, Statistic.ENRICHMENT, channel=Channel.G1, raw_value=1.0),
        g2: _point(Surface.SOURCE_NATIVE, Statistic.ENRICHMENT, channel=Channel.G2, raw_value=3.0),
        jh: _point(Surface.SOURCE_NATIVE, Statistic.ENRICHMENT, channel=Channel.JOINT_HIGH, raw_value=0.5),
        jl: _point(Surface.SOURCE_NATIVE, Statistic.ENRICHMENT, channel=Channel.JOINT_LOW,
                    raw_value=None, raw_status=SATURATION),
    }
    cons_over = {
        g1: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.ENRICHMENT, channel=Channel.G1, raw_value=1.0),
        g2: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.ENRICHMENT, channel=Channel.G2, raw_value=3.0),
        jh: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.ENRICHMENT, channel=Channel.JOINT_HIGH, raw_value=0.5),
        jl: _point(Surface.CONSENSUS_ENSEMBLE, Statistic.ENRICHMENT, channel=Channel.JOINT_LOW,
                    raw_value=None, raw_status=SATURATION),
    }
    df = build_roi_cross_grain_sensitivity_df(_complete_inputs(
        source_overrides=src_over, consensus_overrides=cons_over,
    ))
    r1 = _row(df, Surface.CONSENSUS_ENSEMBLE, statistic="enrichment", channel="g1")
    assert r1["reference_effect_value"] == pytest.approx(0.0)
    assert r1["reference_sign"] == Sign.ZERO.value

    r2 = _row(df, Surface.CONSENSUS_ENSEMBLE, statistic="enrichment", channel="g2")
    assert r2["reference_effect_value"] == pytest.approx(3.0 / 4.0 - 0.5)

    r3 = _row(df, Surface.CONSENSUS_ENSEMBLE, statistic="enrichment", channel="joint_high")
    assert r3["reference_effect_value"] == pytest.approx(0.5 / 1.5 - 0.5)

    r4 = _row(df, Surface.CONSENSUS_ENSEMBLE, statistic="enrichment", channel="joint_low")
    assert r4["reference_effect_value"] == pytest.approx(0.5)
    assert r4["target_effect_value"] == pytest.approx(0.5)
    assert r4["comparison_status"] == ComparisonStatus.BOTH_COMPARABLE.value


def test_typed_unavailable_comparison_variants():
    """All four ComparisonStatus states arise from explicitly-supplied
    unavailable observations (never from missing rows)."""
    key = (Statistic.CORRELATION, Channel.NONE, "")
    unavailable_source = _point(
        Surface.SOURCE_NATIVE, Statistic.CORRELATION,
        raw_value=None, raw_status="insufficient_source_spots",
    )
    unavailable_cons = _point(
        Surface.CONSENSUS_ENSEMBLE, Statistic.CORRELATION,
        raw_value=None, raw_status="insufficient_hex_support",
    )
    unavailable_tess = _tess(
        Statistic.CORRELATION, raw_summary=None, raw_status="zero_variance",
        effect_median=None, effect_iqr=None,
        n_comparable_replicates=0, n_planned_replicates=10,
    )
    finite_cons = _point(
        Surface.CONSENSUS_ENSEMBLE, Statistic.CORRELATION, raw_value=0.2,
    )
    finite_tess = _tess(
        Statistic.CORRELATION, raw_summary=0.15, effect_median=0.15, effect_iqr=0.02,
    )

    # target unavailable
    df = build_roi_cross_grain_sensitivity_df(_complete_inputs(
        tess_overrides={key: unavailable_tess},
        consensus_overrides={key: unavailable_cons},
    ))
    for surface in (Surface.TESSELLATION_MEDIAN, Surface.CONSENSUS_ENSEMBLE):
        row = _row(df, surface, statistic="correlation")
        assert row["comparison_status"] == ComparisonStatus.TARGET_UNAVAILABLE.value
        assert pd.isna(row["delta_effect"])
        assert pd.isna(row["abs_delta_effect"])
        assert pd.isna(row["sign_preserved"])
        assert row["reference_effect_value"] == pytest.approx(0.0)
        assert pd.isna(row["target_effect_value"])

    # reference unavailable
    df2 = build_roi_cross_grain_sensitivity_df(_complete_inputs(
        source_overrides={key: unavailable_source},
        tess_overrides={key: finite_tess},
        consensus_overrides={key: finite_cons},
    ))
    for surface in (Surface.TESSELLATION_MEDIAN, Surface.CONSENSUS_ENSEMBLE):
        row = _row(df2, surface, statistic="correlation")
        assert row["comparison_status"] == ComparisonStatus.REFERENCE_UNAVAILABLE.value
        assert pd.isna(row["reference_effect_value"])

    # both unavailable
    df3 = build_roi_cross_grain_sensitivity_df(_complete_inputs(
        source_overrides={key: unavailable_source},
        tess_overrides={key: unavailable_tess},
        consensus_overrides={key: unavailable_cons},
    ))
    for surface in (Surface.TESSELLATION_MEDIAN, Surface.CONSENSUS_ENSEMBLE):
        row = _row(df3, surface, statistic="correlation")
        assert row["comparison_status"] == ComparisonStatus.BOTH_UNAVAILABLE.value


def test_tessellation_summary_wired_through_verbatim():
    """The producer takes effect_median + effect_iqr + counts from the
    tessellation summary observation without recomputation, so the point
    estimate and spread describe the same replicate population.
    """
    key = (Statistic.ENRICHMENT, Channel.G1, "")
    replicate_raws = [0.5, 1.0, 3.0, 2.0]
    replicate_effects = [r / (1 + r) - 0.5 for r in replicate_raws]
    adapter_effect_median = float(np.median(replicate_effects))
    adapter_effect_iqr = float(
        np.quantile(replicate_effects, 0.75) - np.quantile(replicate_effects, 0.25)
    )
    adapter_raw_summary = float(np.median(replicate_raws))
    tess_obs = _tess(
        Statistic.ENRICHMENT, channel=Channel.G1,
        raw_summary=adapter_raw_summary,
        effect_median=adapter_effect_median,
        effect_iqr=adapter_effect_iqr,
        n_comparable_replicates=4, n_planned_replicates=4,
    )
    df = build_roi_cross_grain_sensitivity_df(_complete_inputs(
        tess_overrides={key: tess_obs},
    ))
    row = _row(df, Surface.TESSELLATION_MEDIAN, statistic="enrichment", channel="g1")
    assert row["target_effect_value"] == pytest.approx(adapter_effect_median)
    assert row["tessellation_effect_iqr"] == pytest.approx(adapter_effect_iqr)
    assert row["n_comparable_replicates"] == 4
    assert row["n_planned_replicates"] == 4
    assert row["target_raw_value"] == pytest.approx(adapter_raw_summary)


def test_fail_loud_closure_and_schema_invariants():
    """Missing observations raise; canonical column set is exact;
    row set is 14 x |roi_ids|; unique key spans g1_id/g2_id."""
    key = (Statistic.CORRELATION, Channel.NONE, "")
    inputs = _complete_inputs()
    consensus_missing = CrossGrainSensitivityInputs(
        sample_id=inputs.sample_id, g1_id=inputs.g1_id, g2_id=inputs.g2_id,
        roi_ids=inputs.roi_ids,
        source_observations=inputs.source_observations,
        tessellation_observations=inputs.tessellation_observations,
        consensus_observations=[
            obs for obs in inputs.consensus_observations
            if (obs.statistic, obs.channel, obs.moran_direction) != key
        ],
    )
    with pytest.raises(KeyError, match="consensus_ensemble"):
        build_roi_cross_grain_sensitivity_df(consensus_missing)

    source_missing = CrossGrainSensitivityInputs(
        sample_id=inputs.sample_id, g1_id=inputs.g1_id, g2_id=inputs.g2_id,
        roi_ids=inputs.roi_ids,
        source_observations=[
            obs for obs in inputs.source_observations
            if (obs.statistic, obs.channel, obs.moran_direction) != key
        ],
        tessellation_observations=inputs.tessellation_observations,
        consensus_observations=inputs.consensus_observations,
    )
    with pytest.raises(KeyError, match="source_native"):
        build_roi_cross_grain_sensitivity_df(source_missing)

    df = build_roi_cross_grain_sensitivity_df(inputs)
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert len(df) == 14
    key_cols = ["sample_id", "roi_id", "g1_id", "g2_id",
                "statistic", "channel", "moran_direction", "target_surface"]
    assert not df[key_cols].duplicated().any()
    assert (df["sample_id"] == SAMPLE).all()
    assert (df["g1_id"] == G1_ID).all()
    assert (df["g2_id"] == G2_ID).all()
    consensus_rows = df[df["target_surface"] == Surface.CONSENSUS_ENSEMBLE.value]
    for col in ("tessellation_effect_iqr", "n_comparable_replicates", "n_planned_replicates"):
        assert consensus_rows[col].isna().all()

    empty = CrossGrainSensitivityInputs(
        sample_id=SAMPLE, g1_id=G1_ID, g2_id=G2_ID, roi_ids=[],
        source_observations=[], tessellation_observations=[],
        consensus_observations=[],
    )
    empty_df = build_roi_cross_grain_sensitivity_df(empty)
    assert list(empty_df.columns) == list(CANONICAL_COLUMNS)
    assert len(empty_df) == 0


def test_compute_effect_raises_on_malformed_inputs():
    """The transform validates the raw/status combination: saturation
    outside enrichment, non-finite value under a finite status, and
    negative enrichment all raise."""
    with pytest.raises(ValueError, match="enrichment-only"):
        compute_effect(0.5, SATURATION, Statistic.CORRELATION)
    with pytest.raises(ValueError, match="enrichment-only"):
        compute_effect(0.5, SATURATION, Statistic.MORAN)
    with pytest.raises(ValueError, match="finite"):
        compute_effect(None, FINITE, Statistic.CORRELATION)
    with pytest.raises(ValueError, match="finite"):
        compute_effect(float("nan"), FINITE, Statistic.ENRICHMENT)
    with pytest.raises(ValueError, match="non-negative"):
        compute_effect(-0.1, FINITE, Statistic.ENRICHMENT)

    assert compute_effect(None, UNAVAILABLE_ENRICHMENT, Statistic.ENRICHMENT) is None
    assert compute_effect(None, "insufficient_spots", Statistic.CORRELATION) is None
