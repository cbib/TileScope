"""Contract pins for the strict grain-neutral enrichment primitive."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.cols import (
    ENRICHMENT_RATIO_STATUS_COLUMNS,
    GENE1_DENSITY, GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS,
    GENE2_DENSITY, GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS,
    N_BOTH_HIGH, N_BOTH_LOW, N_G1_HIGH, N_G1_LOW, N_G2_HIGH, N_G2_LOW,
)
from tilescope.metrics.enrichment import (
    EnrichmentRatioStatus,
    compute_enrichment_ratios_from_counts,
)


def _coherent_counts() -> pd.DataFrame:
    # counts satisfy: each HIGH/LOW <= denominator; both <= min(g1, g2)
    return pd.DataFrame({
        N_G1_HIGH: [2, 3, 1, 4],
        N_G2_HIGH: [1, 2, 2, 3],
        N_BOTH_HIGH: [1, 1, 0, 2],
        N_G1_LOW: [1, 0, 2, 0],
        N_G2_LOW: [2, 1, 1, 0],
        N_BOTH_LOW: [1, 0, 1, 0],
    })


def _example_baselines() -> dict:
    return {
        'ctx_n_g1_high': 20, 'ctx_n_g2_high': 15, 'ctx_n_both_high': 8,
        'ctx_n_g1_low': 25, 'ctx_n_g2_low': 22, 'ctx_n_both_low': 10,
        'ctx_n_total': 100,
    }


_RATIO_COLS = (
    GENE1_ENRICHMENT, GENE2_ENRICHMENT,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
)


# ---------------------------------------------------------------------------
# 1. Side-effect-free + minimal-output-columns + index preservation
# ---------------------------------------------------------------------------


def test_primitive_returns_ratios_and_per_ratio_status_columns():
    counts = _coherent_counts()
    counts.index = pd.Index(["r0", "r1", "r2", "r3"], name="hex")
    counts_with_density = counts.copy()
    counts_with_density[GENE1_DENSITY] = -999.0
    counts_with_density[GENE2_DENSITY] = -888.0
    denominators = pd.Series([10, 12, 8, 15], index=counts.index)

    out = compute_enrichment_ratios_from_counts(
        counts=counts_with_density, denominators=denominators,
        context_baselines=_example_baselines(),
    )
    assert (counts_with_density[GENE1_DENSITY] == -999.0).all()
    assert (counts_with_density[GENE2_DENSITY] == -888.0).all()
    assert list(out.columns) == list(_RATIO_COLS) + list(ENRICHMENT_RATIO_STATUS_COLUMNS)
    assert out.index.equals(counts.index)
    for status_col in ENRICHMENT_RATIO_STATUS_COLUMNS:
        assert (out[status_col] == EnrichmentRatioStatus.FINITE.value).all()


# ---------------------------------------------------------------------------
# 2. Primitive validation — no SUM_G / n_obs_col required
# ---------------------------------------------------------------------------


def test_primitive_does_not_require_sum_or_denominator_columns():
    counts = _coherent_counts()
    denominators = pd.Series([10, 12, 8, 15])
    baselines = _example_baselines()

    out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=denominators,
        context_baselines=baselines,
    )
    assert len(out) == 4
    with pytest.raises(ValueError, match="missing"):
        compute_enrichment_ratios_from_counts(
            counts=counts.drop(columns=[N_G1_HIGH]),
            denominators=denominators, context_baselines=baselines,
        )
    incomplete = dict(baselines)
    del incomplete['ctx_n_total']
    with pytest.raises(ValueError, match="ctx_n_total"):
        compute_enrichment_ratios_from_counts(
            counts=counts, denominators=denominators,
            context_baselines=incomplete,
        )


# ---------------------------------------------------------------------------
# 3. Grain-coherence enforcement
# ---------------------------------------------------------------------------


def test_primitive_rejects_mixed_grain_inputs():
    # Legacy tessellation-level mixed grain: numerator carries
    # summed-across-tiles spot counts, denominator carries tile
    # count. Per-row counts can (and typically do) exceed the
    # denominator — the coherence check catches this.
    #
    # Row 0: 12 high spots aggregated across 3 tiles (denominator=3).
    # 12 > 3 → grain incoherent.
    mixed_counts = pd.DataFrame({
        N_G1_HIGH: [12, 5, 8, 4],
        N_G2_HIGH: [10, 4, 6, 3],
        N_BOTH_HIGH: [4, 2, 3, 1],
        N_G1_LOW: [3, 1, 2, 0],
        N_G2_LOW: [4, 2, 2, 0],
        N_BOTH_LOW: [1, 0, 1, 0],
    })
    tile_counts = pd.Series([3, 2, 3, 2])   # denominator == N_TILES
    with pytest.raises(ValueError, match="grain coherence"):
        compute_enrichment_ratios_from_counts(
            counts=mixed_counts, denominators=tile_counts,
            context_baselines=_example_baselines(),
        )


# ---------------------------------------------------------------------------
# 4. Undefined context + coherent spot/unit-grain sanity
# ---------------------------------------------------------------------------


_ZERO_KEY_TO_STATUS_COLUMNS = {
    'ctx_n_g1_high': (GENE1_ENRICHMENT_STATUS, JOINT_HIGH_ENRICHMENT_STATUS),
    'ctx_n_g2_high': (GENE2_ENRICHMENT_STATUS, JOINT_HIGH_ENRICHMENT_STATUS),
    'ctx_n_both_high': (JOINT_HIGH_ENRICHMENT_STATUS,),
    'ctx_n_both_low': (JOINT_LOW_ENRICHMENT_STATUS,),
}
_ZERO_KEY_TO_VALUE_COLUMNS = {
    'ctx_n_g1_high': (GENE1_ENRICHMENT, JOINT_HIGH_ENRICHMENT_FC),
    'ctx_n_g2_high': (GENE2_ENRICHMENT, JOINT_HIGH_ENRICHMENT_FC),
    'ctx_n_both_high': (JOINT_HIGH_ENRICHMENT_FC,),
    'ctx_n_both_low': (JOINT_LOW_ENRICHMENT_FC,),
}


@pytest.mark.parametrize(
    "zero_key",
    ['ctx_n_g1_high', 'ctx_n_g2_high', 'ctx_n_both_high', 'ctx_n_both_low'],
)
def test_primitive_marks_zero_context_ratios_and_nulls_only_their_values(
    zero_key,
):
    # Context proportion == 0 no longer raises: the affected per-ratio
    # status becomes positive_over_zero_context (numerator > 0) or
    # zero_over_zero_context, and only that ratio's numeric value is
    # NaN. Unaffected ratios remain finite.
    counts = _coherent_counts()
    denominators = pd.Series([10, 12, 8, 15])
    baselines = _example_baselines()
    baselines[zero_key] = 0
    if zero_key in ('ctx_n_g1_high', 'ctx_n_g2_high'):
        baselines['ctx_n_both_high'] = 0

    out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=denominators,
        context_baselines=baselines,
    )

    affected_status_cols = _ZERO_KEY_TO_STATUS_COLUMNS[zero_key]
    affected_value_cols = _ZERO_KEY_TO_VALUE_COLUMNS[zero_key]
    non_finite_statuses = {
        EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value,
        EnrichmentRatioStatus.ZERO_OVER_ZERO_CONTEXT.value,
    }
    for status_col in affected_status_cols:
        assert set(out[status_col].unique()) <= non_finite_statuses, (
            f"{status_col} should be non-finite under zero context"
        )
    for value_col in affected_value_cols:
        assert out[value_col].isna().all(), (
            f"{value_col} must be NaN when its per-ratio status is non-finite"
        )
    unaffected_status_cols = tuple(
        c for c in ENRICHMENT_RATIO_STATUS_COLUMNS if c not in affected_status_cols
    )
    for status_col in unaffected_status_cols:
        assert (out[status_col] == EnrichmentRatioStatus.FINITE.value).all()


def test_primitive_records_positive_vs_zero_over_zero_context_per_row():
    # Same zero-context baselines, but row 2 has numerator zero → its
    # status must be zero_over_zero_context while other rows carry
    # positive_over_zero_context.
    counts = _coherent_counts()
    counts.loc[2, N_G1_HIGH] = 0
    denominators = pd.Series([10, 12, 8, 15])
    baselines = _example_baselines()
    baselines['ctx_n_g1_high'] = 0
    baselines['ctx_n_both_high'] = 0

    out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=denominators,
        context_baselines=baselines,
    )
    statuses = out[GENE1_ENRICHMENT_STATUS].tolist()
    assert statuses[2] == EnrichmentRatioStatus.ZERO_OVER_ZERO_CONTEXT.value
    for i in (0, 1, 3):
        assert statuses[i] == (
            EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value
        )
    assert out[GENE1_ENRICHMENT].isna().all()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda b: b.update({'ctx_n_g1_high': -1}), "is negative"),
        (lambda b: b.update({'ctx_n_g1_high': b['ctx_n_total'] + 1}), "exceeds ctx_n_total"),
        (lambda b: b.update({'ctx_n_g1_high': float("nan")}), "finite"),
        (lambda b: b.update({'ctx_n_both_high': min(b['ctx_n_g1_high'],
                                                    b['ctx_n_g2_high']) + 1}),
         "exceeds min"),
    ],
)
def test_primitive_validates_context_baselines(mutate, match):
    # Context counts must be finite and in [0, ctx_n_total]; joint
    # counts must not exceed marginals. Any violation raises.
    counts = _coherent_counts()
    denominators = pd.Series([10, 12, 8, 15])
    baselines = _example_baselines()
    mutate(baselines)
    with pytest.raises(ValueError, match=match):
        compute_enrichment_ratios_from_counts(
            counts=counts, denominators=denominators,
            context_baselines=baselines,
        )


def test_primitive_is_grain_neutral_on_coherent_inputs():
    counts = _coherent_counts()
    baselines = _example_baselines()

    # Spot-grain: denominators are per-row spot counts.
    spot_denoms = pd.Series([10, 12, 8, 15])
    # Unit-grain: denominators are per-row unit counts. Coherent
    # variant — each count ≤ denominator holds.
    unit_denoms = pd.Series([5, 4, 3, 5])

    spot_out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=spot_denoms,
        context_baselines=baselines,
    )
    unit_out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=unit_denoms,
        context_baselines=baselines,
    )
    for col in (GENE1_ENRICHMENT, JOINT_HIGH_ENRICHMENT_FC):
        assert np.isfinite(spot_out[col]).all()
        assert np.isfinite(unit_out[col]).all()
        assert not np.allclose(spot_out[col], unit_out[col])


# ---------------------------------------------------------------------------
# 5. Float-drift tolerance (CORE-NUM-01)
# ---------------------------------------------------------------------------


def _coherent_counts_float() -> pd.DataFrame:
    """Coherent counts as float dtype so ULP-scale float assignments
    survive without pandas int-coercion warnings."""
    return _coherent_counts().astype(float)


def test_primitive_accepts_several_ulp_upper_bound_drift():
    """Sums-of-partial-sums (ensemble-weighted masses) can push
    n_g1_high a few ULPs above the denominator without any coherence
    breach. The 1e-9 tolerance absorbs that noise."""
    counts = _coherent_counts_float()
    denominators = pd.Series([10.0, 12.0, 8.0, 15.0])
    # Nudge one row's n_g1_high just past the denominator by 2e-15
    # (single-ULP scale at these magnitudes). Keep n_both_high at its
    # coherent value so this test isolates the denominator check.
    counts.loc[0, N_G1_HIGH] = float(denominators.iloc[0]) + 2e-15
    counts.loc[0, N_G2_HIGH] = float(denominators.iloc[0]) + 2e-15
    out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=denominators,
        context_baselines=_example_baselines(),
    )
    assert np.isfinite(out[GENE1_ENRICHMENT].iloc[0])


def test_primitive_rejects_upper_bound_breach_beyond_tolerance():
    """A breach larger than the 1e-9 tolerance still fails; the error
    reports the observed excess magnitude."""
    counts = _coherent_counts_float()
    denominators = pd.Series([10.0, 12.0, 8.0, 15.0])
    counts.loc[0, N_G1_HIGH] = float(denominators.iloc[0]) + 1e-6
    with pytest.raises(ValueError, match=r"exceeds denominator.*by up to 1\.000e-06"):
        compute_enrichment_ratios_from_counts(
            counts=counts, denominators=denominators,
            context_baselines=_example_baselines(),
        )


def test_primitive_accepts_several_ulp_joint_marginal_drift():
    """n_both_high a few ULPs above min(n_g1_high, n_g2_high) is
    absorbed as float noise, not a coherence breach."""
    counts = _coherent_counts_float()
    denominators = pd.Series([10.0, 12.0, 8.0, 15.0])
    marginal_min = min(counts.loc[0, N_G1_HIGH], counts.loc[0, N_G2_HIGH])
    counts.loc[0, N_BOTH_HIGH] = float(marginal_min) + 2e-15
    out = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=denominators,
        context_baselines=_example_baselines(),
    )
    assert np.isfinite(out[JOINT_HIGH_ENRICHMENT_FC].iloc[0])


def test_primitive_rejects_joint_marginal_breach_beyond_tolerance():
    counts = _coherent_counts_float()
    denominators = pd.Series([10.0, 12.0, 8.0, 15.0])
    marginal_min = min(counts.loc[0, N_G1_HIGH], counts.loc[0, N_G2_HIGH])
    counts.loc[0, N_BOTH_HIGH] = float(marginal_min) + 1e-6
    with pytest.raises(
        ValueError, match=r"n_both_high exceeds min.*by up to 1\.000e-06",
    ):
        compute_enrichment_ratios_from_counts(
            counts=counts, denominators=denominators,
            context_baselines=_example_baselines(),
        )
