"""Pure computational functions for enrichment analysis.

Per-ratio status contract: every enrichment ratio row carries both a
numeric value column and a status column keyed by
:class:`EnrichmentRatioStatus`. A ratio is ``finite`` iff its
specific context proportion is strictly positive; otherwise it is
``positive_over_zero_context`` (target proportion > 0) or
``zero_over_zero_context`` (target proportion also 0). Numeric value
is ``NaN`` when the status is either non-finite variant. Exact zero
remains exact: no tolerance, epsilon, or pseudocount.

Grain coherence + baseline validation still raise; those are
schema-level violations. A per-ratio non-finite status means
strictly "validated context proportion equals zero", never
"unknown".
"""

from enum import Enum

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from tilescope.cols import (
    GENE1_ENRICHMENT, GENE2_ENRICHMENT,
    GENE1_ENRICHMENT_STATUS, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
    JOINT_HIGH_ENRICHMENT_STATUS, JOINT_LOW_ENRICHMENT_STATUS,
    N_BOTH_HIGH, N_BOTH_LOW,
    N_G1_HIGH, N_G1_LOW, N_G2_HIGH, N_G2_LOW,
)


class EnrichmentRatioStatus(str, Enum):
    """Per-ratio outcome of the enrichment primitive.

    ``FINITE``: context proportion > 0; numeric value is the ratio.
    ``POSITIVE_OVER_ZERO_CONTEXT``: target count > 0 AND context total
    > 0 AND relevant context count == 0.
    ``ZERO_OVER_ZERO_CONTEXT``: target count == 0 AND context total
    > 0 AND relevant context count == 0.

    An empty context population (``ctx_n_total == 0``) is structural
    unavailability handled upstream in the analyzers; it never reaches
    this enum.
    """
    FINITE = "finite"
    POSITIVE_OVER_ZERO_CONTEXT = "positive_over_zero_context"
    ZERO_OVER_ZERO_CONTEXT = "zero_over_zero_context"


_REQUIRED_COUNT_COLS = [
    N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
    N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
]
_REQUIRED_BASELINE_KEYS = [
    'ctx_n_g1_high', 'ctx_n_g2_high', 'ctx_n_both_high',
    'ctx_n_g1_low', 'ctx_n_g2_low', 'ctx_n_both_low',
    'ctx_n_total',
]


def compute_enrichment_ratios_from_counts(
    *,
    counts: pd.DataFrame,
    denominators: pd.Series,
    context_baselines: Dict[str, Any],
) -> pd.DataFrame:
    """Grain-neutral enrichment ratios with per-ratio status.

    Numerator: per-row HIGH / LOW / BOTH counts (must be at the SAME
    grain as the denominator).

    Denominator: per-row observation count at the caller's grain.

    Context baselines: proportions from a same-grain context population.

    Grain coherence + baseline validation raise on violation (schema
    breaches). Per-ratio zero-context behaviour is NEVER an exception:

    - context proportion > 0            → ``finite``,        numeric = ratio
    - context proportion == 0, num > 0  → ``positive_over_zero_context``, NaN
    - context proportion == 0, num == 0 → ``zero_over_zero_context``,     NaN

    Exact zero remains exact: no epsilon-floor. A non-finite status
    means strictly "validated context proportion equals zero", never
    "unknown".

    Returns a NEW DataFrame with 8 columns (index preserved from
    ``counts``): four numeric ratios then four per-ratio statuses. Status
    columns use pandas string dtype and take values from
    :class:`EnrichmentRatioStatus`.
    """
    missing = [c for c in _REQUIRED_COUNT_COLS if c not in counts.columns]
    if missing:
        raise ValueError(
            f"compute_enrichment_ratios_from_counts: counts frame missing "
            f"required columns: {missing}"
        )
    for key in _REQUIRED_BASELINE_KEYS:
        if key not in context_baselines:
            raise ValueError(
                f"compute_enrichment_ratios_from_counts: context baselines "
                f"missing required key {key!r}"
            )
    if len(denominators) != len(counts):
        raise ValueError(
            f"compute_enrichment_ratios_from_counts: denominators length "
            f"({len(denominators)}) does not match counts frame "
            f"({len(counts)})"
        )

    _validate_grain_coherence(counts, denominators, context_baselines)
    _validate_context_baselines(context_baselines)

    ctx_total = float(context_baselines['ctx_n_total'])
    ctx_prop_g1_high = float(context_baselines['ctx_n_g1_high']) / ctx_total
    ctx_prop_g2_high = float(context_baselines['ctx_n_g2_high']) / ctx_total
    ctx_prop_both_high = float(context_baselines['ctx_n_both_high']) / ctx_total
    ctx_prop_both_low = float(context_baselines['ctx_n_both_low']) / ctx_total

    denom = denominators.to_numpy(dtype=float)
    g1_hi = counts[N_G1_HIGH].to_numpy(dtype=float)
    g2_hi = counts[N_G2_HIGH].to_numpy(dtype=float)
    both_hi = counts[N_BOTH_HIGH].to_numpy(dtype=float)
    both_lo = counts[N_BOTH_LOW].to_numpy(dtype=float)

    g1_value, g1_status = _per_ratio(g1_hi, denom, ctx_prop_g1_high)
    g2_value, g2_status = _per_ratio(g2_hi, denom, ctx_prop_g2_high)
    jh_value, jh_status = _per_ratio(both_hi, denom, ctx_prop_both_high)
    jl_value, jl_status = _per_ratio(both_lo, denom, ctx_prop_both_low)

    out = pd.DataFrame(index=counts.index)
    out[GENE1_ENRICHMENT] = g1_value
    out[GENE2_ENRICHMENT] = g2_value
    out[JOINT_HIGH_ENRICHMENT_FC] = jh_value
    out[JOINT_LOW_ENRICHMENT_FC] = jl_value
    out[GENE1_ENRICHMENT_STATUS] = pd.array(g1_status, dtype="string")
    out[GENE2_ENRICHMENT_STATUS] = pd.array(g2_status, dtype="string")
    out[JOINT_HIGH_ENRICHMENT_STATUS] = pd.array(jh_status, dtype="string")
    out[JOINT_LOW_ENRICHMENT_STATUS] = pd.array(jl_status, dtype="string")
    return out


def _per_ratio(
    numerator_counts: np.ndarray, denominators: np.ndarray,
    context_proportion: float,
) -> Tuple[np.ndarray, list]:
    """Return (value_array, status_list) per row for one ratio."""
    n = numerator_counts.shape[0]
    if context_proportion > 0.0:
        value = (numerator_counts / denominators) / context_proportion
        status = [EnrichmentRatioStatus.FINITE.value] * n
        return value, status
    value = np.full(n, np.nan, dtype=float)
    status = [
        EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value
        if numerator_counts[i] > 0.0
        else EnrichmentRatioStatus.ZERO_OVER_ZERO_CONTEXT.value
        for i in range(n)
    ]
    return value, status


# Upper-bound grain-coherence checks tolerate several-ULP float drift.
# Sums-of-partial-sums (ensemble-weighted hex masses) can carry
# last-bit excesses that a strict `>` treats as a coherence breach.
# The surrounding consensus contracts already tolerate 1e-9; matching
# that bound here preserves catch-real-breach behaviour while
# accepting the harmless float noise. Masses / ratios are never clipped.
_GRAIN_COHERENCE_ABS_TOL = 1e-9


def aggregate_context_baselines_from_counts(
    counts: pd.DataFrame,
    denominators: pd.Series,
) -> Dict[str, Any]:
    """Grain-neutral summation of per-row HIGH / LOW / BOTH count
    primitives + denominators into the context-baselines dict shape
    that :func:`compute_enrichment_ratios_from_counts` expects.

    Validates the per-row inputs via :func:`_validate_grain_coherence`
    on a stub baselines dict (every row must be finite / non-negative
    / joint <= marginal), then sums each count column and the
    denominators into the canonical ``ctx_n_*`` + ``ctx_n_total``
    dict.

    Used by:
      - the unit-enrichment orchestrator to build per-target context
        baselines from a set of context units' primitives;
      - single-row aggregation call sites (e.g. the partition
        coordinator) as a substitute for hand-written dict
        construction, standardising the boundary contract.
    """
    ctx_total = float(np.asarray(denominators, dtype=float).sum())
    # Reuse the strict per-row validator; ctx_n_total on the stub
    # matches the sum so the row-level "count <= denominator" checks
    # are meaningful and the stub itself passes the total > 0 gate.
    if ctx_total <= 0.0:
        raise ValueError(
            "aggregate_context_baselines_from_counts: denominators sum "
            f"must be positive; got {ctx_total!r}"
        )
    stub_baselines = {'ctx_n_total': ctx_total}
    _validate_grain_coherence(counts, denominators, stub_baselines)
    return {
        'ctx_n_g1_high': float(counts[N_G1_HIGH].sum()),
        'ctx_n_g2_high': float(counts[N_G2_HIGH].sum()),
        'ctx_n_both_high': float(counts[N_BOTH_HIGH].sum()),
        'ctx_n_g1_low': float(counts[N_G1_LOW].sum()),
        'ctx_n_g2_low': float(counts[N_G2_LOW].sum()),
        'ctx_n_both_low': float(counts[N_BOTH_LOW].sum()),
        'ctx_n_total': ctx_total,
    }


def _validate_context_baselines(context_baselines: Dict[str, Any]) -> None:
    """Every context count finite and within [0, ctx_n_total]; joint
    context counts must not exceed the corresponding marginals. Applies
    to the strict primitive only.
    """
    ctx_total = context_baselines['ctx_n_total']
    for key in (
        'ctx_n_g1_high', 'ctx_n_g2_high', 'ctx_n_both_high',
        'ctx_n_g1_low', 'ctx_n_g2_low', 'ctx_n_both_low',
    ):
        value = context_baselines[key]
        if not np.isfinite(value):
            raise ValueError(
                f"grain coherence: context baseline {key!r} must be finite; "
                f"got {value!r}"
            )
        if value < 0:
            raise ValueError(
                f"grain coherence: context baseline {key!r}={value!r} is "
                f"negative"
            )
        if value - ctx_total > _GRAIN_COHERENCE_ABS_TOL:
            raise ValueError(
                f"grain coherence: context baseline {key!r}={value!r} exceeds "
                f"ctx_n_total={ctx_total!r} by {value - ctx_total:.3e} "
                f"(tolerance {_GRAIN_COHERENCE_ABS_TOL:.0e})"
            )
    excess = context_baselines['ctx_n_both_high'] - min(
        context_baselines['ctx_n_g1_high'],
        context_baselines['ctx_n_g2_high'],
    )
    if excess > _GRAIN_COHERENCE_ABS_TOL:
        raise ValueError(
            f"grain coherence: ctx_n_both_high exceeds "
            f"min(ctx_n_g1_high, ctx_n_g2_high) by {excess:.3e} "
            f"(tolerance {_GRAIN_COHERENCE_ABS_TOL:.0e})"
        )
    excess = context_baselines['ctx_n_both_low'] - min(
        context_baselines['ctx_n_g1_low'],
        context_baselines['ctx_n_g2_low'],
    )
    if excess > _GRAIN_COHERENCE_ABS_TOL:
        raise ValueError(
            f"grain coherence: ctx_n_both_low exceeds "
            f"min(ctx_n_g1_low, ctx_n_g2_low) by {excess:.3e} "
            f"(tolerance {_GRAIN_COHERENCE_ABS_TOL:.0e})"
        )


def _validate_grain_coherence(
    counts: pd.DataFrame,
    denominators: pd.Series,
    context_baselines: Dict[str, Any],
) -> None:
    """Reject any incoherent-grain input to the primitive."""
    ctx_total = context_baselines['ctx_n_total']
    if not (np.isfinite(ctx_total) and ctx_total > 0):
        raise ValueError(
            f"grain coherence: ctx_n_total must be finite and positive; "
            f"got {ctx_total!r}"
        )
    denom_arr = np.asarray(denominators, dtype=float)
    if not (np.isfinite(denom_arr).all() and (denom_arr > 0).all()):
        raise ValueError(
            "grain coherence: denominators must be finite and positive "
            "for every row"
        )
    for col in _REQUIRED_COUNT_COLS:
        vals = np.asarray(counts[col], dtype=float)
        if not (np.isfinite(vals).all() and (vals >= 0).all()):
            raise ValueError(
                f"grain coherence: {col} must be finite and non-negative "
                f"for every row"
            )
    for col in (N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
                N_G1_LOW, N_G2_LOW, N_BOTH_LOW):
        excess = np.asarray(counts[col], dtype=float) - denom_arr
        max_excess = float(excess.max()) if excess.size else 0.0
        if max_excess > _GRAIN_COHERENCE_ABS_TOL:
            raise ValueError(
                f"grain coherence: {col} exceeds denominator on at least "
                f"one row by up to {max_excess:.3e} (tolerance "
                f"{_GRAIN_COHERENCE_ABS_TOL:.0e}) — counts and denominator "
                f"are not at the same grain (e.g. spot-count numerator with "
                f"unit-count denominator)"
            )
    marginal_high = np.minimum(
        np.asarray(counts[N_G1_HIGH], dtype=float),
        np.asarray(counts[N_G2_HIGH], dtype=float),
    )
    excess = np.asarray(counts[N_BOTH_HIGH], dtype=float) - marginal_high
    max_excess = float(excess.max()) if excess.size else 0.0
    if max_excess > _GRAIN_COHERENCE_ABS_TOL:
        raise ValueError(
            f"grain coherence: {N_BOTH_HIGH} exceeds min({N_G1_HIGH}, "
            f"{N_G2_HIGH}) on at least one row by up to {max_excess:.3e} "
            f"(tolerance {_GRAIN_COHERENCE_ABS_TOL:.0e})"
        )
    marginal_low = np.minimum(
        np.asarray(counts[N_G1_LOW], dtype=float),
        np.asarray(counts[N_G2_LOW], dtype=float),
    )
    excess = np.asarray(counts[N_BOTH_LOW], dtype=float) - marginal_low
    max_excess = float(excess.max()) if excess.size else 0.0
    if max_excess > _GRAIN_COHERENCE_ABS_TOL:
        raise ValueError(
            f"grain coherence: {N_BOTH_LOW} exceeds min({N_G1_LOW}, "
            f"{N_G2_LOW}) on at least one row by up to {max_excess:.3e} "
            f"(tolerance {_GRAIN_COHERENCE_ABS_TOL:.0e})"
        )
