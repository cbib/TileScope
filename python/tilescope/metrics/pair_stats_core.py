"""
Scale-agnostic sufficient-statistic and distributional-effect engine.

Array-first design (no pandas in core) for crisp boundaries.
Wraps existing metrics functions to avoid duplication.
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import trim_mean, mstats

from tilescope.defaults import (
    DEFAULT_TRIM_PROPORTION, FC_SMALL_SAMPLE_THRESHOLD, FC_WINSORIZE_THRESHOLD,
    EPSILON
)


@dataclass(frozen=True)
class SufficientStats:
    """Sufficient statistics for pair-association analysis (scale-agnostic container).

    Field names use generic 'n_obs' (not N_SPOTS/N_TILES) to remain scale-agnostic.
    Mapping to column names happens in schema.sufficient_stats.sufficient_stats_row.
    """
    n_obs: int
    sum_g1: float
    sum_g2: float
    n_g1_high: int
    n_g2_high: int
    n_both_high: int
    n_g1_low: int
    n_g2_low: int
    n_both_low: int


class PairStatsCore:
    """Scale-agnostic pair-association statistical engine.

    Array-first design: operates on NumPy arrays only (no pandas in core).
    Adapters handle DataFrame→array conversion externally.

    NA Policy: Drop rows where either gene is NA.
    Constant operands are valid: threshold counts and distributional effects
    do not share correlation's variance requirement.
    """

    def sufficient_stats(self,
                        g1_vals: np.ndarray,
                        g2_vals: np.ndarray, *,
                        q_hi_g1: float,
                        q_hi_g2: float,
                        q_lo_g1: float,
                        q_lo_g2: float) -> SufficientStats:
        """Compute sufficient statistics for enrichment analysis."""
        if len(g1_vals) != len(g2_vals):
            raise ValueError(f"Gene arrays must have same length: {len(g1_vals)} != {len(g2_vals)}")

        # NA handling: Drop rows where either gene is NA
        valid_mask = ~(np.isnan(g1_vals) | np.isnan(g2_vals))
        g1_clean = g1_vals[valid_mask]
        g2_clean = g2_vals[valid_mask]

        if len(g1_clean) == 0:
            raise ValueError("All observations have NA values")

        # Compute counts
        n_obs = len(g1_clean)
        sum_g1 = float(g1_clean.sum())
        sum_g2 = float(g2_clean.sum())

        # High threshold counts
        g1_high_mask = g1_clean > q_hi_g1
        g2_high_mask = g2_clean > q_hi_g2
        n_g1_high = int(g1_high_mask.sum())
        n_g2_high = int(g2_high_mask.sum())
        n_both_high = int((g1_high_mask & g2_high_mask).sum())

        # Low threshold counts
        g1_low_mask = g1_clean < q_lo_g1
        g2_low_mask = g2_clean < q_lo_g2
        n_g1_low = int(g1_low_mask.sum())
        n_g2_low = int(g2_low_mask.sum())
        n_both_low = int((g1_low_mask & g2_low_mask).sum())

        return SufficientStats(
            n_obs=n_obs,
            sum_g1=sum_g1,
            sum_g2=sum_g2,
            n_g1_high=n_g1_high,
            n_g2_high=n_g2_high,
            n_both_high=n_both_high,
            n_g1_low=n_g1_low,
            n_g2_low=n_g2_low,
            n_both_low=n_both_low
        )

    def log2fc(self,
              roi_values: np.ndarray,
              context_values: np.ndarray,
              gene_min_value: float,
              eps: float = EPSILON) -> float:
        """Compute log2 fold change using sample-size adaptive robust estimators."""
        n_roi = len(roi_values)
        n_ctx = len(context_values)

        # Choose robust estimator based on smallest sample size
        n_min = min(n_roi, n_ctx)

        if n_min < FC_SMALL_SAMPLE_THRESHOLD:
            # Small sample: use median
            roi_est = np.median(roi_values)
            ctx_est = np.median(context_values)
        elif n_min < FC_WINSORIZE_THRESHOLD:
            # Borderline sample: use winsorized mean
            roi_est = mstats.winsorize(roi_values, limits=(0.1, 0.1)).mean()
            ctx_est = mstats.winsorize(context_values, limits=(0.1, 0.1)).mean()
        else:
            # Large sample: use trimmed mean
            roi_est = trim_mean(roi_values, DEFAULT_TRIM_PROPORTION)
            ctx_est = trim_mean(context_values, DEFAULT_TRIM_PROPORTION)

        # Compute shift parameter
        lam = max(eps, -float(gene_min_value) + eps)

        # Apply log2 transformation with shift
        try:
            log2fc = np.log2((roi_est + lam) / (ctx_est + lam))
            return log2fc
        except (ValueError, ZeroDivisionError):
            return np.nan

    def cles(self,
            roi_values: np.ndarray,
            context_values: np.ndarray) -> float:
        """Common Language Effect Size = P(X_roi > Y_context).

        Linear-memory implementation via sorted-context binary search:
        for each roi value r, ``searchsorted(sorted_ctx, r, side='left')``
        returns the count of context values strictly less than r (the
        ``side='left'`` boundary places equal values above the search
        point). Summing across r gives the same n_greater count as the
        ``roi[:, None] > context`` boolean matrix: exact integer
        equivalence, no quadratic allocation.

        NaN semantics preserve the legacy contract: any ``r > c``
        comparison involving a NaN operand is False, so NaN operands
        contribute zero to the numerator but STILL enter the
        denominator ``len(roi) * len(context)``. NaNs are stripped
        from both operands BEFORE the search (numpy sort would place
        NaN after all finites and searchsorted would then falsely
        count them as smaller than a NaN roi value); the original
        lengths are retained for the denominator. ±inf comparisons
        follow standard IEEE ordering and the finite-search path
        handles them correctly.
        """
        roi_arr = np.asarray(roi_values)
        ctx_arr = np.asarray(context_values)
        n_total = roi_arr.size * ctx_arr.size
        if n_total == 0:
            return 0.5
        finite_roi = roi_arr[~np.isnan(roi_arr)]
        finite_ctx = ctx_arr[~np.isnan(ctx_arr)]
        if finite_roi.size == 0 or finite_ctx.size == 0:
            # No pair can satisfy strict `>` when at least one side is
            # all-NaN. Denominator retained from the raw inputs.
            return 0.0 / n_total
        sorted_ctx = np.sort(finite_ctx)
        n_greater = int(np.searchsorted(
            sorted_ctx, finite_roi, side="left",
        ).sum())
        return n_greater / n_total
