"""Benjamini-Hochberg FDR primitives for the ROI-grain surfaces:
the raw ``benjamini_hochberg`` array function and the grouped
``apply_bh_within_families`` DataFrame operation. Both are pure.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from tilescope.cols import FDR_FAMILY_SIZE


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values for a family of p-values.

    NaN entries pass through as NaN and do not contribute to the
    family size. All finite entries must lie in [0, 1]; non-1D,
    non-finite (excluding NaN), negative, or > 1 inputs raise
    ``ValueError`` so upstream contract violations are not silently
    clipped.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError(
            f"benjamini_hochberg requires 1D input; got ndim={p.ndim}, "
            f"shape={p.shape}."
        )
    finite_mask = ~np.isnan(p)
    finite_p = p[finite_mask]
    if finite_p.size:
        bad_inf = ~np.isfinite(finite_p)
        if bad_inf.any():
            raise ValueError(
                "benjamini_hochberg received non-finite non-NaN p-values "
                f"(count={int(bad_inf.sum())}); refusing to silently clip. "
                f"Fix the upstream p-value producer."
            )
        bad_range = (finite_p < 0.0) | (finite_p > 1.0)
        if bad_range.any():
            raise ValueError(
                f"benjamini_hochberg received p-values outside [0, 1] "
                f"(count={int(bad_range.sum())}, "
                f"min={float(finite_p.min())}, max={float(finite_p.max())}); "
                f"refusing to silently clip. Fix the upstream p-value producer."
            )
    q = np.full_like(p, np.nan, dtype=float)
    n = int(finite_mask.sum())
    if n == 0:
        return q
    order = np.argsort(finite_p, kind="mergesort")
    ranks = np.arange(1, n + 1, dtype=float)
    sorted_p = finite_p[order]
    raw = sorted_p * n / ranks
    # BH step-up: q_i = min_{k >= i} raw_k, then clip to [0, 1].
    cum_min = np.minimum.accumulate(raw[::-1])[::-1]
    cum_min = np.clip(cum_min, 0.0, 1.0)
    q_sorted = np.empty(n, dtype=float)
    q_sorted[order] = cum_min
    q[finite_mask] = q_sorted
    return q


def apply_bh_within_families(
    df: pd.DataFrame,
    *,
    family_keys: Sequence[str],
    p_to_q_columns: Sequence[Tuple[str, str]],
    populated_mask: pd.Series,
    family_size_column: str = FDR_FAMILY_SIZE,
) -> pd.DataFrame:
    """Per-family BH q-value attachment.

    For each group defined by ``family_keys`` (canonical
    ``sort=False`` groupby order preserved), count rows where
    ``populated_mask`` is True, write that count to every row in the
    group under ``family_size_column``, and, when the count is > 0,
    call :func:`benjamini_hochberg` on each ``(p_col, q_col)`` pair
    over the populated subset, writing the q values back. Non-populated
    rows keep NaN q. Returns a copy; the input is never mutated.

    ``family_size_column`` is coerced to pandas ``Int64`` at the end
    so every emitted family-size value is nullable-int typed uniformly.
    """
    out = df.copy()
    # Order of column creation matters for CSV output. Add q columns
    # first (so they appear before family_size on frames that lack
    # both), then reset family_size; matches historical column-append
    # order.
    for _, q_col in p_to_q_columns:
        if q_col not in out.columns:
            out[q_col] = np.nan
    out[family_size_column] = pd.NA
    for _, idx in out.groupby(list(family_keys), sort=False).groups.items():
        group_idx = list(idx)
        group_populated = [i for i in group_idx if populated_mask.loc[i]]
        family_size = len(group_populated)
        for i in group_idx:
            out.at[i, family_size_column] = family_size
        if family_size == 0:
            continue
        for p_col, q_col in p_to_q_columns:
            p_vals = out.loc[group_populated, p_col].to_numpy(dtype=float)
            q_vals = benjamini_hochberg(p_vals)
            for pos, i in enumerate(group_populated):
                out.at[i, q_col] = float(q_vals[pos])
    out[family_size_column] = out[family_size_column].astype("Int64")
    return out
