"""Shared primitives for the three ROI-grain bivariate Moran analyzers.

Centralises direction naming, per-direction compute + zero-variance
translation, region-blocked block-id assembly, and finite-row coupling
enforcement. Grain-specific data extraction and adjacency construction
stay in the per-grain analyzers.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import spmatrix

from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.cols import (
    MORAN_DIRECTION, MORAN_REASON, MORAN_VALUE, PARENT_MORAN_STATUS,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import assert_columns_populated_iff
from tilescope.stats.spatial_autocorr import (
    UndefinedMoranError, UndefinedMoranReason, bivariate_morans_i,
)


MORAN_DIRECTIONS: Tuple[str, str] = ("g1_lag_g2", "g2_lag_g1")

MIN_UNITS_FOR_MORAN = 3
MIN_ADJACENCY_SUPPORTED_FRACTION = 0.5


def adjacency_support(W: np.ndarray | spmatrix) -> Tuple[int, float]:
    """Return units with a positive spatial lag row and their fraction."""
    if sparse.issparse(W):
        shape = W.shape
        row_sums = np.asarray(W.sum(axis=1)).ravel()
    else:
        weights = np.asarray(W, dtype=float)
        shape = weights.shape
        row_sums = weights.sum(axis=1) if weights.ndim == 2 else np.array([])
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"W must be square; got shape {shape!r}")
    n_units = int(shape[0])
    n_supported = int(np.count_nonzero(row_sums > 0))
    fraction = n_supported / n_units if n_units else 0.0
    return n_supported, float(fraction)


def has_sufficient_adjacency_support(
    n_supported: int, fraction_supported: float,
) -> bool:
    """Common Moran eligibility gate for every technology and grain."""
    return (
        n_supported >= MIN_UNITS_FOR_MORAN
        and fraction_supported >= MIN_ADJACENCY_SUPPORTED_FRACTION
    )


def all_adjacency_blocks_complete(W, block_ids: np.ndarray) -> bool:
    """Whether every region block connects every unit to every other.

    Such a graph is independent of within-block coordinates and cannot
    distinguish near from far spatial organization.
    """
    blocks = np.asarray(block_ids)
    if blocks.ndim != 1 or W.shape != (blocks.size, blocks.size):
        raise ValueError(
            f"W shape {W.shape!r} and block_ids shape {blocks.shape!r} "
            "must describe the same square graph."
        )
    if blocks.size < 2:
        return False
    expected_entries = 0
    observed_within_entries = 0
    for block_id in np.unique(blocks):
        positions = np.flatnonzero(blocks == block_id)
        n_block = int(positions.size)
        expected_entries += n_block * (n_block - 1)
        if sparse.issparse(W):
            observed_within_entries += int(W[positions][:, positions].nnz)
        else:
            observed_within_entries += int(np.count_nonzero(
                np.asarray(W)[np.ix_(positions, positions)],
            ))
    observed_total = (
        int(W.nnz) if sparse.issparse(W) else int(np.count_nonzero(W))
    )
    return (
        expected_entries > 0
        and observed_within_entries == expected_entries
        and observed_total == expected_entries
    )


def block_ids_from_component_region_tuples(
    tuples: Sequence[Tuple[str, int]],
) -> np.ndarray:
    """Contiguous integer block ids under natural (component, region) order."""
    unique = sorted(
        set(tuples),
        key=lambda item: (natural_sort_key(item[0]), int(item[1])),
    )
    lookup = {item: index for index, item in enumerate(unique)}
    return np.asarray([lookup[item] for item in tuples], dtype=int)


def translate_zero_variance(
    reason: UndefinedMoranReason, direction: str,
) -> ParentMoranStatus:
    """Map an underlying UndefinedMoranReason to biological-A/B status.

    Direction 1 (`g1_lag_g2`) passes X=A, Y=B; direction 2 passes X=B,
    Y=A. ZERO_VARIANCE_X therefore names the *first argument*, whose
    biological identity depends on the direction being computed.
    """
    if reason is UndefinedMoranReason.BOTH_ZERO_VARIANCE:
        return ParentMoranStatus.ZERO_VARIANCE_A_AND_B
    if reason in (
        UndefinedMoranReason.INSUFFICIENT_WEIGHTS,
        UndefinedMoranReason.INSUFFICIENT_WEIGHTS_IN_REGION,
    ):
        return ParentMoranStatus.NO_ADJACENCY_EDGES
    x_is_a = direction == MORAN_DIRECTIONS[0]
    if reason is UndefinedMoranReason.ZERO_VARIANCE_X:
        return (
            ParentMoranStatus.ZERO_VARIANCE_A if x_is_a
            else ParentMoranStatus.ZERO_VARIANCE_B
        )
    if reason is UndefinedMoranReason.ZERO_VARIANCE_Y:
        return (
            ParentMoranStatus.ZERO_VARIANCE_B if x_is_a
            else ParentMoranStatus.ZERO_VARIANCE_A
        )
    return ParentMoranStatus.UNDEFINED


def compute_both_directions(
    a: np.ndarray, b: np.ndarray, W: np.ndarray | spmatrix,
) -> List[Tuple[Optional[float], ParentMoranStatus]]:
    """Try both directions; return `(value_or_none, status)` per direction."""
    results: List[Tuple[Optional[float], ParentMoranStatus]] = []
    for direction, x, y in (
        (MORAN_DIRECTIONS[0], a, b),
        (MORAN_DIRECTIONS[1], b, a),
    ):
        try:
            value, _ = bivariate_morans_i(x, y, W, permutations=0, seed=0)
        except UndefinedMoranError as exc:
            results.append((None, translate_zero_variance(exc.reason, direction)))
        else:
            results.append((float(value), ParentMoranStatus.FINITE))
    return results


def emit_uniform_rows(
    base: dict, status: ParentMoranStatus,
) -> List[dict]:
    """Emit one non-finite row per direction with the same failure status."""
    return [_row_from(base, d, status, None) for d in MORAN_DIRECTIONS]


def emit_direction_rows(
    base: dict,
    per_direction: Sequence[Tuple[Optional[float], ParentMoranStatus]],
) -> List[dict]:
    """Emit one row per direction from per-direction (value, status) results."""
    if len(per_direction) != len(MORAN_DIRECTIONS):
        raise ValueError(
            f"per_direction must have {len(MORAN_DIRECTIONS)} entries; "
            f"got {len(per_direction)}"
        )
    return [
        _row_from(base, direction, status, value)
        for direction, (value, status) in zip(MORAN_DIRECTIONS, per_direction)
    ]


def _row_from(
    base: dict, direction: str, status: ParentMoranStatus,
    value: Optional[float],
) -> dict:
    row = dict(base)
    row[MORAN_DIRECTION] = direction
    row[PARENT_MORAN_STATUS] = status.value
    if status is ParentMoranStatus.FINITE and value is not None:
        row[MORAN_VALUE] = float(value)
        row[MORAN_REASON] = None
    else:
        row[MORAN_VALUE] = float("nan")
        row[MORAN_REASON] = status.value
    return row


def enforce_finite_coupling(
    frame: pd.DataFrame,
    surface_name: str,
    extra_finite_columns: Sequence[str] = (),
) -> None:
    """Coupling: MORAN_VALUE (+ optional extras) populated iff status is FINITE."""
    finite = frame[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value
    coupled = (MORAN_VALUE,) + tuple(extra_finite_columns)
    assert_columns_populated_iff(
        frame,
        coupled,
        finite,
        surface_name=surface_name,
        condition_label="parent_moran_status == 'finite'",
    )
    if not frame.loc[finite, MORAN_REASON].isna().all():
        raise ValueError(
            f"{surface_name}: FINITE row(s) carry a non-null moran_reason."
        )


__all__ = [
    "MIN_ADJACENCY_SUPPORTED_FRACTION",
    "MIN_UNITS_FOR_MORAN",
    "MORAN_DIRECTIONS",
    "adjacency_support",
    "all_adjacency_blocks_complete",
    "block_ids_from_component_region_tuples",
    "compute_both_directions",
    "emit_direction_rows",
    "emit_uniform_rows",
    "enforce_finite_coupling",
    "has_sufficient_adjacency_support",
    "translate_zero_variance",
]
