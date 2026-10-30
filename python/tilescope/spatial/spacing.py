"""Canonical spatial-spacing primitive: median nearest-neighbour distance.

Single source of truth for the "typical inter-spot spacing" scalar consumed
by tessellation d1, canvas pitch, splat radius, support-overlay ring size,
and support-eligibility graph threshold. Callers pass the point set they
actually mean: dedup (or don't) at the call site.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def median_nn_distance(points_xy: np.ndarray) -> float:
    """Median of every point's Euclidean distance to its nearest other point.

    `points_xy` must be a finite float array of shape (N, 2). Returns NaN
    for N < 2 (nearest-neighbour is undefined). Malformed shape or
    non-finite values raise ValueError.
    """
    arr = np.asarray(points_xy)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"median_nn_distance expects shape (N, 2); got {arr.shape!r}."
        )
    if arr.size and not np.isfinite(arr).all():
        raise ValueError("median_nn_distance rejects non-finite coordinates.")
    if arr.shape[0] < 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2).fit(arr)
    distances, _ = nn.kneighbors(arr)
    return float(np.median(distances[:, 1]))


def median_nn_distance_from_df(
    df: pd.DataFrame, x_col: str, y_col: str,
) -> Optional[float]:
    """DataFrame convenience. Returns None on empty / <2-row input."""
    if df is None or len(df) < 2:
        return None
    coords = np.column_stack([df[x_col].to_numpy(dtype=float),
                              df[y_col].to_numpy(dtype=float)])
    return median_nn_distance(coords)
