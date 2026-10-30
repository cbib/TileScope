"""ROI-grain spot/cell bivariate Moran on the canonical spot/cell adjacency.

One row per source ROI × direction. Values are per-spot spot/cell expression
directly; adjacency is the sample-wide `SourceSpotAdjacency` restricted to
the ROI's spots (edges kept iff both endpoints belong to the ROI).
"""
from __future__ import annotations

from typing import List, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row, resolve_sample_id_from_typed_record,
)
from tilescope.analyzers.parent_roi_moran_common import (
    MIN_UNITS_FOR_MORAN, adjacency_support, compute_both_directions,
    emit_direction_rows, emit_uniform_rows, enforce_finite_coupling,
    has_sufficient_adjacency_support,
)
from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.cols import (
    ADJACENCY_SUPPORTED_FRACTION, G1_ID, G2_ID, N_ADJACENCY_ENTRIES,
    N_ADJACENCY_SUPPORTED_UNITS, N_FINITE_SPOTS, N_SOURCE_SPOTS, N_UNITS,
    ROI_ID, SAMPLE_ID,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import RoiSourceMoranSchema
from tilescope.spatial.source_spot_adjacency import SourceSpotAdjacency


def build_roi_source_moran_df(
    *,
    source_spot_adjacency: SourceSpotAdjacency,
    source_values_g1: np.ndarray,
    source_values_g2: np.ndarray,
    support_status_df: pd.DataFrame,
    sample_id: str,
    moran_enabled: bool,
    g1_id: str,
    g2_id: str,
    weight_cache: Optional[MutableMapping[tuple, csr_matrix]] = None,
) -> pd.DataFrame:
    """One row per source ROI × direction on spot/cell values + adjacency."""
    resolved_sample_id = resolve_sample_id_from_typed_record(str(sample_id), "")
    values_g1 = np.asarray(source_values_g1, dtype=float)
    values_g2 = np.asarray(source_values_g2, dtype=float)
    n_spots = source_spot_adjacency.n_spots
    if values_g1.shape != (n_spots,):
        raise ValueError(
            f"source_values_g1 length {values_g1.shape!r} disagrees with "
            f"adjacency n_spots={n_spots}."
        )
    if values_g2.shape != (n_spots,):
        raise ValueError(
            f"source_values_g2 length {values_g2.shape!r} disagrees with "
            f"adjacency n_spots={n_spots}."
        )
    roi_labels = np.asarray(source_spot_adjacency.source_spot_table.roi_id)
    roi_ids = sorted(
        support_status_df[ROI_ID].astype(str).unique(), key=natural_sort_key,
    )
    rows: List[dict] = []
    for roi_id in roi_ids:
        lookup_parent_support_row(support_status_df, roi_id)
        in_roi = roi_labels == roi_id
        row_positions = np.where(in_roi)[0]
        a_all = values_g1[row_positions]
        b_all = values_g2[row_positions]
        finite_mask = np.isfinite(a_all) & np.isfinite(b_all)
        finite_positions = row_positions[finite_mask]
        a = a_all[finite_mask]
        b = b_all[finite_mask]
        n_source = int(row_positions.size)
        n_finite = int(finite_positions.size)

        base = {
            SAMPLE_ID: resolved_sample_id,
            ROI_ID: str(roi_id),
            G1_ID: str(g1_id),
            G2_ID: str(g2_id),
            N_SOURCE_SPOTS: n_source,
            N_FINITE_SPOTS: n_finite,
            N_UNITS: n_finite,
            N_ADJACENCY_ENTRIES: 0,
            N_ADJACENCY_SUPPORTED_UNITS: 0,
            ADJACENCY_SUPPORTED_FRACTION: 0.0,
        }
        if not moran_enabled:
            rows.extend(emit_uniform_rows(
                base, ParentMoranStatus.MORAN_DISABLED,
            ))
            continue
        if n_finite < MIN_UNITS_FOR_MORAN:
            rows.extend(emit_uniform_rows(
                base, ParentMoranStatus.INSUFFICIENT_UNITS,
            ))
            continue

        W, n_directed = _row_standardized_subgraph(
            source_spot_adjacency, finite_positions, cache=weight_cache,
        )
        base[N_ADJACENCY_ENTRIES] = n_directed
        if n_directed == 0:
            rows.extend(emit_uniform_rows(
                base, ParentMoranStatus.NO_ADJACENCY_EDGES,
            ))
            continue
        n_supported, supported_fraction = adjacency_support(W)
        base[N_ADJACENCY_SUPPORTED_UNITS] = n_supported
        base[ADJACENCY_SUPPORTED_FRACTION] = supported_fraction
        if not has_sufficient_adjacency_support(
            n_supported, supported_fraction,
        ):
            rows.extend(emit_uniform_rows(
                base, ParentMoranStatus.INSUFFICIENT_ADJACENCY_SUPPORT,
            ))
            continue
        rows.extend(emit_direction_rows(
            base, compute_both_directions(a, b, W),
        ))

    frame = pd.DataFrame(rows, columns=list(RoiSourceMoranSchema.columns))
    if frame.empty:
        return frame
    frame = RoiSourceMoranSchema.validate(frame, lazy=True)
    enforce_finite_coupling(frame, "roi_source_moran_df")
    return frame


def _row_standardized_subgraph(
    adjacency: SourceSpotAdjacency,
    retained_positions: np.ndarray,
    *,
    cache: Optional[MutableMapping[tuple, csr_matrix]] = None,
) -> Tuple[csr_matrix, int]:
    """Row-standardized binary adjacency restricted to `retained_positions`.

    Sample-wide undirected edges become two directed entries (i→j, j→i)
    after symmetrization; `n_directed` returns the count of nonzero
    directed adjacency entries; the same convention every grain reports.
    """
    n = int(retained_positions.size)
    cache_key = (
        "source_subgraph", np.ascontiguousarray(retained_positions).tobytes(),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached, int(cached.nnz)
    empty = csr_matrix((n, n), dtype=np.float64)
    if n == 0 or adjacency.edges.shape[0] == 0:
        if cache is not None:
            cache[cache_key] = empty
        return empty, 0
    retained = set(int(p) for p in retained_positions.tolist())
    index_by_position = {int(p): i for i, p in enumerate(retained_positions)}
    rows = []
    cols = []
    for i_pos, j_pos in adjacency.edges:
        if int(i_pos) in retained and int(j_pos) in retained:
            i = index_by_position[int(i_pos)]
            j = index_by_position[int(j_pos)]
            rows.extend((i, j))
            cols.extend((j, i))
    if not rows:
        if cache is not None:
            cache[cache_key] = empty
        return empty, 0
    W = csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n),
    )
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums, dtype=float)
    positive = row_sums > 0
    inverse[positive] = 1.0 / row_sums[positive]
    W = (sparse.diags(inverse) @ W).tocsr()
    if cache is not None:
        cache[cache_key] = W
    return W, int(W.nnz)
