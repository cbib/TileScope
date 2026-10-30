"""ROI-grain tessellation bivariate Moran on per-tile density per replicate.

Per replicate: pool tile rows via the canonical `TessellationPoolingIndex`,
build region-blocked kNN adjacency on tile centroids, compute bivariate
Moran per direction, then reduce across *complete* contributing replicates
to a median + Q25 + Q75 spread. A replicate contributes iff it covers
every eligible sibling component (mirrors the consensus complete-support
gate one grain down); incomplete replicates are silently skipped. If no
replicate covers full support, the ROI emits NO_VALID_REPLICATES.
Replicates must also satisfy the common Moran adjacency-support gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely import wkt as _wkt

from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row, resolve_sample_id_from_typed_record,
)
from tilescope.analyzers.parent_roi_moran_common import (
    MIN_UNITS_FOR_MORAN, MORAN_DIRECTIONS, adjacency_support,
    all_adjacency_blocks_complete, block_ids_from_component_region_tuples,
    compute_both_directions,
    enforce_finite_coupling, has_sufficient_adjacency_support,
)
from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.analyzers.parent_roi_tessellation_common import (
    RoiReplicatePool, TessellationPoolingIndex,
)
from tilescope.cols import (
    ADJACENCY_SUPPORTED_FRACTION, COMPONENT_ID, DOMAIN_REGION_ID, G1_ID, G2_ID,
    MORAN_DIRECTION, MORAN_REASON, MORAN_VALUE,
    MORAN_VALUE_P25_ACROSS_REPLICATES, MORAN_VALUE_P75_ACROSS_REPLICATES,
    N_ADJACENCY_ENTRIES, N_ADJACENCY_SUPPORTED_UNITS,
    N_FINITE_REPLICATES, N_REPLICATES, N_UNITS,
    PARENT_MORAN_STATUS, ROI_ID, SAMPLE_ID, TILE_GEOMETRY_WKT,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import RoiTessellationMoranSchema
from tilescope.stats.spatial_autocorr import (
    UndefinedMoranError, build_region_blocked_knn_adjacency,
)


@dataclass(frozen=True)
class _ReplicateMoranOutcome:
    status: ParentMoranStatus
    value_a: Optional[float]
    value_b: Optional[float]
    n_units: int
    n_directed: int
    n_supported: int
    supported_fraction: float


def build_roi_tessellation_moran_df(
    *,
    pooling_index: TessellationPoolingIndex,
    tessellation_tiles_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
    k_neighbors: int,
    moran_enabled: bool,
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
    weight_cache: Optional[MutableMapping[tuple, csr_matrix]] = None,
) -> pd.DataFrame:
    """One row per source ROI × direction; median over complete replicates."""
    resolved_sample_id = resolve_sample_id_from_typed_record(
        pooling_index.sample_id, sample_id,
    )
    n_planned = pooling_index.n_replicates_planned
    roi_ids = sorted(
        support_status_df[ROI_ID].astype(str).unique(), key=natural_sort_key,
    )
    tiles = tessellation_tiles_df.reset_index(drop=True)

    rows: List[dict] = []
    for roi_id in roi_ids:
        lookup_parent_support_row(support_status_df, roi_id)
        support = pooling_index.support_by_roi[roi_id]
        n_eligible = len(support.eligible_component_ids)
        base = {
            SAMPLE_ID: resolved_sample_id,
            ROI_ID: str(roi_id),
            G1_ID: str(g1_id),
            G2_ID: str(g2_id),
            N_REPLICATES: n_planned,
            N_FINITE_REPLICATES: 0,
            N_UNITS: 0,
            N_ADJACENCY_ENTRIES: 0,
            N_ADJACENCY_SUPPORTED_UNITS: 0,
            ADJACENCY_SUPPORTED_FRACTION: 0.0,
        }
        if not moran_enabled:
            rows.extend(_summary_rows(
                base, ParentMoranStatus.MORAN_DISABLED, [], [],
            ))
            continue
        if n_eligible == 0:
            rows.extend(_summary_rows(
                base,
                ParentMoranStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT, [], [],
            ))
            continue

        per_direction: Dict[str, List[float]] = {
            MORAN_DIRECTIONS[0]: [], MORAN_DIRECTIONS[1]: [],
        }
        per_replicate_units: List[int] = []
        per_replicate_entries: List[int] = []
        per_replicate_supported: List[int] = []
        per_replicate_supported_fraction: List[float] = []
        attempted_outcomes: List[_ReplicateMoranOutcome] = []
        for replicate in range(n_planned):
            pool = pooling_index.pool_by_roi_replicate.get(
                (str(roi_id), int(replicate))
            )
            if pool is None or pool.n_contributing_components < n_eligible:
                # Incomplete-support replicate: skip, matches the
                # consensus INCOMPLETE_SUPPORT_COVERAGE gate one grain down.
                continue
            outcome = _try_replicate(
                pool, tiles, k_neighbors, pooling_index,
                weight_cache=weight_cache,
            )
            attempted_outcomes.append(outcome)
            if outcome.status is not ParentMoranStatus.FINITE:
                continue
            assert outcome.value_a is not None and outcome.value_b is not None
            per_direction[MORAN_DIRECTIONS[0]].append(outcome.value_a)
            per_direction[MORAN_DIRECTIONS[1]].append(outcome.value_b)
            per_replicate_units.append(outcome.n_units)
            per_replicate_entries.append(outcome.n_directed)
            per_replicate_supported.append(outcome.n_supported)
            per_replicate_supported_fraction.append(outcome.supported_fraction)

        if not per_replicate_units:
            status = ParentMoranStatus.NO_VALID_REPLICATES
            if attempted_outcomes and all(
                outcome.status
                is ParentMoranStatus.INSUFFICIENT_ADJACENCY_SUPPORT
                for outcome in attempted_outcomes
            ):
                status = ParentMoranStatus.INSUFFICIENT_ADJACENCY_SUPPORT
                base[N_UNITS] = int(np.median([
                    outcome.n_units for outcome in attempted_outcomes
                ]))
                base[N_ADJACENCY_ENTRIES] = int(np.median([
                    outcome.n_directed for outcome in attempted_outcomes
                ]))
                base[N_ADJACENCY_SUPPORTED_UNITS] = int(np.median([
                    outcome.n_supported for outcome in attempted_outcomes
                ]))
                base[ADJACENCY_SUPPORTED_FRACTION] = float(np.median([
                    outcome.supported_fraction for outcome in attempted_outcomes
                ]))
            elif attempted_outcomes and all(
                outcome.status is ParentMoranStatus.NO_SPATIAL_DISCRIMINATION
                for outcome in attempted_outcomes
            ):
                status = ParentMoranStatus.NO_SPATIAL_DISCRIMINATION
                base[N_UNITS] = int(np.median([
                    outcome.n_units for outcome in attempted_outcomes
                ]))
                base[N_ADJACENCY_ENTRIES] = int(np.median([
                    outcome.n_directed for outcome in attempted_outcomes
                ]))
                base[N_ADJACENCY_SUPPORTED_UNITS] = int(np.median([
                    outcome.n_supported for outcome in attempted_outcomes
                ]))
                base[ADJACENCY_SUPPORTED_FRACTION] = float(np.median([
                    outcome.supported_fraction for outcome in attempted_outcomes
                ]))
            rows.extend(_summary_rows(
                base, status, [], [],
            ))
            continue
        base[N_FINITE_REPLICATES] = len(per_replicate_units)
        base[N_UNITS] = int(np.median(per_replicate_units))
        base[N_ADJACENCY_ENTRIES] = int(np.median(per_replicate_entries))
        base[N_ADJACENCY_SUPPORTED_UNITS] = int(np.median(
            per_replicate_supported,
        ))
        base[ADJACENCY_SUPPORTED_FRACTION] = float(np.median(
            per_replicate_supported_fraction,
        ))
        rows.extend(_summary_rows(
            base, ParentMoranStatus.FINITE,
            per_direction[MORAN_DIRECTIONS[0]],
            per_direction[MORAN_DIRECTIONS[1]],
        ))

    frame = pd.DataFrame(rows, columns=list(RoiTessellationMoranSchema.columns))
    if frame.empty:
        return frame
    frame = RoiTessellationMoranSchema.validate(frame, lazy=True)
    enforce_finite_coupling(
        frame, "roi_tessellation_moran_df",
        extra_finite_columns=(
            MORAN_VALUE_P25_ACROSS_REPLICATES,
            MORAN_VALUE_P75_ACROSS_REPLICATES,
        ),
    )
    return frame


def _try_replicate(
    pool: RoiReplicatePool,
    tiles: pd.DataFrame,
    k_neighbors: int,
    pooling_index: TessellationPoolingIndex,
    *,
    weight_cache: Optional[MutableMapping[tuple, csr_matrix]] = None,
) -> _ReplicateMoranOutcome:
    """Compute both Moran directions and retain typed support diagnostics."""
    if not pool.tile_row_positions:
        return _failed_replicate(ParentMoranStatus.INSUFFICIENT_UNITS)
    positions = np.asarray(pool.tile_row_positions, dtype=np.int64)
    a_all, b_all = pooling_index.density_values(pool)
    finite = np.isfinite(a_all) & np.isfinite(b_all)
    n_units = int(finite.sum())
    if n_units < MIN_UNITS_FOR_MORAN:
        return _failed_replicate(
            ParentMoranStatus.INSUFFICIENT_UNITS, n_units=n_units,
        )
    positions_finite = positions[finite]
    a = a_all[finite]
    b = b_all[finite]
    slice_ = tiles.iloc[positions_finite].reset_index(drop=True)
    centers = _tile_centroids(slice_)
    blocks = block_ids_from_component_region_tuples(
        list(zip(
            slice_[COMPONENT_ID].astype(str),
            slice_[DOMAIN_REGION_ID].astype(int),
        ))
    )
    try:
        W = build_region_blocked_knn_adjacency(
            centers, blocks, k=k_neighbors, cache=weight_cache,
        )
    except UndefinedMoranError:
        return _failed_replicate(
            ParentMoranStatus.NO_ADJACENCY_EDGES, n_units=n_units,
        )
    n_directed = int(W.nnz)
    if n_directed == 0:
        return _failed_replicate(
            ParentMoranStatus.NO_ADJACENCY_EDGES,
            n_units=n_units, n_directed=n_directed,
        )
    n_supported, supported_fraction = adjacency_support(W)
    if all_adjacency_blocks_complete(W, blocks):
        return _failed_replicate(
            ParentMoranStatus.NO_SPATIAL_DISCRIMINATION,
            n_units=n_units, n_directed=n_directed,
            n_supported=n_supported, supported_fraction=supported_fraction,
        )
    if not has_sufficient_adjacency_support(
        n_supported, supported_fraction,
    ):
        return _failed_replicate(
            ParentMoranStatus.INSUFFICIENT_ADJACENCY_SUPPORT,
            n_units=n_units, n_directed=n_directed,
            n_supported=n_supported, supported_fraction=supported_fraction,
        )
    results = compute_both_directions(a, b, W)
    (value_a, status_a), (value_b, status_b) = results[0], results[1]
    if status_a is not ParentMoranStatus.FINITE or value_a is None:
        return _failed_replicate(
            status_a, n_units=n_units, n_directed=n_directed,
            n_supported=n_supported, supported_fraction=supported_fraction,
        )
    if status_b is not ParentMoranStatus.FINITE or value_b is None:
        return _failed_replicate(
            status_b, n_units=n_units, n_directed=n_directed,
            n_supported=n_supported, supported_fraction=supported_fraction,
        )
    return _ReplicateMoranOutcome(
        status=ParentMoranStatus.FINITE,
        value_a=float(value_a), value_b=float(value_b),
        n_units=n_units, n_directed=n_directed,
        n_supported=n_supported, supported_fraction=supported_fraction,
    )


def _failed_replicate(
    status: ParentMoranStatus,
    *,
    n_units: int = 0,
    n_directed: int = 0,
    n_supported: int = 0,
    supported_fraction: float = 0.0,
) -> _ReplicateMoranOutcome:
    return _ReplicateMoranOutcome(
        status=status, value_a=None, value_b=None,
        n_units=n_units, n_directed=n_directed,
        n_supported=n_supported, supported_fraction=supported_fraction,
    )


def _tile_centroids(slice_: pd.DataFrame) -> np.ndarray:
    if slice_.empty:
        return np.empty((0, 2), dtype=float)
    return np.array(
        [_wkt.loads(w).centroid.coords[0] for w in slice_[TILE_GEOMETRY_WKT]],
        dtype=float,
    )


def _summary_rows(
    base: dict, status: ParentMoranStatus,
    values_g1_lag_g2: Sequence[float],
    values_g2_lag_g1: Sequence[float],
) -> List[dict]:
    """Emit one row per direction with median + Q25 + Q75 summaries."""
    rows: List[dict] = []
    for direction, values in (
        (MORAN_DIRECTIONS[0], values_g1_lag_g2),
        (MORAN_DIRECTIONS[1], values_g2_lag_g1),
    ):
        row = dict(base)
        row[MORAN_DIRECTION] = direction
        if status is ParentMoranStatus.FINITE and values:
            arr = np.asarray(values, dtype=float)
            row[PARENT_MORAN_STATUS] = ParentMoranStatus.FINITE.value
            row[MORAN_VALUE] = float(np.median(arr))
            row[MORAN_VALUE_P25_ACROSS_REPLICATES] = float(
                np.quantile(arr, 0.25)
            )
            row[MORAN_VALUE_P75_ACROSS_REPLICATES] = float(
                np.quantile(arr, 0.75)
            )
            row[MORAN_REASON] = None
        else:
            row[PARENT_MORAN_STATUS] = status.value
            row[MORAN_VALUE] = float("nan")
            row[MORAN_VALUE_P25_ACROSS_REPLICATES] = float("nan")
            row[MORAN_VALUE_P75_ACROSS_REPLICATES] = float("nan")
            row[MORAN_REASON] = status.value
        rows.append(row)
    return rows
