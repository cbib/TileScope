"""ROI-grain consensus bivariate Moran on per-hex densities per surface.

Adjacency: region-blocked kNN on hex centres keyed by (component_id,
domain_region_id). Values are the per-hex densities (ensemble + direct).
Emits one row per source ROI × surface × direction. Complete-support
gate mirrors `roi_consensus_correlations`: any surviving-but-incomplete
sibling coverage suppresses Moran with INCOMPLETE_SUPPORT_COVERAGE.
"""
from __future__ import annotations

from typing import List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from tilescope.analyzers.consensus_frames import ConsensusAnalysisFrames
from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row, resolve_sample_id_from_typed_record,
)
from tilescope.analyzers.parent_roi_moran_common import (
    MIN_UNITS_FOR_MORAN, adjacency_support, all_adjacency_blocks_complete,
    block_ids_from_component_region_tuples,
    compute_both_directions, emit_direction_rows, emit_uniform_rows,
    enforce_finite_coupling, has_sufficient_adjacency_support,
)
from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.cols import (
    ADJACENCY_SUPPORTED_FRACTION, CENTER_X, CENTER_Y,
    COMPONENT_ID, DOMAIN_REGION_ID,
    G1_ID, G2_ID,
    GENE1_DENSITY_DIRECT, GENE1_DENSITY_ENSEMBLE,
    GENE2_DENSITY_DIRECT, GENE2_DENSITY_ENSEMBLE,
    N_ADJACENCY_ENTRIES, N_ADJACENCY_SUPPORTED_UNITS,
    N_CONTRIBUTING_COMPONENTS_CONSENSUS,
    N_ELIGIBLE_CONSENSUS_COMPONENTS,
    N_HEXES_POOLED, N_HEXES_WITH_FINITE_DENSITY_POOLED,
    N_UNITS, ROI_ID, SAMPLE_ID, VALUE_SURFACE,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import RoiConsensusMoranSchema
from tilescope.stats.spatial_autocorr import (
    UndefinedMoranError, build_region_blocked_knn_adjacency,
)


_SURFACE_DENSITY_COLUMNS = {
    PartitionValueSurface.CONSENSUS_ENSEMBLE: (
        GENE1_DENSITY_ENSEMBLE, GENE2_DENSITY_ENSEMBLE,
    ),
    PartitionValueSurface.CONSENSUS_DIRECT: (
        GENE1_DENSITY_DIRECT, GENE2_DENSITY_DIRECT,
    ),
}


def build_roi_consensus_moran_df(
    *,
    consensus_frames: ConsensusAnalysisFrames,
    support_status_df: pd.DataFrame,
    k_neighbors: int,
    moran_enabled: bool,
    surfaces: Sequence[PartitionValueSurface] = (
        PartitionValueSurface.CONSENSUS_ENSEMBLE,
        PartitionValueSurface.CONSENSUS_DIRECT,
    ),
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
    weight_cache: Optional[MutableMapping[tuple, csr_matrix]] = None,
) -> pd.DataFrame:
    """One row per source ROI × surface × direction on hex-density arrays."""
    resolved_sample_id = resolve_sample_id_from_typed_record(
        consensus_frames.sample_id, sample_id,
    )
    roi_ids = sorted(
        support_status_df[ROI_ID].astype(str).unique(), key=natural_sort_key,
    )
    rows: List[dict] = []
    for roi_id in roi_ids:
        lookup_parent_support_row(support_status_df, roi_id)
        for surface in surfaces:
            support = consensus_frames.support(roi_id, surface)
            base = {
                SAMPLE_ID: resolved_sample_id,
                ROI_ID: str(roi_id),
                VALUE_SURFACE: surface.value,
                G1_ID: str(g1_id),
                G2_ID: str(g2_id),
                N_HEXES_POOLED: support.n_hexes_constructed,
                N_HEXES_WITH_FINITE_DENSITY_POOLED: (
                    support.n_hexes_with_finite_density
                ),
                N_ELIGIBLE_CONSENSUS_COMPONENTS: len(
                    support.eligible_component_ids,
                ),
                N_CONTRIBUTING_COMPONENTS_CONSENSUS: len(
                    support.contributing_component_ids,
                ),
                N_UNITS: 0,
                N_ADJACENCY_ENTRIES: 0,
                N_ADJACENCY_SUPPORTED_UNITS: 0,
                ADJACENCY_SUPPORTED_FRACTION: 0.0,
            }
            if not moran_enabled:
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.MORAN_DISABLED,
                ))
                continue
            if not support.eligible_component_ids:
                rows.extend(emit_uniform_rows(
                    base,
                    ParentMoranStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT,
                ))
                continue
            if support.n_hexes_constructed == 0:
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.NO_CONSENSUS_HEXES,
                ))
                continue
            if support.n_incomplete_components > 0:
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.INCOMPLETE_SUPPORT_COVERAGE,
                ))
                continue

            finite_hexes = consensus_frames.finite_hexes(roi_id, surface)
            centers, blocks, a, b = _extract_hex_arrays(finite_hexes, surface)
            n_units = int(a.size)
            base[N_UNITS] = n_units
            if n_units < MIN_UNITS_FOR_MORAN:
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.INSUFFICIENT_UNITS,
                ))
                continue
            try:
                W = build_region_blocked_knn_adjacency(
                    centers, blocks, k=k_neighbors, cache=weight_cache,
                )
            except UndefinedMoranError:
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.NO_ADJACENCY_EDGES,
                ))
                continue
            n_directed = int(W.nnz)
            base[N_ADJACENCY_ENTRIES] = n_directed
            if n_directed == 0:
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.NO_ADJACENCY_EDGES,
                ))
                continue
            n_supported, supported_fraction = adjacency_support(W)
            base[N_ADJACENCY_SUPPORTED_UNITS] = n_supported
            base[ADJACENCY_SUPPORTED_FRACTION] = supported_fraction
            if all_adjacency_blocks_complete(W, blocks):
                rows.extend(emit_uniform_rows(
                    base, ParentMoranStatus.NO_SPATIAL_DISCRIMINATION,
                ))
                continue
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

    frame = pd.DataFrame(rows, columns=list(RoiConsensusMoranSchema.columns))
    if frame.empty:
        return frame
    frame = RoiConsensusMoranSchema.validate(frame, lazy=True)
    enforce_finite_coupling(frame, "roi_consensus_moran_df")
    return frame


def _extract_hex_arrays(
    finite_hexes: pd.DataFrame, surface: PartitionValueSurface,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if finite_hexes.empty:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )
    g1_col, g2_col = _SURFACE_DENSITY_COLUMNS[surface]
    ordered = finite_hexes.copy()
    ordered["_cid_key"] = ordered[COMPONENT_ID].astype(str).map(natural_sort_key)
    ordered["_region_key"] = ordered[DOMAIN_REGION_ID].astype(int)
    ordered = ordered.sort_values(
        ["_cid_key", "_region_key"], kind="mergesort",
    ).reset_index(drop=True)
    a = pd.to_numeric(ordered[g1_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(ordered[g2_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    ordered = ordered.loc[finite].reset_index(drop=True)
    a = a[finite]
    b = b[finite]
    centers = ordered[[CENTER_X, CENTER_Y]].to_numpy(dtype=float)
    blocks = block_ids_from_component_region_tuples(
        list(zip(
            ordered[COMPONENT_ID].astype(str),
            ordered[DOMAIN_REGION_ID].astype(int),
        ))
    )
    return centers, blocks, a, b
