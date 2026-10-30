"""
ROI splitting for disconnected spatial components.

Handles real spatial transcriptomics data where tissue regions may be fragmented
into spatially disconnected components. Uses k-NN graph connectivity for
scale-invariant component detection that works consistently across coordinate
systems (grid vs world).
"""

from __future__ import annotations

from shapely.geometry import MultiPoint

import logging
import numpy as np
import pandas as pd
from typing import Optional, Literal

from tilescope.cols import ROI_ID, COORD_X, COORD_Y, SOURCE_ROW_ID
from tilescope.spatial.connectivity_graph import knn_components
from tilescope.spatial.source_spot_adjacency import SourceSpotAdjacency

logger = logging.getLogger(__name__)


def split_rois_from_source_adjacency(
    df: pd.DataFrame,
    adjacency: SourceSpotAdjacency,
    *,
    split_min_component_observations: int = 3,
) -> pd.DataFrame:
    """Split ROIs using induced components of the canonical source graph.

    This is the production splitter. The older kNN splitter remains available
    to direct callers, but is no longer a second connectivity authority in the
    TileScope pipeline.
    """
    if SOURCE_ROW_ID not in df.columns:
        raise ValueError(
            "split_rois_from_source_adjacency requires source_row_id."
        )
    if split_min_component_observations < 1:
        raise ValueError("split_min_component_observations must be >= 1.")
    source_ids = tuple(str(v) for v in df[SOURCE_ROW_ID].to_numpy())
    if source_ids != adjacency.source_spot_table.spot_id:
        raise ValueError(
            "split_rois_from_source_adjacency requires the frame in the "
            "same canonical row order as SourceSpotAdjacency."
        )
    roi_labels = np.asarray(adjacency.source_spot_table.roi_id)
    if tuple(str(v) for v in df[ROI_ID].to_numpy()) != tuple(roi_labels):
        raise ValueError(
            "split_rois_from_source_adjacency ROI labels differ from the "
            "canonical SourceSpotTable."
        )

    out = df.copy()
    cc_assignment = np.full(len(out), -1, dtype=np.int64)
    total_components = 0
    components_by_roi: dict[str, list] = {}
    for component in adjacency.within_roi_components:
        if component.n_spots >= split_min_component_observations:
            components_by_roi.setdefault(component.roi_id, []).append(component)
    for roi_id in sorted(np.unique(roi_labels)):
        components = components_by_roi.get(str(roi_id), [])
        for rank, component in enumerate(components):
            cc_assignment[component.positions] = rank
        total_components += len(components)
        if len(components) > 1:
            logger.info(
                "[ROI_SPLIT] ROI %s: source graph produced %d components "
                "(sizes: %s)",
                roi_id, len(components),
                [component.n_spots for component in components],
            )

    out["cc_idx"] = cc_assignment
    out = out[out["cc_idx"] != -1].copy()
    keys = (
        out[[ROI_ID, "cc_idx"]].drop_duplicates()
        .sort_values([ROI_ID, "cc_idx"])
        .itertuples(index=False, name=None)
    )
    id_map = {key: i for i, key in enumerate(keys)}
    out["comp_id"] = [
        id_map[(roi, cc)] for roi, cc in zip(out[ROI_ID], out["cc_idx"])
    ]
    logger.info(
        "[ROI_SPLIT] Canonical source graph produced %d retained pre-filter "
        "components", total_components,
    )
    return out


def split_disconnected_rois(
    df: pd.DataFrame,
    k: int = 10,
    gamma: Optional[float] = None,
    threshold: Literal["local", "global"] = "local",
    symmetrize: Literal["mutual", "union"] = "mutual",
    min_component_spots: int = 3,
    canvas =None
) -> pd.DataFrame:
    """Split spatially disconnected ROIs using k-NN graph connectivity."""
    df = df.copy()
    df['cc_idx'] = -1

    logger.info(f"[ROI_SPLIT] Splitting disconnected ROIs (k-NN graph connectivity)")
    logger.info(f"[ROI_SPLIT] k={k}, gamma={gamma}, threshold={threshold}, min_component_spots={min_component_spots}")

    total_original_rois = 0
    total_pseudo_rois = 0

    for roi_id in sorted(df[ROI_ID].unique()):
        roi_mask = df[ROI_ID] == roi_id
        roi_coords = df.loc[roi_mask, [COORD_X, COORD_Y]].values

        if len(roi_coords) < min_component_spots:
            # Too few points - keep as single component
            df.loc[roi_mask, 'cc_idx'] = 0
            total_original_rois += 1
            total_pseudo_rois += 1
            continue

        # Find connected components using k-NN graph
        labels, n_components = knn_components(
            roi_coords,
            k=k,
            gamma=gamma,
            threshold=threshold,
            symmetrize=symmetrize,
            canvas =canvas
        )

        # Build component metadata for deterministic ranking
        components = []
        for comp_label in range(n_components):
            component_mask_local = labels == comp_label
            if component_mask_local.sum() < min_component_spots:
                continue  # Skip small components

            component_indices = np.where(roi_mask)[0][component_mask_local]
            n_spots = len(component_indices)
            x_centroid = float(roi_coords[component_mask_local, 0].mean())

            components.append({
                'label': comp_label,
                'n_spots': n_spots,
                'x_centroid': x_centroid,
                'indices': component_indices
            })

        if len(components) == 0:
            # All components were too small
            logger.warning(f"[ROI_SPLIT] ROI {roi_id}: all components < {min_component_spots} points - skipping")
            continue

        # Deterministic ordering: size descending, then x-centroid ascending
        components.sort(key=lambda c: (-c['n_spots'], c['x_centroid']))

        # Assign cc_idx (component index within this ROI)
        for rank, comp in enumerate(components):
            df.loc[comp['indices'], 'cc_idx'] = rank

        total_original_rois += 1
        total_pseudo_rois += len(components)

        if len(components) > 1:
            logger.info(f"[ROI_SPLIT] ROI {roi_id}: split into {len(components)} components (sizes: {[c['n_spots'] for c in components]})")

    # Remove points not assigned to any component
    df = df[df['cc_idx'] != -1].copy()

    # Build unique comp_id from (roi_id, cc_idx) pairs
    # Deterministic enumeration: sorted by roi_id, then cc_idx
    keys = (df[[ROI_ID, 'cc_idx']]
            .drop_duplicates()
            .sort_values([ROI_ID, 'cc_idx'])
            .itertuples(index=False, name=None))

    id_map = {k: i for i, k in enumerate(keys)}  # (roi_id, cc_idx) -> comp_id
    df['comp_id'] = [id_map[(r, c)] for r, c in zip(df[ROI_ID], df['cc_idx'])]

    logger.info(f"[ROI_SPLIT] Split {total_original_rois} original ROIs → {total_pseudo_rois} components")
    logger.info(f"[ROI_SPLIT] Created comp_id mapping: {len(id_map)} components [0..{len(id_map)-1}]")
    logger.info(f"[ROI_SPLIT] ROI_ID preserved as parent ROI, cc_idx tracks component index")

    return df


def filter_components(
    pseudo_df: pd.DataFrame,
    min_component_spots: int,
    min_density: Optional[float] = None,
    min_points_small: int = 8
) -> pd.DataFrame:
    """Adaptive filtering: keep dense islands OR large components."""
    if 'comp_id' not in pseudo_df.columns:
        raise ValueError("DataFrame must have 'comp_id' column (run split_disconnected_rois first)")

    logger.info(f"[ROI_SPLIT] Filtering components...")
    logger.info(f"[ROI_SPLIT] min_component_spots: {min_component_spots}")
    if min_density is not None:
        logger.info(f"[ROI_SPLIT] min_density: {min_density:.4f}, min_points_small: {min_points_small}")

    # Group by comp_id and compute metrics
    comp_ids_to_keep = []
    for comp_id in pseudo_df['comp_id'].unique():
        mask = pseudo_df['comp_id'] == comp_id
        n_spots = mask.sum()

        # Compute density if threshold provided
        if min_density is not None:
            coords = pseudo_df.loc[mask, [COORD_X, COORD_Y]].values
            # Estimate area from convex hull (fast proxy)
            hull = MultiPoint([(float(x), float(y)) for x, y in coords]).convex_hull
            area = hull.area if hull.area > 0 else 1.0
            density = n_spots / area
        else:
            density = None

        # Adaptive filtering logic
        keep_absolute = n_spots >= min_component_spots
        keep_density = (
            min_density is not None and
            density is not None and
            density >= min_density and
            n_spots >= min_points_small
        )

        if keep_absolute or keep_density:
            comp_ids_to_keep.append(comp_id)

    # Filter DataFrame
    filtered_df = pseudo_df[pseudo_df['comp_id'].isin(comp_ids_to_keep)].copy()

    n_kept = len(comp_ids_to_keep)
    n_total = pseudo_df['comp_id'].nunique()
    n_dropped = n_total - n_kept

    logger.info(f"[ROI_SPLIT] Kept {n_kept}/{n_total} components (dropped {n_dropped})")

    return filtered_df
