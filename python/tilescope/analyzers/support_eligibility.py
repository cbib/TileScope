"""Scattered-ROI eligibility gate: four-class `spatial_support_class` per biological ROI.

Diagnostic-only for mixed ROIs; does not reassign COMPONENT_ID or touch adaptive gamma.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tilescope.cols import ROI_ID
from tilescope.schema.helpers import natural_sort_key
from tilescope.spatial.source_spot_adjacency import (
    DEFAULT_SAMPLE_SUPPORT_FACTOR, SourceSpotAdjacency,
    build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import (
    build_source_spot_table_from_frame,
)


__all__ = [
    "DEFAULT_SAMPLE_SUPPORT_FACTOR", "SPATIAL_SUPPORT_CLASSES",
    "classify_sample_support", "component_sizes_by_roi_from_adjacency",
    "derive_roi_support_status",
]


# The four spatial-support classes are a partition of biological ROIs
# under a chosen `min_component_spots`. See `derive_roi_support_status`.
SPATIAL_SUPPORT_CLASSES = (
    "scattered",
    "mixed",
    "fully_connected",
    "fully_supported_multicomponent",
)


def component_sizes_by_roi_from_adjacency(
    adjacency: SourceSpotAdjacency,
) -> Dict[str, List[int]]:
    """Project the adjacency's canonical ROI-component memberships to sizes."""
    sizes_by_roi: Dict[str, List[int]] = {}
    for component in adjacency.within_roi_components:
        sizes_by_roi.setdefault(component.roi_id, []).append(component.n_spots)
    for sizes in sizes_by_roi.values():
        sizes.sort(reverse=True)
    return sizes_by_roi


def classify_sample_support(
    df: pd.DataFrame,
    factor: float = DEFAULT_SAMPLE_SUPPORT_FACTOR,
) -> Tuple[Dict[str, List[int]], float]:
    """Per-ROI same-label support-component sizes DESC via the canonical spot/cell adjacency.

    Cross-label edges scale the graph but never connect within a parent ROI.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"factor must be positive and finite; got {factor!r}")
    if len(df) == 0:
        return {}, float("nan")

    source_spot_table = build_source_spot_table_from_frame(df)
    adjacency = build_source_spot_adjacency(
        source_spot_table, radius_factor=factor,
    )
    if (
        source_spot_table.n_spots >= 2
        and not np.isfinite(adjacency.sample_median_nn)
    ):
        raise ValueError(
            "Cannot compute a nearest-neighbour scale from fewer than two "
            "unique spot positions; all input rows share the same (x, y)."
        )
    return (
        component_sizes_by_roi_from_adjacency(adjacency),
        adjacency.sample_median_nn,
    )


_STATUS_COLUMNS: Tuple[str, ...] = (
    "roi_id",
    "source_spots",
    "support_component_count",
    "support_qualified_component_count",
    "support_qualified_spots",
    "support_qualified_fraction",
    "largest_support_component_spots",
    "largest_support_component_fraction",
    "spatial_support_class",
)


def derive_roi_support_status(
    component_sizes_by_roi: Dict[str, List[int]],
    min_component_spots: int,
) -> pd.DataFrame:
    """Per-ROI four-class classification + support-diagnostic counts.

    Four-class partition on chosen `min_component_spots` (the
    geometry-eligible support-component threshold):

      - `scattered`: no geometry-eligible support component
         (all support components below min_component_spots).
      - `mixed`: at least one geometry-eligible support component
         coexists with at least one sub-threshold component.
      - `fully_connected`: every spot lies in ONE geometry-eligible
         support component.
      - `fully_supported_multicomponent`: every spot lies in a
         geometry-eligible support component, and there are MULTIPLE
         such components.

    Columns (9): `roi_id`, `source_spots`, `support_component_count`,
    `support_qualified_component_count`, `support_qualified_spots`,
    `support_qualified_fraction`, `largest_support_component_spots`,
    `largest_support_component_fraction`, `spatial_support_class`.

    Empty input returns a DataFrame with the 9 columns and zero rows
    (stable schema).

    Raises:
        ValueError: if `min_component_spots < 1`. min_component_spots
            is a positive-integer count; zero or negative is undefined.
    """
    if min_component_spots < 1:
        raise ValueError(
            f"min_component_spots must be ≥ 1; got {min_component_spots!r}"
        )
    if not component_sizes_by_roi:
        return pd.DataFrame(columns=list(_STATUS_COLUMNS))
    rows = []
    for roi_id, sizes in sorted(
        component_sizes_by_roi.items(), key=lambda kv: natural_sort_key(kv[0])
    ):
        source_spots = int(sum(sizes))
        support_component_count = len(sizes)
        qualifying = [s for s in sizes if s >= min_component_spots]
        support_qualified_component_count = len(qualifying)
        support_qualified_spots = int(sum(qualifying))
        support_qualified_fraction = (
            support_qualified_spots / source_spots if source_spots else 0.0
        )
        largest = max(sizes) if sizes else 0
        largest_fraction = largest / source_spots if source_spots else 0.0

        if support_qualified_component_count == 0:
            cls = "scattered"
        elif support_qualified_spots == source_spots:
            cls = (
                "fully_connected"
                if support_qualified_component_count == 1
                else "fully_supported_multicomponent"
            )
        else:
            cls = "mixed"

        rows.append({
            "roi_id": roi_id,
            "source_spots": source_spots,
            "support_component_count": support_component_count,
            "support_qualified_component_count": support_qualified_component_count,
            "support_qualified_spots": support_qualified_spots,
            "support_qualified_fraction": support_qualified_fraction,
            "largest_support_component_spots": largest,
            "largest_support_component_fraction": largest_fraction,
            "spatial_support_class": cls,
        })

    return pd.DataFrame(rows)
