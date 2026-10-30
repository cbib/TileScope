"""Pre-analysis pipeline: source snapshot, splitter+filter, scattered gate.

Consumed by `run_analysis`. Every transform validates `source_row_id`
lineage so downstream geometry construction can identify supported
versus ineligible source rows by set difference.

Boundary: this module owns the input-frame transforms and support
classification. Sample-level geometry and component datasets belong to
``sample_authority.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional

import numpy as np
import pandas as pd

from tilescope.analyzers.roi_splitter import (
    filter_components,
    split_disconnected_rois,
    split_rois_from_source_adjacency,
)
from tilescope.analyzers.support_eligibility import (
    DEFAULT_SAMPLE_SUPPORT_FACTOR,
    component_sizes_by_roi_from_adjacency,
    derive_roi_support_status,
)
from tilescope.cols import COMPONENT_ID, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial.source_spot_adjacency import (
    SourceSpotAdjacency, build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import (
    SourceSpotTable, build_source_spot_table_from_frame,
)

logger = logging.getLogger(__name__)


def _validate_source_row_identity(
    frame: pd.DataFrame,
    *,
    allowed_ids: frozenset,
    stage: str,
) -> None:
    """Fail loud if the source_row_id column drifts through a pipeline stage.

    Verifies at each stage that the column is present, non-null int64,
    unique (no row duplication), and every ID is present in the
    **immediately preceding stage's** ID set (`allowed_ids`). Stage-by-
    stage subset checks, rather than a global baseline check, prevent
    a row dropped by an earlier stage from being reintroduced later.
    Explicit `raise` rather than `assert` so the invariant holds under
    `python -O`.
    """
    if SOURCE_ROW_ID not in frame.columns:
        raise ValueError(
            f"[source_row_id contract] stage={stage!r}: column "
            f"{SOURCE_ROW_ID!r} missing from frame."
        )
    col = frame[SOURCE_ROW_ID]
    if col.isna().any():
        raise ValueError(
            f"[source_row_id contract] stage={stage!r}: "
            f"{int(col.isna().sum())} null value(s) in {SOURCE_ROW_ID!r}."
        )
    if col.dtype != np.int64:
        raise ValueError(
            f"[source_row_id contract] stage={stage!r}: {SOURCE_ROW_ID!r} "
            f"dtype is {col.dtype!r}, expected int64."
        )
    if col.duplicated().any():
        raise ValueError(
            f"[source_row_id contract] stage={stage!r}: "
            f"{int(col.duplicated().sum())} duplicate ID(s) in "
            f"{SOURCE_ROW_ID!r} — row duplication would break identity."
        )
    unexpected = set(col.tolist()) - allowed_ids
    if unexpected:
        sample = sorted(unexpected)[:10]
        raise ValueError(
            f"[source_row_id contract] stage={stage!r}: {len(unexpected)} "
            f"ID(s) not present in the preceding-stage set: "
            f"{sample}{'...' if len(unexpected) > 10 else ''}."
        )


def prepare_source_frame(
    expression_df: pd.DataFrame,
) -> pd.DataFrame:
    """Snapshot the input expression frame + assign stable SOURCE_ROW_ID.

    `SOURCE_ROW_ID` is a reserved internal column assigned here; the
    caller's input must not already contain it. Raises `ValueError` if
    the column is already present so we never silently overwrite a
    caller-supplied identifier.
    """
    if SOURCE_ROW_ID in expression_df.columns:
        raise ValueError(
            f"Input expression_df already contains the reserved internal "
            f"column {SOURCE_ROW_ID!r}. This column is assigned by "
            f"prepare_source_frame; the caller's input frame must not "
            f"carry it."
        )
    source_expression_df = expression_df.copy()
    source_expression_df[SOURCE_ROW_ID] = np.arange(
        len(source_expression_df), dtype=np.int64,
    )
    baseline_ids: frozenset = frozenset(source_expression_df[SOURCE_ROW_ID].tolist())
    _validate_source_row_identity(
        source_expression_df,
        allowed_ids=baseline_ids,
        stage="source_snapshot",
    )
    return source_expression_df


class SplitterFilterResult(NamedTuple):
    """Two frames the orchestrator + spot-ledger builder need.

    `split_df` is the raw `split_disconnected_rois` output (with
    `comp_id`, pre-filter). The spot ledger uses its ``SOURCE_ROW_ID``
    values (not positional index) to detect filter-dropped spots;
    join key is the integer source-row identity regardless of the
    caller's DataFrame index.
    `component_expression_df` is the post-filter frame with the
    splitter's `comp_id` normalized to the canonical `COMPONENT_ID`
    string column and `ROI_ID` coerced to string.
    """
    split_df: pd.DataFrame
    component_expression_df: pd.DataFrame


def apply_splitter_and_filter(
    source_expression_df: pd.DataFrame,
    *,
    min_component_spots: int,
    canvas: object = None,
    source_spot_adjacency: Optional[SourceSpotAdjacency] = None,
    knn_k: int = 10,
    knn_gamma: Optional[float] = None,
    knn_threshold: Literal["local", "global"] = "local",
    knn_symmetrize: Literal["mutual", "union"] = "mutual",
) -> SplitterFilterResult:
    """Split disconnected ROIs + adaptive component filter.

    Fed the SOURCE_ROW_ID-carrying snapshot so the column propagates
    through both stages. Validates lineage stage-by-stage (split ⊆
    source; filtered ⊆ split); a global-baseline check would allow a
    split-dropped row to be reintroduced by the filter. Normalizes the
    splitter's `comp_id` output column to the canonical `COMPONENT_ID`
    string column and coerces `ROI_ID` to string.
    """
    source_ids: frozenset = frozenset(
        source_expression_df[SOURCE_ROW_ID].tolist()
    )
    if source_spot_adjacency is None:
        # Backward-compatible path for focused low-level callers. Production
        # always supplies the sample's canonical adjacency.
        split_df = split_disconnected_rois(
            source_expression_df,
            k=knn_k,
            gamma=knn_gamma,
            threshold=knn_threshold,
            symmetrize=knn_symmetrize,
            canvas=canvas,
        )
    else:
        split_df = split_rois_from_source_adjacency(
            source_expression_df, source_spot_adjacency,
        )
    _validate_source_row_identity(
        split_df,
        allowed_ids=source_ids,
        stage="split_disconnected_rois",
    )
    split_ids: frozenset = frozenset(split_df[SOURCE_ROW_ID].tolist())

    filtered_df = filter_components(
        split_df,
        min_component_spots=min_component_spots,
    )
    _validate_source_row_identity(
        filtered_df,
        allowed_ids=split_ids,
        stage="filter_components",
    )

    component_expression_df = filtered_df
    if 'comp_id' in component_expression_df.columns:
        component_expression_df[COMPONENT_ID] = (
            component_expression_df['comp_id'].astype(str)
        )
    component_expression_df[ROI_ID] = component_expression_df[ROI_ID].astype(str)

    logger.info(
        f" Retained {len(component_expression_df)} spots after splitting/filtering"
    )
    return SplitterFilterResult(
        split_df=split_df,
        component_expression_df=component_expression_df,
    )


@dataclass(frozen=True)
class SampleSupportClassification:
    """Per-ROI sample-support classification result.

    Carries the sample-wide `SourceSpotTable` and its canonical
    `SourceSpotAdjacency` so downstream stages consume the same
    materialised graph rather than rebuilding it.
    """
    sample_median_nn: float
    support_status_df: pd.DataFrame
    scattered_rois: frozenset
    all_scattered: bool
    source_spots: SourceSpotTable
    source_spot_adjacency: SourceSpotAdjacency


def classify_sample_support_and_status(
    source_expression_df: pd.DataFrame,
    *,
    min_component_spots: int,
    technology: Literal["visium", "xenium"] = "visium",
    radius_factor: float = DEFAULT_SAMPLE_SUPPORT_FACTOR,
    local_factor: float = 1.5,
    global_cap_factor: float = 4.0,
) -> SampleSupportClassification:
    """Classify sample-support connectivity per biological ROI.

    Runs on `source_expression_df`: safe on any non-empty source,
    decoupled from whether `filter_components` retained any rows.
    Materialises the canonical spot/cell adjacency exactly once.
    """
    source_spots = build_source_spot_table_from_frame(source_expression_df)
    adjacency = build_source_spot_adjacency(
        source_spots, technology=technology, radius_factor=radius_factor,
        local_factor=local_factor, global_cap_factor=global_cap_factor,
    )
    if source_spots.n_spots >= 2 and not np.isfinite(adjacency.sample_median_nn):
        raise ValueError(
            "Cannot compute a nearest-neighbour scale from fewer than two "
            "unique spot positions; all input rows share the same (x, y)."
        )
    support_sizes = component_sizes_by_roi_from_adjacency(adjacency)
    support_status_df = derive_roi_support_status(
        support_sizes,
        min_component_spots=min_component_spots,
    )
    scattered_rois: frozenset = frozenset(
        support_status_df.loc[
            support_status_df["spatial_support_class"] == "scattered",
            "roi_id",
        ].astype(str)
    )
    all_scattered = (
        len(support_status_df) > 0
        and len(scattered_rois) == len(support_status_df)
    )
    return SampleSupportClassification(
        sample_median_nn=adjacency.sample_median_nn,
        support_status_df=support_status_df,
        scattered_rois=scattered_rois,
        all_scattered=all_scattered,
        source_spots=source_spots,
        source_spot_adjacency=adjacency,
    )


def apply_scattered_gate_mask(
    component_expression_df: pd.DataFrame,
    scattered_rois: frozenset,
    *,
    sample_median_nn: float,
    stage_label: str = "scattered_gate",
) -> pd.DataFrame:
    """Drop rows for scattered ROIs; validate SOURCE_ROW_ID lineage.

    Validates gated IDs are a subset of the input component frame's
    IDs (immediately preceding stage). No-op (returns
    `component_expression_df` unchanged) when `scattered_rois` is
    empty. Boolean-mask preserves the `SOURCE_ROW_ID` column.
    """
    input_ids: frozenset = frozenset(
        component_expression_df[SOURCE_ROW_ID].tolist()
    )
    if scattered_rois:
        gated = component_expression_df[
            ~component_expression_df[ROI_ID].astype(str).isin(scattered_rois)
        ].copy()
        logger.info(
            f"[SUPPORT] Scattered-ROI gate: {len(scattered_rois)} ROIs classified "
            f"scattered ({sorted(scattered_rois)}); "
            f"{len(component_expression_df) - len(gated)} spot(s) excluded from "
            f"spatial pipeline (sample_median_nn={sample_median_nn:.4f}, "
            f"factor={DEFAULT_SAMPLE_SUPPORT_FACTOR})."
        )
    else:
        gated = component_expression_df

    _validate_source_row_identity(
        gated, allowed_ids=input_ids, stage=stage_label,
    )
    return gated
