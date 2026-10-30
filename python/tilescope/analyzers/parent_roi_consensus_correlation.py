"""ROI-pooled consensus correlation on ensemble and direct hex densities."""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from tilescope.analyzers.consensus_frames import ConsensusAnalysisFrames
from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row,
    resolve_sample_id_from_typed_record,
)
from tilescope.analyzers.parent_roi_outcomes import (
    ParentConsensusCorrelationStatus,
    parent_consensus_correlation_status_from_reason,
)
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.cols import (
    CORR_P_ANALYTIC,
    CORR_Q_ANALYTIC,
    FDR_FAMILY_SIZE,
    G1_ID,
    G2_ID,
    MODE,
    N_CONTRIBUTING_COMPONENTS_CONSENSUS,
    N_ELIGIBLE_CONSENSUS_COMPONENTS,
    N_HEXES_POOLED,
    N_HEXES_WITH_FINITE_DENSITY_POOLED,
    PARENT_CONSENSUS_CORRELATION_STATUS,
    ROI_ID,
    SAMPLE_ID,
    SAMPLE_SIZE,
    VALUE_SURFACE,
)
from tilescope.enums import CorrelationMode
from tilescope.metrics.correlation import (
    UndefinedCorrelationError,
    compute_correlation_from_arrays,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import (
    RoiConsensusCorrelationSchema,
    assert_columns_populated_iff,
)
from tilescope.stats.fdr import apply_bh_within_families


MIN_HEXES_FOR_CORRELATION = 3
SURFACE_DENSITY_COLUMNS = {
    PartitionValueSurface.CONSENSUS_ENSEMBLE: (
        "gene1_density_ensemble", "gene2_density_ensemble",
    ),
    PartitionValueSurface.CONSENSUS_DIRECT: (
        "gene1_density_direct", "gene2_density_direct",
    ),
}
CONSENSUS_FDR_GROUP_KEYS = (
    SAMPLE_ID, VALUE_SURFACE, G1_ID, G2_ID, MODE,
)


def build_roi_consensus_correlations_df(
    *,
    consensus_frames: ConsensusAnalysisFrames,
    support_status_df: pd.DataFrame,
    surfaces: Sequence[PartitionValueSurface] = (
        PartitionValueSurface.CONSENSUS_ENSEMBLE,
        PartitionValueSurface.CONSENSUS_DIRECT,
    ),
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
    mode: str = CorrelationMode.PEARSON.value,
) -> pd.DataFrame:
    """Emit one status-typed row per source ROI and declared surface."""
    resolved_sample_id = resolve_sample_id_from_typed_record(
        consensus_frames.sample_id, sample_id,
    )
    roi_ids = sorted(
        support_status_df[ROI_ID].astype(str).unique(), key=natural_sort_key,
    )
    rows = []
    for roi_id in roi_ids:
        lookup_parent_support_row(support_status_df, roi_id)
        for surface in surfaces:
            support = consensus_frames.support(roi_id, surface)
            row = {
                SAMPLE_ID: resolved_sample_id,
                ROI_ID: roi_id,
                VALUE_SURFACE: surface.value,
                G1_ID: str(g1_id),
                G2_ID: str(g2_id),
                MODE: mode,
                N_ELIGIBLE_CONSENSUS_COMPONENTS: len(
                    support.eligible_component_ids,
                ),
                N_CONTRIBUTING_COMPONENTS_CONSENSUS: len(
                    support.contributing_component_ids,
                ),
                N_HEXES_POOLED: support.n_hexes_constructed,
                N_HEXES_WITH_FINITE_DENSITY_POOLED: (
                    support.n_hexes_with_finite_density
                ),
                "correlation_value": float("nan"),
                SAMPLE_SIZE: pd.NA,
                CORR_P_ANALYTIC: float("nan"),
                CORR_Q_ANALYTIC: float("nan"),
                FDR_FAMILY_SIZE: pd.NA,
            }
            status = _correlation_status_or_compute(
                row, consensus_frames, roi_id, surface, support, mode,
            )
            row[PARENT_CONSENSUS_CORRELATION_STATUS] = status.value
            rows.append(row)

    frame = pd.DataFrame(
        rows, columns=list(RoiConsensusCorrelationSchema.columns),
    )
    if not frame.empty:
        frame[SAMPLE_SIZE] = frame[SAMPLE_SIZE].astype("Int64")
        finite = frame[PARENT_CONSENSUS_CORRELATION_STATUS] == (
            ParentConsensusCorrelationStatus.FINITE.value
        )
        frame = apply_bh_within_families(
            frame,
            family_keys=CONSENSUS_FDR_GROUP_KEYS,
            p_to_q_columns=((CORR_P_ANALYTIC, CORR_Q_ANALYTIC),),
            populated_mask=finite,
        )
    frame = RoiConsensusCorrelationSchema.validate(frame, lazy=True)
    _enforce_coupling(frame)
    return frame


def _correlation_status_or_compute(
    row: dict,
    frames: ConsensusAnalysisFrames,
    roi_id: str,
    surface: PartitionValueSurface,
    support,
    mode: str,
) -> ParentConsensusCorrelationStatus:
    if not support.eligible_component_ids:
        return ParentConsensusCorrelationStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT
    if support.n_hexes_constructed == 0:
        return ParentConsensusCorrelationStatus.NO_CONSENSUS_HEXES
    if support.n_incomplete_components > 0:
        return ParentConsensusCorrelationStatus.INCOMPLETE_SUPPORT_COVERAGE
    if support.n_hexes_with_finite_density < MIN_HEXES_FOR_CORRELATION:
        return ParentConsensusCorrelationStatus.INSUFFICIENT_FINITE_HEXES

    g1_col, g2_col = SURFACE_DENSITY_COLUMNS[surface]
    observations = frames.finite_hexes(roi_id, surface)
    try:
        result = compute_correlation_from_arrays(
            observations[g1_col].to_numpy(dtype=float),
            observations[g2_col].to_numpy(dtype=float),
            mode=mode,
            n_perms=0,
        )
    except UndefinedCorrelationError as exc:
        return parent_consensus_correlation_status_from_reason(exc.reason)
    value = float(result["corr_obs"])
    if not (math.isfinite(value) and -1.0 <= value <= 1.0):
        return ParentConsensusCorrelationStatus.UNDEFINED_CORRELATION_OTHER
    row["correlation_value"] = value
    row[SAMPLE_SIZE] = int(result["sample_size"])
    row[CORR_P_ANALYTIC] = float(result[CORR_P_ANALYTIC])
    return ParentConsensusCorrelationStatus.FINITE


def _enforce_coupling(frame: pd.DataFrame) -> None:
    finite = frame[PARENT_CONSENSUS_CORRELATION_STATUS] == (
        ParentConsensusCorrelationStatus.FINITE.value
    )
    assert_columns_populated_iff(
        frame,
        ("correlation_value", SAMPLE_SIZE, CORR_P_ANALYTIC),
        finite,
        surface_name="roi_consensus_correlations_df",
        condition_label="parent_consensus_correlation_status == 'finite'",
    )
