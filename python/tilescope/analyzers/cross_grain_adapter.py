"""Per-pair adapter: canonical pair frames -> CrossGrainSensitivityInputs.

Sole mapper of the canonical ROI-grain frames into the typed
observation records the pure producer consumes. Column names come from
``tilescope.cols`` and status vocabularies from the analyzer + metrics
authorities; no strings are re-invented here.

The adapter is invoked once per pair while the typed pair result is built.
The disk backfill path supplies the same frame bundle.

Enrichment status precedence
----------------------------
Per-ratio channel statuses (`g1_enrichment_status`, etc.) take precedence
over the parent row status whenever they are populated with a valid
``EnrichmentRatioStatus``. This preserves the distinction between a
structural parent-row failure (INSUFFICIENT_ROI_SPOTS,
NO_TESSELLATION_ELIGIBLE_COMPONENT, ...) and per-channel signals that
coexist with a partly-populated row (CONTEXT_PROPORTION_ZERO commonly
leaves some channels finite while flagging others saturated).

Tessellation-grain effect population
------------------------------------
For enrichment tessellation observations, ``effect_median``,
``effect_iqr``, and ``n_comparable_replicates`` all describe the same
population: per-ratio ``FINITE`` (raw contributes ``E(r)``) plus
``POSITIVE_OVER_ZERO_CONTEXT`` (typed saturation contributes ``+0.5``).
Other per-ratio statuses (e.g. ``ZERO_OVER_ZERO_CONTEXT``) are excluded
as genuinely unavailable.

For correlation and Moran tessellation, the identity transform means raw
quantiles equal effect quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tilescope.analyzers.cross_grain_types import (
    Channel,
    Statistic,
    Surface,
    UnitKind,
)
from tilescope.analyzers.parent_roi_moran_common import MORAN_DIRECTIONS
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.analyzers.roi_cross_grain_sensitivity import (
    CrossGrainSensitivityInputs,
    PointSurfaceObservation,
    TessellationSummaryObservation,
    compute_effect,
)

from tilescope.cols import (
    CORR_MEDIAN, CORR_OBS, CORR_Q25, CORR_Q75,
    CORRELATION_VALUE,
    GENE1_ENRICHMENT, GENE2_ENRICHMENT,
    GENE1_ENRICHMENT_STATUS, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
    JOINT_HIGH_ENRICHMENT_STATUS, JOINT_LOW_ENRICHMENT_STATUS,
    MORAN_DIRECTION, MORAN_VALUE,
    MORAN_VALUE_P25_ACROSS_REPLICATES, MORAN_VALUE_P75_ACROSS_REPLICATES,
    N_CONTEXT_HEXES_POOLED,
    N_CONTEXT_TILES_CONTRIBUTED,
    N_CTX,
    N_FINITE_REPLICATES,
    N_HEXES_POOLED, N_HEXES_WITH_FINITE_DENSITY_POOLED,
    N_PLANNED_REPLICATES,
    N_REPLICATES,
    N_ROI,
    N_TILES_CONTRIBUTED, N_TILES_CONTRIBUTED_MEDIAN,
    N_UNITS,
    PARENT_CONSENSUS_CORRELATION_STATUS,
    PARENT_CONSENSUS_ENRICHMENT_STATUS,
    PARENT_CORRELATION_STATUS,
    PARENT_ENRICHMENT_STATUS,
    PARENT_MORAN_STATUS,
    PARENT_REPLICATE_CORRELATION_STATUS,
    PARENT_REPLICATE_ENRICHMENT_STATUS,
    PARENT_TESSELLATION_CORRELATION_STATUS,
    ROI_ID,
    VALUE_SURFACE,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus

FINITE = EnrichmentRatioStatus.FINITE.value
SATURATION = EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value
_VALID_RATIO_STATUSES: frozenset = frozenset(s.value for s in EnrichmentRatioStatus)
_COMPARABLE_RATIO_STATUSES: frozenset = frozenset({FINITE, SATURATION})
_NO_COMPARABLE_REPLICATES = "no_comparable_replicates"

_CONSENSUS_ENSEMBLE = PartitionValueSurface.CONSENSUS_ENSEMBLE.value

# Per-channel column names on the enrichment frames.
_ENRICHMENT_CHANNELS: Tuple[Tuple[Channel, str, str], ...] = (
    (Channel.G1, GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS),
    (Channel.G2, GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS),
    (Channel.JOINT_HIGH, JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS),
    (Channel.JOINT_LOW, JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS),
)


# ---------------------------------------------------------------------
# Typed input bundle
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CrossGrainPairFrames:
    """The exact frame set the adapter reads for one pair-run.

    Live analysis constructs this bundle from the canonical frames before
    finalizing ``PairStatisticResult``. The disk backfill loader constructs
    the same bundle from canonical CSVs. Both paths hit the identical adapter
    logic; no second disk-specific calculation lives elsewhere.
    """
    feature_g1_id: str
    feature_g2_id: str
    roi_correlations_df: pd.DataFrame
    roi_enrichment_df: pd.DataFrame
    roi_source_moran_df: pd.DataFrame
    roi_replicate_correlations_df: pd.DataFrame
    roi_tessellation_correlations_df: pd.DataFrame
    roi_replicate_enrichment_df: pd.DataFrame
    roi_tessellation_moran_df: pd.DataFrame
    roi_consensus_correlations_df: pd.DataFrame
    roi_consensus_enrichment_df: pd.DataFrame
    roi_consensus_moran_df: pd.DataFrame

# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def build_cross_grain_inputs(
    sample_id: str,
    pair_frames: CrossGrainPairFrames,
) -> CrossGrainSensitivityInputs:
    """Extract typed cross-grain observations from one pair-run."""
    roi_ids = _enumerate_roi_ids(pair_frames)
    return CrossGrainSensitivityInputs(
        sample_id=sample_id,
        g1_id=pair_frames.feature_g1_id,
        g2_id=pair_frames.feature_g2_id,
        roi_ids=roi_ids,
        source_observations=list(_extract_source(pair_frames, roi_ids)),
        tessellation_observations=list(_extract_tessellation(pair_frames, roi_ids)),
        consensus_observations=list(_extract_consensus(pair_frames, roi_ids)),
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _enumerate_roi_ids(pair_frames: CrossGrainPairFrames) -> List[str]:
    """Source-correlation frame is the ROI universe (one row per ROI,
    with typed availability when correlation is unavailable). Sorted for
    deterministic row order."""
    df = pair_frames.roi_correlations_df
    if df.empty:
        return []
    return sorted(df[ROI_ID].astype(str).unique().tolist())


def _lookup_row(df: pd.DataFrame, **filters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].astype(str) == str(val)
    matches = df[mask]
    if len(matches) != 1:
        raise KeyError(
            f"expected one row for {filters!r} in frame with "
            f"columns={list(df.columns)!r}; got {len(matches)}."
        )
    return matches.iloc[0]


def _opt_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _opt_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _row_status(row: pd.Series, status_col: str) -> str:
    raw = row.get(status_col)
    if raw is None or pd.isna(raw):
        return ""
    return str(raw)


def _resolve_ratio_status(row: pd.Series, ratio_status_col: str, parent_status: str) -> str:
    """Per-ratio status precedence.

    When the per-ratio column carries a valid ``EnrichmentRatioStatus``,
    use it (channel-specific signal). Otherwise fall back to the parent
    row status (structural unavailability with no per-ratio detail)."""
    per_ratio = _row_status(row, ratio_status_col)
    if per_ratio in _VALID_RATIO_STATUSES:
        return per_ratio
    return parent_status


# ---------------------------------------------------------------------
# Shared enrichment-row extractor (source + consensus)
# ---------------------------------------------------------------------

def _emit_point_enrichment(
    row: pd.Series,
    roi_id: str,
    *,
    parent_status_col: str,
    surface: Surface,
    target_unit_kind: UnitKind,
    context_unit_kind: UnitKind,
    n_target_units: Optional[float],
    n_context_units: Optional[float],
) -> Iterable[PointSurfaceObservation]:
    parent_status = _row_status(row, parent_status_col)
    for channel, value_col, status_col in _ENRICHMENT_CHANNELS:
        resolved = _resolve_ratio_status(row, status_col, parent_status)
        if resolved == FINITE:
            raw = _opt_float(row.get(value_col))
        else:
            raw = None
        undefined = "" if resolved == FINITE else resolved
        yield PointSurfaceObservation(
            roi_id=roi_id, surface=surface,
            statistic=Statistic.ENRICHMENT, channel=channel, moran_direction="",
            raw_value=raw, raw_status=resolved, undefined_reason=undefined,
            n_target_units=n_target_units, n_context_units=n_context_units,
            target_unit_kind=target_unit_kind, context_unit_kind=context_unit_kind,
        )


# ---------------------------------------------------------------------
# Source-native extraction
# ---------------------------------------------------------------------

def _extract_source(
    pair_frames: CrossGrainPairFrames,
    roi_ids: Sequence[str],
) -> Iterable[PointSurfaceObservation]:
    corr_df = pair_frames.roi_correlations_df
    enr_df = pair_frames.roi_enrichment_df
    moran_df = pair_frames.roi_source_moran_df

    for roi_id in roi_ids:
        yield _source_correlation(corr_df, roi_id)
        row = _lookup_row(enr_df, **{ROI_ID: roi_id})
        yield from _emit_point_enrichment(
            row, roi_id,
            parent_status_col=PARENT_ENRICHMENT_STATUS,
            surface=Surface.SOURCE_NATIVE,
            target_unit_kind=UnitKind.SOURCE_SPOT,
            context_unit_kind=UnitKind.SOURCE_SPOT,
            n_target_units=_opt_float(row.get(N_ROI)),
            n_context_units=_opt_float(row.get(N_CTX)),
        )
        yield from _source_moran(moran_df, roi_id)


def _source_correlation(df: pd.DataFrame, roi_id: str) -> PointSurfaceObservation:
    row = _lookup_row(df, **{ROI_ID: roi_id})
    status = _row_status(row, PARENT_CORRELATION_STATUS)
    is_finite = status == FINITE
    # Source Pearson r lives under CORR_OBS on roi_correlations_df;
    # the tessellation-replicate and consensus frames use CORRELATION_VALUE.
    return PointSurfaceObservation(
        roi_id=roi_id, surface=Surface.SOURCE_NATIVE,
        statistic=Statistic.CORRELATION, channel=Channel.NONE, moran_direction="",
        raw_value=_opt_float(row[CORR_OBS]) if is_finite else None,
        raw_status=status,
        undefined_reason="" if is_finite else status,
        n_target_units=_opt_float(row.get(N_ROI)), n_context_units=None,
        target_unit_kind=UnitKind.SOURCE_SPOT, context_unit_kind=None,
    )


def _source_moran(
    df: pd.DataFrame, roi_id: str,
) -> Iterable[PointSurfaceObservation]:
    for direction in MORAN_DIRECTIONS:
        row = _lookup_row(df, **{ROI_ID: roi_id, MORAN_DIRECTION: direction})
        status = _row_status(row, PARENT_MORAN_STATUS)
        is_finite = status == FINITE
        yield PointSurfaceObservation(
            roi_id=roi_id, surface=Surface.SOURCE_NATIVE,
            statistic=Statistic.MORAN, channel=Channel.NONE, moran_direction=direction,
            raw_value=_opt_float(row[MORAN_VALUE]) if is_finite else None,
            raw_status=status,
            undefined_reason="" if is_finite else status,
            n_target_units=_opt_float(row.get(N_UNITS)),
            n_context_units=None,
            target_unit_kind=UnitKind.SOURCE_SPOT, context_unit_kind=None,
        )


# ---------------------------------------------------------------------
# Tessellation extraction
# ---------------------------------------------------------------------

def _extract_tessellation(
    pair_frames: CrossGrainPairFrames,
    roi_ids: Sequence[str],
) -> Iterable[TessellationSummaryObservation]:
    corr_summary = pair_frames.roi_tessellation_correlations_df
    corr_replicates = pair_frames.roi_replicate_correlations_df
    enr_replicates = pair_frames.roi_replicate_enrichment_df
    moran_summary = pair_frames.roi_tessellation_moran_df

    for roi_id in roi_ids:
        yield _tessellation_correlation(corr_summary, corr_replicates, roi_id)
        yield from _tessellation_enrichment(enr_replicates, roi_id)
        yield from _tessellation_moran(moran_summary, roi_id)


def _median_iqr(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return (None, None)
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    if len(arr) < 2:
        return (median, None)
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    return (median, float(q75 - q25))


def _tessellation_correlation(
    summary_df: pd.DataFrame,
    replicates_df: pd.DataFrame,
    roi_id: str,
) -> TessellationSummaryObservation:
    summary_row = _lookup_row(summary_df, **{ROI_ID: roi_id})
    summary_status = _row_status(summary_row, PARENT_TESSELLATION_CORRELATION_STATUS)
    n_planned = _opt_int(summary_row.get(N_PLANNED_REPLICATES))
    if n_planned is None:
        n_planned = int((replicates_df[ROI_ID].astype(str) == str(roi_id)).sum())
    n_target_median = _opt_float(summary_row.get(N_TILES_CONTRIBUTED_MEDIAN))

    roi_replicates = replicates_df[replicates_df[ROI_ID].astype(str) == str(roi_id)]
    finite_mask = roi_replicates[PARENT_REPLICATE_CORRELATION_STATUS].astype(str) == FINITE
    finite_values = [
        v for v in roi_replicates.loc[finite_mask, CORRELATION_VALUE].tolist()
        if v is not None and not pd.isna(v)
    ]
    n_comparable = len(finite_values)
    effect_median, effect_iqr = _median_iqr(finite_values)

    if summary_status == FINITE:
        raw_summary = _opt_float(summary_row.get(CORR_MEDIAN))
        raw_status = FINITE
        undefined = ""
    else:
        raw_summary = None
        raw_status = summary_status
        undefined = summary_status

    return TessellationSummaryObservation(
        roi_id=roi_id,
        statistic=Statistic.CORRELATION, channel=Channel.NONE, moran_direction="",
        raw_summary=raw_summary, raw_status=raw_status, undefined_reason=undefined,
        effect_median=effect_median if n_comparable > 0 else None,
        effect_iqr=effect_iqr,
        n_comparable_replicates=n_comparable,
        n_planned_replicates=n_planned or 0,
        n_target_units=n_target_median, n_context_units=None,
        target_unit_kind=UnitKind.TILE, context_unit_kind=None,
    )


def _tessellation_enrichment(
    replicates_df: pd.DataFrame,
    roi_id: str,
) -> Iterable[TessellationSummaryObservation]:
    """Per-channel tessellation summary.

    Effect population = per-ratio ``FINITE`` (contributes ``E(r)``) plus
    ``POSITIVE_OVER_ZERO_CONTEXT`` (contributes ``+0.5``). Other statuses
    are excluded as genuinely unavailable.
    """
    roi_replicates = replicates_df[replicates_df[ROI_ID].astype(str) == str(roi_id)]
    n_planned = len(roi_replicates)
    n_target_median = (
        _opt_float(roi_replicates[N_TILES_CONTRIBUTED].median())
        if not roi_replicates.empty and N_TILES_CONTRIBUTED in roi_replicates.columns
        else None
    )
    n_context_median = (
        _opt_float(roi_replicates[N_CONTEXT_TILES_CONTRIBUTED].median())
        if not roi_replicates.empty and N_CONTEXT_TILES_CONTRIBUTED in roi_replicates.columns
        else None
    )
    parent_statuses = (
        roi_replicates[PARENT_REPLICATE_ENRICHMENT_STATUS].astype(str).tolist()
        if not roi_replicates.empty else []
    )

    for channel, value_col, status_col in _ENRICHMENT_CHANNELS:
        effects: List[float] = []
        finite_raws: List[float] = []
        per_ratio_statuses: List[str] = []
        for _, row in roi_replicates.iterrows():
            parent_status = _row_status(row, PARENT_REPLICATE_ENRICHMENT_STATUS)
            resolved = _resolve_ratio_status(row, status_col, parent_status)
            per_ratio_statuses.append(resolved)
            if resolved not in _COMPARABLE_RATIO_STATUSES:
                continue
            raw = _opt_float(row.get(value_col))
            if resolved == FINITE:
                if raw is None:
                    continue
                finite_raws.append(raw)
                effect = compute_effect(raw, FINITE, Statistic.ENRICHMENT)
            else:
                # POSITIVE_OVER_ZERO_CONTEXT: no raw; effect = +0.5
                effect = compute_effect(None, SATURATION, Statistic.ENRICHMENT)
            if effect is not None:
                effects.append(effect)

        n_comparable = len(effects)
        raw_summary = float(np.median(finite_raws)) if finite_raws else None
        effect_median, effect_iqr = _median_iqr(effects)

        if n_comparable > 0:
            # Finite raws present -> FINITE; only saturation contributed ->
            # POSITIVE_OVER_ZERO_CONTEXT (raw_summary is null in that case).
            raw_status = FINITE if finite_raws else SATURATION
            undefined = ""
        else:
            raw_status = _no_comparable_status(per_ratio_statuses, parent_statuses)
            undefined = raw_status

        yield TessellationSummaryObservation(
            roi_id=roi_id,
            statistic=Statistic.ENRICHMENT, channel=channel, moran_direction="",
            raw_summary=raw_summary, raw_status=raw_status, undefined_reason=undefined,
            effect_median=effect_median if n_comparable else None,
            effect_iqr=effect_iqr,
            n_comparable_replicates=n_comparable,
            n_planned_replicates=n_planned,
            n_target_units=n_target_median, n_context_units=n_context_median,
            target_unit_kind=UnitKind.TILE, context_unit_kind=UnitKind.TILE,
        )


def _no_comparable_status(
    per_ratio_statuses: Sequence[str],
    parent_statuses: Sequence[str],
) -> str:
    """Aggregate a channel-level status when no replicate is comparable.

    Prefer a single, unanimous per-ratio status (e.g. every replicate
    reports ``zero_over_zero_context``). Otherwise, if per-ratio was
    never populated (structural failure), report the unique parent
    status. Fall back to ``no_comparable_replicates``.
    """
    populated_per_ratio = [
        s for s in per_ratio_statuses if s in _VALID_RATIO_STATUSES
    ]
    unique_per_ratio = set(populated_per_ratio)
    if len(unique_per_ratio) == 1:
        return next(iter(unique_per_ratio))
    if not populated_per_ratio:
        unique_parents = {p for p in parent_statuses if p and p != FINITE}
        if len(unique_parents) == 1:
            return next(iter(unique_parents))
    return _NO_COMPARABLE_REPLICATES


def _tessellation_moran(
    summary_df: pd.DataFrame,
    roi_id: str,
) -> Iterable[TessellationSummaryObservation]:
    for direction in MORAN_DIRECTIONS:
        row = _lookup_row(summary_df, **{ROI_ID: roi_id, MORAN_DIRECTION: direction})
        status = _row_status(row, PARENT_MORAN_STATUS)
        is_finite = status == FINITE
        n_planned = _opt_int(row.get(N_REPLICATES)) or 0
        n_comparable = _opt_int(row.get(N_FINITE_REPLICATES)) or 0
        n_target = _opt_float(row.get(N_UNITS))

        if is_finite:
            raw_summary = _opt_float(row.get(MORAN_VALUE))
            effect_median = raw_summary
            p25 = _opt_float(row.get(MORAN_VALUE_P25_ACROSS_REPLICATES))
            p75 = _opt_float(row.get(MORAN_VALUE_P75_ACROSS_REPLICATES))
            effect_iqr = (p75 - p25) if (p25 is not None and p75 is not None) else None
            raw_status = FINITE
            undefined = ""
        else:
            raw_summary = None
            effect_median = None
            effect_iqr = None
            raw_status = status
            undefined = status

        yield TessellationSummaryObservation(
            roi_id=roi_id,
            statistic=Statistic.MORAN, channel=Channel.NONE, moran_direction=direction,
            raw_summary=raw_summary, raw_status=raw_status, undefined_reason=undefined,
            effect_median=effect_median, effect_iqr=effect_iqr,
            n_comparable_replicates=n_comparable,
            n_planned_replicates=n_planned,
            n_target_units=n_target, n_context_units=None,
            target_unit_kind=UnitKind.TILE, context_unit_kind=None,
        )


# ---------------------------------------------------------------------
# Consensus-ensemble extraction
# ---------------------------------------------------------------------

def _consensus_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a consensus frame to value_surface == consensus_ensemble."""
    if VALUE_SURFACE not in df.columns:
        return df
    mask = df[VALUE_SURFACE].astype(str) == _CONSENSUS_ENSEMBLE
    return df[mask]


def _extract_consensus(
    pair_frames: CrossGrainPairFrames,
    roi_ids: Sequence[str],
) -> Iterable[PointSurfaceObservation]:
    corr_df = _consensus_ensemble(pair_frames.roi_consensus_correlations_df)
    enr_df = _consensus_ensemble(pair_frames.roi_consensus_enrichment_df)
    moran_df = _consensus_ensemble(pair_frames.roi_consensus_moran_df)

    for roi_id in roi_ids:
        yield _consensus_correlation(corr_df, roi_id)
        row = _lookup_row(enr_df, **{ROI_ID: roi_id})
        yield from _emit_point_enrichment(
            row, roi_id,
            parent_status_col=PARENT_CONSENSUS_ENRICHMENT_STATUS,
            surface=Surface.CONSENSUS_ENSEMBLE,
            target_unit_kind=UnitKind.HEX,
            context_unit_kind=UnitKind.HEX,
            n_target_units=_opt_float(row.get(N_HEXES_POOLED)),
            n_context_units=_opt_float(row.get(N_CONTEXT_HEXES_POOLED)),
        )
        yield from _consensus_moran(moran_df, roi_id)


def _consensus_correlation(df: pd.DataFrame, roi_id: str) -> PointSurfaceObservation:
    row = _lookup_row(df, **{ROI_ID: roi_id})
    status = _row_status(row, PARENT_CONSENSUS_CORRELATION_STATUS)
    is_finite = status == FINITE
    return PointSurfaceObservation(
        roi_id=roi_id, surface=Surface.CONSENSUS_ENSEMBLE,
        statistic=Statistic.CORRELATION, channel=Channel.NONE, moran_direction="",
        raw_value=_opt_float(row[CORRELATION_VALUE]) if is_finite else None,
        raw_status=status,
        undefined_reason="" if is_finite else status,
        n_target_units=_opt_float(row.get(N_HEXES_WITH_FINITE_DENSITY_POOLED)),
        n_context_units=None,
        target_unit_kind=UnitKind.HEX, context_unit_kind=None,
    )


def _consensus_moran(
    df: pd.DataFrame, roi_id: str,
) -> Iterable[PointSurfaceObservation]:
    for direction in MORAN_DIRECTIONS:
        row = _lookup_row(df, **{ROI_ID: roi_id, MORAN_DIRECTION: direction})
        status = _row_status(row, PARENT_MORAN_STATUS)
        is_finite = status == FINITE
        yield PointSurfaceObservation(
            roi_id=roi_id, surface=Surface.CONSENSUS_ENSEMBLE,
            statistic=Statistic.MORAN, channel=Channel.NONE, moran_direction=direction,
            raw_value=_opt_float(row[MORAN_VALUE]) if is_finite else None,
            raw_status=status,
            undefined_reason="" if is_finite else status,
            n_target_units=_opt_float(row.get(N_UNITS)), n_context_units=None,
            target_unit_kind=UnitKind.HEX, context_unit_kind=None,
        )
