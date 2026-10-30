"""Typed enrichment/effect analysis for one observation partition.

This module deliberately excludes correlation. Correlation and enrichment
have different availability contracts and separate public builders; the old
coordinator recomputed Pearson during enrichment and discarded the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from tilescope.cols import (
    ENRICHMENT_RATIO_STATUS_PAIRS,
)
from tilescope.metrics.enrichment import (
    EnrichmentRatioStatus,
    aggregate_context_baselines_from_counts,
    compute_enrichment_ratios_from_counts,
)
from tilescope.metrics.enrichment_thresholds import PercentileThresholds
from tilescope.metrics.normalization import ScaleNormalizationParams
from tilescope.metrics.pair_stats_core import PairStatsCore, SufficientStats
from tilescope.schema.sufficient_stats import sufficient_stats_row


class PartitionValueSurface(str, Enum):
    """Provenance of the paired observation values."""

    VORONOI_TILE_MEAN = "voronoi_tile_mean"
    CONSENSUS_ENSEMBLE = "consensus_ensemble"
    CONSENSUS_DIRECT = "consensus_direct"


class PartitionEnrichmentStatus(str, Enum):
    """Aggregate partition-level enrichment status.

    Structural: ``INSUFFICIENT_UNITS`` (empty target) and
    ``INSUFFICIENT_CONTEXT_UNITS`` (empty context): per-ratio statuses
    and values are all null. ``FINITE`` iff every per-ratio status is
    FINITE. ``CONTEXT_PROPORTION_ZERO`` iff at least one per-ratio
    status is non-finite while the context population is non-empty;
    individual ratios may still be finite: always consult the per-
    ratio status.
    """
    FINITE = "finite"
    INSUFFICIENT_UNITS = "insufficient_units"
    INSUFFICIENT_CONTEXT_UNITS = "insufficient_context_units"
    CONTEXT_PROPORTION_ZERO = "context_proportion_zero"


_STRUCTURAL_ENRICHMENT_STATUSES = frozenset({
    PartitionEnrichmentStatus.INSUFFICIENT_UNITS,
    PartitionEnrichmentStatus.INSUFFICIENT_CONTEXT_UNITS,
})


class PartitionEffectSizeStatus(str, Enum):
    FINITE = "finite"
    INSUFFICIENT_TARGET_UNITS = "insufficient_target_units"
    INSUFFICIENT_CONTEXT_UNITS = "insufficient_context_units"


@dataclass(frozen=True)
class PartitionValues:
    """Paired observation arrays with explicit value-surface provenance."""

    value_surface: PartitionValueSurface
    gene1: np.ndarray
    gene2: np.ndarray

    def __post_init__(self) -> None:
        if len(self.gene1) != len(self.gene2):
            raise ValueError(
                "PartitionValues gene arrays must have equal length; "
                f"got {len(self.gene1)} and {len(self.gene2)}."
            )


@dataclass(frozen=True)
class PartitionEnrichmentResult:
    """Enrichment, distributional effects, and target sufficient stats.

    Per-ratio statuses (``enrichment_*_status``) mirror the primitive's
    :class:`~tilescope.metrics.enrichment.EnrichmentRatioStatus`. All
    four are None iff the aggregate status is structurally unavailable
    (INSUFFICIENT_UNITS or INSUFFICIENT_CONTEXT_UNITS); otherwise all
    four are populated and each numeric value is finite iff its own
    per-ratio status is FINITE.
    """

    n_units_total: int
    n_units_finite: int
    enrichment_g1_high: Optional[float]
    enrichment_g2_high: Optional[float]
    enrichment_joint_high_fc: Optional[float]
    enrichment_joint_low_fc: Optional[float]
    enrichment_g1_high_status: Optional[EnrichmentRatioStatus]
    enrichment_g2_high_status: Optional[EnrichmentRatioStatus]
    enrichment_joint_high_fc_status: Optional[EnrichmentRatioStatus]
    enrichment_joint_low_fc_status: Optional[EnrichmentRatioStatus]
    enrichment_status: PartitionEnrichmentStatus
    log2fc_g1: Optional[float]
    log2fc_g2: Optional[float]
    cles_g1: Optional[float]
    cles_g2: Optional[float]
    effect_size_status: PartitionEffectSizeStatus
    target_stats: Optional[SufficientStats]

    def __post_init__(self) -> None:
        per_ratio_values = (
            self.enrichment_g1_high, self.enrichment_g2_high,
            self.enrichment_joint_high_fc, self.enrichment_joint_low_fc,
        )
        per_ratio_statuses = (
            self.enrichment_g1_high_status, self.enrichment_g2_high_status,
            self.enrichment_joint_high_fc_status,
            self.enrichment_joint_low_fc_status,
        )
        for status in per_ratio_statuses:
            if status is not None and not isinstance(
                status, EnrichmentRatioStatus,
            ):
                raise ValueError(
                    f"per-ratio status must be an EnrichmentRatioStatus or "
                    f"None; got {status!r}"
                )
        if self.enrichment_status in _STRUCTURAL_ENRICHMENT_STATUSES:
            if not all(v is None for v in per_ratio_values):
                raise ValueError(
                    f"{self.enrichment_status.name} requires all ratios None:"
                    f" {per_ratio_values!r}"
                )
            if not all(s is None for s in per_ratio_statuses):
                raise ValueError(
                    f"{self.enrichment_status.name} requires all per-ratio "
                    f"statuses None: {per_ratio_statuses!r}"
                )
        else:
            if any(s is None for s in per_ratio_statuses):
                raise ValueError(
                    f"numerically eligible status requires all per-ratio "
                    f"statuses populated: {per_ratio_statuses!r}"
                )
            for value, status in zip(per_ratio_values, per_ratio_statuses):
                is_finite_status = status is EnrichmentRatioStatus.FINITE
                if is_finite_status:
                    if value is None or not np.isfinite(value):
                        raise ValueError(
                            f"per-ratio FINITE status requires finite value; "
                            f"got status={status!r} value={value!r}"
                        )
                elif value is not None:
                    raise ValueError(
                        f"per-ratio non-finite status requires None value; "
                        f"got status={status!r} value={value!r}"
                    )
            all_finite = all(
                s is EnrichmentRatioStatus.FINITE
                for s in per_ratio_statuses
            )
            expected = (
                PartitionEnrichmentStatus.FINITE if all_finite
                else PartitionEnrichmentStatus.CONTEXT_PROPORTION_ZERO
            )
            if self.enrichment_status is not expected:
                raise ValueError(
                    f"aggregate enrichment_status {self.enrichment_status!r} "
                    f"disagrees with per-ratio statuses {per_ratio_statuses!r}"
                )
        _assert_values_match_status(
            self.effect_size_status is PartitionEffectSizeStatus.FINITE,
            (self.log2fc_g1, self.log2fc_g2, self.cles_g1, self.cles_g2),
            "effect size",
        )
        if self.target_stats is not None:
            if self.target_stats.n_obs != self.n_units_finite:
                raise ValueError(
                    "target_stats.n_obs must equal n_units_finite; got "
                    f"{self.target_stats.n_obs} and {self.n_units_finite}."
                )


def analyze_partition_enrichment(
    *,
    target_values: PartitionValues,
    context_values: PartitionValues,
    thresholds: PercentileThresholds,
    normalization_params: ScaleNormalizationParams,
) -> PartitionEnrichmentResult:
    """Compute enrichment and effects once on finite target/context values."""
    if target_values.value_surface is not context_values.value_surface:
        raise ValueError(
            "target and context value surfaces differ: "
            f"{target_values.value_surface.value!r} != "
            f"{context_values.value_surface.value!r}."
        )
    n_units_total = len(target_values.gene1)
    target = _finite_subset(target_values)
    context = _finite_subset(context_values)
    n_units_finite = len(target.gene1)
    stats = (
        _sufficient_stats(target, thresholds)
        if n_units_finite else None
    )
    enrichment_values, enrichment_statuses, enrichment_status = _enrichment(
        target, context, thresholds, stats,
    )
    effects, effect_status = _effect_sizes(
        target, context, normalization_params,
    )
    return PartitionEnrichmentResult(
        n_units_total=n_units_total,
        n_units_finite=n_units_finite,
        enrichment_g1_high=enrichment_values[0],
        enrichment_g2_high=enrichment_values[1],
        enrichment_joint_high_fc=enrichment_values[2],
        enrichment_joint_low_fc=enrichment_values[3],
        enrichment_g1_high_status=enrichment_statuses[0],
        enrichment_g2_high_status=enrichment_statuses[1],
        enrichment_joint_high_fc_status=enrichment_statuses[2],
        enrichment_joint_low_fc_status=enrichment_statuses[3],
        enrichment_status=enrichment_status,
        log2fc_g1=effects[0],
        log2fc_g2=effects[1],
        cles_g1=effects[2],
        cles_g2=effects[3],
        effect_size_status=effect_status,
        target_stats=stats,
    )


_NO_VALUES: Tuple[Optional[float], ...] = (None, None, None, None)


def _finite_subset(values: PartitionValues) -> PartitionValues:
    mask = np.isfinite(values.gene1) & np.isfinite(values.gene2)
    if bool(mask.all()):
        return values
    return PartitionValues(
        value_surface=values.value_surface,
        gene1=values.gene1[mask],
        gene2=values.gene2[mask],
    )


def _sufficient_stats(
    values: PartitionValues,
    thresholds: PercentileThresholds,
) -> SufficientStats:
    return PairStatsCore().sufficient_stats(
        values.gene1,
        values.gene2,
        q_hi_g1=thresholds.q_g1_threshold,
        q_hi_g2=thresholds.q_g2_threshold,
        q_lo_g1=thresholds.q_g1_low_threshold,
        q_lo_g2=thresholds.q_g2_low_threshold,
    )


_NO_STATUSES: Tuple[Optional[EnrichmentRatioStatus], ...] = (
    None, None, None, None,
)


def _enrichment(
    target: PartitionValues,
    context: PartitionValues,
    thresholds: PercentileThresholds,
    target_stats: Optional[SufficientStats],
) -> tuple[
    Tuple[Optional[float], ...],
    Tuple[Optional[EnrichmentRatioStatus], ...],
    PartitionEnrichmentStatus,
]:
    if target_stats is None:
        return (
            _NO_VALUES,
            _NO_STATUSES,
            PartitionEnrichmentStatus.INSUFFICIENT_UNITS,
        )
    if len(context.gene1) == 0:
        return (
            _NO_VALUES,
            _NO_STATUSES,
            PartitionEnrichmentStatus.INSUFFICIENT_CONTEXT_UNITS,
        )
    context_stats = _sufficient_stats(context, thresholds)
    ratios = compute_enrichment_ratios_from_counts(
        counts=pd.DataFrame([sufficient_stats_row(target_stats)]),
        denominators=pd.Series([target_stats.n_obs]),
        context_baselines=aggregate_context_baselines_from_counts(
            pd.DataFrame([sufficient_stats_row(context_stats)]),
            pd.Series([context_stats.n_obs]),
        ),
    ).iloc[0]
    per_ratio_statuses = tuple(
        EnrichmentRatioStatus(ratios[status_col])
        for _, status_col in ENRICHMENT_RATIO_STATUS_PAIRS
    )
    per_ratio_values = tuple(
        float(ratios[value_col]) if status is EnrichmentRatioStatus.FINITE
        else None
        for (value_col, _), status in zip(
            ENRICHMENT_RATIO_STATUS_PAIRS, per_ratio_statuses,
        )
    )
    all_finite = all(s is EnrichmentRatioStatus.FINITE for s in per_ratio_statuses)
    parent_status = (
        PartitionEnrichmentStatus.FINITE if all_finite
        else PartitionEnrichmentStatus.CONTEXT_PROPORTION_ZERO
    )
    return per_ratio_values, per_ratio_statuses, parent_status


def _effect_sizes(
    target: PartitionValues,
    context: PartitionValues,
    normalization_params: ScaleNormalizationParams,
) -> tuple[Tuple[Optional[float], ...], PartitionEffectSizeStatus]:
    if len(target.gene1) == 0:
        return _NO_VALUES, PartitionEffectSizeStatus.INSUFFICIENT_TARGET_UNITS
    if len(context.gene1) == 0:
        return _NO_VALUES, PartitionEffectSizeStatus.INSUFFICIENT_CONTEXT_UNITS
    core = PairStatsCore()
    return (
        (
            float(core.log2fc(
                target.gene1, context.gene1,
                normalization_params.gene1_min_density,
            )),
            float(core.log2fc(
                target.gene2, context.gene2,
                normalization_params.gene2_min_density,
            )),
            float(core.cles(target.gene1, context.gene1)),
            float(core.cles(target.gene2, context.gene2)),
        ),
        PartitionEffectSizeStatus.FINITE,
    )


def _assert_values_match_status(
    finite_status: bool,
    values: tuple[Optional[float], ...],
    label: str,
) -> None:
    if finite_status:
        if not all(value is not None and np.isfinite(value) for value in values):
            raise ValueError(f"finite {label} status requires finite values: {values!r}")
    elif not all(value is None for value in values):
        raise ValueError(f"non-finite {label} status requires null values: {values!r}")
