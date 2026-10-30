"""Parent-ROI tessellation enrichment: ROI × replicate primary
builder + ROI-grain reducer.

Same-replicate cross-ROI context; sibling-component tile densities are
pooled per (ROI, replicate) before invoking the shared partition-
analysis primitive. Two typed availability statuses per row
(enrichment vs effect-size) kept independent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row,
)
from tilescope.analyzers.parent_roi_outcomes import (
    ParentReplicateEffectSizeStatus,
    ParentReplicateEnrichmentStatus,
    ParentTessellationEffectSizeStatus,
    ParentTessellationEnrichmentStatus,
    parent_replicate_effect_size_status_from_partition,
    parent_replicate_enrichment_status_from_partition,
)
from tilescope.analyzers.parent_roi_tessellation_common import (
    TessellationPoolingIndex,
)
from tilescope.analyzers.partition_analysis import (
    PartitionValueSurface,
    PartitionValues,
    analyze_partition_enrichment,
)
from tilescope.cols import (
    CLES_G1, CLES_G2,
    CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION,
    CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION,
    ELIGIBLE_SPOT_COVERAGE_FRACTION,
    ENRICHMENT_DIRECTION,
    ENRICHMENT_RATIO_STATUS_COLUMNS,
    ENRICHMENT_RATIO_STATUS_PAIRS,
    G1_ID, G2_ID,
    GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS,
    GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS,
    LOG2FC_G1, LOG2FC_G2,
    N_BOTH_HIGH, N_BOTH_LOW,
    N_CONTEXT_CONTRIBUTED_SPOTS,
    N_CONTEXT_CONTRIBUTING_COMPONENTS, N_CONTEXT_CONTRIBUTING_REGIONS,
    N_CONTEXT_ELIGIBLE_COMPONENTS, N_CONTEXT_ELIGIBLE_REGIONS,
    N_CONTEXT_ELIGIBLE_SPOTS, N_CONTEXT_SOURCE_SPOTS,
    N_CONTEXT_TILES_CONTRIBUTED,
    N_CONTRIBUTED_SPOTS, N_CONTRIBUTING_COMPONENTS, N_CONTRIBUTING_REGIONS,
    N_ELIGIBLE_COMPONENTS, N_ELIGIBLE_REGIONS, N_ELIGIBLE_SPOTS,
    N_G1_HIGH, N_G1_LOW, N_G2_HIGH, N_G2_LOW,
    N_PLANNED_REPLICATES,
    N_SOURCE_SPOTS, N_SPOTS, N_TILES_CONTRIBUTED, N_TILES_CONTRIBUTED_MEDIAN,
    PARENT_REPLICATE_EFFECT_SIZE_STATUS,
    PARENT_REPLICATE_ENRICHMENT_STATUS,
    PARENT_TESSELLATION_EFFECT_SIZE_STATUS,
    PARENT_TESSELLATION_ENRICHMENT_STATUS,
    REPLICATE_INDEX, ROI_ID, SAMPLE_ID,
    SOURCE_ROI_SPOT_COVERAGE_FRACTION,
    SUM_G1, SUM_G2,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus
from tilescope.policies.enrichment_concordance import (
    DEFAULT_CONCORDANCE_POLICY,
    ConcordancePolicy,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import (
    RoiReplicateEnrichmentSchema,
    RoiTessellationEnrichmentSchema,
    assert_columns_populated_iff,
    assert_per_ratio_contract,
)

if TYPE_CHECKING:
    from tilescope.metrics.enrichment_thresholds import PercentileThresholds
    from tilescope.metrics.normalization import ScaleNormalizationParams


_ROI_REPLICATE_ENRICH_COLS = list(RoiReplicateEnrichmentSchema.columns)


def _pooled_partition_values(
    gene1_density: np.ndarray, gene2_density: np.ndarray,
) -> PartitionValues:
    """Wrap pooled tile densities as a VORONOI_TILE_MEAN PartitionValues
    bundle. Finite-only masking is applied inside the primitive."""
    return PartitionValues(
        value_surface=PartitionValueSurface.VORONOI_TILE_MEAN,
        gene1=np.asarray(gene1_density, dtype=float),
        gene2=np.asarray(gene2_density, dtype=float),
    )


def build_roi_replicate_enrichment_df(
    *,
    pooling_index: TessellationPoolingIndex,
    support_status_df: pd.DataFrame,
    thresholds: "PercentileThresholds",
    normalization_params: "ScaleNormalizationParams",
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
) -> pd.DataFrame:
    """ROI × replicate primary table for tessellation enrichment.

    Target = pooled tile-density pairs from every tessellation-eligible
    region of the source ROI at replicate ``t``. Context = pooled
    tile-density pairs from every OTHER biological ROI's tessellation-
    eligible regions at the SAME replicate ``t`` (never pooled across
    replicates).

    Two typed statuses per row: ``parent_replicate_enrichment_status``
    gates the fold-enrichment ratios; ``parent_replicate_effect_size_status``
    gates log2FC / CLES (survives ``context_proportion_zero``).
    Sufficient stats are populated whenever the target has ≥ 1 finite
    tile.
    """
    typed_sample_id = pooling_index.sample_id
    caller_sample_id = str(sample_id) if sample_id else ""
    if caller_sample_id and caller_sample_id != typed_sample_id:
        raise ValueError(
            f"sample_id identity conflict: caller supplied "
            f"{caller_sample_id!r} but the typed tessellation sample_id is "
            f"{typed_sample_id!r}. Two identity authorities are not "
            "permitted; pass only one, or leave sample_id unset."
        )
    resolved_sample_id = typed_sample_id
    n_planned = pooling_index.n_replicates_planned
    all_parents = pooling_index.roi_ids

    scaffold: List[dict] = []

    for parent_roi in all_parents:
        support = pooling_index.support_by_roi[parent_roi]
        context_support = pooling_index.context_support_by_roi[parent_roi]
        support_row = lookup_parent_support_row(support_status_df, parent_roi)
        if int(support_row["source_spots"]) != support.n_source_spots:
            raise ValueError(
                f"Source-support disagreement for ROI {parent_roi!r}."
            )
        n_source_spots = support.n_source_spots
        n_eligible_spots_roi = support.n_eligible_spots
        n_eligible_regions_roi = support.n_eligible_regions
        n_eligible_components_roi = support.n_eligible_components
        n_context_source_spots = context_support.n_source_spots
        n_context_eligible_components = context_support.n_eligible_components
        n_context_eligible_regions = context_support.n_eligible_regions
        n_context_eligible_spots = context_support.n_eligible_spots

        base_row = {
            SAMPLE_ID: resolved_sample_id,
            ROI_ID: str(parent_roi),
            G1_ID: str(g1_id),
            G2_ID: str(g2_id),
            N_SOURCE_SPOTS: n_source_spots,
            N_ELIGIBLE_SPOTS: n_eligible_spots_roi,
            N_ELIGIBLE_COMPONENTS: n_eligible_components_roi,
            N_ELIGIBLE_REGIONS: n_eligible_regions_roi,
            N_CONTEXT_SOURCE_SPOTS: n_context_source_spots,
            N_CONTEXT_ELIGIBLE_SPOTS: n_context_eligible_spots,
            N_CONTEXT_ELIGIBLE_COMPONENTS: n_context_eligible_components,
            N_CONTEXT_ELIGIBLE_REGIONS: n_context_eligible_regions,
        }

        for t in range(n_planned):
            target_pool = pooling_index.pool_by_roi_replicate[(parent_roi, t)]
            context_pool = pooling_index.context_pool_by_roi_replicate[
                (parent_roi, t)
            ]
            n_contributing_components_roi = (
                target_pool.n_contributing_components
            )
            n_contributing_regions_roi = target_pool.n_contributing_regions
            n_tiles_contributed_roi = target_pool.n_tiles
            n_contributed_spots_roi = target_pool.n_spots

            if n_contributed_spots_roi > n_eligible_spots_roi:
                raise ValueError(
                    f"Invariant violation for parent ROI {parent_roi!r} "
                    f"at replicate {t}: n_contributed_spots "
                    f"({n_contributed_spots_roi}) > n_eligible_spots "
                    f"({n_eligible_spots_roi}). Pooled tile assignments "
                    "overshot the eligible source-row set — construction "
                    "failure."
                )

            eligible_cov = (
                n_contributed_spots_roi / n_eligible_spots_roi
                if n_eligible_spots_roi > 0 else float("nan")
            )
            source_cov = (
                n_contributed_spots_roi / n_source_spots
                if n_source_spots > 0 else float("nan")
            )

            n_ctx_contributing_components = (
                context_pool.n_contributing_components
            )
            n_ctx_contributing_regions = context_pool.n_contributing_regions
            n_ctx_tiles_contributed = context_pool.n_tiles
            n_ctx_contributed_spots = context_pool.n_spots

            if n_ctx_contributed_spots > n_context_eligible_spots:
                raise ValueError(
                    f"Invariant violation for parent ROI {parent_roi!r} "
                    f"at replicate {t}: n_context_contributed_spots "
                    f"({n_ctx_contributed_spots}) > "
                    f"n_context_eligible_spots ({n_context_eligible_spots}). "
                    "Context-side pooled tile assignments overshot the "
                    "eligible source-row set."
                )

            ctx_eligible_cov = (
                n_ctx_contributed_spots / n_context_eligible_spots
                if n_context_eligible_spots > 0 else float("nan")
            )
            ctx_source_cov = (
                n_ctx_contributed_spots / n_context_source_spots
                if n_context_source_spots > 0 else float("nan")
            )

            row = {
                **base_row,
                REPLICATE_INDEX: t,
                N_CONTRIBUTING_COMPONENTS: n_contributing_components_roi,
                N_CONTRIBUTING_REGIONS: n_contributing_regions_roi,
                N_TILES_CONTRIBUTED: n_tiles_contributed_roi,
                N_CONTRIBUTED_SPOTS: n_contributed_spots_roi,
                ELIGIBLE_SPOT_COVERAGE_FRACTION: eligible_cov,
                SOURCE_ROI_SPOT_COVERAGE_FRACTION: source_cov,
                N_CONTEXT_CONTRIBUTING_COMPONENTS: n_ctx_contributing_components,
                N_CONTEXT_CONTRIBUTING_REGIONS: n_ctx_contributing_regions,
                N_CONTEXT_TILES_CONTRIBUTED: n_ctx_tiles_contributed,
                N_CONTEXT_CONTRIBUTED_SPOTS: n_ctx_contributed_spots,
                CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION: ctx_eligible_cov,
                CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION: ctx_source_cov,
                PARENT_REPLICATE_EFFECT_SIZE_STATUS: (
                    ParentReplicateEffectSizeStatus.NOT_COMPUTED.value
                ),
                GENE1_ENRICHMENT: float("nan"),
                GENE2_ENRICHMENT: float("nan"),
                JOINT_HIGH_ENRICHMENT_FC: float("nan"),
                JOINT_LOW_ENRICHMENT_FC: float("nan"),
                GENE1_ENRICHMENT_STATUS: pd.NA,
                GENE2_ENRICHMENT_STATUS: pd.NA,
                JOINT_HIGH_ENRICHMENT_STATUS: pd.NA,
                JOINT_LOW_ENRICHMENT_STATUS: pd.NA,
                LOG2FC_G1: float("nan"),
                LOG2FC_G2: float("nan"),
                CLES_G1: float("nan"),
                CLES_G2: float("nan"),
                N_SPOTS: pd.NA,
                N_G1_HIGH: pd.NA,
                N_G2_HIGH: pd.NA,
                N_BOTH_HIGH: pd.NA,
                N_G1_LOW: pd.NA,
                N_G2_LOW: pd.NA,
                N_BOTH_LOW: pd.NA,
                SUM_G1: float("nan"),
                SUM_G2: float("nan"),
            }

            status: ParentReplicateEnrichmentStatus
            if n_eligible_components_roi == 0:
                status = (
                    ParentReplicateEnrichmentStatus
                    .NO_TESSELLATION_ELIGIBLE_COMPONENT
                )
            elif (
                n_contributing_regions_roi < n_eligible_regions_roi
                or n_contributed_spots_roi < n_eligible_spots_roi
            ):
                status = (
                    ParentReplicateEnrichmentStatus
                    .INCOMPLETE_SUPPORT_COVERAGE
                )
            elif n_context_eligible_components == 0:
                status = (
                    ParentReplicateEnrichmentStatus.NO_ELIGIBLE_CONTEXT
                )
            elif (
                n_ctx_contributing_regions < n_context_eligible_regions
                or n_ctx_contributed_spots < n_context_eligible_spots
            ):
                status = (
                    ParentReplicateEnrichmentStatus
                    .INCOMPLETE_CONTEXT_COVERAGE
                )
            else:
                target_g1, target_g2 = pooling_index.density_values(target_pool)
                context_g1, context_g2 = pooling_index.density_values(context_pool)
                result = analyze_partition_enrichment(
                    target_values=_pooled_partition_values(target_g1, target_g2),
                    context_values=_pooled_partition_values(context_g1, context_g2),
                    thresholds=thresholds,
                    normalization_params=normalization_params,
                )
                enrichment_status_parent = (
                    parent_replicate_enrichment_status_from_partition(
                        result.enrichment_status,
                    )
                )
                row[PARENT_REPLICATE_ENRICHMENT_STATUS] = (
                    enrichment_status_parent.value
                )
                per_ratio_values = (
                    result.enrichment_g1_high, result.enrichment_g2_high,
                    result.enrichment_joint_high_fc,
                    result.enrichment_joint_low_fc,
                )
                per_ratio_statuses = (
                    result.enrichment_g1_high_status,
                    result.enrichment_g2_high_status,
                    result.enrichment_joint_high_fc_status,
                    result.enrichment_joint_low_fc_status,
                )
                for (value_col, status_col), per_ratio_value, per_ratio_status in zip(
                    ENRICHMENT_RATIO_STATUS_PAIRS,
                    per_ratio_values,
                    per_ratio_statuses,
                ):
                    row[status_col] = (
                        per_ratio_status.value if per_ratio_status is not None
                        else pd.NA
                    )
                    if per_ratio_value is not None:
                        row[value_col] = per_ratio_value
                effect_size_status_parent = (
                    parent_replicate_effect_size_status_from_partition(
                        result.effect_size_status,
                    )
                )
                row[PARENT_REPLICATE_EFFECT_SIZE_STATUS] = (
                    effect_size_status_parent.value
                )
                if effect_size_status_parent is (
                    ParentReplicateEffectSizeStatus.FINITE
                ):
                    row[LOG2FC_G1] = result.log2fc_g1
                    row[LOG2FC_G2] = result.log2fc_g2
                    row[CLES_G1] = result.cles_g1
                    row[CLES_G2] = result.cles_g2
                stats = result.target_stats
                if stats is not None:
                    row[N_SPOTS] = int(stats.n_obs)
                    row[N_G1_HIGH] = int(stats.n_g1_high)
                    row[N_G2_HIGH] = int(stats.n_g2_high)
                    row[N_BOTH_HIGH] = int(stats.n_both_high)
                    row[N_G1_LOW] = int(stats.n_g1_low)
                    row[N_G2_LOW] = int(stats.n_g2_low)
                    row[N_BOTH_LOW] = int(stats.n_both_low)
                    row[SUM_G1] = float(stats.sum_g1)
                    row[SUM_G2] = float(stats.sum_g2)
                scaffold.append(row)
                continue

            row[PARENT_REPLICATE_ENRICHMENT_STATUS] = status.value
            scaffold.append(row)

    df = pd.DataFrame(scaffold, columns=_ROI_REPLICATE_ENRICH_COLS)
    if not df.empty:
        for col in (N_SPOTS, N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
                    N_G1_LOW, N_G2_LOW, N_BOTH_LOW):
            df[col] = df[col].astype("Int64")
    df = RoiReplicateEnrichmentSchema.validate(df, lazy=True)
    _enforce_enrichment_finite_null_coupling(df)
    return df


_ENRICHMENT_EFFECT_COLS = (LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2)
_ENRICHMENT_TARGET_STATS_FLOAT_COLS = (SUM_G1, SUM_G2)
_ENRICHMENT_TARGET_STATS_INT_COLS = (
    N_SPOTS, N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
    N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
)
_AGGREGATE_STATUSES_WITH_PER_RATIO = (
    ParentReplicateEnrichmentStatus.FINITE.value,
    ParentReplicateEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value,
)


def _enforce_enrichment_finite_null_coupling(df: pd.DataFrame) -> None:
    """Per-ratio value/status coupling + effect-size independence.

    Aggregate status FINITE or CONTEXT_PROPORTION_ZERO ⇒ per-ratio
    statuses populated; other aggregate statuses ⇒ per-ratio statuses
    null. Numeric ratio finite iff its own per-ratio status is FINITE.
    Effect sizes populated iff effect-size status FINITE (independent;
    survives context_proportion_zero). Target sufficient stats
    populated whenever target has ≥ 1 finite tile.
    """
    effect_size_finite = ParentReplicateEffectSizeStatus.FINITE.value
    effect_size_context_only = (
        ParentReplicateEffectSizeStatus.INSUFFICIENT_CONTEXT_UNITS.value
    )
    surface = "roi_replicate_enrichment_df"

    assert_per_ratio_contract(
        df,
        aggregate_status_col=PARENT_REPLICATE_ENRICHMENT_STATUS,
        aggregate_statuses_with_per_ratio=_AGGREGATE_STATUSES_WITH_PER_RATIO,
        ratio_pairs=ENRICHMENT_RATIO_STATUS_PAIRS,
        surface_name=surface,
    )
    assert_columns_populated_iff(
        df, _ENRICHMENT_EFFECT_COLS,
        df[PARENT_REPLICATE_EFFECT_SIZE_STATUS] == effect_size_finite,
        surface_name=surface,
        condition_label="effect-size status FINITE",
    )
    assert_columns_populated_iff(
        df,
        _ENRICHMENT_TARGET_STATS_FLOAT_COLS + _ENRICHMENT_TARGET_STATS_INT_COLS,
        df[PARENT_REPLICATE_EFFECT_SIZE_STATUS].isin(
            (effect_size_finite, effect_size_context_only)
        ),
        surface_name=surface,
        condition_label=(
            "target has ≥ 1 finite tile "
            "(effect-size status FINITE or INSUFFICIENT_CONTEXT_UNITS)"
        ),
    )


_ENRICHMENT_RATIO_SUMMARY_KEYS = (
    (GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS,
     "g1_enrichment_median", "g1_enrichment_q25", "g1_enrichment_q75",
     "n_valid_g1_enrichment_replicates"),
    (GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS,
     "g2_enrichment_median", "g2_enrichment_q25", "g2_enrichment_q75",
     "n_valid_g2_enrichment_replicates"),
    (JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS,
     "joint_high_enrichment_fc_median",
     "joint_high_enrichment_fc_q25",
     "joint_high_enrichment_fc_q75",
     "n_valid_joint_high_enrichment_replicates"),
    (JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS,
     "joint_low_enrichment_fc_median",
     "joint_low_enrichment_fc_q25",
     "joint_low_enrichment_fc_q75",
     "n_valid_joint_low_enrichment_replicates"),
)
_EFFECT_SIZE_SUMMARY_KEYS = (
    (LOG2FC_G1, "log2fc_g1_median", "log2fc_g1_q25", "log2fc_g1_q75"),
    (LOG2FC_G2, "log2fc_g2_median", "log2fc_g2_q25", "log2fc_g2_q75"),
    (CLES_G1, "CLES_g1_median", "CLES_g1_q25", "CLES_g1_q75"),
    (CLES_G2, "CLES_g2_median", "CLES_g2_q25", "CLES_g2_q75"),
)


_TESS_ENRICH_SUMMARY_COLS = list(RoiTessellationEnrichmentSchema.columns)


def _median_iqr_finite_only(series: pd.Series) -> tuple:
    finite = series.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.25)),
        float(np.quantile(finite, 0.75)),
    )


def _median_nan_safe(series: pd.Series) -> float:
    arr = series.astype(float).to_numpy()
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def compute_roi_tessellation_enrichment(
    *,
    roi_replicate_enrichment_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
    concordance_policy: ConcordancePolicy = DEFAULT_CONCORDANCE_POLICY,
) -> pd.DataFrame:
    """Reader-facing ROI tessellation-enrichment summary: pure
    reduction of the per-replicate table. No inference.
    """
    all_parents = sorted(
        support_status_df[ROI_ID].astype(str).unique(),
        key=natural_sort_key,
    )
    frame_parents = set(
        roi_replicate_enrichment_df[ROI_ID].astype(str).unique()
    ) if not roi_replicate_enrichment_df.empty else set()
    unknown_parents = frame_parents - set(all_parents)
    if unknown_parents:
        raise ValueError(
            f"compute_roi_tessellation_enrichment: per-replicate frame "
            f"carries ROI(s) {sorted(unknown_parents)!r} absent from "
            "support_status_df — identity drift between builder and "
            "reducer."
        )
    frame_sample_ids = set(
        roi_replicate_enrichment_df[SAMPLE_ID].astype(str).unique()
    ) if not roi_replicate_enrichment_df.empty else set()
    if len(frame_sample_ids) > 1:
        raise ValueError(
            f"compute_roi_tessellation_enrichment: per-replicate frame "
            f"carries multiple sample_id values {sorted(frame_sample_ids)!r}."
        )
    caller_sample_id = str(sample_id) if sample_id else ""
    frame_sample_id = next(iter(frame_sample_ids), "")
    if caller_sample_id and frame_sample_id and caller_sample_id != frame_sample_id:
        raise ValueError(
            f"sample_id identity conflict: caller supplied "
            f"{caller_sample_id!r} but per-replicate frame carries "
            f"{frame_sample_id!r}."
        )
    resolved_sample_id = frame_sample_id or caller_sample_id

    effect_size_finite = ParentReplicateEffectSizeStatus.FINITE.value
    no_eligible_component = (
        ParentReplicateEnrichmentStatus
        .NO_TESSELLATION_ELIGIBLE_COMPONENT.value
    )

    rows: List[dict] = []
    for parent_roi in all_parents:
        support_row = lookup_parent_support_row(support_status_df, parent_roi)
        support_class = str(support_row["spatial_support_class"])

        replicate_rows = roi_replicate_enrichment_df.loc[
            roi_replicate_enrichment_df[ROI_ID].astype(str)
            == str(parent_roi)
        ]
        n_planned = int(len(replicate_rows))
        if n_planned == 0:
            raise ValueError(
                f"Invariant violation: roi_replicate_enrichment_df has "
                f"no rows for parent ROI {parent_roi!r}."
            )
        actual_indices = sorted(replicate_rows[REPLICATE_INDEX].tolist())
        expected_indices = list(range(n_planned))
        if actual_indices != expected_indices:
            raise ValueError(
                f"Invariant violation for parent ROI {parent_roi!r}: "
                f"replicate_index set {actual_indices} does not equal "
                f"expected 0..{n_planned - 1} (duplicates or gaps)."
            )

        per_ratio_finite_masks = {
            status_col: (
                replicate_rows[status_col]
                == EnrichmentRatioStatus.FINITE.value
            )
            for _, status_col, *_ in _ENRICHMENT_RATIO_SUMMARY_KEYS
        }
        per_ratio_valid_counts = {
            status_col: int(mask.sum())
            for status_col, mask in per_ratio_finite_masks.items()
        }
        effect_size_mask = (
            replicate_rows[PARENT_REPLICATE_EFFECT_SIZE_STATUS]
            == effect_size_finite
        )
        n_valid_effect_size = int(effect_size_mask.sum())

        no_eligible = bool((
            replicate_rows[PARENT_REPLICATE_ENRICHMENT_STATUS]
            == no_eligible_component
        ).all())

        if no_eligible:
            enrichment_status = (
                ParentTessellationEnrichmentStatus
                .NO_TESSELLATION_ELIGIBLE_COMPONENT.value
            )
            effect_size_status = (
                ParentTessellationEffectSizeStatus
                .NO_TESSELLATION_ELIGIBLE_COMPONENT.value
            )
        else:
            has_any_finite = any(c > 0 for c in per_ratio_valid_counts.values())
            has_any_missing = any(c == 0 for c in per_ratio_valid_counts.values())
            if has_any_finite and not has_any_missing:
                enrichment_status = (
                    ParentTessellationEnrichmentStatus.FINITE.value
                )
            elif has_any_finite:
                enrichment_status = (
                    ParentTessellationEnrichmentStatus
                    .PARTIAL_RATIO_AVAILABILITY.value
                )
            else:
                enrichment_status = (
                    ParentTessellationEnrichmentStatus.NO_VALID_REPLICATES.value
                )
            effect_size_status = (
                ParentTessellationEffectSizeStatus.FINITE.value
                if n_valid_effect_size > 0
                else ParentTessellationEffectSizeStatus.NO_VALID_REPLICATES.value
            )

        row = {
            SAMPLE_ID: resolved_sample_id,
            ROI_ID: str(parent_roi),
            "spatial_support_class": support_class,
            G1_ID: str(g1_id),
            G2_ID: str(g2_id),
            PARENT_TESSELLATION_ENRICHMENT_STATUS: enrichment_status,
            PARENT_TESSELLATION_EFFECT_SIZE_STATUS: effect_size_status,
            N_PLANNED_REPLICATES: n_planned,
            "n_valid_effect_size_replicates": n_valid_effect_size,
        }

        for value_col, status_col, median_col, q25_col, q75_col, count_col in (
            _ENRICHMENT_RATIO_SUMMARY_KEYS
        ):
            slice_df = replicate_rows.loc[per_ratio_finite_masks[status_col]]
            median, q25, q75 = _median_iqr_finite_only(slice_df[value_col])
            row[median_col] = median
            row[q25_col] = q25
            row[q75_col] = q75
            row[count_col] = per_ratio_valid_counts[status_col]

        effect_slice = replicate_rows.loc[effect_size_mask]
        for col, median_col, q25_col, q75_col in _EFFECT_SIZE_SUMMARY_KEYS:
            median, q25, q75 = _median_iqr_finite_only(effect_slice[col])
            row[median_col] = median
            row[q25_col] = q25
            row[q75_col] = q75

        marginal_ratios_finite = (
            per_ratio_valid_counts[GENE1_ENRICHMENT_STATUS] > 0
            and per_ratio_valid_counts[GENE2_ENRICHMENT_STATUS] > 0
        )
        if marginal_ratios_finite:
            row[ENRICHMENT_DIRECTION] = (
                concordance_policy.format_enrichment_direction(
                    gene1_enrichment=row["g1_enrichment_median"],
                    gene2_enrichment=row["g2_enrichment_median"],
                )
            )
        else:
            row[ENRICHMENT_DIRECTION] = None

        row[N_TILES_CONTRIBUTED_MEDIAN] = _median_nan_safe(
            replicate_rows[N_TILES_CONTRIBUTED],
        )
        row["n_context_tiles_contributed_median"] = _median_nan_safe(
            replicate_rows[N_CONTEXT_TILES_CONTRIBUTED],
        )
        row["eligible_spot_coverage_fraction_median"] = _median_nan_safe(
            replicate_rows[ELIGIBLE_SPOT_COVERAGE_FRACTION],
        )
        row["source_roi_spot_coverage_fraction_median"] = _median_nan_safe(
            replicate_rows[SOURCE_ROI_SPOT_COVERAGE_FRACTION],
        )
        row["context_eligible_spot_coverage_fraction_median"] = _median_nan_safe(
            replicate_rows[CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION],
        )
        row["context_source_roi_spot_coverage_fraction_median"] = _median_nan_safe(
            replicate_rows[CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION],
        )

        rows.append(row)

    df = pd.DataFrame(rows, columns=_TESS_ENRICH_SUMMARY_COLS)
    if df.empty:
        return df
    return RoiTessellationEnrichmentSchema.validate(df, lazy=True)
