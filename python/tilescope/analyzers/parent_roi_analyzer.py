"""Parent-ROI analyzer: biological headline statistics.

Pools every source-labelled spot for each biological ROI (regardless of
component eligibility) and computes parent correlation (Pearson + within-ROI
permutation, gated on `min_roi_spots`) and parent enrichment (percentile
ratio + LOG2FC + CLES + sufficient counts, gated on both `min_roi_spots`
and `min_context_spots`). Correlation and enrichment are computed
independently per ROI; correlation p-values carry BH-adjusted q-values
within each (sample_id, g1_id, g2_id, mode) family over FINITE rows.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tilescope.cols import (
    ROI_ID, SAMPLE_ID, GENE1, GENE2, G1_ID, G2_ID,
    MODE, N_PERMUTATIONS, N_SPOTS, N_ROI, N_CTX,
    MIN_ROI_SPOTS, MIN_CONTEXT_SPOTS,
    PARENT_CORRELATION_STATUS, PARENT_ENRICHMENT_STATUS,
    PARENT_EFFECT_SIZE_STATUS,
    ENRICHMENT_DIRECTION,
    ENRICHMENT_RATIO_STATUS_COLUMNS,
    ENRICHMENT_RATIO_STATUS_PAIRS,
    GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS,
    GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS,
    LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2,
    N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH, N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
    SUM_G1, SUM_G2,
    CORR_OBS, CORR_P_ANALYTIC, CORR_Q_ANALYTIC, CORR_Q_PERM,
    CORR_P_PERM, CORR_PERM_MEAN, CORR_PERM_SD, SAMPLE_SIZE,
    Q_G1_THRESHOLD, Q_G2_THRESHOLD, Q_G1_LOW_THRESHOLD, Q_G2_LOW_THRESHOLD,
)
from tilescope.defaults import NORM_GENE1_MIN, NORM_GENE2_MIN
from tilescope.enums import CorrelationMode
from tilescope.metrics.correlation import (
    compute_correlation_from_arrays,
    UndefinedCorrelationError,
)
from tilescope.analyzers.parent_roi_outcomes import (
    ParentCorrelationStatus,
    ParentEffectSizeStatus,
    ParentEnrichmentStatus,
    parent_correlation_status_from_reason,
)
from tilescope.analyzers.parent_roi_projection import (
    FeatureParentRoiProjection, ParentRoiSupportIndex,
    build_feature_parent_roi_projection, build_parent_roi_support_index,
)
from tilescope.stats.fdr import apply_bh_within_families
from tilescope.metrics.enrichment import (
    EnrichmentRatioStatus,
    aggregate_context_baselines_from_counts,
    compute_enrichment_ratios_from_counts,
)
from tilescope.policies.enrichment_concordance import DEFAULT_CONCORDANCE_POLICY
from tilescope.schema.table_validation import (
    ParentCorrelationSchema,
    ParentEnrichmentSchema,
    assert_columns_populated_iff,
    assert_per_ratio_contract,
)

logger = logging.getLogger(__name__)


def analyze_parent_rois(
    source_expression_df: pd.DataFrame,
    thresholds: Dict[str, float],
    normalization_params: Dict[str, float],
    min_roi_spots: int,
    min_context_spots: int,
    *,
    sample_id: str = "",
    seed: int = 42,
    n_permutations: int = 100,
    gene1_col: str = GENE1,
    gene2_col: str = GENE2,
    g1_id: str = GENE1,
    g2_id: str = GENE2,
    support_index: Optional[ParentRoiSupportIndex] = None,
    feature_projection_g1: Optional[FeatureParentRoiProjection] = None,
    feature_projection_g2: Optional[FeatureParentRoiProjection] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute parent-ROI correlation + enrichment on pooled spots or cells.

    Returns `(roi_enrichment_df, roi_correlations_df)`, both total over source
    parent ROIs (one row per parent, keyed by `(sample_id, roi_id)`). Numeric
    estimands are null iff the corresponding status column is not FINITE.
    """
    if source_expression_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if support_index is None:
        support_index = build_parent_roi_support_index(source_expression_df)
    if feature_projection_g1 is None:
        feature_projection_g1 = build_feature_parent_roi_projection(
            feature_name=gene1_col,
            source_values=source_expression_df[gene1_col].to_numpy(dtype=float),
            high_threshold=float(thresholds[Q_G1_THRESHOLD]),
            low_threshold=float(thresholds[Q_G1_LOW_THRESHOLD]),
            support_index=support_index,
            gene_min_value=float(normalization_params[NORM_GENE1_MIN]),
            min_roi_spots=min_roi_spots,
            min_context_spots=min_context_spots,
        )
    if feature_projection_g2 is None:
        feature_projection_g2 = build_feature_parent_roi_projection(
            feature_name=gene2_col,
            source_values=source_expression_df[gene2_col].to_numpy(dtype=float),
            high_threshold=float(thresholds[Q_G2_THRESHOLD]),
            low_threshold=float(thresholds[Q_G2_LOW_THRESHOLD]),
            support_index=support_index,
            gene_min_value=float(normalization_params[NORM_GENE2_MIN]),
            min_roi_spots=min_roi_spots,
            min_context_spots=min_context_spots,
        )
    _validate_parent_projections(
        support_index=support_index,
        projection_g1=feature_projection_g1,
        projection_g2=feature_projection_g2,
        min_roi_spots=min_roi_spots,
        min_context_spots=min_context_spots,
    )
    rng = np.random.default_rng(seed)
    all_roi_ids = support_index.roi_ids
    all_g1 = feature_projection_g1.source_values
    all_g2 = feature_projection_g2.source_values
    total_both_high = int((
        (all_g1 > thresholds[Q_G1_THRESHOLD])
        & (all_g2 > thresholds[Q_G2_THRESHOLD])
    ).sum())
    total_both_low = int((
        (all_g1 < thresholds[Q_G1_LOW_THRESHOLD])
        & (all_g2 < thresholds[Q_G2_LOW_THRESHOLD])
    ).sum())

    enrichment_rows: list = []
    correlation_rows: list = []

    for roi_id in all_roi_ids:
        target_positions = support_index.target_positions_by_roi[roi_id]
        context_positions = support_index.context_positions_by_roi[roi_id]
        n_roi = len(target_positions)
        n_ctx = len(context_positions)
        g1_vals = feature_projection_g1.target_values(roi_id)
        g2_vals = feature_projection_g2.target_values(roi_id)

        # Correlation: total table, typed status. Identity + gate context on every row.
        base_corr_row = {
            SAMPLE_ID: str(sample_id),
            ROI_ID: roi_id,
            G1_ID: g1_id,
            G2_ID: g2_id,
            MODE: CorrelationMode.PEARSON.value,
            N_ROI: int(n_roi),
            N_PERMUTATIONS: int(n_permutations),
            MIN_ROI_SPOTS: int(min_roi_spots),
        }
        if n_roi < min_roi_spots:
            correlation_rows.append(_undefined_correlation_row(
                base_corr_row, ParentCorrelationStatus.INSUFFICIENT_ROI_SPOTS,
            ))
            logger.info(
                f"[PARENT] ROI {roi_id!r} correlation "
                f"{ParentCorrelationStatus.INSUFFICIENT_ROI_SPOTS.value}: "
                f"{n_roi} spots < min_roi_spots={min_roi_spots}"
            )
        else:
            try:
                corr_dict = compute_correlation_from_arrays(
                    g1_array=g1_vals,
                    g2_array=g2_vals,
                    mode=CorrelationMode.PEARSON.value,
                    n_perms=n_permutations,
                    rng=rng,
                )
                correlation_rows.append({
                    **base_corr_row,
                    PARENT_CORRELATION_STATUS: ParentCorrelationStatus.FINITE.value,
                    **corr_dict,
                })
            except UndefinedCorrelationError as e:
                status = parent_correlation_status_from_reason(e.reason)
                correlation_rows.append(_undefined_correlation_row(
                    base_corr_row, status,
                ))
                logger.warning(
                    f"[PARENT] ROI {roi_id!r} correlation "
                    f"{status.value}: {e}"
                )

        # Enrichment: total table, typed status.
        base_enrich_row = {
            SAMPLE_ID: str(sample_id),
            ROI_ID: roi_id,
            G1_ID: g1_id,
            G2_ID: g2_id,
            N_ROI: int(n_roi),
            N_CTX: int(n_ctx),
            MIN_ROI_SPOTS: int(min_roi_spots),
            MIN_CONTEXT_SPOTS: int(min_context_spots),
        }
        if n_roi < min_roi_spots:
            enrichment_rows.append(_undefined_enrichment_row(
                base_enrich_row,
                ParentEnrichmentStatus.INSUFFICIENT_ROI_SPOTS,
                ParentEffectSizeStatus.INSUFFICIENT_ROI_SPOTS,
            ))
            continue

        marginal_g1 = feature_projection_g1.marginals_by_roi[roi_id]
        marginal_g2 = feature_projection_g2.marginals_by_roi[roi_id]
        target_both_high = int((
            (g1_vals > thresholds[Q_G1_THRESHOLD])
            & (g2_vals > thresholds[Q_G2_THRESHOLD])
        ).sum())
        target_both_low = int((
            (g1_vals < thresholds[Q_G1_LOW_THRESHOLD])
            & (g2_vals < thresholds[Q_G2_LOW_THRESHOLD])
        ).sum())
        target_stats = _pair_sufficient_stats_row(
            n_obs=n_roi,
            sum_g1=marginal_g1.target_sum,
            sum_g2=marginal_g2.target_sum,
            n_g1_high=marginal_g1.target_high,
            n_g2_high=marginal_g2.target_high,
            n_both_high=target_both_high,
            n_g1_low=marginal_g1.target_low,
            n_g2_low=marginal_g2.target_low,
            n_both_low=target_both_low,
        )
        if n_ctx < min_context_spots:
            row = _undefined_enrichment_row(
                base_enrich_row,
                ParentEnrichmentStatus.INSUFFICIENT_CONTEXT_SPOTS,
                ParentEffectSizeStatus.INSUFFICIENT_CONTEXT_SPOTS,
            )
            row.update(target_stats)
            enrichment_rows.append(row)
            continue

        context_stats = _pair_sufficient_stats_row(
            n_obs=n_ctx,
            sum_g1=marginal_g1.context_sum,
            sum_g2=marginal_g2.context_sum,
            n_g1_high=marginal_g1.context_high,
            n_g2_high=marginal_g2.context_high,
            n_both_high=total_both_high - target_both_high,
            n_g1_low=marginal_g1.context_low,
            n_g2_low=marginal_g2.context_low,
            n_both_low=total_both_low - target_both_low,
        )
        row = _undefined_enrichment_row(
            base_enrich_row,
            ParentEnrichmentStatus.CONTEXT_PROPORTION_ZERO,
            ParentEffectSizeStatus.FINITE,
        )
        row.update(target_stats)
        row.update({
            LOG2FC_G1: marginal_g1.log2fc,
            LOG2FC_G2: marginal_g2.log2fc,
            CLES_G1: marginal_g1.cles,
            CLES_G2: marginal_g2.cles,
        })
        context_baselines = aggregate_context_baselines_from_counts(
            pd.DataFrame([context_stats]),
            pd.Series([n_ctx]),
        )
        ratios = compute_enrichment_ratios_from_counts(
            counts=pd.DataFrame([target_stats]),
            denominators=pd.Series([n_roi]),
            context_baselines=context_baselines,
        ).iloc[0]
        per_ratio_status = {
            value_col: str(ratios[status_col])
            for value_col, status_col in ENRICHMENT_RATIO_STATUS_PAIRS
        }
        per_ratio_finite = {
            col: status == EnrichmentRatioStatus.FINITE.value
            for col, status in per_ratio_status.items()
        }
        all_finite = all(per_ratio_finite.values())
        row[PARENT_ENRICHMENT_STATUS] = (
            ParentEnrichmentStatus.FINITE.value if all_finite
            else ParentEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value
        )
        for value_col, status_col in ENRICHMENT_RATIO_STATUS_PAIRS:
            row[status_col] = per_ratio_status[value_col]
            if per_ratio_finite[value_col]:
                row[value_col] = float(ratios[value_col])
        if per_ratio_finite[GENE1_ENRICHMENT] and per_ratio_finite[GENE2_ENRICHMENT]:
            row[ENRICHMENT_DIRECTION] = (
                DEFAULT_CONCORDANCE_POLICY.format_enrichment_direction(
                    row[GENE1_ENRICHMENT], row[GENE2_ENRICHMENT],
                )
            )
        if not all_finite:
            logger.warning(
                "[PARENT] ROI %r enrichment %s: per-ratio status=%s",
                roi_id,
                ParentEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value,
                per_ratio_status,
            )
        enrichment_rows.append(row)

    enrichment_df = pd.DataFrame(enrichment_rows) if enrichment_rows else pd.DataFrame()
    correlation_df = pd.DataFrame(correlation_rows) if correlation_rows else pd.DataFrame()

    if not enrichment_df.empty:
        enrichment_df = _enforce_parent_enrichment_coupling(enrichment_df)
        enrichment_df = ParentEnrichmentSchema.validate(enrichment_df, lazy=True)

    if not correlation_df.empty:
        correlation_df = _apply_correlation_fdr(correlation_df)
        correlation_df = ParentCorrelationSchema.validate(correlation_df, lazy=True)

    return enrichment_df, correlation_df


def _validate_parent_projections(
    *, support_index: ParentRoiSupportIndex,
    projection_g1: FeatureParentRoiProjection,
    projection_g2: FeatureParentRoiProjection,
    min_roi_spots: int,
    min_context_spots: int,
) -> None:
    for projection in (projection_g1, projection_g2):
        if projection.support_index is not support_index:
            raise ValueError(
                f"parent ROI projection {projection.feature_name!r} is not "
                "attached to the supplied support index"
            )
        if (
            projection.min_roi_spots != min_roi_spots
            or projection.min_context_spots != min_context_spots
        ):
            raise ValueError(
                f"parent ROI projection {projection.feature_name!r} uses "
                "different eligibility gates"
            )


def _pair_sufficient_stats_row(
    *, n_obs: int,
    sum_g1: Optional[float], sum_g2: Optional[float],
    n_g1_high: Optional[int], n_g2_high: Optional[int], n_both_high: int,
    n_g1_low: Optional[int], n_g2_low: Optional[int], n_both_low: int,
) -> dict:
    marginal_values = (
        sum_g1, sum_g2, n_g1_high, n_g2_high, n_g1_low, n_g2_low,
    )
    if any(value is None for value in marginal_values):
        raise ValueError(
            "pair sufficient statistics require eligible per-feature marginals"
        )
    return {
        N_SPOTS: int(n_obs),
        SUM_G1: float(sum_g1),
        SUM_G2: float(sum_g2),
        N_G1_HIGH: int(n_g1_high),
        N_G2_HIGH: int(n_g2_high),
        N_BOTH_HIGH: int(n_both_high),
        N_G1_LOW: int(n_g1_low),
        N_G2_LOW: int(n_g2_low),
        N_BOTH_LOW: int(n_both_low),
    }


_FDR_GROUP_KEYS: tuple = (SAMPLE_ID, G1_ID, G2_ID, MODE)


def _apply_correlation_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """BH q-values on native ROI correlations, family-keyed by
    (sample_id, g1_id, g2_id, mode). Non-FINITE rows contribute no p and get no q."""
    finite_mask = (
        df[PARENT_CORRELATION_STATUS] == ParentCorrelationStatus.FINITE.value
    )
    return apply_bh_within_families(
        df,
        family_keys=_FDR_GROUP_KEYS,
        p_to_q_columns=(
            (CORR_P_ANALYTIC, CORR_Q_ANALYTIC),
            (CORR_P_PERM, CORR_Q_PERM),
        ),
        populated_mask=finite_mask,
    )


def _undefined_correlation_row(
    base: dict, status: ParentCorrelationStatus,
) -> dict:
    # sample_size is the n fed to the correlation primitive, not the eligibility
    # spot count; on a non-FINITE row nothing was computed so it is genuinely null.
    # n_roi (eligibility count) stays populated via `base`.
    return {
        **base,
        PARENT_CORRELATION_STATUS: status.value,
        CORR_OBS: float("nan"),
        CORR_P_ANALYTIC: float("nan"),
        CORR_PERM_MEAN: float("nan"),
        CORR_PERM_SD: float("nan"),
        CORR_P_PERM: float("nan"),
        SAMPLE_SIZE: pd.NA,
    }


def _undefined_enrichment_row(
    base: dict,
    status: ParentEnrichmentStatus,
    effect_size_status: ParentEffectSizeStatus,
) -> dict:
    # Estimator fields (including count sufficient statistics) are null so they
    # are not mistaken for biological absence. `enrichment_direction` is null
    # (not "") so downstream readers do not confuse it with the neutral verdict.
    return {
        **base,
        PARENT_ENRICHMENT_STATUS: status.value,
        PARENT_EFFECT_SIZE_STATUS: effect_size_status.value,
        N_SPOTS: pd.NA,
        SUM_G1: float("nan"),
        SUM_G2: float("nan"),
        N_G1_HIGH: pd.NA,
        N_G2_HIGH: pd.NA,
        N_BOTH_HIGH: pd.NA,
        N_G1_LOW: pd.NA,
        N_G2_LOW: pd.NA,
        N_BOTH_LOW: pd.NA,
        LOG2FC_G1: float("nan"),
        LOG2FC_G2: float("nan"),
        CLES_G1: float("nan"),
        CLES_G2: float("nan"),
        GENE1_ENRICHMENT: float("nan"),
        GENE2_ENRICHMENT: float("nan"),
        JOINT_HIGH_ENRICHMENT_FC: float("nan"),
        JOINT_LOW_ENRICHMENT_FC: float("nan"),
        GENE1_ENRICHMENT_STATUS: pd.NA,
        GENE2_ENRICHMENT_STATUS: pd.NA,
        JOINT_HIGH_ENRICHMENT_STATUS: pd.NA,
        JOINT_LOW_ENRICHMENT_STATUS: pd.NA,
        ENRICHMENT_DIRECTION: pd.NA,
    }


def _enforce_parent_enrichment_coupling(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ratio value/status coupling + independent effect-size and
    direction gates."""
    assert_per_ratio_contract(
        df,
        aggregate_status_col=PARENT_ENRICHMENT_STATUS,
        aggregate_statuses_with_per_ratio=(
            ParentEnrichmentStatus.FINITE.value,
            ParentEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value,
        ),
        ratio_pairs=ENRICHMENT_RATIO_STATUS_PAIRS,
        surface_name="roi_enrichment",
    )
    marginal_direction_available = (
        (df[GENE1_ENRICHMENT_STATUS] == EnrichmentRatioStatus.FINITE.value)
        & (df[GENE2_ENRICHMENT_STATUS] == EnrichmentRatioStatus.FINITE.value)
    )
    assert_columns_populated_iff(
        df,
        columns=(ENRICHMENT_DIRECTION,),
        populated_mask=marginal_direction_available,
        surface_name="roi_enrichment",
        condition_label="g1 and g2 per-ratio statuses both finite",
    )
    assert_columns_populated_iff(
        df,
        columns=(LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2),
        populated_mask=(
            df[PARENT_EFFECT_SIZE_STATUS]
            == ParentEffectSizeStatus.FINITE.value
        ),
        surface_name="roi_enrichment",
        condition_label="parent_effect_size_status is finite",
    )
    assert_columns_populated_iff(
        df,
        columns=(
            N_SPOTS,
            SUM_G1,
            SUM_G2,
            N_G1_HIGH,
            N_G2_HIGH,
            N_BOTH_HIGH,
            N_G1_LOW,
            N_G2_LOW,
            N_BOTH_LOW,
        ),
        populated_mask=df[N_ROI] >= df[MIN_ROI_SPOTS],
        surface_name="roi_enrichment",
        condition_label="target has sufficient ROI spots",
    )
    return df
