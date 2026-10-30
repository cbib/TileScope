"""Pandera schema validators for TileScope tables."""


from pandera.pandas import Column, DataFrameSchema, Check
import pandas as pd

from tilescope.metrics.enrichment import EnrichmentRatioStatus

from tilescope.analyzers.parent_roi_outcomes import (
    ParentConsensusCorrelationStatus,
    ParentConsensusEffectSizeStatus,
    ParentConsensusEnrichmentStatus,
    ParentCorrelationStatus,
    ParentEffectSizeStatus,
    ParentEnrichmentStatus,
    ParentMoranStatus,
    ParentReplicateCorrelationStatus,
    ParentReplicateEffectSizeStatus,
    ParentReplicateEnrichmentStatus,
    ParentTessellationCorrelationStatus,
    ParentTessellationEffectSizeStatus,
    ParentTessellationEnrichmentStatus,
)

from tilescope.cols import (
    COMPONENT_ID, ROI_ID, SAMPLE_ID,
    CORR_MEDIAN, CORR_Q25, CORR_Q75,
    DOMAIN_REGION_ID,
    GENE1_DENSITY, GENE2_DENSITY,
    G1_ID, G2_ID, MODE, N_PERMUTATIONS,
    MIN_ROI_SPOTS, MIN_CONTEXT_SPOTS,
    N_PLANNED_REPLICATES, N_TILES_CONTRIBUTED_MEDIAN,
    PARENT_CORRELATION_STATUS, PARENT_ENRICHMENT_STATUS,
    PARENT_EFFECT_SIZE_STATUS,
    PARENT_REPLICATE_CORRELATION_STATUS,
    PARENT_REPLICATE_EFFECT_SIZE_STATUS,
    PARENT_REPLICATE_ENRICHMENT_STATUS,
    PARENT_TESSELLATION_EFFECT_SIZE_STATUS,
    PARENT_TESSELLATION_ENRICHMENT_STATUS,
    PARENT_CONSENSUS_CORRELATION_STATUS,
    PARENT_CONSENSUS_EFFECT_SIZE_STATUS,
    PARENT_CONSENSUS_ENRICHMENT_STATUS,
    PARENT_MORAN_STATUS,
    VALUE_SURFACE, MORAN_DIRECTION, MORAN_VALUE, MORAN_REASON,
    N_HEXES_POOLED, N_HEXES_WITH_FINITE_DENSITY_POOLED,
    N_CONTRIBUTING_COMPONENTS_CONSENSUS,
    N_ELIGIBLE_CONSENSUS_COMPONENTS,
    N_CONTEXT_HEXES_POOLED, N_CONTEXT_HEXES_WITH_FINITE_DENSITY_POOLED,
    N_CONTEXT_ELIGIBLE_CONSENSUS_COMPONENTS,
    N_CONTEXT_CONTRIBUTING_COMPONENTS_CONSENSUS,
    N_UNITS, N_ADJACENCY_ENTRIES, N_ADJACENCY_SUPPORTED_UNITS,
    ADJACENCY_SUPPORTED_FRACTION,
    N_FINITE_SPOTS,
    N_REPLICATES, N_FINITE_REPLICATES,
    MORAN_VALUE_P25_ACROSS_REPLICATES,
    MORAN_VALUE_P75_ACROSS_REPLICATES,
    TESSELLATION_ID, TILE_ID,
    ENRICHMENT_DIRECTION,
    GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS,
    GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS,
    LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2,
    N_SPOTS,
    N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH, N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
    SUM_G1, SUM_G2, N_ROI, N_CTX,
    TILE_GEOMETRY_WKT,
    OBS_EQUIV_SPOT_MASS,
    OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G2_HIGH_MASS, OBS_EQUIV_BOTH_HIGH_MASS,
    OBS_EQUIV_G1_LOW_MASS, OBS_EQUIV_G2_LOW_MASS, OBS_EQUIV_BOTH_LOW_MASS,
    OBS_EQUIV_G1_SUM_MASS, OBS_EQUIV_G2_SUM_MASS,
    CONTEXT_OBS_EQUIV_SPOT_MASS,
    CONTEXT_OBS_EQUIV_G1_HIGH_MASS, CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
    CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS,
    CONTEXT_OBS_EQUIV_G1_LOW_MASS, CONTEXT_OBS_EQUIV_G2_LOW_MASS,
    CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_SUM_MASS, CONTEXT_OBS_EQUIV_G2_SUM_MASS,
    CORR_OBS, CORR_P_ANALYTIC, CORR_Q_ANALYTIC, CORR_Q_PERM,
    CORR_P_ANALYTIC_CONDITIONAL, CORR_Q_ANALYTIC_CONDITIONAL,
    CORR_P_PERM_REDUCED, CORR_Q_PERM_REDUCED,
    N_PERMUTATIONS_REDUCED, N_VALID_PERMUTATIONS, PERM_SEED,
    FDR_FAMILY_SIZE,
    CORR_PERM_MEAN, CORR_PERM_SD, CORR_P_PERM, SAMPLE_SIZE,
    TILE_AREA,
    CORRELATION_VALUE, REPLICATE_INDEX,
    N_SOURCE_SPOTS, N_ELIGIBLE_SPOTS, N_CONTRIBUTED_SPOTS,
    N_ELIGIBLE_COMPONENTS, N_CONTRIBUTING_COMPONENTS,
    N_ELIGIBLE_REGIONS, N_CONTRIBUTING_REGIONS, N_TILES_CONTRIBUTED,
    ELIGIBLE_SPOT_COVERAGE_FRACTION, SOURCE_ROI_SPOT_COVERAGE_FRACTION,
    N_CONTEXT_SOURCE_SPOTS, N_CONTEXT_ELIGIBLE_SPOTS,
    N_CONTEXT_CONTRIBUTED_SPOTS, N_CONTEXT_ELIGIBLE_COMPONENTS,
    N_CONTEXT_CONTRIBUTING_COMPONENTS, N_CONTEXT_ELIGIBLE_REGIONS,
    N_CONTEXT_CONTRIBUTING_REGIONS, N_CONTEXT_TILES_CONTRIBUTED,
    CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION,
    CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION,
)


# =============================================================================
# COLUMN COUPLING CHECKS
# =============================================================================

def assert_columns_populated_iff(
    df: pd.DataFrame,
    columns,
    populated_mask: "pd.Series",
    *,
    surface_name: str,
    condition_label: str,
) -> None:
    """Assert each column is non-NA where ``populated_mask`` is True and NA elsewhere."""
    if df.empty:
        return
    for column in columns:
        if df.loc[populated_mask, column].isna().any():
            raise ValueError(
                f"{surface_name}: row(s) where {condition_label} carry NA "
                f"{column!r} — the field must be populated there."
            )
        if not df.loc[~populated_mask, column].isna().all():
            raise ValueError(
                f"{surface_name}: row(s) where NOT ({condition_label}) carry "
                f"a non-NA {column!r} — the field must be NA there."
            )


def assert_per_ratio_contract(
    df: pd.DataFrame,
    *,
    aggregate_status_col: str,
    aggregate_statuses_with_per_ratio,
    ratio_pairs,
    surface_name: str,
) -> None:
    """Enforce per-ratio status/value coupling against an aggregate status."""
    # Materialize once because a generator would be exhausted by the first pass
    # and silently skip every per-ratio value/status check.
    ratio_pairs = tuple(ratio_pairs)
    if df.empty:
        return
    populated = df[aggregate_status_col].astype(str).isin(
        aggregate_statuses_with_per_ratio,
    )
    status_cols = tuple(status_col for _, status_col in ratio_pairs)
    assert_columns_populated_iff(
        df, status_cols, populated,
        surface_name=surface_name,
        condition_label=(
            f"{aggregate_status_col!r} in "
            f"{sorted(aggregate_statuses_with_per_ratio)!r}"
        ),
    )
    finite_value = EnrichmentRatioStatus.FINITE.value
    for value_col, status_col in ratio_pairs:
        ratio_finite = df[status_col].fillna("").astype(str) == finite_value
        assert_columns_populated_iff(
            df, (value_col,), ratio_finite,
            surface_name=surface_name,
            condition_label=f"{status_col} == 'finite'",
        )


# =============================================================================
# PANDERA SCHEMAS
# =============================================================================

# Correlation tables are ROI-grain only; reported statistics pool to the
# biological ROI in every published output.


_PARENT_CORRELATION_STATUS_VALUES = (
    *(status.value for status in ParentCorrelationStatus),
)

ParentCorrelationSchema = DataFrameSchema({
    SAMPLE_ID: Column(str),
    ROI_ID: Column(str),
    G1_ID: Column(str),
    G2_ID: Column(str),
    MODE: Column(str),
    N_ROI: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_PERMUTATIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    MIN_ROI_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    PARENT_CORRELATION_STATUS: Column(
        str, checks=Check.isin(_PARENT_CORRELATION_STATUS_VALUES),
    ),
    # Numeric estimands nullable; status disambiguates FINITE from every
    # eligibility / undefined branch.
    CORR_OBS: Column(float, checks=Check.between(-1, 1), nullable=True),
    CORR_P_ANALYTIC: Column(float, checks=Check.between(0, 1), nullable=True),
    CORR_PERM_MEAN: Column(float, checks=Check.between(-1, 1), nullable=True),
    CORR_PERM_SD: Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    CORR_P_PERM: Column(float, checks=Check.between(0, 1), nullable=True),
    # BH q-values within FDR family (sample_id, g1_id, g2_id, mode).
    # fdr_family_size is populated on EVERY row of the family (including
    # non-finite rows) so consumers can distinguish "family of 0" from
    # "family missing".
    CORR_Q_ANALYTIC: Column(float, checks=Check.between(0, 1), nullable=True),
    CORR_Q_PERM: Column(float, checks=Check.between(0, 1), nullable=True),
    FDR_FAMILY_SIZE: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=False,
    ),
    SAMPLE_SIZE: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
}, coerce=True, strict=True, unique=[SAMPLE_ID, ROI_ID],
    checks=[
        Check(
            lambda df: (
                (df[PARENT_CORRELATION_STATUS] == "finite")
                == df[CORR_Q_ANALYTIC].notna()
            ),
            error="corr_q_analytic populated iff parent_correlation_status == 'finite'",
        ),
        Check(
            lambda df: df[CORR_P_PERM].notna() == df[CORR_Q_PERM].notna(),
            error="corr_q_perm populated iff corr_p_perm is populated",
        ),
    ],
)


# Parent-ROI enrichment: one row per biological ROI x gene pair. Pools every
# source-labelled spot of the ROI (including spots in components below the
# tessellation threshold, since the parent estimand is not conditioned on
# component geometry) vs a context of other biological ROIs. MUST NOT carry
# COMPONENT_ID: that would signal a grain leak.
_PARENT_ENRICHMENT_STATUS_VALUES = (
    *(status.value for status in ParentEnrichmentStatus),
)
_PARENT_EFFECT_SIZE_STATUS_VALUES = (
    *(status.value for status in ParentEffectSizeStatus),
)

_PER_RATIO_STATUS_VALUES = tuple(s.value for s in EnrichmentRatioStatus)


ParentEnrichmentSchema = DataFrameSchema({
    SAMPLE_ID: Column(str),
    ROI_ID: Column(str),
    G1_ID: Column(str),
    G2_ID: Column(str),
    N_ROI: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CTX: Column(int, checks=Check.greater_than_or_equal_to(0)),
    MIN_ROI_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    MIN_CONTEXT_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    PARENT_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PARENT_ENRICHMENT_STATUS_VALUES),
    ),
    PARENT_EFFECT_SIZE_STATUS: Column(
        str, checks=Check.isin(_PARENT_EFFECT_SIZE_STATUS_VALUES),
    ),
    ENRICHMENT_DIRECTION: Column(str, nullable=True),
    GENE1_ENRICHMENT: Column(float, nullable=True),
    GENE2_ENRICHMENT: Column(float, nullable=True),
    JOINT_HIGH_ENRICHMENT_FC: Column(float, nullable=True),
    JOINT_LOW_ENRICHMENT_FC: Column(float, nullable=True),
    # Per-ratio primitive status populated iff the parent status is
    # numerically eligible (FINITE / CONTEXT_PROPORTION_ZERO).
    GENE1_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    GENE2_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    JOINT_HIGH_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    JOINT_LOW_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    LOG2FC_G1: Column(float, nullable=True),
    LOG2FC_G2: Column(float, nullable=True),
    CLES_G1: Column(float, checks=Check.between(0, 1), nullable=True),
    CLES_G2: Column(float, checks=Check.between(0, 1), nullable=True),
    # Sufficient stats on pooled source-ROI spots; populated whenever
    # target support passes, independently of context and ratio availability.
    N_SPOTS: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    N_G1_HIGH: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    N_G2_HIGH: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    N_BOTH_HIGH: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    N_G1_LOW: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    N_G2_LOW: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    N_BOTH_LOW: Column("Int64", checks=Check.greater_than_or_equal_to(0), nullable=True),
    SUM_G1: Column(float, nullable=True),
    SUM_G2: Column(float, nullable=True),
}, coerce=True, strict=True, unique=[SAMPLE_ID, ROI_ID])


# =============================================================================
# Parent-ROI spatial statistics
# =============================================================================

# ROI-grain tessellation correlation. Per-replicate symmetric pooling of
# sibling-component tiles, Pearson within replicate, summarised as
# median + Q25/Q75 across FINITE replicates. Every parent from
# roi_support_status.csv emits exactly one row.
_PARENT_TESS_CORR_STATUS_VALUES = (
    *(status.value for status in ParentTessellationCorrelationStatus),
)

RoiTessellationCorrelationSchema = DataFrameSchema({
    SAMPLE_ID: Column(str),
    ROI_ID: Column(str),
    "spatial_support_class": Column(str),
    G1_ID: Column(str),
    G2_ID: Column(str),
    "mode": Column(str, checks=Check.isin(["pearson", "spearman"])),
    "parent_tessellation_correlation_status": Column(
        str, checks=Check.isin(_PARENT_TESS_CORR_STATUS_VALUES),
    ),
    # n_valid_replicates counts FINITE replicate rows in the primary
    # roi_replicate_correlations_df; n_planned_replicates counts every row.
    "n_valid_replicates": Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_PLANNED_REPLICATES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    CORR_MEDIAN: Column(float, checks=Check.between(-1, 1), nullable=True),
    CORR_Q25: Column(float, checks=Check.between(-1, 1), nullable=True),
    CORR_Q75: Column(float, checks=Check.between(-1, 1), nullable=True),
    # Reduced-estimator permutation null on T = median_t(r_t).
    CORR_P_PERM_REDUCED: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    CORR_Q_PERM_REDUCED: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    N_PERMUTATIONS_REDUCED: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=False,
    ),
    N_VALID_PERMUTATIONS: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    PERM_SEED: Column("Int64", nullable=True),
    FDR_FAMILY_SIZE: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=False,
    ),
    # Tile + coverage medians over ALL planned replicates including
    # zero-contribution rows (discrete observation coverage).
    N_TILES_CONTRIBUTED_MEDIAN: Column(
        float, checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    "n_source_spots": Column(int, checks=Check.greater_than_or_equal_to(0)),
    "eligible_spot_coverage_fraction_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    "source_roi_spot_coverage_fraction_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID],
    checks=[
        # When the null was not requested (n_permutations_reduced == 0)
        # every null-related field must be NA even on finite rows.
        Check(
            lambda df: (
                (
                    (df["parent_tessellation_correlation_status"] == "finite")
                    & (df[N_PERMUTATIONS_REDUCED].astype("Int64") > 0)
                )
                == df[CORR_P_PERM_REDUCED].notna()
            ),
            error=(
                "corr_p_perm_reduced populated iff "
                "parent_tessellation_correlation_status == 'finite' AND "
                "n_permutations_reduced > 0"
            ),
        ),
        Check(
            lambda df: (
                df[CORR_P_PERM_REDUCED].notna()
                == df[CORR_Q_PERM_REDUCED].notna()
            ),
            error=(
                "corr_q_perm_reduced populated iff "
                "corr_p_perm_reduced populated"
            ),
        ),
        Check(
            lambda df: (
                df[CORR_P_PERM_REDUCED].notna()
                == df[N_VALID_PERMUTATIONS].notna()
            ),
            error=(
                "n_valid_permutations populated iff "
                "corr_p_perm_reduced populated"
            ),
        ),
        Check(
            lambda df: (
                df[CORR_P_PERM_REDUCED].notna()
                == df[PERM_SEED].notna()
            ),
            error=(
                "perm_seed populated iff corr_p_perm_reduced populated"
            ),
        ),
    ],
)


# ROI x replicate correlation primary table: one row per (sample, roi,
# planned replicate). Analytic p / q are conditional diagnostics: (a) on
# this tessellation realisation, (b) on the independent-tiles assumption
# which spatial adjacency violates. BH is applied per replicate at family
# key (sample_id, replicate_index, g1_id, g2_id, mode); q's are NEVER
# aggregated across replicates.
_PARENT_REPLICATE_CORR_STATUS_VALUES = (
    *(status.value for status in ParentReplicateCorrelationStatus),
)

RoiReplicateCorrelationSchema = DataFrameSchema({
    SAMPLE_ID: Column(str),
    ROI_ID: Column(str),
    REPLICATE_INDEX: Column(int, checks=Check.greater_than_or_equal_to(0)),
    G1_ID: Column(str),
    G2_ID: Column(str),
    MODE: Column(str, checks=Check.isin(["pearson", "spearman"])),
    PARENT_REPLICATE_CORRELATION_STATUS: Column(
        str, checks=Check.isin(_PARENT_REPLICATE_CORR_STATUS_VALUES),
    ),
    CORRELATION_VALUE: Column(float, checks=Check.between(-1, 1), nullable=True),
    SAMPLE_SIZE: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    # Conditional analytic p + BH q within replicate. Family key:
    # (sample_id, replicate_index, g1_id, g2_id, mode). Never aggregated
    # across replicates.
    CORR_P_ANALYTIC_CONDITIONAL: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    CORR_Q_ANALYTIC_CONDITIONAL: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    FDR_FAMILY_SIZE: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=False,
    ),
    N_SOURCE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ELIGIBLE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTED_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ELIGIBLE_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTING_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ELIGIBLE_REGIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTING_REGIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_TILES_CONTRIBUTED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    # Null when denominator undefined (no eligible support => eligible
    # fraction null; zero spots or cells => source fraction null).
    ELIGIBLE_SPOT_COVERAGE_FRACTION: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    SOURCE_ROI_SPOT_COVERAGE_FRACTION: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID, REPLICATE_INDEX],
    checks=[
        Check(
            lambda df: (
                (df[PARENT_REPLICATE_CORRELATION_STATUS] == "finite")
                == df[CORR_Q_ANALYTIC_CONDITIONAL].notna()
            ),
            error=(
                "corr_q_analytic_conditional populated iff "
                "parent_replicate_correlation_status == 'finite'"
            ),
        ),
    ],
)


# ROI x replicate enrichment primary table. Target = pooled sibling-
# component tiles of the ROI at replicate t; context = pooled tiles of
# every OTHER ROI at the SAME replicate t (never mixed across replicates).
# Two typed statuses gate the numeric fields independently:
# parent_replicate_enrichment_status disambiguates ratio FINITE;
# parent_replicate_effect_size_status disambiguates effect-size FINITE
# (log2FC / CLES survive context_proportion_zero).
_PARENT_REPLICATE_ENRICH_STATUS_VALUES = (
    *(status.value for status in ParentReplicateEnrichmentStatus),
)

_PARENT_REPLICATE_EFFECT_SIZE_STATUS_VALUES = (
    *(status.value for status in ParentReplicateEffectSizeStatus),
)


RoiReplicateEnrichmentSchema = DataFrameSchema({
    # mode is deliberately absent: enrichment has no analogous mode enum.
    SAMPLE_ID: Column(str),
    ROI_ID: Column(str),
    REPLICATE_INDEX: Column(int, checks=Check.greater_than_or_equal_to(0)),
    G1_ID: Column(str),
    G2_ID: Column(str),
    PARENT_REPLICATE_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PARENT_REPLICATE_ENRICH_STATUS_VALUES),
    ),
    PARENT_REPLICATE_EFFECT_SIZE_STATUS: Column(
        str, checks=Check.isin(_PARENT_REPLICATE_EFFECT_SIZE_STATUS_VALUES),
    ),
    N_SOURCE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ELIGIBLE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTED_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ELIGIBLE_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTING_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ELIGIBLE_REGIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTING_REGIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_TILES_CONTRIBUTED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    ELIGIBLE_SPOT_COVERAGE_FRACTION: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    SOURCE_ROI_SPOT_COVERAGE_FRACTION: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    N_CONTEXT_SOURCE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_ELIGIBLE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_CONTRIBUTED_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_ELIGIBLE_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_CONTRIBUTING_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_ELIGIBLE_REGIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_CONTRIBUTING_REGIONS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_TILES_CONTRIBUTED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION: Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    GENE1_ENRICHMENT: Column(float, nullable=True),
    GENE2_ENRICHMENT: Column(float, nullable=True),
    JOINT_HIGH_ENRICHMENT_FC: Column(float, nullable=True),
    JOINT_LOW_ENRICHMENT_FC: Column(float, nullable=True),
    GENE1_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    GENE2_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    JOINT_HIGH_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    JOINT_LOW_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    LOG2FC_G1: Column(float, nullable=True),
    LOG2FC_G2: Column(float, nullable=True),
    CLES_G1: Column(float, checks=Check.between(0, 1), nullable=True),
    CLES_G2: Column(float, checks=Check.between(0, 1), nullable=True),
    # Sufficient stats on the target; populated whenever the target has
    # >= 1 finite tile.
    N_SPOTS: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    N_G1_HIGH: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    N_G2_HIGH: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    N_BOTH_HIGH: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    N_G1_LOW: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    N_G2_LOW: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    N_BOTH_LOW: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    SUM_G1: Column(float, nullable=True),
    SUM_G2: Column(float, nullable=True),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID, REPLICATE_INDEX])


# ROI tessellation-enrichment summary: pure reduction of the per-replicate
# table. Enrichment ratios reduce over enrichment-FINITE replicates;
# log2FC / CLES reduce independently over effect-size-FINITE replicates.
# Descriptive enrichment_direction derived once at ROI grain from the
# summary median marginal ratios via ConcordancePolicy.
_PARENT_TESS_ENRICH_STATUS_VALUES = (
    *(status.value for status in ParentTessellationEnrichmentStatus),
)

_PARENT_TESS_EFFECT_SIZE_STATUS_VALUES = (
    *(status.value for status in ParentTessellationEffectSizeStatus),
)

RoiTessellationEnrichmentSchema = DataFrameSchema({
    SAMPLE_ID: Column(str),
    ROI_ID: Column(str),
    "spatial_support_class": Column(str),
    G1_ID: Column(str),
    G2_ID: Column(str),
    PARENT_TESSELLATION_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PARENT_TESS_ENRICH_STATUS_VALUES),
    ),
    PARENT_TESSELLATION_EFFECT_SIZE_STATUS: Column(
        str, checks=Check.isin(_PARENT_TESS_EFFECT_SIZE_STATUS_VALUES),
    ),
    N_PLANNED_REPLICATES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    # Each counts replicates whose own per-ratio status was FINITE.
    "n_valid_g1_enrichment_replicates": Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    "n_valid_g2_enrichment_replicates": Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    "n_valid_joint_high_enrichment_replicates": Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    "n_valid_joint_low_enrichment_replicates": Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    "n_valid_effect_size_replicates": Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    # Fold-enrichment ratios: median / Q25 / Q75 over the ratio's own
    # per-ratio finite replicates. Null iff that ratio has zero valid
    # replicates. No range constraint: undefined-context cases already
    # gated by the per-ratio status upstream.
    "g1_enrichment_median": Column(float, nullable=True),
    "g1_enrichment_q25": Column(float, nullable=True),
    "g1_enrichment_q75": Column(float, nullable=True),
    "g2_enrichment_median": Column(float, nullable=True),
    "g2_enrichment_q25": Column(float, nullable=True),
    "g2_enrichment_q75": Column(float, nullable=True),
    "joint_high_enrichment_fc_median": Column(float, nullable=True),
    "joint_high_enrichment_fc_q25": Column(float, nullable=True),
    "joint_high_enrichment_fc_q75": Column(float, nullable=True),
    "joint_low_enrichment_fc_median": Column(float, nullable=True),
    "joint_low_enrichment_fc_q25": Column(float, nullable=True),
    "joint_low_enrichment_fc_q75": Column(float, nullable=True),
    "log2fc_g1_median": Column(float, nullable=True),
    "log2fc_g1_q25": Column(float, nullable=True),
    "log2fc_g1_q75": Column(float, nullable=True),
    "log2fc_g2_median": Column(float, nullable=True),
    "log2fc_g2_q25": Column(float, nullable=True),
    "log2fc_g2_q75": Column(float, nullable=True),
    "CLES_g1_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    "CLES_g1_q25": Column(float, checks=Check.between(0, 1), nullable=True),
    "CLES_g1_q75": Column(float, checks=Check.between(0, 1), nullable=True),
    "CLES_g2_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    "CLES_g2_q25": Column(float, checks=Check.between(0, 1), nullable=True),
    "CLES_g2_q75": Column(float, checks=Check.between(0, 1), nullable=True),
    # Non-null iff enrichment status FINITE.
    ENRICHMENT_DIRECTION: Column(str, nullable=True),
    # Diagnostics: medians over ALL planned replicates.
    N_TILES_CONTRIBUTED_MEDIAN: Column(
        float, checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    "n_context_tiles_contributed_median": Column(
        float, checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    "eligible_spot_coverage_fraction_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    "source_roi_spot_coverage_fraction_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    "context_eligible_spot_coverage_fraction_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
    "context_source_roi_spot_coverage_fraction_median": Column(
        float, checks=Check.between(0, 1), nullable=True,
    ),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID])


# ---------------------------------------------------------------------------
# ROI consensus + ROI Moran schemas
# ---------------------------------------------------------------------------


_PARENT_CONSENSUS_CORR_STATUS_VALUES = (
    *(status.value for status in ParentConsensusCorrelationStatus),
)

_CONSENSUS_VALUE_SURFACE_VALUES = (
    "consensus_ensemble", "consensus_direct",
)


RoiConsensusCorrelationSchema = DataFrameSchema({
    SAMPLE_ID: Column(str), ROI_ID: Column(str),
    VALUE_SURFACE: Column(
        str, checks=Check.isin(_CONSENSUS_VALUE_SURFACE_VALUES),
    ),
    G1_ID: Column(str), G2_ID: Column(str),
    MODE: Column(str, checks=Check.isin(["pearson", "spearman"])),
    PARENT_CONSENSUS_CORRELATION_STATUS: Column(
        str, checks=Check.isin(_PARENT_CONSENSUS_CORR_STATUS_VALUES),
    ),
    "correlation_value": Column(
        float, checks=Check.between(-1, 1), nullable=True,
    ),
    SAMPLE_SIZE: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=True,
    ),
    # Analytic p + BH q within value_surface family. Family key:
    # (sample_id, value_surface, g1_id, g2_id, mode).
    CORR_P_ANALYTIC: Column(float, checks=Check.between(0, 1), nullable=True),
    CORR_Q_ANALYTIC: Column(float, checks=Check.between(0, 1), nullable=True),
    FDR_FAMILY_SIZE: Column(
        "Int64", checks=Check.greater_than_or_equal_to(0), nullable=False,
    ),
    N_ELIGIBLE_CONSENSUS_COMPONENTS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    N_CONTRIBUTING_COMPONENTS_CONSENSUS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    N_HEXES_POOLED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_HEXES_WITH_FINITE_DENSITY_POOLED: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID, VALUE_SURFACE],
    checks=[
        Check(
            lambda df: (
                (df[PARENT_CONSENSUS_CORRELATION_STATUS] == "finite")
                == df[CORR_Q_ANALYTIC].notna()
            ),
            error=(
                "corr_q_analytic populated iff "
                "parent_consensus_correlation_status == 'finite'"
            ),
        ),
    ],
)


_PARENT_CONSENSUS_ENRICH_STATUS_VALUES = (
    *(status.value for status in ParentConsensusEnrichmentStatus),
)
_PARENT_CONSENSUS_EFFECT_SIZE_STATUS_VALUES = (
    *(status.value for status in ParentConsensusEffectSizeStatus),
)


RoiConsensusEnrichmentSchema = DataFrameSchema({
    SAMPLE_ID: Column(str), ROI_ID: Column(str),
    VALUE_SURFACE: Column(
        str, checks=Check.isin(_CONSENSUS_VALUE_SURFACE_VALUES),
    ),
    G1_ID: Column(str), G2_ID: Column(str),
    PARENT_CONSENSUS_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PARENT_CONSENSUS_ENRICH_STATUS_VALUES),
    ),
    PARENT_CONSENSUS_EFFECT_SIZE_STATUS: Column(
        str, checks=Check.isin(_PARENT_CONSENSUS_EFFECT_SIZE_STATUS_VALUES),
    ),
    N_ELIGIBLE_CONSENSUS_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTRIBUTING_COMPONENTS_CONSENSUS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_HEXES_POOLED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_HEXES_WITH_FINITE_DENSITY_POOLED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_ELIGIBLE_CONSENSUS_COMPONENTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_CONTRIBUTING_COMPONENTS_CONSENSUS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_HEXES_POOLED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_CONTEXT_HEXES_WITH_FINITE_DENSITY_POOLED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    GENE1_ENRICHMENT: Column(float, nullable=True),
    GENE2_ENRICHMENT: Column(float, nullable=True),
    JOINT_HIGH_ENRICHMENT_FC: Column(float, nullable=True),
    JOINT_LOW_ENRICHMENT_FC: Column(float, nullable=True),
    GENE1_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    GENE2_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    JOINT_HIGH_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    JOINT_LOW_ENRICHMENT_STATUS: Column(
        str, checks=Check.isin(_PER_RATIO_STATUS_VALUES), nullable=True,
    ),
    LOG2FC_G1: Column(float, nullable=True),
    LOG2FC_G2: Column(float, nullable=True),
    CLES_G1: Column(float, checks=Check.between(0, 1), nullable=True),
    CLES_G2: Column(float, checks=Check.between(0, 1), nullable=True),
    # Target sufficient stats: observation-equivalent masses. Float64 by
    # construction (per-hex ensemble-weighted accumulator divided by
    # n_tessellations_covered before pooling); the ratios above
    # reconstruct exactly from these. Never rounded to Int64.
    OBS_EQUIV_SPOT_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_G1_HIGH_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_G2_HIGH_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_BOTH_HIGH_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_G1_LOW_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_G2_LOW_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_BOTH_LOW_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    OBS_EQUIV_G1_SUM_MASS: Column(float, nullable=True),
    OBS_EQUIV_G2_SUM_MASS: Column(float, nullable=True),
    # Context sufficient stats: symmetric mirror. Populated iff target is
    # available AND context has >= 1 finite hex (independent of ratio-
    # computation status).
    CONTEXT_OBS_EQUIV_SPOT_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_G1_HIGH_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_G2_HIGH_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_G1_LOW_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_G2_LOW_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_BOTH_LOW_MASS: Column(
        float, checks=Check.greater_than_or_equal_to(0.0), nullable=True,
    ),
    CONTEXT_OBS_EQUIV_G1_SUM_MASS: Column(float, nullable=True),
    CONTEXT_OBS_EQUIV_G2_SUM_MASS: Column(float, nullable=True),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID, VALUE_SURFACE])


_PARENT_MORAN_STATUS_VALUES = (
    *(status.value for status in ParentMoranStatus),
)
_MORAN_DIRECTION_VALUES = ("g1_lag_g2", "g2_lag_g1")


# ROI-grain spot/cell bivariate Moran: spot/cell values + canonical
# spot/cell adjacency; one row per source ROI x direction.
RoiSourceMoranSchema = DataFrameSchema({
    SAMPLE_ID: Column(str), ROI_ID: Column(str),
    G1_ID: Column(str), G2_ID: Column(str),
    MORAN_DIRECTION: Column(
        str, checks=Check.isin(_MORAN_DIRECTION_VALUES),
    ),
    PARENT_MORAN_STATUS: Column(
        str, checks=Check.isin(_PARENT_MORAN_STATUS_VALUES),
    ),
    MORAN_VALUE: Column(float, nullable=True),
    MORAN_REASON: Column(str, nullable=True),
    N_SOURCE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_FINITE_SPOTS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_UNITS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ADJACENCY_ENTRIES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ADJACENCY_SUPPORTED_UNITS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    ADJACENCY_SUPPORTED_FRACTION: Column(
        float, checks=Check.between(0.0, 1.0),
    ),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID, MORAN_DIRECTION])


# ROI-grain tessellation bivariate Moran. Per-replicate compute on
# tile-density arrays + region-blocked kNN adjacency; the ROI-grain
# moran_value is the median across *complete* contributing replicates,
# with Q25 + Q75 for spread. Replicates that do not cover every eligible
# sibling component are silently skipped; if none survive the ROI emits
# NO_VALID_REPLICATES.
RoiTessellationMoranSchema = DataFrameSchema({
    SAMPLE_ID: Column(str), ROI_ID: Column(str),
    G1_ID: Column(str), G2_ID: Column(str),
    MORAN_DIRECTION: Column(
        str, checks=Check.isin(_MORAN_DIRECTION_VALUES),
    ),
    PARENT_MORAN_STATUS: Column(
        str, checks=Check.isin(_PARENT_MORAN_STATUS_VALUES),
    ),
    MORAN_VALUE: Column(float, nullable=True),
    MORAN_VALUE_P25_ACROSS_REPLICATES: Column(float, nullable=True),
    MORAN_VALUE_P75_ACROSS_REPLICATES: Column(float, nullable=True),
    MORAN_REASON: Column(str, nullable=True),
    N_REPLICATES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_FINITE_REPLICATES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_UNITS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ADJACENCY_ENTRIES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ADJACENCY_SUPPORTED_UNITS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    ADJACENCY_SUPPORTED_FRACTION: Column(
        float, checks=Check.between(0.0, 1.0),
    ),
}, strict=True, ordered=False, unique=[SAMPLE_ID, ROI_ID, MORAN_DIRECTION])


# ROI-grain consensus bivariate Moran: per-hex densities + region-blocked
# kNN adjacency on hex centres; both consensus surfaces.
RoiConsensusMoranSchema = DataFrameSchema({
    SAMPLE_ID: Column(str), ROI_ID: Column(str),
    VALUE_SURFACE: Column(
        str, checks=Check.isin(("consensus_ensemble", "consensus_direct")),
    ),
    G1_ID: Column(str), G2_ID: Column(str),
    MORAN_DIRECTION: Column(
        str, checks=Check.isin(_MORAN_DIRECTION_VALUES),
    ),
    PARENT_MORAN_STATUS: Column(
        str, checks=Check.isin(_PARENT_MORAN_STATUS_VALUES),
    ),
    MORAN_VALUE: Column(float, nullable=True),
    MORAN_REASON: Column(str, nullable=True),
    N_HEXES_POOLED: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_HEXES_WITH_FINITE_DENSITY_POOLED: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    N_ELIGIBLE_CONSENSUS_COMPONENTS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    N_CONTRIBUTING_COMPONENTS_CONSENSUS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    N_UNITS: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ADJACENCY_ENTRIES: Column(int, checks=Check.greater_than_or_equal_to(0)),
    N_ADJACENCY_SUPPORTED_UNITS: Column(
        int, checks=Check.greater_than_or_equal_to(0),
    ),
    ADJACENCY_SUPPORTED_FRACTION: Column(
        float, checks=Check.between(0.0, 1.0),
    ),
}, strict=True, ordered=False, unique=[
    SAMPLE_ID, ROI_ID, VALUE_SURFACE, MORAN_DIRECTION,
])


# =============================================================================
# Tessellation tiles: canonical per-tile export
# =============================================================================


def _density_matches_sum_over_n_spots(
    df: pd.DataFrame,
    *,
    sum_col: str,
    density_col: str,
) -> bool:
    """density is null iff n_spots == 0, else density == sum / n_spots exactly."""
    n_spots = df[N_SPOTS].astype("int64")
    zero = n_spots == 0
    density = df[density_col]
    if not density[zero].isna().all():
        return False
    if not density[~zero].notna().all():
        return False
    if (~zero).any():
        expected = df.loc[~zero, sum_col].astype(float) / n_spots[~zero].astype(float)
        actual = density[~zero].astype(float)
        # Same division, same operands: exact equality is meaningful.
        if not (expected.to_numpy() == actual.to_numpy()).all():
            return False
    return True


TessellationTilesSchema = DataFrameSchema({
    SAMPLE_ID: Column("string"),
    ROI_ID: Column("string"),
    COMPONENT_ID: Column("string"),
    DOMAIN_REGION_ID: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    REPLICATE_INDEX: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    TESSELLATION_ID: Column("string"),
    TILE_ID: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    N_SPOTS: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    SUM_G1: Column("Float64"),
    SUM_G2: Column("Float64"),
    N_G1_HIGH: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    N_G2_HIGH: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    N_BOTH_HIGH: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    N_G1_LOW: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    N_G2_LOW: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    N_BOTH_LOW: Column("Int64", checks=Check.greater_than_or_equal_to(0)),
    # Derived densities: nullable iff n_spots == 0.
    GENE1_DENSITY: Column("Float64", nullable=True),
    GENE2_DENSITY: Column("Float64", nullable=True),
    # tile_area strictly positive (typed Tile contract). NEVER weights the estimator.
    TILE_GEOMETRY_WKT: Column("string"),
    TILE_AREA: Column("Float64", checks=Check.greater_than(0.0)),
}, coerce=True, strict=True, ordered=False, unique=[
    SAMPLE_ID, COMPONENT_ID, DOMAIN_REGION_ID, REPLICATE_INDEX, TILE_ID,
], checks=[
    Check(
        lambda df: _density_matches_sum_over_n_spots(
            df, sum_col=SUM_G1, density_col=GENE1_DENSITY,
        ),
        name="g1_density_iff_n_spots_and_matches_sum",
    ),
    Check(
        lambda df: _density_matches_sum_over_n_spots(
            df, sum_col=SUM_G2, density_col=GENE2_DENSITY,
        ),
        name="g2_density_iff_n_spots_and_matches_sum",
    ),
])
