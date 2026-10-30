"""ROI-pooled consensus enrichment on observation-equivalent hex mass."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from tilescope.analyzers.consensus_frames import (
    COUNT_MASS_COLUMNS,
    ConsensusAnalysisFrames,
    ConsensusMassAggregate,
)
from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row,
    resolve_sample_id_from_typed_record,
)
from tilescope.analyzers.parent_roi_outcomes import (
    ParentConsensusEffectSizeStatus,
    ParentConsensusEnrichmentStatus,
)
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.cols import (
    CLES_G1,
    CLES_G2,
    CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS,
    CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_HIGH_MASS,
    CONTEXT_OBS_EQUIV_G1_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_SUM_MASS,
    CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
    CONTEXT_OBS_EQUIV_G2_LOW_MASS,
    CONTEXT_OBS_EQUIV_G2_SUM_MASS,
    CONTEXT_OBS_EQUIV_SPOT_MASS,
    ENRICHMENT_RATIO_STATUS_COLUMNS,
    ENRICHMENT_RATIO_STATUS_PAIRS,
    G1_ID,
    G2_ID,
    GENE1_DENSITY_ENSEMBLE,
    GENE1_ENRICHMENT,
    GENE1_ENRICHMENT_STATUS,
    GENE2_DENSITY_ENSEMBLE,
    GENE2_ENRICHMENT,
    GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC,
    JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC,
    JOINT_LOW_ENRICHMENT_STATUS,
    LOG2FC_G1,
    LOG2FC_G2,
    N_CONTEXT_CONTRIBUTING_COMPONENTS_CONSENSUS,
    N_CONTEXT_ELIGIBLE_CONSENSUS_COMPONENTS,
    N_CONTEXT_HEXES_POOLED,
    N_CONTEXT_HEXES_WITH_FINITE_DENSITY_POOLED,
    N_CONTRIBUTING_COMPONENTS_CONSENSUS,
    N_ELIGIBLE_CONSENSUS_COMPONENTS,
    N_HEXES_POOLED,
    N_HEXES_WITH_FINITE_DENSITY_POOLED,
    OBS_EQUIV_BOTH_HIGH_MASS,
    OBS_EQUIV_BOTH_LOW_MASS,
    OBS_EQUIV_G1_HIGH_MASS,
    OBS_EQUIV_G1_LOW_MASS,
    OBS_EQUIV_G1_SUM_MASS,
    OBS_EQUIV_G2_HIGH_MASS,
    OBS_EQUIV_G2_LOW_MASS,
    OBS_EQUIV_G2_SUM_MASS,
    OBS_EQUIV_SPOT_MASS,
    PARENT_CONSENSUS_EFFECT_SIZE_STATUS,
    PARENT_CONSENSUS_ENRICHMENT_STATUS,
    ROI_ID,
    SAMPLE_ID,
    VALUE_SURFACE,
)
from tilescope.metrics.enrichment import (
    EnrichmentRatioStatus,
    aggregate_context_baselines_from_counts,
    compute_enrichment_ratios_from_counts,
)
from tilescope.metrics.pair_stats_core import PairStatsCore
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import (
    RoiConsensusEnrichmentSchema,
    assert_columns_populated_iff,
    assert_per_ratio_contract,
)

if TYPE_CHECKING:
    from tilescope.metrics.normalization import ScaleNormalizationParams


MIN_HEXES_FOR_EFFECT_SIZE = 1
MASS_COHERENCE_TOL = 1e-9
TARGET_ZERO_HEX_STATUSES = frozenset({
    ParentConsensusEnrichmentStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT.value,
    ParentConsensusEnrichmentStatus.NO_CONSENSUS_HEXES.value,
})
COUNT_MASS_OUTPUTS = tuple(
    (internal, target, context)
    for (internal, target), context in zip(
        COUNT_MASS_COLUMNS,
        (
            CONTEXT_OBS_EQUIV_G1_HIGH_MASS,
            CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
            CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS,
            CONTEXT_OBS_EQUIV_G1_LOW_MASS,
            CONTEXT_OBS_EQUIV_G2_LOW_MASS,
            CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
        ),
    )
)
NUMERIC_OUTPUTS = (
    GENE1_ENRICHMENT, GENE2_ENRICHMENT,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_FC,
    LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2,
    OBS_EQUIV_SPOT_MASS,
    OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G2_HIGH_MASS,
    OBS_EQUIV_BOTH_HIGH_MASS, OBS_EQUIV_G1_LOW_MASS,
    OBS_EQUIV_G2_LOW_MASS, OBS_EQUIV_BOTH_LOW_MASS,
    OBS_EQUIV_G1_SUM_MASS, OBS_EQUIV_G2_SUM_MASS,
    CONTEXT_OBS_EQUIV_SPOT_MASS,
    CONTEXT_OBS_EQUIV_G1_HIGH_MASS, CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
    CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS, CONTEXT_OBS_EQUIV_G1_LOW_MASS,
    CONTEXT_OBS_EQUIV_G2_LOW_MASS, CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
    CONTEXT_OBS_EQUIV_G1_SUM_MASS, CONTEXT_OBS_EQUIV_G2_SUM_MASS,
)


def build_roi_consensus_enrichment_df(
    *,
    consensus_frames: ConsensusAnalysisFrames,
    support_status_df: pd.DataFrame,
    normalization_params: "ScaleNormalizationParams",
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
) -> pd.DataFrame:
    """Emit the ensemble-surface enrichment/effect total table."""
    resolved_sample_id = resolve_sample_id_from_typed_record(
        consensus_frames.sample_id, sample_id,
    )
    roi_ids = sorted(
        support_status_df[ROI_ID].astype(str).unique(), key=natural_sort_key,
    )
    surface = PartitionValueSurface.CONSENSUS_ENSEMBLE
    target_mass = {
        roi_id: consensus_frames.ensemble_mass(roi_id) for roi_id in roi_ids
    }
    missing = [
        roi_id for roi_id, mass in target_mass.items()
        if mass.n_hexes > 0 and not mass.threshold_masses_available
    ]
    if missing:
        raise ValueError(
            "ROI consensus enrichment requires threshold masses on every "
            f"finite ensemble hex; missing for ROI(s) {missing!r}."
        )

    core = PairStatsCore()
    rows = []
    for roi_id in roi_ids:
        lookup_parent_support_row(support_status_df, roi_id)
        target_support = consensus_frames.support(roi_id, surface)
        context_support = consensus_frames.context_support(roi_id, surface)
        row = _empty_row(
            resolved_sample_id, roi_id, g1_id, g2_id,
            target_support, context_support, surface,
        )
        target_gate = _target_gate(target_support)
        if target_gate is not None:
            if target_gate is (
                ParentConsensusEnrichmentStatus.INCOMPLETE_SUPPORT_COVERAGE
            ):
                _write_mass_columns(row, target_mass[roi_id])
                if context_support.n_hexes_with_finite_density > 0:
                    _write_mass_columns(
                        row, consensus_frames.context_ensemble_mass(roi_id),
                        context=True,
                    )
                row[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] = (
                    ParentConsensusEffectSizeStatus
                    .INCOMPLETE_SUPPORT_COVERAGE.value
                )
            row[PARENT_CONSENSUS_ENRICHMENT_STATUS] = target_gate.value
            rows.append(row)
            continue

        target_aggregate = target_mass[roi_id]
        context_aggregate = consensus_frames.context_ensemble_mass(roi_id)
        _write_mass_columns(row, target_aggregate)
        if context_support.n_hexes_with_finite_density > 0:
            _write_mass_columns(row, context_aggregate, context=True)

        status = _enrichment_status_and_values(
            row, target_aggregate, context_aggregate, context_support,
        )
        effect_status = _effect_status_and_values(
            row=row,
            frames=consensus_frames,
            roi_id=roi_id,
            surface=surface,
            enrichment_status=status,
            normalization_params=normalization_params,
            core=core,
        )
        row[PARENT_CONSENSUS_ENRICHMENT_STATUS] = status.value
        row[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] = effect_status.value
        rows.append(row)

    frame = pd.DataFrame(
        rows, columns=list(RoiConsensusEnrichmentSchema.columns),
    )
    frame = RoiConsensusEnrichmentSchema.validate(frame, lazy=True)
    _enforce_coupling(frame)
    return frame


def _empty_row(
    sample_id, roi_id, g1_id, g2_id, target, context, surface,
) -> dict:
    return {
        SAMPLE_ID: sample_id,
        ROI_ID: str(roi_id),
        VALUE_SURFACE: surface.value,
        G1_ID: str(g1_id),
        G2_ID: str(g2_id),
        N_ELIGIBLE_CONSENSUS_COMPONENTS: len(target.eligible_component_ids),
        N_CONTRIBUTING_COMPONENTS_CONSENSUS: len(
            target.contributing_component_ids,
        ),
        N_HEXES_POOLED: target.n_hexes_constructed,
        N_HEXES_WITH_FINITE_DENSITY_POOLED: (
            target.n_hexes_with_finite_density
        ),
        N_CONTEXT_ELIGIBLE_CONSENSUS_COMPONENTS: len(
            context.eligible_component_ids,
        ),
        N_CONTEXT_CONTRIBUTING_COMPONENTS_CONSENSUS: len(
            context.contributing_component_ids,
        ),
        N_CONTEXT_HEXES_POOLED: context.n_hexes_constructed,
        N_CONTEXT_HEXES_WITH_FINITE_DENSITY_POOLED: (
            context.n_hexes_with_finite_density
        ),
        PARENT_CONSENSUS_EFFECT_SIZE_STATUS: (
            ParentConsensusEffectSizeStatus.NOT_COMPUTED.value
        ),
        **{column: float("nan") for column in NUMERIC_OUTPUTS},
        **{column: pd.NA for column in ENRICHMENT_RATIO_STATUS_COLUMNS},
    }


def _target_gate(support) -> Optional[ParentConsensusEnrichmentStatus]:
    if not support.eligible_component_ids:
        return ParentConsensusEnrichmentStatus.NO_TESSELLATION_ELIGIBLE_COMPONENT
    if support.n_hexes_constructed == 0:
        return ParentConsensusEnrichmentStatus.NO_CONSENSUS_HEXES
    if support.n_incomplete_components > 0:
        return ParentConsensusEnrichmentStatus.INCOMPLETE_SUPPORT_COVERAGE
    return None


def _write_mass_columns(
    row: dict, aggregate: ConsensusMassAggregate, *, context: bool = False,
) -> None:
    for internal, target_column, context_column in COUNT_MASS_OUTPUTS:
        row[context_column if context else target_column] = float(
            aggregate.counts[internal]
        )
    row[CONTEXT_OBS_EQUIV_SPOT_MASS if context else OBS_EQUIV_SPOT_MASS] = (
        float(aggregate.spot_mass)
    )
    row[
        CONTEXT_OBS_EQUIV_G1_SUM_MASS if context else OBS_EQUIV_G1_SUM_MASS
    ] = float(aggregate.g1_sum_mass)
    row[
        CONTEXT_OBS_EQUIV_G2_SUM_MASS if context else OBS_EQUIV_G2_SUM_MASS
    ] = float(aggregate.g2_sum_mass)


def _enrichment_status_and_values(
    row: dict,
    target: ConsensusMassAggregate,
    context: ConsensusMassAggregate,
    context_support,
) -> ParentConsensusEnrichmentStatus:
    if not context_support.eligible_component_ids:
        return ParentConsensusEnrichmentStatus.NO_ELIGIBLE_CONTEXT
    if context_support.n_incomplete_components > 0:
        return ParentConsensusEnrichmentStatus.INCOMPLETE_CONTEXT_COVERAGE
    ratios = compute_enrichment_ratios_from_counts(
        counts=pd.DataFrame([target.counts]),
        denominators=pd.Series([target.spot_mass]),
        context_baselines=aggregate_context_baselines_from_counts(
            pd.DataFrame([context.counts]),
            pd.Series([context.spot_mass]),
        ),
    ).iloc[0]
    per_ratio_status = {
        value_col: str(ratios[status_col])
        for value_col, status_col in ENRICHMENT_RATIO_STATUS_PAIRS
    }
    per_ratio_finite = {
        col: status == EnrichmentRatioStatus.FINITE.value
        for col, status in per_ratio_status.items()
    }
    for value_col, status_col in ENRICHMENT_RATIO_STATUS_PAIRS:
        row[status_col] = per_ratio_status[value_col]
        if per_ratio_finite[value_col]:
            row[value_col] = float(ratios[value_col])
    if all(per_ratio_finite.values()):
        return ParentConsensusEnrichmentStatus.FINITE
    return ParentConsensusEnrichmentStatus.CONTEXT_PROPORTION_ZERO


def _effect_status_and_values(
    *, row, frames, roi_id, surface, enrichment_status,
    normalization_params, core,
) -> ParentConsensusEffectSizeStatus:
    if enrichment_status is (
        ParentConsensusEnrichmentStatus.INCOMPLETE_CONTEXT_COVERAGE
    ):
        return ParentConsensusEffectSizeStatus.INCOMPLETE_CONTEXT_COVERAGE
    target = frames.finite_hexes(roi_id, surface)
    context = frames.context_finite_hexes(roi_id, surface)
    if len(target) < MIN_HEXES_FOR_EFFECT_SIZE:
        return ParentConsensusEffectSizeStatus.INSUFFICIENT_TARGET_HEXES
    if len(context) < MIN_HEXES_FOR_EFFECT_SIZE:
        return ParentConsensusEffectSizeStatus.INSUFFICIENT_CONTEXT_HEXES
    target_g1 = target[GENE1_DENSITY_ENSEMBLE].to_numpy(dtype=float)
    target_g2 = target[GENE2_DENSITY_ENSEMBLE].to_numpy(dtype=float)
    context_g1 = context[GENE1_DENSITY_ENSEMBLE].to_numpy(dtype=float)
    context_g2 = context[GENE2_DENSITY_ENSEMBLE].to_numpy(dtype=float)
    row[LOG2FC_G1] = float(core.log2fc(
        target_g1, context_g1, normalization_params.gene1_min_density,
    ))
    row[LOG2FC_G2] = float(core.log2fc(
        target_g2, context_g2, normalization_params.gene2_min_density,
    ))
    row[CLES_G1] = float(core.cles(target_g1, context_g1))
    row[CLES_G2] = float(core.cles(target_g2, context_g2))
    return ParentConsensusEffectSizeStatus.FINITE


_CONSENSUS_AGGREGATE_STATUSES_WITH_PER_RATIO = (
    ParentConsensusEnrichmentStatus.FINITE.value,
    ParentConsensusEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value,
)


def _enforce_coupling(frame: pd.DataFrame) -> None:
    surface = "roi_consensus_enrichment_df"
    assert_per_ratio_contract(
        frame,
        aggregate_status_col=PARENT_CONSENSUS_ENRICHMENT_STATUS,
        aggregate_statuses_with_per_ratio=(
            _CONSENSUS_AGGREGATE_STATUSES_WITH_PER_RATIO
        ),
        ratio_pairs=ENRICHMENT_RATIO_STATUS_PAIRS,
        surface_name=surface,
    )
    effect_finite = frame[PARENT_CONSENSUS_EFFECT_SIZE_STATUS] == (
        ParentConsensusEffectSizeStatus.FINITE.value
    )
    assert_columns_populated_iff(
        frame, (LOG2FC_G1, LOG2FC_G2, CLES_G1, CLES_G2), effect_finite,
        surface_name=surface,
        condition_label="effect-size status FINITE",
    )
    target_available = ~frame[PARENT_CONSENSUS_ENRICHMENT_STATUS].isin(
        TARGET_ZERO_HEX_STATUSES,
    )
    target_mass_columns = (
        OBS_EQUIV_SPOT_MASS,
        OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G2_HIGH_MASS,
        OBS_EQUIV_BOTH_HIGH_MASS,
        OBS_EQUIV_G1_LOW_MASS, OBS_EQUIV_G2_LOW_MASS,
        OBS_EQUIV_BOTH_LOW_MASS,
        OBS_EQUIV_G1_SUM_MASS, OBS_EQUIV_G2_SUM_MASS,
    )
    assert_columns_populated_iff(
        frame, target_mass_columns, target_available,
        surface_name=surface,
        condition_label="target pool has finite-density hexes",
    )
    if frame.empty:
        return
    context_available = target_available & (
        frame[N_CONTEXT_HEXES_WITH_FINITE_DENSITY_POOLED] > 0
    )
    assert_columns_populated_iff(
        frame,
        (
            CONTEXT_OBS_EQUIV_SPOT_MASS,
            CONTEXT_OBS_EQUIV_G1_HIGH_MASS,
            CONTEXT_OBS_EQUIV_G2_HIGH_MASS,
            CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS,
            CONTEXT_OBS_EQUIV_G1_LOW_MASS,
            CONTEXT_OBS_EQUIV_G2_LOW_MASS,
            CONTEXT_OBS_EQUIV_BOTH_LOW_MASS,
            CONTEXT_OBS_EQUIV_G1_SUM_MASS,
            CONTEXT_OBS_EQUIV_G2_SUM_MASS,
        ),
        context_available,
        surface_name=surface,
        condition_label="target available and context has finite hexes",
    )
    _enforce_mass_coherence(frame.loc[target_available])


def _enforce_mass_coherence(frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    def reject(mask: pd.Series, message: str) -> None:
        if bool(mask.any()):
            example = frame.loc[mask].head(1).to_dict(orient="records")[0]
            raise ValueError(
                "roi_consensus_enrichment_df: mass coherence violated — "
                f"{message}. Example row: {example!r}"
            )

    spot = frame[OBS_EQUIV_SPOT_MASS].astype(float)
    for column in (
        OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G2_HIGH_MASS,
        OBS_EQUIV_G1_LOW_MASS, OBS_EQUIV_G2_LOW_MASS,
    ):
        reject(
            frame[column].astype(float) > spot + MASS_COHERENCE_TOL,
            f"{column} > obs_equiv_spot_mass",
        )
    for joint, marginals in (
        (OBS_EQUIV_BOTH_HIGH_MASS,
         (OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G2_HIGH_MASS)),
        (OBS_EQUIV_BOTH_LOW_MASS,
         (OBS_EQUIV_G1_LOW_MASS, OBS_EQUIV_G2_LOW_MASS)),
    ):
        for marginal in marginals:
            reject(
                frame[joint].astype(float)
                > frame[marginal].astype(float) + MASS_COHERENCE_TOL,
                f"{joint} > {marginal}",
            )
    for high, low in (
        (OBS_EQUIV_G1_HIGH_MASS, OBS_EQUIV_G1_LOW_MASS),
        (OBS_EQUIV_G2_HIGH_MASS, OBS_EQUIV_G2_LOW_MASS),
    ):
        reject(
            frame[high].astype(float) + frame[low].astype(float)
            > spot + MASS_COHERENCE_TOL,
            f"{high} + {low} > obs_equiv_spot_mass",
        )
