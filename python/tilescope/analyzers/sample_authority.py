"""Internal multi-pair authority engine.

`_run_sample_authority` runs the substrate stages exactly once
(support bank, tessellation, consensus spatial bank), projects each
unique feature exactly once, and computes each ordered pair exactly
once. Returns a typed `SampleAnalysisResult`; the public boundary
is `tilescope.run_sample` and the legacy `run_analysis` adapter is
in `tilescope.analyzers.orchestrator`.
"""
from __future__ import annotations

from tilescope.schema.helpers import natural_sort_key
from tilescope.analyzers.analysis_output import assemble_pair_output


import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tilescope.analyzers.analysis_output import build_spot_ledger_frame
from tilescope.analyzers.consensus_frames import (
    ConsensusSpatialFrame, FeatureConsensusFrame,
    build_consensus_analysis_frames, build_consensus_spatial_frame,
    build_feature_consensus_frame,
)
from tilescope.analyzers.consensus_stage import (
    SampleConsensusResult, build_consensus_spatial_bank,
    run_pair_consensus_from_bank,
)
from tilescope.analyzers.cross_grain_adapter import (
    CrossGrainPairFrames, build_cross_grain_inputs,
)
from tilescope.analyzers.feature_projection_stage import (
    build_feature_spatial_projection,
)
from tilescope.analyzers.parent_roi_analyzer import analyze_parent_rois
from tilescope.analyzers.parent_roi_projection import (
    FeatureParentRoiProjection, ParentRoiSupportIndex,
    build_feature_parent_roi_projection, build_parent_roi_support_index,
)
from tilescope.analyzers.component_radial_shell_correlogram import (
    PreparedRadialShellGeometry,
    build_component_radial_shell_correlogram_df,
    prepare_component_radial_shell_geometry,
)
from tilescope.analyzers.parent_roi_consensus_correlation import (
    build_roi_consensus_correlations_df,
)
from tilescope.analyzers.parent_roi_consensus_enrichment import (
    build_roi_consensus_enrichment_df,
)
from tilescope.analyzers.parent_roi_interface_response import (
    build_sample_feature_interface_response_df,
    select_pair_interface_response_df,
)
from tilescope.analyzers.parent_roi_consensus_moran import (
    build_roi_consensus_moran_df,
)
from tilescope.analyzers.parent_roi_source_moran import (
    build_roi_source_moran_df,
)
from tilescope.analyzers.parent_roi_tessellation_moran import (
    build_roi_tessellation_moran_df,
)
from tilescope.analyzers.parent_roi_tessellation_common import (
    bind_tessellation_pooling_index, build_tessellation_support_index,
)
from tilescope.analyzers.parent_roi_tessellation_correlation import (
    build_roi_replicate_correlations_df,
    compute_roi_tessellation_correlations,
)
from tilescope.analyzers.parent_roi_tessellation_enrichment import (
    build_roi_replicate_enrichment_df,
    compute_roi_tessellation_enrichment,
)
from tilescope.analyzers.pre_analysis import (
    apply_scattered_gate_mask, apply_splitter_and_filter,
    SampleSupportClassification, classify_sample_support_and_status,
    prepare_source_frame,
)
from tilescope.analyzers.roi_cross_grain_sensitivity import (
    build_roi_cross_grain_sensitivity_df,
)
from tilescope.analyzers.sample_analysis_result import (
    PairKey, PairStatisticResult, SampleAnalysisResult, SampleSupportOutputs,
)
from tilescope.analyzers.sample_support_stage import build_empty_sample_support_bank
from tilescope.analyzers.tessellation_analysis import (
    SampleTessellationResult, run_tessellation_stage,
)
from tilescope.cols import (
    COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID,
)
from tilescope.component_data import ComponentData
from tilescope.defaults import NORM_GENE1_MIN, NORM_GENE2_MIN
from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.metrics.enrichment_thresholds import PercentileThresholds
from tilescope.metrics.normalization import ScaleNormalizationParams
from tilescope.analyzers.consensus_hex_surface import build_consensus_hexes_df
from tilescope.analyzers.sample_support_frames import (
    build_component_domain_regions, build_component_geometry,
)
from tilescope.analyzers.tessellation_outcome_frames import (
    build_tessellation_outcome_tables,
)
from tilescope.analyzers.tessellation_tile_frames import (
    build_tessellation_tile_support, build_tessellation_tiles_df,
)
from tilescope.settings import ResolvedRunConfig
from tilescope.spatial import Canvas, make_canvas, precision_grid_size
from tilescope.spatial.component_domains import (
    ComponentDomainDiagnostic, DomainRegionDiagnostic,
    build_component_polygons,
)
from tilescope.spatial.consensus_bank import SampleSpatialBank
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
)
from tilescope.spatial.source_spot_table import reject_feature_nonfinite
from tilescope.spatial.spacing import median_nn_distance_from_df

logger = logging.getLogger(__name__)


def _pair_thresholds_from_projections(
    projection_g1: FeatureSpatialProjection,
    projection_g2: FeatureSpatialProjection,
    *,
    min_points_threshold: int,
) -> PercentileThresholds:
    """Compose pair-scope thresholds from the two projections' stored
    per-feature `FeatureThresholds`; no re-quantile pass. Projection
    is the sole marginal-threshold authority."""
    t1 = projection_g1.thresholds
    t2 = projection_g2.thresholds
    if (t1.q_high, t1.q_low) != (t2.q_high, t2.q_low):
        raise ValueError(
            "_pair_thresholds_from_projections: projections carry "
            f"different quantile bases ({(t1.q_high, t1.q_low)!r} vs "
            f"{(t2.q_high, t2.q_low)!r})."
        )
    return PercentileThresholds(
        q_g1_threshold=t1.resolved_high, q_g2_threshold=t2.resolved_high,
        q_percentile=t1.q_high,
        q_g1_low_threshold=t1.resolved_low, q_g2_low_threshold=t2.resolved_low,
        q_low_percentile=t1.q_low,
        min_points_threshold=int(min_points_threshold),
    )


def _project_feature(
    *, feature_name: str, source_expression_df: pd.DataFrame,
    config: ResolvedRunConfig, spatial_bank,
) -> FeatureSpatialProjection:
    """Materialise one feature's projection over the bank."""
    values = source_expression_df[feature_name].to_numpy(dtype=float)
    high = float(np.quantile(values, config.resolved_policy.thresholds.q_high))
    low = float(np.quantile(values, config.resolved_policy.thresholds.q_low))
    return build_feature_spatial_projection(
        feature_name=feature_name,
        source_values=values,
        thresholds=FeatureThresholds(
            q_high=float(config.resolved_policy.thresholds.q_high),
            q_low=float(config.resolved_policy.thresholds.q_low),
            resolved_high=high, resolved_low=low,
            n_source_values_used=int(values.size),
        ),
        spatial_bank=spatial_bank,
    )


@dataclass(frozen=True)
class _FeatureAuthorityBundle:
    """All once-per-feature authorities for one sample invocation."""

    spatial: Mapping[str, FeatureSpatialProjection]
    consensus: Mapping[str, FeatureConsensusFrame]
    parent_roi: Mapping[str, FeatureParentRoiProjection]


def _build_feature_authorities(
    *,
    feature_names: Sequence[str],
    source_expression_df: pd.DataFrame,
    config: ResolvedRunConfig,
    spatial_bank,
    consensus_spatial_frame: ConsensusSpatialFrame,
    parent_roi_support_index: ParentRoiSupportIndex,
) -> _FeatureAuthorityBundle:
    """Build each requested feature's projections exactly once."""
    spatial = {}
    consensus = {}
    parent_roi = {}
    for feature in feature_names:
        spatial[feature] = _project_feature(
            feature_name=feature,
            source_expression_df=source_expression_df,
            config=config, spatial_bank=spatial_bank,
        )
        consensus[feature] = build_feature_consensus_frame(
            spatial_frame=consensus_spatial_frame,
            feature_projection=spatial[feature],
        )
        parent_roi[feature] = build_feature_parent_roi_projection(
            feature_name=feature,
            source_values=spatial[feature].source_values,
            high_threshold=spatial[feature].thresholds.resolved_high,
            low_threshold=spatial[feature].thresholds.resolved_low,
            support_index=parent_roi_support_index,
            gene_min_value=float(source_expression_df[feature].min()),
            min_roi_spots=config.run_config.roi_filters.min_roi_spots,
            min_context_spots=config.run_config.roi_filters.min_context_spots,
        )
    return _FeatureAuthorityBundle(
        spatial=MappingProxyType(spatial),
        consensus=MappingProxyType(consensus),
        parent_roi=MappingProxyType(parent_roi),
    )


def _normalization_params(
    *, gene1_min_spot, gene2_min_spot, observations,
    gene1_density_column, gene2_density_column,
) -> ScaleNormalizationParams:
    def _dmin(frame, col):
        if frame.empty or col not in frame.columns:
            return 0.0
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        return float(finite.min()) if finite.size else 0.0
    return ScaleNormalizationParams(
        gene1_min_spot=gene1_min_spot, gene2_min_spot=gene2_min_spot,
        gene1_min_density=_dmin(observations, gene1_density_column),
        gene2_min_density=_dmin(observations, gene2_density_column),
    )


@dataclass(frozen=True)
class PairMixedFrames:
    """Per-pair mixed-shape frames: transient (drop after one flatten)."""
    tessellation_tiles_df: pd.DataFrame
    consensus_hexes_df: pd.DataFrame
    tessellation_pooling_index: Any


@dataclass(frozen=True)
class PairPipelineResult:
    """Transient container returned by `_run_pair_pipeline`. The pair
    loop retains only `pair_statistic` in `SampleAnalysisResult`;
    `mixed_frames` + `thresholds` flow to the streaming sink and are
    dropped before the next pair."""
    pair_statistic: PairStatisticResult
    mixed_frames: PairMixedFrames
    thresholds: PercentileThresholds


PairSink = Callable[[PairKey, Dict[str, object]], None]


@dataclass(frozen=True)
class _PreparedSampleSpatialState:
    """Supported or empty spatial substrate feeding one common continuation."""
    canvas: Canvas
    component_to_roi: Dict[str, str]
    component_polygons: Dict[str, Any]
    component_domain_diagnostics: Tuple[ComponentDomainDiagnostic, ...]
    region_diagnostics: Tuple[DomainRegionDiagnostic, ...]
    retained_components: Dict[str, pd.DataFrame]
    sample_tessellation: SampleTessellationResult
    spatial_bank: SampleSpatialBank
    precision: float
    median_nn_spacing: float


def _prepare_sample_spatial_state(
    *,
    source_expression_df: pd.DataFrame,
    component_expression_df: pd.DataFrame,
    initial_canvas: Canvas,
    support: SampleSupportClassification,
    all_features: Tuple[str, ...],
    config: ResolvedRunConfig,
    dataset_id: str,
) -> _PreparedSampleSpatialState:
    """Build either supported geometry or its typed empty counterpart."""
    if support.all_scattered:
        empty_bank = build_empty_sample_support_bank(
            sample_id=dataset_id,
            source_spot_adjacency=support.source_spot_adjacency,
            config=config,
            n_replicates=config.run_config.tessellation.n,
            base_seed=config.run_config.seed,
        )
        sample_tessellation = SampleTessellationResult(
            sample_id=dataset_id,
            components=MappingProxyType({}),
            n_replicates_planned=config.run_config.tessellation.n,
            support_bank=empty_bank,
        )
        spatial_bank = SampleConsensusResult.from_components(
            sample_id=dataset_id,
            components=MappingProxyType({}),
            n_replicates_planned=config.run_config.tessellation.n,
            sample_anchor_x=0.0,
            sample_anchor_y=0.0,
            support_bank=empty_bank,
        ).spatial_bank
        return _PreparedSampleSpatialState(
            canvas=initial_canvas,
            component_to_roi={},
            component_polygons={},
            component_domain_diagnostics=(),
            region_diagnostics=(),
            retained_components={},
            sample_tessellation=sample_tessellation,
            spatial_bank=spatial_bank,
            precision=0.0,
            median_nn_spacing=0.0,
        )

    component_to_roi = _build_component_to_roi(component_expression_df)
    canvas = make_canvas(
        component_expression_df,
        target_px_per_step=1,
        target_px_per_nn=4,
        margin_px=20,
    )
    spatial_expression_df = apply_scattered_gate_mask(
        component_expression_df,
        support.scattered_rois,
        sample_median_nn=support.sample_median_nn,
    )
    retained_components = _prepare_component_datasets(spatial_expression_df)
    if not retained_components:
        raise ValueError("_run_sample_authority: no components retained.")

    polygon_build = build_component_polygons(
        source_expression_df,
        spatial_expression_df,
        component_to_roi,
        source_spot_adjacency=support.source_spot_adjacency,
        sample_median_nn=support.sample_median_nn,
        component_support_radius_factor=(
            config.resolved_policy.component_domain_radii
            .component_support_radius_factor
        ),
        acquisition_support_radius_factor=(
            config.resolved_policy.component_domain_radii
            .acquisition_support_radius_factor
        ),
        ineligible_spot_exclusion_radius_factor=(
            config.resolved_policy.component_domain_radii
            .ineligible_spot_exclusion_radius_factor
        ),
    )
    spatial_components: Dict[str, ComponentData] = {}
    for component_id, geometry in polygon_build.component_polygons.items():
        if geometry.is_empty:
            continue
        component_rows = retained_components[str(component_id)]
        spatial_components[str(component_id)] = ComponentData(
            component_id=str(component_id),
            points=component_rows[[COORD_X, COORD_Y]].to_numpy(dtype=float),
            canvas=canvas,
            polygon=geometry,
            expression_data=component_rows[
                [SOURCE_ROW_ID] + list(all_features) + [COORD_X, COORD_Y]
            ].copy(),
        )
    sample_tessellation = run_tessellation_stage(
        sample_id=dataset_id,
        spatial_components=spatial_components,
        domain_partitions=polygon_build.partitions,
        component_to_roi=component_to_roi,
        n_replicates=config.run_config.tessellation.n,
        tile_min_points=(
            config.run_config.tessellation.min_observations_per_tile
        ),
        voronoi_fraction=config.run_config.tessellation.voronoi_fraction,
        base_seed=config.run_config.seed,
        source_spot_adjacency=support.source_spot_adjacency,
        config=config,
    )
    spatial_bank = build_consensus_spatial_bank(
        sample_tessellation_result=sample_tessellation,
        canvas=canvas,
        sample_anchor=(
            float(source_expression_df[COORD_X].min()),
            float(source_expression_df[COORD_Y].min()),
        ),
        consensus_config=config,
    )
    return _PreparedSampleSpatialState(
        canvas=canvas,
        component_to_roi=component_to_roi,
        component_polygons=dict(polygon_build.component_polygons),
        component_domain_diagnostics=tuple(polygon_build.component_diagnostics),
        region_diagnostics=tuple(polygon_build.region_diagnostics),
        retained_components=retained_components,
        sample_tessellation=sample_tessellation,
        spatial_bank=spatial_bank,
        precision=precision_grid_size(canvas),
        median_nn_spacing=median_nn_distance_from_df(
            component_expression_df, COORD_X, COORD_Y,
        ),
    )


def _resolve_pair_identity(
    pair_key: PairKey,
    pair_identities: Optional[Dict[PairKey, Tuple[str, str]]],
) -> Tuple[str, str]:
    """Return semantic IDs, rejecting reserved canonical column names."""
    ids = pair_identities.get(pair_key) if pair_identities is not None else None
    candidate = ids if ids is not None else pair_key
    return require_semantic_feature_ids(
        candidate[0], candidate[1], owner=f"pair identity {pair_key!r}",
    )


def _resolve_feature_identities(
    pair_features: Sequence[PairKey],
    pair_identities: Optional[Dict[PairKey, Tuple[str, str]]],
) -> Mapping[str, str]:
    """Resolve one stable semantic identity per computational feature."""
    resolved: Dict[str, str] = {}
    feature_by_id: Dict[str, str] = {}
    for pair_key in pair_features:
        pair_ids = _resolve_pair_identity(pair_key, pair_identities)
        for feature_name, feature_id in zip(pair_key, pair_ids):
            previous = resolved.setdefault(feature_name, feature_id)
            if previous != feature_id:
                raise ValueError(
                    f"Feature {feature_name!r} has conflicting semantic "
                    f"identities {previous!r} and {feature_id!r} across pairs."
                )
            previous_feature = feature_by_id.setdefault(
                feature_id, feature_name,
            )
            if previous_feature != feature_name:
                raise ValueError(
                    f"Semantic identity {feature_id!r} is assigned to both "
                    f"{previous_feature!r} and {feature_name!r}."
                )
    return MappingProxyType(resolved)


def _run_pair_pipeline(
    *,
    feature_g1: str, feature_g2: str,
    feature_g1_id: str, feature_g2_id: str,
    source_expression_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
    sample_tess_result,
    tessellation_tile_support,
    tessellation_support_index,
    spatial_bank,
    projection_g1: FeatureSpatialProjection,
    projection_g2: FeatureSpatialProjection,
    consensus_spatial_frame: ConsensusSpatialFrame,
    consensus_feature_g1: FeatureConsensusFrame,
    consensus_feature_g2: FeatureConsensusFrame,
    parent_roi_support_index: ParentRoiSupportIndex,
    parent_projection_g1: FeatureParentRoiProjection,
    parent_projection_g2: FeatureParentRoiProjection,
    sample_feature_interface_response_df: pd.DataFrame,
    radial_geometry_by_component: Mapping[
        str, PreparedRadialShellGeometry
    ],
    moran_weight_cache: Dict[tuple, Any],
    dataset_id: str,
    config: ResolvedRunConfig,
) -> PairPipelineResult:
    """Run one pair through the ROI-grain builders on a shared bank.

    Returns the transient `PairPipelineResult`: durable `pair_statistic`
    (kept in `SampleAnalysisResult`) plus one-shot `mixed_frames` +
    `thresholds` for the export sink to consume before the next pair.

    `feature_g1_id`/`feature_g2_id` are the SEMANTIC identities that
    become the `g1_id`/`g2_id` columns on every emitted CSV frame.
    They are required whenever the computational columns are generic.
    """
    thresholds = _pair_thresholds_from_projections(
        projection_g1, projection_g2,
        min_points_threshold=(
            config.run_config.tessellation.min_observations_per_tile
        ),
    )
    gene1_min_spot = float(source_expression_df[feature_g1].min())
    gene2_min_spot = float(source_expression_df[feature_g2].min())
    roi_enrichment_df, roi_correlations_df = analyze_parent_rois(
        source_expression_df=source_expression_df,
        thresholds=thresholds.to_dict(),
        normalization_params={
            NORM_GENE1_MIN: gene1_min_spot, NORM_GENE2_MIN: gene2_min_spot,
        },
        min_roi_spots=config.run_config.roi_filters.min_roi_spots,
        min_context_spots=config.run_config.roi_filters.min_context_spots,
        sample_id=dataset_id, seed=config.run_config.seed,
        n_permutations=config.run_config.permutations,
        gene1_col=feature_g1, gene2_col=feature_g2,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
        support_index=parent_roi_support_index,
        feature_projection_g1=parent_projection_g1,
        feature_projection_g2=parent_projection_g2,
    )
    tiles_df = build_tessellation_tiles_df(
        tile_support=tessellation_tile_support,
        thresholds=thresholds,
        projection_g1=projection_g1, projection_g2=projection_g2,
    )
    voronoi_norm = _normalization_params(
        gene1_min_spot=gene1_min_spot, gene2_min_spot=gene2_min_spot,
        observations=tiles_df,
        gene1_density_column="g1_density", gene2_density_column="g2_density",
    )
    consensus_result = run_pair_consensus_from_bank(
        spatial_bank=spatial_bank,
        sample_tessellation_result=sample_tess_result,
        tessellation_tiles_df=tiles_df, thresholds=thresholds,
    )
    consensus_frames = build_consensus_analysis_frames(
        sample_consensus_result=consensus_result,
        spatial_frame=consensus_spatial_frame,
        feature_frame_g1=consensus_feature_g1,
        feature_frame_g2=consensus_feature_g2,
    )
    consensus_norm = _normalization_params(
        gene1_min_spot=gene1_min_spot, gene2_min_spot=gene2_min_spot,
        observations=consensus_frames.hex_observations,
        gene1_density_column="gene1_density_ensemble",
        gene2_density_column="gene2_density_ensemble",
    )
    pooling = bind_tessellation_pooling_index(
        support_index=tessellation_support_index,
        tessellation_tiles_df=tiles_df,
    )
    replicate_corr = build_roi_replicate_correlations_df(
        pooling_index=pooling, support_status_df=support_status_df,
        sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
    )
    tess_corr = compute_roi_tessellation_correlations(
        roi_replicate_correlations_df=replicate_corr,
        support_status_df=support_status_df,
        sample_tess_result=sample_tess_result,
        projection_g1=projection_g1, projection_g2=projection_g2,
        n_permutations=int(config.run_config.permutations), seed=int(config.run_config.seed),
        sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
    )
    replicate_enr = build_roi_replicate_enrichment_df(
        pooling_index=pooling, support_status_df=support_status_df,
        thresholds=thresholds, normalization_params=voronoi_norm,
        sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
    )
    tess_enr = compute_roi_tessellation_enrichment(
        roi_replicate_enrichment_df=replicate_enr,
        support_status_df=support_status_df, sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
    )
    consensus_corr = build_roi_consensus_correlations_df(
        consensus_frames=consensus_frames,
        support_status_df=support_status_df, sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
    )
    consensus_enr = build_roi_consensus_enrichment_df(
        consensus_frames=consensus_frames,
        support_status_df=support_status_df,
        normalization_params=consensus_norm, sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
    )
    hexes_df = build_consensus_hexes_df(
        consensus_frames=consensus_frames,
        roi_consensus_enrichment_df=consensus_enr,
    )
    k_neighbors = int(
        config.resolved_policy.consensus.bivariate_morans.k_neighbors,
    )
    moran_enabled = bool(
        config.resolved_policy.consensus.bivariate_morans.compute,
    )
    source_moran_df = build_roi_source_moran_df(
        source_spot_adjacency=spatial_bank.support_bank.source_spot_adjacency,
        source_values_g1=projection_g1.source_values,
        source_values_g2=projection_g2.source_values,
        support_status_df=support_status_df,
        sample_id=dataset_id,
        moran_enabled=moran_enabled,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
        weight_cache=moran_weight_cache,
    )
    tessellation_moran_df = build_roi_tessellation_moran_df(
        pooling_index=pooling,
        tessellation_tiles_df=tiles_df,
        support_status_df=support_status_df,
        k_neighbors=k_neighbors,
        moran_enabled=moran_enabled,
        sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
        weight_cache=moran_weight_cache,
    )
    consensus_moran_df = build_roi_consensus_moran_df(
        consensus_frames=consensus_frames,
        support_status_df=support_status_df,
        k_neighbors=k_neighbors,
        moran_enabled=moran_enabled,
        sample_id=dataset_id,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
        weight_cache=moran_weight_cache,
    )
    interface_response_df = select_pair_interface_response_df(
        sample_feature_interface_response_df,
        feature_g1_id=feature_g1_id, feature_g2_id=feature_g2_id,
    )
    component_radial_shell_df = build_component_radial_shell_correlogram_df(
        support_bank=spatial_bank.support_bank,
        projection_g1=projection_g1, projection_g2=projection_g2,
        radial_shell_policy=config.resolved_policy.radial_shell,
        g1_id=feature_g1_id, g2_id=feature_g2_id,
        prepared_geometry=radial_geometry_by_component,
    )
    cross_grain_frames = CrossGrainPairFrames(
        feature_g1_id=feature_g1_id,
        feature_g2_id=feature_g2_id,
        roi_correlations_df=roi_correlations_df,
        roi_enrichment_df=roi_enrichment_df,
        roi_source_moran_df=source_moran_df,
        roi_replicate_correlations_df=replicate_corr,
        roi_tessellation_correlations_df=tess_corr,
        roi_replicate_enrichment_df=replicate_enr,
        roi_tessellation_moran_df=tessellation_moran_df,
        roi_consensus_correlations_df=consensus_corr,
        roi_consensus_enrichment_df=consensus_enr,
        roi_consensus_moran_df=consensus_moran_df,
    )
    cross_grain_df = build_roi_cross_grain_sensitivity_df(
        build_cross_grain_inputs(
            sample_id=dataset_id, pair_frames=cross_grain_frames,
        )
    )
    pair_statistic = PairStatisticResult(
        feature_g1=feature_g1, feature_g2=feature_g2,
        feature_g1_id=feature_g1_id, feature_g2_id=feature_g2_id,
        bank_fingerprint=spatial_bank.support_bank.fingerprint,
        roi_correlations_df=roi_correlations_df,
        roi_replicate_correlations_df=replicate_corr,
        roi_tessellation_correlations_df=tess_corr,
        roi_enrichment_df=roi_enrichment_df,
        roi_replicate_enrichment_df=replicate_enr,
        roi_tessellation_enrichment_df=tess_enr,
        roi_consensus_correlations_df=consensus_corr,
        roi_consensus_enrichment_df=consensus_enr,
        roi_source_moran_df=source_moran_df,
        roi_tessellation_moran_df=tessellation_moran_df,
        roi_consensus_moran_df=consensus_moran_df,
        roi_cross_grain_sensitivity_df=cross_grain_df,
        roi_pair_interface_response_df=interface_response_df,
        component_radial_shell_correlogram_df=component_radial_shell_df,
    )
    return PairPipelineResult(
        pair_statistic=pair_statistic,
        mixed_frames=PairMixedFrames(
            tessellation_tiles_df=tiles_df,
            consensus_hexes_df=hexes_df,
            tessellation_pooling_index=pooling,
        ),
        thresholds=thresholds,
    )


def _run_sample_authority(
    *,
    expression_df: pd.DataFrame,
    config: ResolvedRunConfig,
    pair_features: Sequence[PairKey],
    dataset_id: str = "1",
    pair_sink: Optional[PairSink] = None,
    pair_identities: Optional[Dict[PairKey, Tuple[str, str]]] = None,
) -> SampleAnalysisResult:
    """Internal multi-pair engine.

    Substrate built once; each unique feature projected once; each
    ordered pair computed once. If `pair_sink` is provided, it is
    invoked once per pair with the transient legacy-shape dict
    (composed from that pair's `PairPipelineResult`) and dropped
    before the next pair: the streaming per-pair export boundary.
    All bundle-scope state stays local; only the durable
    `SampleAnalysisResult` escapes."""
    if not pair_features:
        raise ValueError("_run_sample_authority: pair_features must be non-empty.")
    # Validate + canonicalise pair list. Duplicates are rejected (silent
    # dedupe hides caller-side errors); ordering is natural-sort ascending
    # so multi-pair results are order-independent by construction.
    seen: Dict[PairKey, None] = {}
    for pair in pair_features:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"_run_sample_authority: pair {pair!r} must be a 2-tuple."
            )
        g1, g2 = pair
        if not isinstance(g1, str) or not isinstance(g2, str):
            raise ValueError(
                f"_run_sample_authority: pair {pair!r} feature names must be str."
            )
        if not g1 or not g2:
            raise ValueError(
                f"_run_sample_authority: pair {pair!r} feature names must be non-empty."
            )
        if (g1, g2) in seen:
            raise ValueError(
                f"_run_sample_authority: duplicate pair {(g1, g2)!r}."
            )
        seen[(g1, g2)] = None
    pair_features = tuple(sorted(
        seen.keys(),
        key=lambda p: (natural_sort_key(p[0]), natural_sort_key(p[1])),
    ))
    feature_identities = _resolve_feature_identities(
        pair_features, pair_identities,
    )

    source_expression_df = prepare_source_frame(expression_df)
    all_features = tuple(sorted(
        {f for pair in pair_features for f in pair},
        key=natural_sort_key,
    ))
    reject_feature_nonfinite(source_expression_df, all_features)
    parent_roi_support_index = build_parent_roi_support_index(
        source_expression_df,
    )

    source_policy = config.resolved_policy.source_adjacency
    support = classify_sample_support_and_status(
        source_expression_df,
        min_component_spots=(
            config.run_config.tessellation.min_component_observations
        ),
        technology=config.run_config.technology,
        radius_factor=(
            source_policy.radius_factor
            if source_policy.radius_factor is not None else 1.5
        ),
        local_factor=(
            source_policy.local_factor
            if source_policy.local_factor is not None else 1.5
        ),
        global_cap_factor=(
            source_policy.global_cap_factor
            if source_policy.global_cap_factor is not None else 4.0
        ),
    )
    canvas = make_canvas(expression_df)
    split_filter = apply_splitter_and_filter(
        source_expression_df,
        min_component_spots=(
            config.run_config.tessellation.min_component_observations
        ),
        canvas=canvas,
        source_spot_adjacency=support.source_spot_adjacency,
    )
    expression_df_split = split_filter.split_df
    component_expression_df = split_filter.component_expression_df
    if component_expression_df.empty and not support.all_scattered:
        raise ValueError(
            "Invariant violation: component_expression_df is empty but "
            "sample-support classification retains supported ROIs."
        )
    prepared = _prepare_sample_spatial_state(
        source_expression_df=source_expression_df,
        component_expression_df=component_expression_df,
        initial_canvas=canvas,
        support=support,
        all_features=all_features,
        config=config,
        dataset_id=dataset_id,
    )
    canvas = prepared.canvas
    component_to_roi = prepared.component_to_roi
    retained_components = prepared.retained_components
    sample_tess = prepared.sample_tessellation
    spatial_bank = prepared.spatial_bank

    # Consensus support rows (identity, WKT, occupancy, ROI/context indices)
    # are feature-independent and materialised EXACTLY once per sample.
    consensus_spatial_frame = build_consensus_spatial_frame(
        spatial_bank=spatial_bank,
        roi_ids=support.support_status_df[ROI_ID].astype(str),
    )

    # A10: every requested feature's spatial, consensus, and parent-ROI
    # projections are built together exactly once.
    feature_authorities = _build_feature_authorities(
        feature_names=all_features,
        source_expression_df=source_expression_df,
        config=config, spatial_bank=spatial_bank,
        consensus_spatial_frame=consensus_spatial_frame,
        parent_roi_support_index=parent_roi_support_index,
    )
    sample_feature_interface_response_df = (
        build_sample_feature_interface_response_df(
            sample_id=dataset_id,
            spatial_bank=spatial_bank,
            feature_projections={
                feature_identities[name]: feature_authorities.spatial[name]
                for name in all_features
            },
            support_status_df=support.support_status_df,
        )
    )
    radial_geometry_by_component = prepare_component_radial_shell_geometry(
        support_bank=spatial_bank.support_bank,
        radial_shell_policy=config.resolved_policy.radial_shell,
    )
    moran_weight_cache: Dict[tuple, Any] = {}

    # Sample-scope support outputs (feature-independent): built BEFORE
    # the pair loop so the streaming sink can compose per-pair export
    # dicts from durable frames + one-shot pair transients.
    tessellation_tables = build_tessellation_outcome_tables(sample_tess)
    component_geometry_df = build_component_geometry(
        sample_id=dataset_id,
        component_polygons=prepared.component_polygons,
        component_to_roi=component_to_roi,
        component_domain_diagnostics=prepared.component_domain_diagnostics,
        sample_median_nn=support.sample_median_nn,
        component_support_radius_factor=(
            config.resolved_policy.component_domain_radii.component_support_radius_factor
        ),
        acquisition_support_radius_factor=(
            config.resolved_policy.component_domain_radii.acquisition_support_radius_factor
        ),
        ineligible_spot_exclusion_radius_factor=(
            config.resolved_policy.component_domain_radii.ineligible_spot_exclusion_radius_factor
        ),
        tessellation_eligibility_frame=(
            tessellation_tables.component_tessellation_eligibility_fields
        ),
    )
    component_regions_df = build_component_domain_regions(
        sample_id=dataset_id,
        region_diagnostics=prepared.region_diagnostics,
        component_to_roi=component_to_roi,
        tessellation_domain_regions_frame=(
            tessellation_tables.component_domain_regions_fields
        ),
    )
    spot_ledger = build_spot_ledger_frame(
        source_expression_df=source_expression_df,
        expression_df_split=expression_df_split,
        component_expression_df=component_expression_df,
        support_status_df=support.support_status_df,
        sample_tess_result=sample_tess,
    )
    sample_support_outputs = SampleSupportOutputs(
        component_geometry_df=component_geometry_df,
        component_domain_regions_df=component_regions_df,
        component_region_replicate_status_df=(
            tessellation_tables.component_region_replicate_status
        ),
        roi_support_status_df=support.support_status_df,
        spot_ledger_df=spot_ledger,
    )
    tessellation_tile_support = build_tessellation_tile_support(
        sample_tess_result=sample_tess, sample_id=dataset_id,
    )
    tessellation_support_index = build_tessellation_support_index(
        sample_tess_result=sample_tess,
        tessellation_tile_support_df=(
            tessellation_tile_support.support_frame
        ),
        support_status_df=support.support_status_df,
    )
    precision = prepared.precision
    d1 = prepared.median_nn_spacing
    component_polygons = prepared.component_polygons

    # Per-pair pipeline (each ordered pair computed once). The transient
    # `PairPipelineResult` feeds the sink then falls out of scope.
    pair_results: Dict[PairKey, PairStatisticResult] = {}
    for pair_key in pair_features:
        g1, g2 = pair_key
        g1_id, g2_id = feature_identities[g1], feature_identities[g2]
        pipeline_result = _run_pair_pipeline(
            feature_g1=g1, feature_g2=g2,
            feature_g1_id=g1_id, feature_g2_id=g2_id,
            source_expression_df=source_expression_df,
            support_status_df=support.support_status_df,
            sample_tess_result=sample_tess,
            tessellation_tile_support=tessellation_tile_support,
            tessellation_support_index=tessellation_support_index,
            spatial_bank=spatial_bank,
            projection_g1=feature_authorities.spatial[g1],
            projection_g2=feature_authorities.spatial[g2],
            consensus_spatial_frame=consensus_spatial_frame,
            consensus_feature_g1=feature_authorities.consensus[g1],
            consensus_feature_g2=feature_authorities.consensus[g2],
            parent_roi_support_index=parent_roi_support_index,
            parent_projection_g1=feature_authorities.parent_roi[g1],
            parent_projection_g2=feature_authorities.parent_roi[g2],
            sample_feature_interface_response_df=(
                sample_feature_interface_response_df
            ),
            radial_geometry_by_component=radial_geometry_by_component,
            moran_weight_cache=moran_weight_cache,
            dataset_id=dataset_id, config=config,
        )
        pair_results[pair_key] = pipeline_result.pair_statistic
        if pair_sink is not None:
            _emit_pair_to_sink(
                pair_sink=pair_sink, pair_key=pair_key,
                pipeline_result=pipeline_result,
                sample_id=str(dataset_id),
                sample_support=sample_support_outputs,
                canvas=canvas, component_polygons=component_polygons,
                component_to_roi=component_to_roi,
                source_expression_df=source_expression_df,
                component_expression_df=component_expression_df,
                retained_component_datasets=retained_components,
                sample_median_nn=support.sample_median_nn,
                precision=precision, d1=d1, config=config,
            )
        # `pipeline_result` (containing mixed_frames + thresholds) is
        # dropped here; no pair-scope state survives across iterations.

    return SampleAnalysisResult(
        sample_id=str(dataset_id),
        spatial_bank=spatial_bank,
        feature_projections=feature_authorities.spatial,
        sample_support_outputs=sample_support_outputs,
        pair_results=MappingProxyType({
            pk: pr for pk, pr in pair_results.items()
        }),
    )


def _emit_pair_to_sink(
    *, pair_sink: PairSink, pair_key: PairKey,
    pipeline_result: PairPipelineResult,
    sample_id: str, sample_support: SampleSupportOutputs,
    canvas: Any, component_polygons: Dict[str, Any],
    component_to_roi: Dict[str, str],
    source_expression_df: pd.DataFrame,
    component_expression_df: pd.DataFrame,
    retained_component_datasets: Dict[str, pd.DataFrame],
    sample_median_nn: float, precision: float, d1: float,
    config: ResolvedRunConfig,
) -> None:
    """Compose the legacy-shape dict for one pair from its transient
    pipeline result + sample-scope substrate, and hand it to the sink.
    Nothing is retained; the caller drops the dict before the next pair."""
    export_dict = assemble_pair_output(
        sample_id=sample_id,
        pair=pipeline_result.pair_statistic,
        sample_support=sample_support,
        canvas=canvas, component_polygons=component_polygons,
        component_to_roi=component_to_roi,
        source_expression_df=source_expression_df,
        component_expression_df=component_expression_df,
        retained_component_datasets=retained_component_datasets,
        sample_median_nn=sample_median_nn,
        precision=precision, d1=d1,
        thresholds=pipeline_result.thresholds, config=config,
        tessellation_tiles_df=pipeline_result.mixed_frames.tessellation_tiles_df,
        consensus_hexes_df=pipeline_result.mixed_frames.consensus_hexes_df,
        tessellation_pooling_index=(
            pipeline_result.mixed_frames.tessellation_pooling_index
        ),
    )
    pair_sink(pair_key, export_dict)



def _build_component_to_roi(component_expression_df: pd.DataFrame) -> Dict[str, str]:
    missing = {c for c in (COMPONENT_ID, ROI_ID)
               if c not in component_expression_df.columns}
    if missing:
        raise ValueError(
            f"component_to_roi mapping requires columns {sorted(missing)!r}."
        )
    pairs = component_expression_df[[COMPONENT_ID, ROI_ID]].copy()
    pairs[COMPONENT_ID] = pairs[COMPONENT_ID].astype(str)
    pairs[ROI_ID] = pairs[ROI_ID].astype(str)
    pairs = pairs.drop_duplicates()
    collisions = pairs.groupby(COMPONENT_ID)[ROI_ID].nunique()
    collisions = collisions[collisions > 1]
    if not collisions.empty:
        raise ValueError(
            f"component_to_roi collision on {list(collisions.index)!r}."
        )
    return dict(zip(pairs[COMPONENT_ID], pairs[ROI_ID]))


def _prepare_component_datasets(
    component_expression_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    if COMPONENT_ID not in component_expression_df.columns:
        raise ValueError("_prepare_component_datasets requires COMPONENT_ID.")
    normalized = component_expression_df.copy()
    normalized[COMPONENT_ID] = normalized[COMPONENT_ID].astype(str)
    return {
        str(cid): group.copy()
        for cid, group in normalized.groupby(COMPONENT_ID, sort=False)
    }
