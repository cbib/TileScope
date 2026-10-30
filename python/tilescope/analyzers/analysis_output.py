"""Flatten a typed pair result to the legacy API dict shape.

Live callers use `sample_authority._run_pair_pipeline` for computation;
this module contributes the shared assembler + spot-ledger + metadata
builders that flow into the legacy per-pair dict returned by
`run_tilescope_from_csv`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tilescope.__version__ import __version__
from tilescope.analyzers.sample_analysis_result import (
    PairStatisticResult, SampleSupportOutputs,
)
from tilescope.analyzers.support_eligibility import DEFAULT_SAMPLE_SUPPORT_FACTOR
from tilescope.cols import COMPONENT_ID, ROI_ID
from tilescope.defaults import SEED_COUNT_FORMULA, SUPPORT_POLICY_VERSION
from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.analyzers.sample_support_frames import (
    build_spot_ledger,
)
from tilescope.schema.helpers import natural_sort_key
from tilescope.settings import RunConfigModel, ResolvedRunConfig



def assemble_pair_output(
    *,
    sample_id: str,
    pair: "PairStatisticResult",
    sample_support: "SampleSupportOutputs",
    canvas,
    component_polygons: Dict,
    component_to_roi: Dict[str, str],
    source_expression_df: pd.DataFrame,
    component_expression_df: pd.DataFrame,
    retained_component_datasets: Dict,
    sample_median_nn: float,
    precision: float,
    d1: float,
    thresholds,
    config: ResolvedRunConfig,
    tessellation_tiles_df: pd.DataFrame,
    consensus_hexes_df: pd.DataFrame,
    tessellation_pooling_index,
) -> Dict:
    """Flatten one pair's statistics into the legacy one-pair dict.

    Sole legacy-flattening path. Reads the caller-supplied `pair` +
    `sample_support` verbatim, inserts the already-materialised transient
    mixed frames unchanged, materialises `run_metadata` (mixed sample /
    feature / pair provenance). Scientific frames are read from the typed
    pair result; the assembler does not compute them.
    """
    support = sample_support
    run_metadata = build_run_metadata(
        source_expression_df=source_expression_df,
        component_expression_df=component_expression_df,
        retained_component_datasets=retained_component_datasets,
        thresholds=thresholds,
        config=config,
        tessellation_pooling_index=tessellation_pooling_index,
        tessellation_tiles_df=tessellation_tiles_df,
        consensus_hexes_df=consensus_hexes_df,
        feature_g1_id=pair.feature_g1_id,
        feature_g2_id=pair.feature_g2_id,
    )
    return {
        "dataset_id": sample_id,
        "analysis_df": component_expression_df,
        "canvas": canvas,
        "component_polygons": component_polygons,
        "roi_enrichment": pair.roi_enrichment_df,
        "roi_correlations": pair.roi_correlations_df,
        "roi_support_status": support.roi_support_status_df,
        "spot_ledger_df": support.spot_ledger_df,
        "sample_median_nn": sample_median_nn,
        "sample_support_factor": DEFAULT_SAMPLE_SUPPORT_FACTOR,
        "source_expression_df": source_expression_df,
        "component_to_roi": component_to_roi,
        "roi_tessellation_correlations": pair.roi_tessellation_correlations_df,
        "component_geometry_df": support.component_geometry_df,
        "component_domain_regions_df": support.component_domain_regions_df,
        "component_region_replicate_status_df": (
            support.component_region_replicate_status_df
        ),
        "roi_replicate_correlations_df": pair.roi_replicate_correlations_df,
        "roi_replicate_enrichment_df": pair.roi_replicate_enrichment_df,
        "roi_tessellation_enrichment_df": pair.roi_tessellation_enrichment_df,
        "roi_consensus_correlations_df": pair.roi_consensus_correlations_df,
        "roi_consensus_enrichment_df": pair.roi_consensus_enrichment_df,
        "roi_source_moran_df": pair.roi_source_moran_df,
        "roi_tessellation_moran_df": pair.roi_tessellation_moran_df,
        "roi_consensus_moran_df": pair.roi_consensus_moran_df,
        "roi_cross_grain_sensitivity_df": pair.roi_cross_grain_sensitivity_df,
        "roi_pair_interface_response_df": pair.roi_pair_interface_response_df,
        "component_radial_shell_correlogram_df": (
            pair.component_radial_shell_correlogram_df
        ),
        "consensus_hexes_df": consensus_hexes_df,
        "tessellation_tiles_df": tessellation_tiles_df,
        "precision": precision,
        "d1": d1,
        "feature_g1_id": pair.feature_g1_id,
        "feature_g2_id": pair.feature_g2_id,
        "feature_g1": pair.feature_g1,
        "feature_g2": pair.feature_g2,
        "run_metadata": run_metadata,
    }


def build_spot_ledger_frame(
    *, source_expression_df, expression_df_split, component_expression_df,
    support_status_df, sample_tess_result,
) -> pd.DataFrame:
    if "cc_idx" not in component_expression_df.columns:
        return pd.DataFrame()
    return build_spot_ledger(
        original_df=source_expression_df,
        post_split_df=expression_df_split,
        post_filter_df=component_expression_df,
        sample_tess_result=sample_tess_result,
        support_status_df=support_status_df,
    )


def build_run_metadata(
    source_expression_df: pd.DataFrame,
    component_expression_df: pd.DataFrame,
    retained_component_datasets: dict,
    thresholds,
    config: ResolvedRunConfig,
    *,
    tessellation_pooling_index=None,
    tessellation_tiles_df: Optional[pd.DataFrame] = None,
    consensus_hexes_df: Optional[pd.DataFrame] = None,
    feature_g1_id: str,
    feature_g2_id: str,
) -> dict:
    """Build biological/component population and construction provenance."""
    resolved_g1_id, resolved_g2_id = require_semantic_feature_ids(
        feature_g1_id, feature_g2_id, owner="build_run_metadata",
    )
    tessellation_config = config.run_config.tessellation
    supports = (
        tuple(tessellation_pooling_index.support_by_roi.values())
        if tessellation_pooling_index is not None else ()
    )
    tessellated_components = {
        component_id
        for support in supports
        for component_id in support.eligible_component_ids
    }
    tile_spots = (
        tessellation_tiles_df["n_spots"].to_numpy(dtype=np.int64)
        if tessellation_tiles_df is not None and not tessellation_tiles_df.empty
        else np.asarray([], dtype=np.int64)
    )
    realized_tiles = {
        "n_tiles": int(tile_spots.size),
        "median": float(np.median(tile_spots)) if tile_spots.size else None,
        "q25": float(np.percentile(tile_spots, 25)) if tile_spots.size else None,
        "q75": float(np.percentile(tile_spots, 75)) if tile_spots.size else None,
    }
    tile_counts = {}
    if tessellation_tiles_df is not None and not tessellation_tiles_df.empty:
        tile_counts = {
            (str(component_id), int(replicate)): len(group)
            for (component_id, replicate), group in tessellation_tiles_df.groupby(
                [COMPONENT_ID, "replicate_index"], sort=False,
            )
        }
    hex_counts = {}
    if consensus_hexes_df is not None and not consensus_hexes_df.empty:
        hex_counts = {
            str(component_id): len(group)
            for component_id, group in consensus_hexes_df.groupby(
                COMPONENT_ID, sort=False,
            )
        }
    component_stats = []
    for component_id, component_frame in retained_component_datasets.items():
        replicate_counts = (
            [
                tile_counts.get((str(component_id), replicate), 0)
                for replicate in range(config.run_config.tessellation.n)
            ]
            if component_id in tessellated_components else []
        )
        component_stats.append({
            "roi_id": str(component_frame[ROI_ID].iloc[0])
            if len(component_frame) else "",
            "component_id": str(component_id),
            "n_spots": int(len(component_frame)),
            "n_tessellations": len(replicate_counts),
            "avg_tiles_per_tess": round(
                sum(replicate_counts) / len(replicate_counts), 1,
            ) if replicate_counts else 0,
            "n_consensus_hexagons": hex_counts.get(str(component_id), 0),
        })
    component_stats.sort(key=lambda row: (
        natural_sort_key(row["roi_id"]),
        natural_sort_key(row["component_id"]),
    ))

    return {
        "tilescope_version": __version__,
        "analysis_policy_version": config.resolved_policy.analysis_policy_version,
        "timestamp": datetime.now().isoformat(),
        "feature_identity": {
            "feature_g1_id": resolved_g1_id,
            "feature_g2_id": resolved_g2_id,
        },
        "thresholds": {
            "q_high_percentile": config.resolved_policy.thresholds.q_high,
            "q_low_percentile": config.resolved_policy.thresholds.q_low,
            "min_spots_per_tile": tessellation_config.min_observations_per_tile,
            "gene1_high_threshold": round(float(thresholds.q_g1_threshold), 4),
            "gene1_low_threshold": round(float(thresholds.q_g1_low_threshold), 4),
            "gene2_high_threshold": round(float(thresholds.q_g2_threshold), 4),
            "gene2_low_threshold": round(float(thresholds.q_g2_low_threshold), 4),
        },
        "threshold_provenance": {
            "source_spot": {
                "source": "source_expression_df",
                "high_percentile": config.resolved_policy.thresholds.q_high,
                "low_percentile": config.resolved_policy.thresholds.q_low,
            },
            "voronoi_tile_mean": {
                "source": "source_spot_thresholds",
                "note": "Tile-mean enrichment uses spot/cell thresholds. "
                        "Surface-specific threshold calibration is an "
                        "unresolved decision.",
            },
            "consensus_ensemble": {
                "source": "source_spot_thresholds",
                "note": "Consensus-ensemble enrichment uses spot/cell "
                        "thresholds. Surface-specific threshold calibration "
                        "is an unresolved decision.",
            },
        },
        "population": {
            "biological_roi_count": int(
                source_expression_df[ROI_ID].astype(str).nunique()
            ),
            "source_spot_count": int(len(source_expression_df)),
            "spatially_supported_component_count": len(
                retained_component_datasets
            ),
            "spatially_supported_spot_count": int(sum(
                len(frame) for frame in retained_component_datasets.values()
            )),
            "tessellated_component_count": len(tessellated_components),
            "tessellated_spot_count": int(sum(
                support.n_eligible_spots for support in supports
            )),
            "component_retained_spot_count": int(len(component_expression_df)),
        },
        "gates": {
            "min_roi_spots": config.run_config.roi_filters.min_roi_spots,
            "min_component_spots": tessellation_config.min_component_observations,
            "min_spots_per_tile": tessellation_config.min_observations_per_tile,
            "min_context_spots": config.run_config.roi_filters.min_context_spots,
        },
        "tessellation": {
            "n_tessellations": config.run_config.tessellation.n,
            "method": "voronoi",
            "technology": config.run_config.technology,
            "source_adjacency": (
                config.resolved_policy.source_adjacency.model_dump()
            ),
            "support_policy_version": SUPPORT_POLICY_VERSION,
            "target_observations_per_tile": (
                tessellation_config.target_observations_per_tile
            ),
            "resolved_voronoi_fraction": tessellation_config.voronoi_fraction,
            "resolved_min_spots_per_tile": (
                tessellation_config.min_observations_per_tile
            ),
            "min_component_observations": (
                tessellation_config.min_component_observations
            ),
            "tile_value_aggregation": (
                config.resolved_policy.tile_value_aggregation
            ),
            "seed_count_formula": SEED_COUNT_FORMULA,
            "realized_tile_spots": realized_tiles,
        },
        "consensus": {
            "min_spots_per_hex": config.run_config.consensus.min_spots_per_hex,
            "min_tess_coverage_fraction": (
                config.run_config.consensus.min_tess_coverage_fraction
            ),
            "min_tess_per_hex": config.resolved_policy.consensus.min_tess_per_hex,
            "base_area_quantile": config.resolved_policy.consensus.base_area_quantile,
            "n_consensus_grids": sum(count > 0 for count in hex_counts.values()),
        },
        "component_stats": component_stats,
    }
