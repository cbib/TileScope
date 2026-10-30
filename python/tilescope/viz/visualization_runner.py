"""Centralized visualization orchestration for TileScope datasets."""
from tilescope.cols import REPLICATE_INDEX


import logging
from pathlib import Path
from typing import Dict, Optional

from tilescope.cols import CORR_OBS, ROI_ID
from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.spatial.spacing import median_nn_distance_from_df
from tilescope.viz.consensus import plot_consensus_grids
from tilescope.viz.expression import plot_expression_summary
from tilescope.viz.provenance import plot_support_provenance
from tilescope.viz.tiles import plot_tile_densities

logger = logging.getLogger(__name__)


class VisualizationRunner:
    """Generate all standard plots for a dataset analysis."""

    def __init__(self, output_dir: Path, viz_config: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_config = viz_config if viz_config is not None else {}

    def generate_all_plots(self, dataset_results: Dict) -> Dict[str, str]:
        # Pair authority is the sole source of biological labels. Generic
        # computational columns must never leak into plots as identities.
        self.feature_a_id, self.feature_b_id = require_semantic_feature_ids(
            dataset_results.get("feature_g1_id"),
            dataset_results.get("feature_g2_id"),
            owner="VisualizationRunner",
        )
        self.feature_a_col = dataset_results.get("feature_g1")
        self.feature_b_col = dataset_results.get("feature_g2")
        if not isinstance(self.feature_a_col, str) or not isinstance(
            self.feature_b_col, str,
        ):
            raise ValueError(
                "VisualizationRunner: computational feature columns are missing."
            )
        # Compute the sample-wide source-frame median NN spacing once and thread
        # it through every renderer: expression, per-replicate tile densities,
        # consensus, and provenance would otherwise recompute the same scalar.
        # Preserves the all-source-rows substrate (no dedup).
        source_expression_df = dataset_results.get("source_expression_df")
        source_median_nn: Optional[float] = median_nn_distance_from_df(
            source_expression_df, "x", "y",
        ) if source_expression_df is not None else None

        paths: Dict[str, str] = {}
        paths.update(self._generate_expression_plots(dataset_results, source_median_nn))
        paths.update(self._generate_tessellation_plots(dataset_results, source_median_nn))
        paths.update(self._generate_consensus_plots(dataset_results, source_median_nn))
        paths.update(self._generate_provenance_plot(dataset_results, source_median_nn))
        return paths

    # -- spot/cell ROI analysis (A/B expression + ROI correlation) ---------

    def _generate_expression_plots(self, dataset_results: Dict,
                                   source_median_nn: Optional[float]) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        source_expression_df = dataset_results.get("source_expression_df")
        roi_correlations_df = dataset_results.get("roi_correlations")
        component_polygons = dataset_results.get("component_polygons")
        canvas = dataset_results.get("canvas")
        spot_ledger_df = dataset_results.get("spot_ledger_df")

        if source_expression_df is None or source_expression_df.empty:
            return paths

        correlation_map: Dict[str, float] = {}
        if roi_correlations_df is not None and not roi_correlations_df.empty:
            for idx, row in roi_correlations_df.iterrows():
                try:
                    correlation_map[str(row[ROI_ID])] = float(row[CORR_OBS])
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"Invalid CORR_OBS type in roi_correlations row {idx}: {e}"
                    ) from e

        component_to_roi: Dict[str, str] = dataset_results.get("component_to_roi", {})

        plot_expression_summary(
            source_expression_df=source_expression_df,
            correlation_map=correlation_map,
            component_to_roi=component_to_roi,
            roi_support_status_df=dataset_results.get("roi_support_status"),
            output_dir=str(self.output_dir),
            component_polygons=component_polygons,
            canvas=canvas,
            spot_ledger_df=spot_ledger_df,
            source_median_nn=source_median_nn,
            feature_a_col=self.feature_a_col,
            feature_b_col=self.feature_b_col,
            feature_a_id=self.feature_a_id,
            feature_b_id=self.feature_b_id,
        )
        paths["spot_expression_plot"] = str(self.output_dir / "expression_summary.pdf")
        return paths

    # -- per-replicate tile densities ---------------------------------------

    def _generate_tessellation_plots(self, dataset_results: Dict,
                                     source_median_nn: Optional[float]) -> Dict[str, str]:

        paths: Dict[str, str] = {}
        tiles_df = dataset_results.get("tessellation_tiles_df")
        if tiles_df is None or tiles_df.empty:
            return paths

        component_polygons = dataset_results.get("component_polygons")
        canvas = dataset_results.get("canvas")
        precision = dataset_results.get("precision")
        d1 = dataset_results.get("d1")
        spot_ledger_df = dataset_results.get("spot_ledger_df")
        source_expression_df = dataset_results.get("source_expression_df")

        replicate_indices = sorted(
            int(v) for v in tiles_df[REPLICATE_INDEX].dropna().unique()
        )
        for replicate_index in replicate_indices:
            plot_tile_densities(
                tessellation_tiles_df=tiles_df,
                replicate_index=replicate_index,
                output_dir=str(self.output_dir),
                component_polygons=component_polygons,
                canvas=canvas,
                precision=precision,
                d1=d1,
                spot_ledger_df=spot_ledger_df,
                source_expression_df=source_expression_df,
                source_median_nn=source_median_nn,
                feature_a_col=self.feature_a_col,
                feature_b_col=self.feature_b_col,
                feature_a_id=self.feature_a_id,
                feature_b_id=self.feature_b_id,
            )
            paths[f"replicate_{replicate_index}_densities"] = str(
                self.output_dir
                / f"tessellation_densities_{replicate_index + 1}.pdf"
            )
        return paths

    # -- consensus (canonical DataFrames) ------------------------------------

    def _generate_consensus_plots(self, dataset_results: Dict,
                                  source_median_nn: Optional[float]) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        consensus_hexes_df = dataset_results.get("consensus_hexes_df")
        if consensus_hexes_df is None or consensus_hexes_df.empty:
            return paths

        plot_consensus_grids(
            consensus_hexes_df=consensus_hexes_df,
            roi_consensus_moran_df=dataset_results.get(
                "roi_consensus_moran_df",
            ),
            roi_consensus_correlations_df=dataset_results.get("roi_consensus_correlations_df"),
            output_dir=str(self.output_dir),
            component_polygons=dataset_results.get("component_polygons"),
            component_to_roi=dataset_results.get("component_to_roi", {}),
            canvas=dataset_results.get("canvas"),
            show_stability=self.viz_config.get('show_stability', False),
            show_bivariate_morans=self.viz_config.get('show_bivariate_morans', True),
            spot_ledger_df=dataset_results.get("spot_ledger_df"),
            source_expression_df=dataset_results.get("source_expression_df"),
            source_median_nn=source_median_nn,
            feature_a_col=self.feature_a_col,
            feature_b_col=self.feature_b_col,
            feature_a_id=self.feature_a_id,
            feature_b_id=self.feature_b_id,
        )
        paths["consensus_plot"] = str(self.output_dir / "consensus_tessellation_summary.pdf")
        return paths

    # -- support provenance (new standalone) ---------------------------------

    def _generate_provenance_plot(self, dataset_results: Dict,
                                  source_median_nn: Optional[float]) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        spot_ledger_df = dataset_results.get("spot_ledger_df")
        source_expression_df = dataset_results.get("source_expression_df")
        if spot_ledger_df is None or source_expression_df is None:
            return paths
        output_path = self.output_dir / "support_provenance.pdf"
        plot_support_provenance(
            spot_ledger_df=spot_ledger_df,
            source_expression_df=source_expression_df,
            component_polygons=dataset_results.get("component_polygons"),
            component_to_roi=dataset_results.get("component_to_roi", {}),
            canvas=dataset_results.get("canvas"),
            output_path=str(output_path),
            source_median_nn=source_median_nn,
            feature_a_col=self.feature_a_col,
            feature_b_col=self.feature_b_col,
        )
        paths["support_provenance"] = str(output_path)
        return paths
