"""Sole CSV writer for TileScope publication frames.

The orchestrator materialises every canonical publication DataFrame on the return dict; this module writes them via :func:`tilescope.tilescope_io.exports.export_dataframe_csv` and dispatches to the visualization runner.
"""

import glob
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from tilescope.tilescope_io.exports import export_dataframe_csv
from tilescope.viz import VisualizationRunner

logger = logging.getLogger(__name__)


# Every canonical publication CSV this exporter writes, purged before write
# so a stale artifact from a prior run cannot masquerade as current output.
_MANAGED_OUTPUTS = (
    "roi_enrichment.csv",
    "roi_correlations.csv",
    "roi_support_status.csv",
    "roi_tessellation_correlations.csv",
    "spot_ledger.csv",
    "component_geometry.csv",
    "component_domain_regions.csv",
    "component_region_replicate_status.csv",
    "roi_replicate_correlations.csv",
    "roi_replicate_enrichment.csv",
    "roi_tessellation_enrichment.csv",
    "roi_consensus_correlations.csv",
    "roi_consensus_enrichment.csv",
    "roi_source_moran.csv",
    "roi_tessellation_moran.csv",
    "roi_consensus_moran.csv",
    "roi_cross_grain_sensitivity.csv",
    "roi_pair_interface_response.csv",
    "component_radial_shell_correlogram.csv",
    "consensus_hexes.csv",
    "tessellation_tiles.csv",
)


# Obsolete artifact names deleted on export startup with a warning so a stale
# copy cannot survive a reused output directory.
_RETIRED_ARTIFACTS = (
    "spot_enrichment.csv",
    "spot_correlation.csv",
    "summary_cross_scale.csv",
    "roi_borders_original_cluster.pdf",
    "consensus_enrichment.csv",
    "spot_ledger_summary.csv",
    "summary_tessellation_tile_verdicts_per_roi.csv",
    "summary_tessellation_correlations.csv",
    "summary_consensus_hexagon_verdicts_per_roi.csv",
    "summary_consensus_stability.csv",
    "roi_moran.csv",
    "component_replicate_correlations.csv",
    "roi_borders_by_parent.pdf",
    "consensus_component_outcomes.csv",
    "correlation_distribution.pdf",
)

_RETIRED_ARTIFACT_GLOBS = (
    "tessellation_*_correlation.csv",
    "tessellation_*_enrichment.csv",
)


def _purge_managed_and_retired(output_dir: Path) -> None:
    """Unlink managed outputs and obsolete artifacts before write."""
    for name in _MANAGED_OUTPUTS:
        stale = output_dir / name
        if stale.exists():
            stale.unlink()
    for name in _RETIRED_ARTIFACTS:
        stale = output_dir / name
        if stale.exists():
            stale.unlink()
            logger.warning(
                "[EXPORT] removed obsolete artifact: %s (no longer emitted)",
                name,
            )
    for pattern in _RETIRED_ARTIFACT_GLOBS:
        for stale_path in glob.glob(str(output_dir / pattern)):
            Path(stale_path).unlink()
            logger.warning(
                "[EXPORT] removed obsolete artifact: %s (no longer emitted)",
                Path(stale_path).name,
            )


def _write_if_present(
    results: Dict[str, object],
    key: str,
    output_dir: Path,
    filename: str,
    export_paths: Dict[str, str],
    export_key: str,
) -> None:
    """Write a canonical publication DataFrame if non-empty."""
    frame = results.get(key)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return
    path = str(output_dir / filename)
    export_dataframe_csv(frame, path, index=False)
    export_paths[export_key] = path
    logger.info(
        "[EXPORT] %s: %s (%d rows)", export_key, filename, len(frame),
    )


def export_results(
    results: Dict[str, object],
    output_dir: Path,
    generate_visualizations: bool = True,
    viz_config: Dict = None,
) -> Dict[str, str]:
    """Export all analysis results to CSV and generate visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _purge_managed_and_retired(output_dir)

    logger.info(f"[EXPORT] Exporting results to {output_dir.resolve()}")

    export_paths: Dict[str, str] = {}

    _write_if_present(results, "roi_enrichment", output_dir, "roi_enrichment.csv", export_paths, "roi_enrichment")
    _write_if_present(results, "roi_correlations", output_dir, "roi_correlations.csv", export_paths, "roi_correlations")
    _write_if_present(results, "roi_support_status", output_dir, "roi_support_status.csv", export_paths, "roi_support_status")
    _write_if_present(
        results, "roi_tessellation_correlations", output_dir,
        "roi_tessellation_correlations.csv", export_paths,
        "roi_tessellation_correlations",
    )

    _write_if_present(results, "spot_ledger_df", output_dir, "spot_ledger.csv", export_paths, "spot_ledger")

    # Component-grain construction / provenance frames.
    _write_if_present(results, "component_geometry_df", output_dir, "component_geometry.csv", export_paths, "component_geometry")
    _write_if_present(results, "component_domain_regions_df", output_dir, "component_domain_regions.csv", export_paths, "component_domain_regions")
    _write_if_present(results, "component_region_replicate_status_df", output_dir, "component_region_replicate_status.csv", export_paths, "component_region_replicate_status")

    # ROI-grain tessellation frames: replicate + summary, correlation + enrichment.
    # Reporting grain is biological ROI; component / region / tile are construction.
    _write_if_present(results, "roi_replicate_correlations_df", output_dir, "roi_replicate_correlations.csv", export_paths, "roi_replicate_correlations")
    _write_if_present(results, "roi_replicate_enrichment_df", output_dir, "roi_replicate_enrichment.csv", export_paths, "roi_replicate_enrichment")
    _write_if_present(results, "roi_tessellation_enrichment_df", output_dir, "roi_tessellation_enrichment.csv", export_paths, "roi_tessellation_enrichment")

    # ROI-grain consensus + Moran frames.
    _write_if_present(results, "roi_consensus_correlations_df", output_dir, "roi_consensus_correlations.csv", export_paths, "roi_consensus_correlations")
    _write_if_present(results, "roi_consensus_enrichment_df", output_dir, "roi_consensus_enrichment.csv", export_paths, "roi_consensus_enrichment")
    _write_if_present(results, "roi_source_moran_df", output_dir, "roi_source_moran.csv", export_paths, "roi_source_moran")
    _write_if_present(results, "roi_tessellation_moran_df", output_dir, "roi_tessellation_moran.csv", export_paths, "roi_tessellation_moran")
    _write_if_present(results, "roi_consensus_moran_df", output_dir, "roi_consensus_moran.csv", export_paths, "roi_consensus_moran")
    _write_if_present(results, "roi_cross_grain_sensitivity_df", output_dir, "roi_cross_grain_sensitivity.csv", export_paths, "roi_cross_grain_sensitivity")
    _write_if_present(results, "roi_pair_interface_response_df", output_dir, "roi_pair_interface_response.csv", export_paths, "roi_pair_interface_response")

    # Component-grain radial shell cross-correlogram: one row per
    # (component, shell). Component-grain because shell normalisation is
    # per-component; pooling across ROI components would change the estimand.
    _write_if_present(results, "component_radial_shell_correlogram_df", output_dir, "component_radial_shell_correlogram.csv", export_paths, "component_radial_shell_correlogram")

    _write_if_present(results, "consensus_hexes_df", output_dir, "consensus_hexes.csv", export_paths, "consensus_hexes")

    # Canonical per-tile surface; single tabular materialisation shared by
    # ROI x replicate estimators, visualization, and this writer.
    _write_if_present(results, "tessellation_tiles_df", output_dir, "tessellation_tiles.csv", export_paths, "tessellation_tiles")

    if generate_visualizations:
        logger.info("[EXPORT] Generating visualizations...")
        viz_opts = viz_config if viz_config is not None else {}
        viz_runner = VisualizationRunner(
            output_dir=output_dir, viz_config=viz_opts,
        )
        viz_paths = viz_runner.generate_all_plots(results)
        export_paths.update(viz_paths)
        logger.info("[EXPORT] Visualizations complete")

    if "run_metadata" in results:
        metadata_path = output_dir / "run_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results["run_metadata"], f, indent=2)
        export_paths["run_metadata"] = str(metadata_path)
        logger.info(f"[EXPORT] Run metadata: {metadata_path.name}")

    # HTML generation is owned by API entry points (api/core.py), which
    # gate on the caller's `html_report` flag.

    return export_paths
