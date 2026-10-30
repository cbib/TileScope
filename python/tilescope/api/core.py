"""TileScope programmatic API.

High-level Python API for running TileScope analysis programmatically.
Designed for integration with other tools (e.g., R/Seurat via reticulate).

For CLI usage, use tilescope.py instead.

Example:
    from tilescope.api import run_tilescope_from_csv, make_run_config

    results = run_tilescope_from_csv(
        input_csv="data.csv",
        profile="standard",
        output_dir="results/",
        export=True,
        feature_a_id="Neuron",
        feature_b_id="AC",
    )

    print(f"Analyzed {len(results['roi_enrichment'])} ROIs")
"""

from __future__ import annotations

from tilescope.logger_config import configure_tilescope_logging


import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tilescope.__version__ import __version__
from tilescope.settings import (
    load_run_config,
    load_dataset_schema,
    RunConfigModel,
    ResolvedRunConfig,
    DatasetSchemaModel,
)
from tilescope.tilescope_io import read_expression_csv
from tilescope.analyzers import run_analysis
from tilescope.reporters import export_results, generate_html_report
from tilescope.defaults import dump_effective_config
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, ROI_ID
from tilescope.feature_identity import require_semantic_feature_ids

logger = logging.getLogger(__name__)

__all__ = [
    "run_tilescope_from_csv",
    "run_tilescope_from_df",
    "make_run_config",
    "validate_run_config",
    "validate_tilescope_dataframe",
    "__version__",
]


_MINIMAL_RESULT_KEYS = (
    "roi_enrichment",
    "roi_correlations",
    "roi_support_status",
    "roi_replicate_correlations_df",
    "roi_tessellation_correlations",
    "roi_replicate_enrichment_df",
    "roi_tessellation_enrichment_df",
    "roi_consensus_correlations_df",
    "roi_consensus_enrichment_df",
    "roi_source_moran_df",
    "roi_tessellation_moran_df",
    "roi_consensus_moran_df",
    "roi_cross_grain_sensitivity_df",
    "roi_pair_interface_response_df",
    "component_radial_shell_correlogram_df",
    "consensus_hexes_df",
    "tessellation_tiles_df",
)


def validate_run_config(
    config: Union[ResolvedRunConfig, RunConfigModel, dict],
) -> ResolvedRunConfig:
    """Validate config dict/model without requiring a dataframe.

    For dicts, this merges with defaults using load_run_config to ensure
    all required fields are populated.

    Args:
        config: ResolvedRunConfig, RunConfigModel, or dict to validate

    Returns:
        Validated ResolvedRunConfig instance

    Raises:
        ValidationError: If config is invalid

    Example:
        config_dict = {"tessellation": {"n": 10}}
        validated = validate_run_config(config_dict)
    """
    if isinstance(config, ResolvedRunConfig):
        # Reconstruct so Pydantic validators catch stale derived policy
        # introduced through model_copy(update=...), which skips validation.
        return ResolvedRunConfig.model_validate(config.model_dump())
    if isinstance(config, RunConfigModel):
        config = config.model_dump()
    if isinstance(config, dict):
        return load_run_config(
            path=None,
            profile=None,  # No profile layer, packaged defaults only
            flag_overrides={},
            set_overrides=config
        )
    raise TypeError(f"Unsupported run configuration: {type(config).__name__}")


def make_run_config(
    profile: Optional[str] = None,
    overrides: Optional[dict] = None,
    **kwargs,
) -> ResolvedRunConfig:
    """Create a resolved run configuration from profile + overrides.

    Args:
        profile: Profile name ('exploratory', 'standard', 'debugging').
            None or "" means no profile is applied: packaged defaults only.
            Pass profile='standard' explicitly to get the standard profile.
        overrides: Nested dict of overrides (e.g., {'tessellation': {'n': 10}})
        **kwargs: Additional top-level overrides (e.g., seed=42, verbose=True)

    Returns:
        ResolvedRunConfig instance

    Example:
        config = make_run_config(
            profile='standard',
            overrides={'tessellation': {'n': 10}},
            seed=42
        )
    """
    # Merge kwargs into overrides
    final_overrides = overrides.copy() if overrides else {}
    if kwargs:
        final_overrides.update(kwargs)

    # Use settings layer to load config with profile + overrides
    return load_run_config(
        path=None,
        profile=profile,  # None/"" = no profile layer; name = apply that profile
        flag_overrides={},  # No CLI flags in library mode
        set_overrides=final_overrides
    )


def validate_tilescope_dataframe(df: pd.DataFrame) -> bool:
    """Validate that DataFrame is in canonical TileScope format.

    This expects canonical column names (already mapped by dataset_schema
    in the CSV route). Required columns: x, y, gene1, gene2, roi_id.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValueError: If DataFrame is invalid with helpful error message

    Example:
        validate_tilescope_dataframe(df)  # Raises if invalid
    """
    required_cols = [COORD_X, COORD_Y, GENE1, GENE2, ROI_ID]
    missing = set(required_cols) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Check for empty dataframe (before type checks)
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Type checks
    for column in (COORD_X, COORD_Y, GENE1, GENE2):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column {column!r} must be numeric")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column!r} contains non-finite values")

    return True


def build_visualization_options(config: ResolvedRunConfig) -> Dict[str, Any]:
    """Assemble the viz_config dict handed to `export_results`.

    Populates `show_bivariate_morans` from
    `consensus.bivariate_morans.compute AND consensus.bivariate_morans.visualize`
    Compute governs the CSV Moran surface; visualize governs only PDF
    panel presentation. Single production authority used by both
    `run_tilescope_from_csv` and `run_tilescope_from_df`.
    """
    viz_options = config.run_config.visualization.model_dump()
    viz_options["show_bivariate_morans"] = (
        config.resolved_policy.consensus.bivariate_morans.compute
        and config.run_config.consensus.bivariate_morans.visualize
    )
    return viz_options


def _prepare_execution(
    *,
    config: ResolvedRunConfig,
    output_dir: Union[str, Path, None],
    export: bool,
    html_report: bool,
    feature_a_id: str,
    feature_b_id: str,
) -> ResolvedRunConfig:
    """Validate shared public-API options before analysis starts."""
    require_semantic_feature_ids(
        feature_a_id, feature_b_id, owner="TileScope public API",
    )
    if html_report and not export:
        raise ValueError("html_report=True requires export=True")
    if output_dir is None:
        return config
    payload = config.model_dump()
    payload["run_config"]["output_dir"] = str(output_dir)
    return ResolvedRunConfig.model_validate(payload)


def _finalize_results(
    *,
    results: dict,
    config: ResolvedRunConfig,
    export: bool,
    html_report: bool,
    return_minimal: bool,
    merge_sources: Sequence[str] = (),
) -> dict:
    """One export/report/minimal-return path for both input loaders."""
    results["config"] = config
    if export:
        final_output_dir = Path(config.run_config.output_dir)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        results["export_paths"] = export_results(
            results,
            final_output_dir,
            viz_config=build_visualization_options(config),
        )
        dump_effective_config(
            config.model_dump(),
            final_output_dir / "effective_config.json",
            metadata={
                "tilescope_version": __version__,
                "merge_sources": list(merge_sources),
            },
        )
        if html_report:
            html_path = generate_html_report(final_output_dir)
            logger.info(f"HTML report: {html_path}")

    if not return_minimal:
        return results
    minimal = {key: results.get(key) for key in _MINIMAL_RESULT_KEYS}
    minimal.update({
        "run_metadata": results["run_metadata"],
        "config": config,
        "export_paths": results.get("export_paths"),
    })
    return minimal


def run_tilescope_from_csv(
    input_csv: Union[str, Path],
    run_config: Union[
        str, Path, dict, RunConfigModel, ResolvedRunConfig, None
    ] = None,
    dataset_schema: Union[str, Path, dict, DatasetSchemaModel, None] = None,
    profile: Optional[str] = None,
    flag_overrides: Optional[dict] = None,
    set_overrides: Optional[dict] = None,
    output_dir: Union[str, Path, None] = None,
    export: bool = True,
    html_report: bool = False,
    dataset_id: int = 1,
    configure_logging: bool = False,
    return_minimal: bool = False,
    *,
    feature_a_id: str,
    feature_b_id: str,
) -> dict:
    """High-level API: read CSV, load config, run analysis, optionally export results.

    This is the primary entry point for programmatic TileScope usage.

    Args:
        input_csv: Path to CSV with columns: x, y, gene1, gene2, roi_id
        run_config: ResolvedRunConfig instance, dict, or path to run.config.json
        dataset_schema: DatasetSchemaModel instance, dict, or path to dataset.schema.json
        profile: Profile name ('exploratory', 'standard', 'debugging').
            None or "" means no profile is applied: packaged defaults only.
            Ignored when run_config is an in-memory configuration.
        flag_overrides: Flat CLI-style overrides (e.g., {'seed': 42, 'verbose': True})
        set_overrides: Nested overrides (e.g., {'tessellation': {'n': 10}})
        output_dir: Output directory override. If omitted, use the resolved
            configuration's output directory.
        export: Whether to export CSV/PDF results
        html_report: Whether to generate HTML report (requires export=True)
        dataset_id: Dataset identifier for metadata
        configure_logging: Whether to configure logging (False for library use)
        return_minimal: If True, return only summary tables (excludes heavy objects)
        feature_a_id: Required biological identity represented by ``gene1``.
        feature_b_id: Required biological identity represented by ``gene2``.

    Returns:
        dict with keys:
            - roi_enrichment: DataFrame, parent-ROI (biological) enrichment,
                pooled over spots or cells (public headline output)
            - roi_correlations: DataFrame, parent-ROI (biological) correlation,
                pooled over spots or cells (public headline output)
            - roi_replicate_correlations_df: DataFrame, one row per
              (sample_id, roi_id, replicate_index) with typed
              parent_replicate_correlation_status. Pearson computed
              once over canonically pooled sibling-component tile
              means per replicate; ROI-grain. Discrete spot / region /
              component coverage counts + eligible/source coverage
              fractions.
            - roi_replicate_enrichment_df: DataFrame, one row per
              (sample_id, roi_id, replicate_index) with typed
              parent_replicate_enrichment_status +
              parent_replicate_effect_size_status. Same-replicate
              cross-ROI context: target = pooled sibling-component
              tiles of the ROI at replicate t; context = pooled tiles
              of every OTHER ROI at the SAME replicate. Fold ratios,
              log2FC, CLES, and target sufficient stats emitted; no
              inference (no p/q, no confidence).
            - roi_tessellation_enrichment_df: DataFrame, one row per
              source parent ROI (total table). Median/Q25/Q75 of
              fold ratios over enrichment-FINITE replicates + of
              log2FC/CLES independently over effect-size-FINITE
              replicates; independent typed statuses.
            - consensus_hexes_df: DataFrame, one row per (component, hex)
              with WKT geometry, ensemble + direct sufficient stats, and
              ROI-context-derived per-hex enrichment ratios keyed by
              `hex_enrichment_status`
            - run_metadata: dict with analysis metadata
            - config: ResolvedRunConfig used for analysis
            - export_paths: dict of exported file paths (if export=True)

        If return_minimal=True, excludes:
            - analysis_df
            - canvas
            - component_polygons

    Example:
        results = run_tilescope_from_csv(
            "data.csv",
            profile="standard",
            output_dir="results/",
            export=True,
            feature_a_id="Neuron",
            feature_b_id="AC",
        )
    """
    # Configure logging if requested
    if configure_logging:
        configure_tilescope_logging(verbose=False, level="INFO")

    # Normalize paths
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load configuration
    if isinstance(run_config, (ResolvedRunConfig, RunConfigModel, dict)):
        # Already a model or dict - validate it
        config = validate_run_config(run_config)
    else:
        # Load from file path or defaults
        config = load_run_config(
            path=run_config,
            profile=profile,  # None/"" = no profile layer; name = apply that profile
            flag_overrides=flag_overrides or {},
            set_overrides=set_overrides or {}
        )

    config = _prepare_execution(
        config=config,
        output_dir=output_dir,
        export=export,
        html_report=html_report,
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
    )

    # Load dataset schema
    if isinstance(dataset_schema, DatasetSchemaModel):
        schema = dataset_schema
    elif isinstance(dataset_schema, dict):
        schema = DatasetSchemaModel(**dataset_schema)
    else:
        # Load from file path or None
        schema = load_dataset_schema(dataset_schema)

    # Load data
    logger.info(f"Loading expression data from {input_csv}")
    expression_df = read_expression_csv(input_csv, dataset_schema=schema)

    # Run analysis
    results = run_analysis(
        expression_df,
        config,
        dataset_id=dataset_id,
        dataset_schema=schema,
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
    )

    merge_sources = []
    if profile:
        merge_sources.append(f"profile:{profile}")
    if run_config and not isinstance(
        run_config, (dict, RunConfigModel, ResolvedRunConfig)
    ):
        merge_sources.append(f"user_json:{run_config}")
    if flag_overrides or set_overrides:
        merge_sources.append("api_overrides")
    return _finalize_results(
        results=results,
        config=config,
        export=export,
        html_report=html_report,
        return_minimal=return_minimal,
        merge_sources=merge_sources,
    )


def run_tilescope_from_df(
    df: pd.DataFrame,
    config: Union[ResolvedRunConfig, dict],
    dataset_id: int = 1,
    output_dir: Union[str, Path, None] = None,
    export: bool = False,
    html_report: bool = False,
    configure_logging: bool = False,
    return_minimal: bool = False,
    *,
    feature_a_id: str,
    feature_b_id: str,
) -> dict:
    """Programmatic API: analyze DataFrame directly (no CSV).

    Useful when data is already loaded in memory (e.g., from Seurat/R).
    DataFrame must already be in canonical format (x, y, gene1, gene2, roi_id).

    Args:
        df: DataFrame with canonical columns (x, y, gene1, gene2, roi_id)
        config: ResolvedRunConfig instance or dict
        dataset_id: Dataset identifier
        output_dir: Output directory override. If omitted, use the resolved
            configuration's output directory.
        export: Whether to export results
        html_report: Whether to generate HTML report
        configure_logging: Whether to configure logging
        return_minimal: If True, return only summary tables
        feature_a_id: Required biological identity represented by ``gene1``.
        feature_b_id: Required biological identity represented by ``gene2``.

    Returns:
        dict with analysis results (same as run_tilescope_from_csv)

    Example:
        df = pd.read_csv("data.csv")
        config = make_run_config(profile="standard")
        results = run_tilescope_from_df(
            df, config, feature_a_id="Neuron", feature_b_id="AC",
        )
    """
    # Configure logging if requested
    if configure_logging:
        configure_tilescope_logging(verbose=False, level="INFO")

    # Validate all shared options before starting the analysis.
    validate_tilescope_dataframe(df)
    config = validate_run_config(config)
    config = _prepare_execution(
        config=config,
        output_dir=output_dir,
        export=export,
        html_report=html_report,
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
    )

    # Run analysis
    results = run_analysis(
        df,
        config,
        dataset_id=dataset_id,
        dataset_schema=None,  # Not applicable for DataFrame route
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
    )

    return _finalize_results(
        results=results,
        config=config,
        export=export,
        html_report=html_report,
        return_minimal=return_minimal,
    )
