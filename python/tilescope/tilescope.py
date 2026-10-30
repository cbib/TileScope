#!/usr/bin/env python3
"""TileScope - Spatial transcriptomics co-expression analysis.

Main analysis script that runs the complete TileScope pipeline on user-provided
expression data. Performs spot-level, tessellation-level, and consensus-level
analyses with enrichment classification and correlation computation.

Input CSV Requirements:
    - gene1: Gene 1 expression values (float)
    - gene2: Gene 2 expression values (float)
    - roi_id: Region of interest identifier (string/int)
    - x: X coordinate (float)
    - y: Y coordinate (float)

No preprocessing is performed by default - data is analyzed as provided.
"""

from __future__ import annotations

from tilescope.logger_config import configure_tilescope_logging
from tilescope.api import run_tilescope_from_csv
import traceback


import argparse
import json
import logging
import sys
from pathlib import Path

from tilescope.__version__ import __version__
from tilescope.settings import load_run_config, load_dataset_schema, parse_set_overrides
from tilescope.defaults import (
    DEFAULT_N_TESSELLATIONS,
    DEFAULT_MIN_ROI_SPOTS,
)

logger = logging.getLogger(__name__)


def build_run_cli_overrides(args: argparse.Namespace) -> dict:
    # Build the nested RunConfigModel override dict from parsed CLI args.
    # Only flags the user actually supplied appear in the result: flags with
    # `default=None` in the parser are omitted, so profile/config values win
    # unless the user typed the flag.
    overrides: dict = {}

    tessellation = {}
    if args.n_tessellations is not None:
        tessellation['n'] = args.n_tessellations
    if tessellation:
        overrides['tessellation'] = tessellation

    roi_filters = {}
    if args.min_roi_spots is not None:
        roi_filters['min_roi_spots'] = args.min_roi_spots
    if roi_filters:
        overrides['roi_filters'] = roi_filters

    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.output is not None:
        overrides['output_dir'] = str(args.output)

    return overrides


def main() -> None:
    """Run TileScope analysis on user-provided expression data."""
    parser = argparse.ArgumentParser(
        description="TileScope - Spatial transcriptomics co-expression analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  tilescope expression_data.csv --feature-a-id Neuron --feature-b-id AC
  tilescope expression_data.csv --feature-a-id Neuron --feature-b-id AC --output results/

Input CSV format:
  Required columns: gene1, gene2, roi_id, x, y
  No preprocessing is performed - data is analyzed as provided.
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'TileScope {__version__}'
    )
    parser.add_argument(
        'input_csv',
        type=Path,
        help='Path to expression CSV file (requires: gene1, gene2, roi_id, x, y)'
    )
    parser.add_argument(
        '--feature-a-id',
        required=True,
        help='Biological name represented by the canonical gene1 column',
    )
    parser.add_argument(
        '--feature-b-id',
        required=True,
        help='Biological name represented by the canonical gene2 column',
    )
    # Override flags: default=None means "not supplied by user"; when None, the
    # value from config/profile/packaged defaults wins. This lets the user
    # explicitly override a profile setting *back* to the packaged default.
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output directory (config default: tilescope_output)'
    )
    parser.add_argument(
        '--n-tessellations',
        type=int,
        default=None,
        help=f'Number of tessellations per ROI (config default: {DEFAULT_N_TESSELLATIONS})'
    )
    parser.add_argument(
        '--min-roi-spots',
        type=int,
        default=None,
        help=f'Minimum source-labelled spots per biological ROI (config default: {DEFAULT_MIN_ROI_SPOTS})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (config default: 42)'
    )
    # JSON-first configuration system
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        choices=['exploratory', 'standard', 'debugging'],
        help='Load preset configuration profile (exploratory, standard, or debugging)'
    )
    parser.add_argument(
        '--set',
        action='append',
        dest='set_overrides',
        metavar='KEY=VALUE',
        help='Override config values (e.g., --set tessellation.n=5 --set verbose=true)'
    )

    # JSON configuration files (optional)
    parser.add_argument(
        '--config-run',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to run.config.json (runtime parameters). CLI flags override JSON values.'
    )
    parser.add_argument(
        '--config-dataset',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to dataset.schema.json (source-to-canonical column mapping)'
    )
    parser.add_argument(
        '--dump-effective-config',
        type=str,
        nargs='?',
        const='-',
        default=None,
        metavar='PATH',
        help='Dump effective configuration after merging defaults/JSON/CLI. Use "-" for stdout or provide a file path.'
    )

    # Logging verbosity controls
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output (DEBUG level logging)'
    )
    verbosity_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress informational messages (WARNING level logging)'
    )

    args = parser.parse_args()

    run_cli_overrides = build_run_cli_overrides(args)

    # Parse --set overrides (keep separate from CLI flag overrides)
    set_overrides = parse_set_overrides(args.set_overrides)

    # Load configs with precedence: --set > CLI flags > user JSON > profile JSON > defaults
    # Settings layer owns all merging: just pass both dicts through
    run_cfg = load_run_config(
        args.config_run,
        profile=args.profile,
        flag_overrides=run_cli_overrides,
        set_overrides=set_overrides
    )
    dataset_schema = load_dataset_schema(args.config_dataset)

    # Report configuration sources
    if args.profile:
        logger.info(f"[CONFIG] Using profile: {args.profile}")
    if args.config_run:
        logger.info(f"[CONFIG] User config: {args.config_run}")
    if args.set_overrides:
        logger.info(f"[CONFIG] CLI overrides: {', '.join(args.set_overrides)}")

    # Handle effective config dump if requested
    if args.dump_effective_config:
        effective_config = {
            'run': run_cfg.model_dump(),
            'dataset': dataset_schema.model_dump() if dataset_schema else None,
        }

        if args.dump_effective_config == '-':
            # Print to stdout
            logger.info(json.dumps(effective_config, indent=2))
            sys.exit(0)
        else:
            # Write to file
            dump_path = Path(args.dump_effective_config)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open('w', encoding='utf-8') as f:
                json.dump(effective_config, f, indent=2)
            logger.info(f"Effective configuration dumped to: {dump_path.resolve()}")
            sys.exit(0)

    config = run_cfg

    # Configure logging based on CLI verbosity flags (CLI entry point responsibility)

    # Determine logging level from CLI flags (overrides config.verbose)
    if args.verbose:
        log_level = 'DEBUG'
    elif args.quiet:
        log_level = 'WARNING'
    else:
        # Default: INFO (or use config.verbose if present)
        log_level = 'DEBUG' if run_cfg.verbose else 'INFO'

    configure_tilescope_logging(verbose=(log_level == 'DEBUG'), level=log_level)

    logger.info("=" * 60)
    logger.info("TileScope - Spatial Transcriptomics Co-expression Analysis")
    logger.info("=" * 60)

    try:
        # Run analysis via API (keeps CLI and API in sync)

        # output_dir is already in run_cli_overrides when --output was supplied;
        # do not forward args.output as a kwarg or it will clobber the merged
        # value even when --output was omitted.
        results = run_tilescope_from_csv(
            input_csv=args.input_csv,
            run_config=run_cfg,
            dataset_schema=dataset_schema,
            profile=args.profile,
            flag_overrides=run_cli_overrides,
            set_overrides=set_overrides,
            export=True,
            html_report=True,
            configure_logging=False,  # Already configured above
            return_minimal=False,
            feature_a_id=args.feature_a_id,
            feature_b_id=args.feature_b_id,
        )

        export_paths = results.get("export_paths", {})

        # Summary
        logger.info("=" * 60)
        logger.info("Analysis Complete")
        logger.info("=" * 60)
        logger.info(f"Results directory: {Path(run_cfg.output_dir).resolve()}")
        logger.info("Summary:")
        logger.info(f"  Parent-ROI analyses: {len(results['roi_enrichment'])} ROIs")
        logger.info(f"  Consensus hexagons: {len(results.get('consensus_hexes_df', []))} hexes")
        logger.info(f"  Visualizations: {len([p for p in export_paths.values() if p.endswith('.pdf')])} plots")
        logger.info(f"  CSV exports: {len([p for p in export_paths.values() if p.endswith('.csv')])} files")
        logger.info(f"  Effective config: effective_config.json")

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
