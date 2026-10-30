#!/usr/bin/env python
"""Gene enrichment diagnostic tool - CLI entry point.

Quick univariate screening to identify enriched/depleted genes per ROI using
LOG2FC from main TileScope (simpler workflow than full tessellation analysis).

Usage:
    python tools/tilescope_enrichment_screening.py --input data.csv --output results/gene_enrichment/

    # With filtering
    python tools/tilescope_enrichment_screening.py --input data.csv --output results/gene_enrichment/ \\
        --min-expression 0.1

    # Specify output formats
    python tools/tilescope_enrichment_screening.py --input data.csv --output results/gene_enrichment/ \\
        --format detailed,wide,summary
"""
import traceback


import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd

from tilescope.logger_config import configure_tilescope_logging
from tilescope.enrichment.screening import (
    screen_gene_enrichment,
    export_screening_results
)
from tilescope.cols import ROI_ID, COORD_X, COORD_Y

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for gene enrichment diagnostic."""
    parser = argparse.ArgumentParser(
        description='Gene enrichment diagnostic - univariate LOG2FC-based enrichment screening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tools/tilescope_enrichment_screening.py --input data.csv --output results/gene_enrichment/

  # Filter low-expression genes
  python tools/tilescope_enrichment_screening.py --input data.csv --output results/gene_enrichment/ \\
      --min-expression 0.1

  # Specify output formats
  python tools/tilescope_enrichment_screening.py --input data.csv --output results/gene_enrichment/ \\
      --format detailed,wide,summary

Output formats:
  detailed: One row per (ROI, gene) with LOG2FC, FC, densities, verdict
  wide:     One row per ROI with semicolon-separated gene lists per verdict
  summary:  One row per ROI with counts per verdict category

Note: This uses LOG2FC from main TileScope (robust estimators) but simpler workflow:
      - No spatial tessellation (direct ROI vs context comparison)
      - No percentile thresholds (FC-based classification only)
      - Context = all spots NOT in current ROI (union of other ROIs)
"""
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file with expression data (x, y, roi_id, gene1, gene2, ..., geneN)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for enrichment results'
    )

    parser.add_argument(
        '--roi-column',
        default=ROI_ID,
        help=f"Name of the ROI column in the input file (default: '{ROI_ID}')"
    )

    parser.add_argument(
        '--min-expression', '-m',
        type=float,
        default=0.0,
        help='Minimum global mean expression to analyze gene (default: 0.0, analyze all)'
    )

    parser.add_argument(
        '--format', '-f',
        default='detailed,wide,summary',
        help='Output formats (comma-separated): detailed, wide, summary (default: all)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (WARNING level only)'
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_level = 'DEBUG'
    elif args.quiet:
        log_level = 'WARNING'
    else:
        log_level = 'INFO'

    configure_tilescope_logging(verbose=(log_level == 'DEBUG'), level=log_level)

    logger.info("=" * 60)
    logger.info("TileScope Gene Enrichment Diagnostic")
    logger.info("=" * 60)

    try:
        # Load expression data (directly with pandas - no gene1/gene2 requirement)
        logger.info(f"Loading expression data from {args.input}")
        expression_df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(expression_df)} spots")

        # Normalize/validate ROI column
        roi_column = args.roi_column
        available_columns = set(expression_df.columns)
        alias_candidates = []

        # Allow common aliases when user keeps default roi_id argument
        if roi_column == ROI_ID:
            alias_candidates = ['ROI', 'roi', 'ROI_ID', 'Roi', 'ROIId']

        roi_column_in_data = None

        if roi_column in available_columns:
            roi_column_in_data = roi_column
        else:
            # Case-insensitive equality check for provided name
            lower_map = {col.lower(): col for col in expression_df.columns}
            roi_lower = roi_column.lower()
            if roi_lower in lower_map:
                roi_column_in_data = lower_map[roi_lower]
            else:
                for alias in alias_candidates:
                    if alias in available_columns:
                        roi_column_in_data = alias
                        break

        if roi_column_in_data is None:
            raise ValueError(
                f"ROI column '{roi_column}' not found in input columns: {sorted(expression_df.columns)}"
            )

        if roi_column_in_data != ROI_ID:
            logger.info(
                f"Renaming ROI column '{roi_column_in_data}' to canonical '{ROI_ID}' for analysis"
            )
            expression_df = expression_df.rename(columns={roi_column_in_data: ROI_ID})

        # Validate required columns
        required_cols = {ROI_ID, COORD_X, COORD_Y}
        missing = required_cols - set(expression_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for at least one gene column
        gene_cols = [c for c in expression_df.columns if c not in required_cols]
        if not gene_cols:
            raise ValueError(f"No gene columns found (only found: {list(expression_df.columns)})")

        logger.info(f"Found {len(gene_cols)} gene columns: {gene_cols[:5]}{'...' if len(gene_cols) > 5 else ''}")

        # Parse output formats
        formats = [f.strip() for f in args.format.split(',')]
        valid_formats = {'detailed', 'wide', 'summary'}
        invalid = set(formats) - valid_formats
        if invalid:
            logger.error(f"Invalid format(s): {invalid}. Valid: {valid_formats}")
            sys.exit(1)

        # Screen enrichment
        logger.info(f"Screening gene enrichment (min_expression={args.min_expression})")
        detailed_df = screen_gene_enrichment(
            expression_df,
            min_mean_expression=args.min_expression
        )

        # Export results
        output_dir = Path(args.output)
        logger.info(f"Exporting results to {output_dir}")
        export_paths = export_screening_results(
            detailed_df,
            str(output_dir),
            formats=formats
        )

        logger.info("=" * 60)
        logger.info("Gene enrichment diagnostic complete!")
        logger.info("=" * 60)
        logger.info("Output files:")
        for fmt, path in export_paths.items():
            logger.info(f"  {fmt:10s}: {path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
