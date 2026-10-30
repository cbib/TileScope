"""Gene enrichment screening tool - multi-gene enrichment screening.

Computes per-ROI enrichment for ALL genes in expression table using LOG2FC.

IMPORTANT: Uses LOG2FC from main TileScope (robust estimators) but simpler workflow:
- LOG2FC computed using adaptive robust estimators (median/winsorized/trimmed mean)
- Context = all spots NOT in current ROI (union of other ROIs)
- No percentile thresholds (unlike main TileScope proportion-based enrichment)
- No spatial tessellation (direct ROI vs context comparison)

Use case: Quick screening to identify which genes are enriched/depleted in each ROI
without running full TileScope tessellation/consensus analysis.

Output formats:
1. Detailed (long format): One row per (ROI, gene) with LOG2FC, FC, densities, verdict
2. Wide (gene lists): One row per ROI with lists of genes per verdict category
3. Summary (counts): One row per ROI with counts per verdict category
"""
from tilescope.enums import GeneVerdict
import os
from tilescope.tilescope_io.exports import export_dataframe_csv


import logging
import pandas as pd
import numpy as np
from typing import List, Dict

from tilescope.cols import ROI_ID, COORD_X, COORD_Y
from tilescope.policies.enrichment_concordance import ConcordancePolicy
from tilescope.metrics.pair_stats_core import PairStatsCore

logger = logging.getLogger(__name__)


def screen_gene_enrichment(expression_df: pd.DataFrame,
                           min_mean_expression: float = 0.0,
                           policy: ConcordancePolicy = None) -> pd.DataFrame:
    """Screen multi-gene enrichment using LOG2FC from main TileScope.

    Args:
        expression_df: Expression data (x, y, roi_id, gene1, gene2, ..., geneN)
        min_mean_expression: Minimum global mean expression to analyze gene (default: 0.0)
        policy: Enrichment policy for FC classification (default: uses defaults.py thresholds)

    Returns:
        DataFrame with columns: roi_id, gene_name, log2fc, fc, roi_mean_density,
                               context_mean_density, verdict
    """
    if policy is None:
        policy = ConcordancePolicy()

    # Initialize core stats engine
    stats_core = PairStatsCore()

    # Identify gene columns (exclude spatial/metadata columns)
    exclude_cols = {ROI_ID, COORD_X, COORD_Y}
    gene_cols = [col for col in expression_df.columns if col not in exclude_cols]

    if not gene_cols:
        raise ValueError("No gene columns found in expression_df")

    logger.info(f"[GENE-SCREEN] Found {len(gene_cols)} gene columns")

    # Filter low-expression genes (optional)
    if min_mean_expression > 0:
        gene_means = expression_df[gene_cols].mean()
        filtered_genes = gene_means[gene_means >= min_mean_expression].index.tolist()
        logger.info(f"[GENE-SCREEN] Filtered to {len(filtered_genes)}/{len(gene_cols)} genes "
                   f"with mean ≥ {min_mean_expression}")
        gene_cols = filtered_genes

    # Compute enrichment for each ROI
    roi_ids = expression_df[ROI_ID].unique()
    logger.info(f"[GENE-SCREEN] Computing enrichment for {len(roi_ids)} ROIs")

    results = []

    for roi_id in roi_ids:
        roi_spots = expression_df[expression_df[ROI_ID] == roi_id]
        context_spots = expression_df[expression_df[ROI_ID] != roi_id]

        if len(roi_spots) == 0:
            logger.warning(f"[GENE-SCREEN] ROI {roi_id}: No spots, skipping")
            continue

        if len(context_spots) == 0:
            logger.warning(f"[GENE-SCREEN] ROI {roi_id}: No context (single ROI dataset), skipping")
            continue

        for gene in gene_cols:
            roi_values = roi_spots[gene].dropna().values
            context_values = context_spots[gene].dropna().values

            if len(roi_values) == 0 or len(context_values) == 0:
                logger.warning(f"[GENE-SCREEN] ROI {roi_id}, {gene}: No valid values, skipping")
                continue

            # Get gene-specific minimum value for LOG2FC shift parameter
            # Computed from entire dataset (all ROIs) for this gene
            gene_min = float(expression_df[gene].min())

            # CALL existing LOG2FC computation from main TileScope (robust estimators)
            log2fc = stats_core.log2fc(roi_values, context_values, gene_min)

            # Convert LOG2FC to FC for classification
            fc = 2 ** log2fc

            # Classify using 5-level policy (reuse existing thresholds)
            verdict = policy.classify_fc(fc)

            # Compute mean densities for reporting
            roi_mean = float(np.mean(roi_values))
            context_mean = float(np.mean(context_values))

            results.append({
                ROI_ID: roi_id,
                'gene_name': gene,
                'log2fc': log2fc,
                'fc': fc,
                'roi_mean_density': roi_mean,
                'context_mean_density': context_mean,
                'verdict': verdict
            })

    results_df = pd.DataFrame(results)
    logger.info(f"[GENE-SCREEN] Computed {len(results_df)} (ROI, gene) enrichment values")

    return results_df


def build_screening_wide_format(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Convert detailed screening results to wide format (gene lists per verdict).

    Args:
        detailed_df: Output from screen_gene_enrichment()

    Returns:
        DataFrame with columns: roi_id, strong_enriched, weak_enriched, neutral,
                               weak_depleted, strong_depleted
                               (each column contains semicolon-separated gene names)
    """

    verdict_cols = [
        GeneVerdict.STRONG_ENRICHED.value,
        GeneVerdict.WEAK_ENRICHED.value,
        GeneVerdict.NEUTRAL.value,
        GeneVerdict.WEAK_DEPLETED.value,
        GeneVerdict.STRONG_DEPLETED.value
    ]

    wide_rows = []

    for roi_id in detailed_df[ROI_ID].unique():
        roi_df = detailed_df[detailed_df[ROI_ID] == roi_id]

        row = {ROI_ID: roi_id}

        for verdict in verdict_cols:
            genes = roi_df[roi_df['verdict'] == verdict]['gene_name'].tolist()
            row[verdict] = ';'.join(genes) if genes else ''

        wide_rows.append(row)

    return pd.DataFrame(wide_rows)


def build_screening_summary_format(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Convert detailed screening results to summary format (counts per verdict).

    Args:
        detailed_df: Output from screen_gene_enrichment()

    Returns:
        DataFrame with columns: roi_id, n_genes_total, n_strong_enriched, n_weak_enriched,
                               n_neutral, n_weak_depleted, n_strong_depleted
    """

    verdict_cols = [
        GeneVerdict.STRONG_ENRICHED.value,
        GeneVerdict.WEAK_ENRICHED.value,
        GeneVerdict.NEUTRAL.value,
        GeneVerdict.WEAK_DEPLETED.value,
        GeneVerdict.STRONG_DEPLETED.value
    ]

    summary_rows = []

    for roi_id in detailed_df[ROI_ID].unique():
        roi_df = detailed_df[detailed_df[ROI_ID] == roi_id]

        row = {
            ROI_ID: roi_id,
            'n_genes_total': len(roi_df)
        }

        for verdict in verdict_cols:
            count = (roi_df['verdict'] == verdict).sum()
            row[f'n_{verdict}'] = count

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def export_screening_results(detailed_df: pd.DataFrame,
                             output_dir: str,
                             formats: List[str] = None) -> Dict[str, str]:
    """Export gene enrichment screening results in multiple formats.

    Args:
        detailed_df: Output from screen_gene_enrichment()
        output_dir: Directory to write output files
        formats: List of formats to export ('detailed', 'wide', 'summary'), default: all

    Returns:
        Dictionary mapping format name to output file path
    """

    os.makedirs(output_dir, exist_ok=True)

    if formats is None:
        formats = ['detailed', 'wide', 'summary']

    export_paths = {}

    # 1. Detailed format (long format)
    if 'detailed' in formats:
        detailed_path = os.path.join(output_dir, 'gene_enrichment_detailed.csv')
        export_dataframe_csv(detailed_df, detailed_path, index=False)
        export_paths['detailed'] = detailed_path
        logger.info(f"[EXPORT] Detailed format: {detailed_path}")

    # 2. Wide format (gene lists)
    if 'wide' in formats:
        wide_df = build_screening_wide_format(detailed_df)
        wide_path = os.path.join(output_dir, 'gene_enrichment_lists.csv')
        export_dataframe_csv(wide_df, wide_path, index=False)
        export_paths['wide'] = wide_path
        logger.info(f"[EXPORT] Wide format (gene lists): {wide_path}")

    # 3. Summary format (counts)
    if 'summary' in formats:
        summary_df = build_screening_summary_format(detailed_df)
        summary_path = os.path.join(output_dir, 'gene_enrichment_counts.csv')
        export_dataframe_csv(summary_df, summary_path, index=False)
        export_paths['summary'] = summary_path
        logger.info(f"[EXPORT] Summary format (counts): {summary_path}")

    return export_paths
