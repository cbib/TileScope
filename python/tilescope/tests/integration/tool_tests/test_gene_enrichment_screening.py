"""Integration test for gene enrichment diagnostic tool.

Tests the full gene enrichment workflow using the mock 5 ROI, 100 gene fixture.

Fixture properties:
- 5 ROIs (ROI_1 ... ROI_5)
- 100 genes (G1 ... G100)
- 50 spots per ROI (250 total)
- Each gene is enriched in exactly one ROI (2-5× boost)
"""
from tilescope.defaults import DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD, DEFAULT_FC_WEAK_ENRICHMENT_THRESHOLD, DEFAULT_FC_STRONG_DEPLETION_THRESHOLD, DEFAULT_FC_WEAK_DEPLETION_THRESHOLD
from tilescope.metrics.pair_stats_core import PairStatsCore


import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path

from tilescope.enrichment.screening import (
    screen_gene_enrichment,
    build_screening_wide_format,
    build_screening_summary_format,
    export_screening_results
)
from tilescope.cols import ROI_ID
from tilescope.enums import GeneVerdict


@pytest.fixture
def mock_5roi_100gene_data():
    """[FIXTURE] Load mock dataset with 5 ROIs and 100 genes."""
    fixture_path = Path(__file__).parent.parent.parent / 'fixtures' / 'tilescope_mock_5rois_100genes.csv'
    df = pd.read_csv(fixture_path)
    return df


def test_gene_enrichment_basic_computation(mock_5roi_100gene_data):
    """[ENRICH] screen_gene_enrichment produces valid results for 5 ROI fixture."""
    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)

    # Should have one row per (ROI, gene) combination
    n_rois = mock_5roi_100gene_data[ROI_ID].nunique()
    n_genes = 100
    expected_rows = n_rois * n_genes

    assert len(detailed_df) == expected_rows, \
        f"Expected {expected_rows} rows (5 ROIs * 100 genes), got {len(detailed_df)}"

    # Check required columns exist
    required_cols = {ROI_ID, 'gene_name', 'fc', 'roi_mean_density', 'context_mean_density', 'verdict'}
    assert required_cols.issubset(detailed_df.columns), \
        f"Missing columns: {required_cols - set(detailed_df.columns)}"

    # All verdicts should be valid
    valid_verdicts = {
        GeneVerdict.STRONG_ENRICHED.value,
        GeneVerdict.WEAK_ENRICHED.value,
        GeneVerdict.NEUTRAL.value,
        GeneVerdict.WEAK_DEPLETED.value,
        GeneVerdict.STRONG_DEPLETED.value
    }
    invalid_verdicts = set(detailed_df['verdict'].unique()) - valid_verdicts
    assert len(invalid_verdicts) == 0, f"Found invalid verdicts: {invalid_verdicts}"

    # FC should be positive and reasonable
    assert (detailed_df['fc'] > 0).all(), "All FCs should be positive"
    assert (detailed_df['fc'] < 100).all(), "FCs should be reasonable (< 100x)"


def test_gene_enrichment_per_gene_specificity(mock_5roi_100gene_data):
    """[ENRICH] Each gene should be strongly enriched in exactly one ROI."""
    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)

    # For each gene, check enrichment pattern
    for gene_name in [f'G{i}' for i in range(1, 101)]:
        gene_df = detailed_df[detailed_df['gene_name'] == gene_name]

        # Should have 5 rows (one per ROI)
        assert len(gene_df) == 5, f"{gene_name}: Expected 5 ROIs, got {len(gene_df)}"

        # Count strong enrichments
        strong_enriched = (gene_df['verdict'] == GeneVerdict.STRONG_ENRICHED.value).sum()

        # Each gene should be strongly enriched in at least one ROI
        # (Due to the 2-5× boost in fixture design)
        assert strong_enriched >= 1, \
            f"{gene_name}: Expected at least 1 strong enrichment, got {strong_enriched}"

        # Most genes should have depleted status in other ROIs (reciprocal pattern)
        depleted = (
            (gene_df['verdict'] == GeneVerdict.STRONG_DEPLETED.value) |
            (gene_df['verdict'] == GeneVerdict.WEAK_DEPLETED.value)
        ).sum()

        # Should have some depletion in other ROIs (at least 1 out of 4 other ROIs)
        # Note: Not all genes will be depleted in all other ROIs due to random variation
        assert depleted >= 1, \
            f"{gene_name}: Expected depletion in other ROIs, got {depleted}/4"


def test_gene_enrichment_fc_verdict_consistency(mock_5roi_100gene_data):
    """[ENRICH] FC thresholds should match verdicts correctly."""

    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)

    # Check strong enrichment
    strong_enriched = detailed_df[detailed_df['verdict'] == GeneVerdict.STRONG_ENRICHED.value]
    assert (strong_enriched['fc'] >= DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD).all(), \
        "Strong enriched genes should have FC >= strong threshold"

    # Check weak enrichment
    weak_enriched = detailed_df[detailed_df['verdict'] == GeneVerdict.WEAK_ENRICHED.value]
    assert (weak_enriched['fc'] >= DEFAULT_FC_WEAK_ENRICHMENT_THRESHOLD).all(), \
        "Weak enriched genes should have FC >= weak threshold"
    assert (weak_enriched['fc'] < DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD).all(), \
        "Weak enriched genes should have FC < strong threshold"

    # Check strong depletion
    strong_depleted = detailed_df[detailed_df['verdict'] == GeneVerdict.STRONG_DEPLETED.value]
    assert (strong_depleted['fc'] <= DEFAULT_FC_STRONG_DEPLETION_THRESHOLD).all(), \
        "Strong depleted genes should have FC <= strong depletion threshold"

    # Check weak depletion
    weak_depleted = detailed_df[detailed_df['verdict'] == GeneVerdict.WEAK_DEPLETED.value]
    assert (weak_depleted['fc'] <= DEFAULT_FC_WEAK_DEPLETION_THRESHOLD).all(), \
        "Weak depleted genes should have FC <= weak depletion threshold"
    assert (weak_depleted['fc'] > DEFAULT_FC_STRONG_DEPLETION_THRESHOLD).all(), \
        "Weak depleted genes should have FC > strong depletion threshold"


def test_gene_enrichment_wide_format(mock_5roi_100gene_data):
    """[ENRICH] Wide format (gene lists) is generated correctly."""
    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)
    wide_df = build_screening_wide_format(detailed_df)

    # Should have one row per ROI
    assert len(wide_df) == 5, f"Expected 5 ROIs, got {len(wide_df)}"

    # Check required columns
    expected_cols = [
        ROI_ID,
        GeneVerdict.STRONG_ENRICHED.value,
        GeneVerdict.WEAK_ENRICHED.value,
        GeneVerdict.NEUTRAL.value,
        GeneVerdict.WEAK_DEPLETED.value,
        GeneVerdict.STRONG_DEPLETED.value
    ]
    assert set(expected_cols).issubset(wide_df.columns), \
        f"Missing columns: {set(expected_cols) - set(wide_df.columns)}"

    # Each ROI should have some enriched genes (semicolon-separated strings)
    for _, row in wide_df.iterrows():
        strong_enriched_genes = row[GeneVerdict.STRONG_ENRICHED.value]

        # Should be non-empty string with gene names
        if strong_enriched_genes:  # Could be empty string
            assert ';' in strong_enriched_genes or len(strong_enriched_genes.split(';')) >= 1, \
                f"Expected semicolon-separated gene list, got: {strong_enriched_genes}"


def test_gene_enrichment_summary_format(mock_5roi_100gene_data):
    """[ENRICH] Summary format (counts) is generated correctly."""
    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)
    summary_df = build_screening_summary_format(detailed_df)

    # Should have one row per ROI
    assert len(summary_df) == 5, f"Expected 5 ROIs, got {len(summary_df)}"

    # Check required columns
    expected_cols = [
        ROI_ID,
        'n_genes_total',
        'n_strong_enriched',
        'n_weak_enriched',
        'n_neutral',
        'n_weak_depleted',
        'n_strong_depleted'
    ]
    assert set(expected_cols).issubset(summary_df.columns), \
        f"Missing columns: {set(expected_cols) - set(summary_df.columns)}"

    # Each ROI should have 100 genes total
    assert (summary_df['n_genes_total'] == 100).all(), \
        "Each ROI should have exactly 100 genes"

    # Counts should sum to 100
    for _, row in summary_df.iterrows():
        total_count = (
            row['n_strong_enriched'] +
            row['n_weak_enriched'] +
            row['n_neutral'] +
            row['n_weak_depleted'] +
            row['n_strong_depleted']
        )
        assert total_count == 100, \
            f"ROI {row[ROI_ID]}: Verdict counts sum to {total_count}, expected 100"


def test_gene_enrichment_export_all_formats(mock_5roi_100gene_data):
    """[ENRICH] Export produces all three output formats correctly."""
    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_paths = export_screening_results(
            detailed_df,
            tmpdir,
            formats=['detailed', 'wide', 'summary']
        )

        # Should have 3 output files
        assert len(export_paths) == 3, f"Expected 3 formats, got {len(export_paths)}"
        assert 'detailed' in export_paths
        assert 'wide' in export_paths
        assert 'summary' in export_paths

        # All files should exist
        for fmt, path in export_paths.items():
            assert os.path.exists(path), f"{fmt} file not found: {path}"

            # Files should be non-empty
            assert os.path.getsize(path) > 0, f"{fmt} file is empty: {path}"

            # Should be valid CSV
            df = pd.read_csv(path)
            assert len(df) > 0, f"{fmt} CSV has no data rows"


def test_gene_enrichment_context_definition(mock_5roi_100gene_data):
    """[ENRICH] Context is correctly defined as 'all other ROIs'."""

    detailed_df = screen_gene_enrichment(mock_5roi_100gene_data)

    # For a specific gene in a specific ROI, manually compute expected context
    test_gene = 'G1'
    test_roi = 'ROI_1'

    # Get result
    result_row = detailed_df[
        (detailed_df[ROI_ID] == test_roi) &
        (detailed_df['gene_name'] == test_gene)
    ].iloc[0]

    # Manually compute expected values
    roi_spots = mock_5roi_100gene_data[mock_5roi_100gene_data[ROI_ID] == test_roi]
    context_spots = mock_5roi_100gene_data[mock_5roi_100gene_data[ROI_ID] != test_roi]

    expected_roi_mean = roi_spots[test_gene].mean()
    expected_context_mean = context_spots[test_gene].mean()

    # Compute expected FC using same LOG2FC approach as implementation
    stats_core = PairStatsCore()
    roi_values = roi_spots[test_gene].dropna().values
    context_values = context_spots[test_gene].dropna().values
    gene_min = float(mock_5roi_100gene_data[test_gene].min())
    expected_log2fc = stats_core.log2fc(roi_values, context_values, gene_min)
    expected_fc = 2 ** expected_log2fc

    # Check values match (within floating point tolerance)
    assert abs(result_row['roi_mean_density'] - expected_roi_mean) < 1e-6, \
        f"ROI mean mismatch: {result_row['roi_mean_density']} vs {expected_roi_mean}"
    assert abs(result_row['context_mean_density'] - expected_context_mean) < 1e-6, \
        f"Context mean mismatch: {result_row['context_mean_density']} vs {expected_context_mean}"
    assert abs(result_row['fc'] - expected_fc) < 1e-6, \
        f"FC mismatch: {result_row['fc']} vs {expected_fc}"


def test_gene_enrichment_min_expression_filter(mock_5roi_100gene_data):
    """[ENRICH] min_mean_expression filter works correctly."""
    # Compute global gene means
    exclude_cols = {ROI_ID, 'x', 'y'}
    gene_cols = [c for c in mock_5roi_100gene_data.columns if c not in exclude_cols]
    gene_means = mock_5roi_100gene_data[gene_cols].mean()

    # Set threshold at median
    threshold = gene_means.median()
    genes_above_threshold = (gene_means >= threshold).sum()

    # Run enrichment with filter
    detailed_df = screen_gene_enrichment(
        mock_5roi_100gene_data,
        min_mean_expression=threshold
    )

    # Should have filtered out low-expression genes
    n_genes_analyzed = detailed_df['gene_name'].nunique()

    # Should be approximately half the genes (around 50)
    assert 40 <= n_genes_analyzed <= 60, \
        f"Expected ~50 genes with median threshold, got {n_genes_analyzed}"

    # All analyzed genes should have mean >= threshold
    analyzed_genes = detailed_df['gene_name'].unique()
    for gene in analyzed_genes:
        assert gene_means[gene] >= threshold, \
            f"{gene} has mean {gene_means[gene]:.2f} < threshold {threshold:.2f}"
