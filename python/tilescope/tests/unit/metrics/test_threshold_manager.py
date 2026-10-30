"""Tests for compute_global_thresholds() pure function."""

import pytest
import numpy as np
import pandas as pd

from tilescope.metrics.enrichment_thresholds import compute_global_thresholds, PercentileThresholds
from tilescope.cols import GENE1, GENE2, ROI_ID
from tilescope.defaults import DEFAULT_Q_PERCENTILE, DEFAULT_Q_LOW_PERCENTILE, DEFAULT_MIN_TILE_SPOTS


def _create_test_dataframe(n_spots: int = 100) -> pd.DataFrame:
    """Helper: create test DataFrame with canonical gene columns."""
    rng = np.random.default_rng(42)

    return pd.DataFrame({
        GENE1: rng.normal(0, 1, n_spots),
        GENE2: rng.normal(0, 1, n_spots),
        ROI_ID: ['roi_0'] * n_spots,
        'x': rng.uniform(0, 100, n_spots),
        'y': rng.uniform(0, 100, n_spots),
    })


def test_compute_global_thresholds_basic():
    """compute_global_thresholds() computes thresholds from gene expression data."""
    df = _create_test_dataframe()

    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1].values,
        gene2_values=df[GENE2].values
    )

    assert isinstance(thresholds, PercentileThresholds)
    assert thresholds.q_percentile == DEFAULT_Q_PERCENTILE
    assert thresholds.q_low_percentile == DEFAULT_Q_LOW_PERCENTILE
    assert thresholds.min_points_threshold == DEFAULT_MIN_TILE_SPOTS
    assert thresholds.q_g1_threshold is not None
    assert thresholds.q_g2_threshold is not None


def test_compute_global_thresholds_custom_percentiles():
    """compute_global_thresholds() respects custom percentile parameters."""
    df = _create_test_dataframe()

    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1].values,
        gene2_values=df[GENE2].values,
        q_percentile=0.90,
        q_low_percentile=0.10,
        min_points_threshold=15
    )

    # Verify parameters were used
    assert thresholds.q_percentile == 0.90
    assert thresholds.q_low_percentile == 0.10
    assert thresholds.min_points_threshold == 15


def test_compute_global_thresholds_returns_immutable_dataclass():
    """compute_global_thresholds() returns immutable PercentileThresholds."""
    df = _create_test_dataframe()

    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1].values,
        gene2_values=df[GENE2].values
    )

    # Verify immutability (frozen=True)
    with pytest.raises(AttributeError):
        thresholds.q_g1_threshold = 999.0

    with pytest.raises(AttributeError):
        thresholds.q_percentile = 0.99


def test_threshold_values_are_reasonable():
    """Thresholds fall within expected ranges for normalized data."""
    df = _create_test_dataframe(n_spots=1000)

    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1].values,
        gene2_values=df[GENE2].values
    )

    # For normalized data (mean~0, std~1), 75th percentile should be ~0.67
    # Allow reasonable range for random data
    assert -2.0 < thresholds.q_g1_threshold < 2.0
    assert -2.0 < thresholds.q_g2_threshold < 2.0

    # 25th percentile should be negative
    assert -2.0 < thresholds.q_g1_low_threshold < 2.0
    assert -2.0 < thresholds.q_g2_low_threshold < 2.0

    # High threshold > low threshold
    assert thresholds.q_g1_threshold > thresholds.q_g1_low_threshold
    assert thresholds.q_g2_threshold > thresholds.q_g2_low_threshold


def test_compute_global_thresholds_pure_function():
    """compute_global_thresholds() is pure - same inputs produce same outputs."""
    df = _create_test_dataframe()

    gene1 = df[GENE1].values
    gene2 = df[GENE2].values

    # Call twice with same inputs
    thresholds1 = compute_global_thresholds(gene1, gene2)
    thresholds2 = compute_global_thresholds(gene1, gene2)

    # Values should be identical (pure function property)
    assert thresholds1.q_g1_threshold == thresholds2.q_g1_threshold
    assert thresholds1.q_g2_threshold == thresholds2.q_g2_threshold
    assert thresholds1.q_g1_low_threshold == thresholds2.q_g1_low_threshold
    assert thresholds1.q_g2_low_threshold == thresholds2.q_g2_low_threshold


def test_compute_global_thresholds_orchestrator_workflow():
    """Orchestrator pattern: compute once at start, pass to all stages."""
    df = _create_test_dataframe()

    # Compute thresholds once
    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1].values,
        gene2_values=df[GENE2].values
    )

    # Reuse threshold values (via .to_dict() or direct access)
    for _ in range(5):
        # Each analysis stage uses the same threshold values
        assert thresholds.q_g1_threshold is not None
        assert thresholds.q_g2_threshold is not None

        # Can serialize to dict for passing to compute functions
        threshold_dict = thresholds.to_dict()
        assert 'q_g1_threshold' in threshold_dict
        assert 'q_g2_threshold' in threshold_dict


def test_compute_global_thresholds_accepts_series():
    """compute_global_thresholds() accepts pandas Series (auto-converts to array)."""
    df = _create_test_dataframe()

    # Test with pandas Series directly
    thresholds = compute_global_thresholds(
        gene1_values=df[GENE1],  # Series, not .values
        gene2_values=df[GENE2]   # Series, not .values
    )

    assert isinstance(thresholds, PercentileThresholds)
    assert thresholds.q_g1_threshold is not None
