"""Unit tests for winsorized mean hex area sizing."""

import numpy as np
import pytest
from tilescope.analyzers.consensus_grid import winsorized_mean


def test_winsorized_mean_reduces_outlier_influence():
    """Winsorized mean should lie between median and mean for right-skewed data."""
    # Simulate Voronoi distribution: 95 interior tiles + 5 huge boundary tiles
    A_tiles = np.array([10.0] * 95 + [1000.0] * 5)

    median = np.median(A_tiles)  # 10.0 (50th percentile)
    mean_plain = A_tiles.mean()  # 59.5 (heavily skewed by top 5)
    mean_winsor = winsorized_mean(A_tiles, 0.0, 0.95)  # Clips top 5% (5 tiles)

    # Verify ordering
    assert median <= mean_winsor <= mean_plain

    # Verify winsor is closer to median than mean (clips outliers)
    assert abs(mean_winsor - median) < abs(mean_plain - median)

    # Verify winsor is reasonable (not inflated by outliers)
    assert mean_winsor < mean_plain, f"Expected winsor < mean, got {mean_winsor} vs {mean_plain}"


def test_winsorized_mean_edge_cases():
    """Test edge cases for winsorized mean."""

    # Empty array
    assert winsorized_mean(np.array([]), 0.0, 1.0) == 0.0

    # No clipping (should equal plain mean)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert winsorized_mean(x, 0.0, 1.0) == x.mean()

    # All identical values
    x = np.array([5.0] * 100)
    assert winsorized_mean(x, 0.0, 0.95) == 5.0


def test_winsorized_mean_invalid_quantiles():
    """Invalid quantiles should raise ValueError."""
    x = np.array([1, 2, 3])

    # q_low > q_high
    with pytest.raises(ValueError, match="Invalid winsor quantiles"):
        winsorized_mean(x, 0.5, 0.3)

    # q_low == q_high
    with pytest.raises(ValueError, match="Invalid winsor quantiles"):
        winsorized_mean(x, 0.5, 0.5)

    # q_low < 0
    with pytest.raises(ValueError, match="Invalid winsor quantiles"):
        winsorized_mean(x, -0.1, 0.9)

    # q_high > 1
    with pytest.raises(ValueError, match="Invalid winsor quantiles"):
        winsorized_mean(x, 0.0, 1.1)


def test_winsorized_mean_symmetric_vs_asymmetric():
    """Test symmetric vs asymmetric winsorization on skewed data."""
    # Right-skewed distribution
    x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 200, 300])

    # Symmetric winsorization (clip both tails equally)
    mean_sym = winsorized_mean(x, 0.05, 0.95)

    # Asymmetric winsorization (clip top tail only)
    mean_asym = winsorized_mean(x, 0.0, 0.95)

    # For right-skewed data, asymmetric should be lower (less influenced by outliers)
    assert mean_asym < x.mean(), "Asymmetric winsor should be < plain mean"
    assert abs(mean_asym - np.median(x)) < abs(mean_sym - np.median(x)), \
        "Asymmetric winsor should be closer to median for right-skewed data"


def test_winsorized_mean_quantile_sensitivity():
    """Test how different quantile thresholds affect winsorization."""
    # Simulated Voronoi tile areas (right-skewed)
    x = np.array([10] * 50 + [20] * 30 + [50] * 15 + [200] * 4 + [1000] * 1)

    # Conservative (clip less)
    mean_98 = winsorized_mean(x, 0.0, 0.98)

    # Moderate (clip more)
    mean_95 = winsorized_mean(x, 0.0, 0.95)

    # Aggressive (clip most)
    mean_90 = winsorized_mean(x, 0.0, 0.90)

    # More aggressive clipping should produce smaller means (for right-skewed data)
    assert mean_90 < mean_95 < mean_98 < x.mean()

    # All should be closer to median than plain mean
    median = np.median(x)
    assert abs(mean_90 - median) < abs(x.mean() - median)
    assert abs(mean_95 - median) < abs(x.mean() - median)
    assert abs(mean_98 - median) < abs(x.mean() - median)
