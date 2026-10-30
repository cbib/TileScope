"""
Unit tests for PairStatsCore.

Tests array-first statistical engine, NA policy, constant handling, and protocol-based adapters.
"""

import numpy as np
import pandas as pd
import pytest

from tilescope.metrics.pair_stats_core import PairStatsCore, SufficientStats
from tilescope.cols import GENE1, GENE2, GENE1_DENSITY, GENE2_DENSITY


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def core():
    return PairStatsCore()


@pytest.fixture
def valid_data():
    """Valid gene pair data for testing."""
    np.random.seed(123)
    n = 100
    g1 = np.random.randn(n) + 1.0  # Mean ~1
    g2 = 0.5 * g1 + 0.5 * np.random.randn(n)  # Correlated with g1
    return g1, g2


@pytest.fixture
def data_with_na():
    """Gene pair data with NA values."""
    g1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    g2 = np.array([2.0, np.nan, 3.0, 4.0, 5.0])
    return g1, g2


# =============================================================================
# SUFFICIENT STATS TESTS
# =============================================================================

def test_sufficient_stats_basic(core, valid_data):
    """Test basic sufficient statistics computation."""
    g1, g2 = valid_data
    q_hi_g1, q_hi_g2 = np.percentile(g1, 75), np.percentile(g2, 75)
    q_lo_g1, q_lo_g2 = np.percentile(g1, 25), np.percentile(g2, 25)

    stats = core.sufficient_stats(
        g1, g2,
        q_hi_g1=q_hi_g1, q_hi_g2=q_hi_g2,
        q_lo_g1=q_lo_g1, q_lo_g2=q_lo_g2
    )

    assert isinstance(stats, SufficientStats)
    assert stats.n_obs == len(g1)
    assert np.isclose(stats.sum_g1, g1.sum())
    assert np.isclose(stats.sum_g2, g2.sum())
    assert stats.n_g1_high > 0
    assert stats.n_g2_high > 0
    assert stats.n_g1_low > 0
    assert stats.n_g2_low > 0


def test_sufficient_stats_na_policy(core, data_with_na):
    """Test NA policy: drop rows where either gene is NA."""
    g1, g2 = data_with_na
    # Valid rows: indices 0, 3, 4 (where neither is NA)
    q_hi_g1, q_hi_g2 = 4.0, 4.0
    q_lo_g1, q_lo_g2 = 2.0, 2.0

    stats = core.sufficient_stats(
        g1, g2,
        q_hi_g1=q_hi_g1, q_hi_g2=q_hi_g2,
        q_lo_g1=q_lo_g1, q_lo_g2=q_lo_g2
    )

    # Should have 3 valid observations (indices 0, 3, 4)
    assert stats.n_obs == 3
    expected_sum_g1 = 1.0 + 4.0 + 5.0  # Valid g1 values
    expected_sum_g2 = 2.0 + 4.0 + 5.0  # Valid g2 values
    assert np.isclose(stats.sum_g1, expected_sum_g1)
    assert np.isclose(stats.sum_g2, expected_sum_g2)


def test_sufficient_stats_all_na(core):
    """Test error when all observations are NA."""
    g1 = np.array([np.nan, np.nan, np.nan])
    g2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="All observations have NA values"):
        core.sufficient_stats(g1, g2, q_hi_g1=1.0, q_hi_g2=2.0, q_lo_g1=0.5, q_lo_g2=1.0)


def test_sufficient_stats_accepts_constant_operands(core):
    """Constant operands are accepted; HIGH / LOW counters remain well-defined.

    A constant vector equal to `c` yields `n = full` when `c > q_hi` (all HIGH)
    and `n = full` when `c < q_lo` (all LOW). Correlation is undefined and
    handled separately by ``compute_correlation_from_arrays``; the sufficient-
    stats layer no longer rejects the input.
    """
    g1 = np.full(100, 5.0)                    # constant, above q_hi=2.0
    g2 = np.linspace(-1.0, 1.0, 100)          # non-constant
    stats = core.sufficient_stats(
        g1, g2,
        q_hi_g1=2.0, q_hi_g2=0.5,
        q_lo_g1=1.0, q_lo_g2=-0.5,
    )
    assert stats.n_obs == 100
    assert stats.n_g1_high == 100             # constant 5.0 > 2.0 for every row
    assert stats.n_g1_low == 0                # constant 5.0 not < 1.0

    g1_below = np.full(100, 0.0)              # constant, below q_lo
    stats_low = core.sufficient_stats(
        g1_below, g2,
        q_hi_g1=2.0, q_hi_g2=0.5,
        q_lo_g1=1.0, q_lo_g2=-0.5,
    )
    assert stats_low.n_g1_high == 0
    assert stats_low.n_g1_low == 100


def test_sufficient_stats_length_mismatch(core):
    """Test error when gene arrays have different lengths."""
    g1 = np.random.randn(100)
    g2 = np.random.randn(50)

    with pytest.raises(ValueError, match="Gene arrays must have same length"):
        core.sufficient_stats(g1, g2, q_hi_g1=1.0, q_hi_g2=1.0, q_lo_g1=0.5, q_lo_g2=0.5)


def test_sufficient_stats_threshold_counts(core):
    """Test correct counting of high/low threshold observations."""
    # Controlled data for predictable counts
    g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    stats = core.sufficient_stats(
        g1, g2,
        q_hi_g1=3.5, q_hi_g2=4.0,  # g1 > 3.5 (2 values: 4, 5), g2 > 4.0 (2 values: 4.5, 5.5)
        q_lo_g1=2.5, q_lo_g2=3.0   # g1 < 2.5 (2 values: 1, 2), g2 < 3.0 (2 values: 1.5, 2.5)
    )

    assert stats.n_g1_high == 2  # 4, 5 > 3.5
    assert stats.n_g2_high == 2  # 4.5, 5.5 > 4.0
    assert stats.n_both_high == 2  # Both genes high: indices 3, 4
    assert stats.n_g1_low == 2  # 1, 2 < 2.5
    assert stats.n_g2_low == 2  # 1.5, 2.5 < 3.0
    assert stats.n_both_low == 2  # Both genes low: indices 0, 1


# =============================================================================
# LOG2FC TESTS
# =============================================================================

def test_log2fc_basic(core):
    """Test log2 fold change computation."""
    roi_values = np.array([3.0, 4.0, 5.0])  # Mean ~4
    context_values = np.array([1.0, 2.0, 3.0])  # Mean ~2
    gene_min_value = 0.0

    log2fc = core.log2fc(roi_values, context_values, gene_min_value)

    assert isinstance(log2fc, (float, np.float64))
    assert log2fc > 0  # ROI > context, so log2fc > 0


def test_log2fc_small_sample(core):
    """Test log2fc uses median for small samples."""
    # Small sample (< FC_SMALL_SAMPLE_THRESHOLD = 5)
    roi_values = np.array([5.0, 10.0])
    context_values = np.array([1.0, 2.0])
    gene_min_value = 0.0

    log2fc = core.log2fc(roi_values, context_values, gene_min_value)

    assert isinstance(log2fc, (float, np.float64))
    assert not np.isnan(log2fc)


def test_log2fc_negative_values(core):
    """Test log2fc with negative minimum value."""
    roi_values = np.array([0.5, 1.0, 1.5])
    context_values = np.array([0.1, 0.2, 0.3])
    gene_min_value = -1.0  # Negative minimum

    log2fc = core.log2fc(roi_values, context_values, gene_min_value)

    assert isinstance(log2fc, (float, np.float64))
    assert not np.isnan(log2fc)


# =============================================================================
# CLES TESTS
# =============================================================================

def test_cles_basic(core):
    """Test Common Language Effect Size computation."""
    roi_values = np.array([3.0, 4.0, 5.0])
    context_values = np.array([1.0, 2.0, 3.0])

    cles = core.cles(roi_values, context_values)

    assert isinstance(cles, (float, np.float64))
    assert 0.0 <= cles <= 1.0
    assert cles > 0.5  # ROI values generally > context values


def test_cles_equal_distributions(core):
    """Test CLES with identical distributions."""
    values = np.array([1.0, 2.0, 3.0])

    cles = core.cles(values, values)

    # When distributions are identical and using strict > comparison,
    # CLES counts only pairs where roi > context (no ties counted as 0.5).
    # For [1,2,3] vs [1,2,3]: 3>1, 3>2, 2>1 = 3 out of 9 = 0.333...
    assert np.isclose(cles, 0.333, atol=0.01)


def test_cles_no_overlap(core):
    """Test CLES with non-overlapping distributions."""
    roi_values = np.array([10.0, 11.0, 12.0])
    context_values = np.array([1.0, 2.0, 3.0])

    cles = core.cles(roi_values, context_values)

    # All ROI values > all context values, so CLES = 1.0
    assert cles == 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_end_to_end_spot_level(core):
    """Spot-level end-to-end: sufficient_stats + correlation via
    PairStatsCore. Arrays extracted from DataFrame columns via
    ``.to_numpy()``."""
    df_roi = pd.DataFrame({
        GENE1: np.random.randn(50) + 2.0,
        GENE2: np.random.randn(50) + 1.0
    })
    df_ctx = pd.DataFrame({
        GENE1: np.random.randn(200),
        GENE2: np.random.randn(200)
    })
    g1_roi = df_roi[GENE1].to_numpy()
    g2_roi = df_roi[GENE2].to_numpy()
    g1_ctx = df_ctx[GENE1].to_numpy()
    g2_ctx = df_ctx[GENE2].to_numpy()

    q_hi_g1, q_hi_g2 = np.percentile(g1_ctx, 75), np.percentile(g2_ctx, 75)
    q_lo_g1, q_lo_g2 = np.percentile(g1_ctx, 25), np.percentile(g2_ctx, 25)

    roi_stats = core.sufficient_stats(
        g1_roi, g2_roi,
        q_hi_g1=q_hi_g1, q_hi_g2=q_hi_g2,
        q_lo_g1=q_lo_g1, q_lo_g2=q_lo_g2
    )

    assert roi_stats.n_obs == 50
    assert isinstance(roi_stats.sum_g1, float)

def test_end_to_end_tile_level(core):
    """Tile-level end-to-end: sufficient_stats + log2fc + CLES via
    PairStatsCore. Arrays extracted from DataFrame density columns via
    ``.to_numpy()``."""
    df_roi = pd.DataFrame({
        GENE1_DENSITY: np.random.rand(30) * 0.5 + 0.5,
        GENE2_DENSITY: np.random.rand(30) * 0.4 + 0.3,
    })
    df_ctx = pd.DataFrame({
        GENE1_DENSITY: np.random.rand(100) * 0.5,
        GENE2_DENSITY: np.random.rand(100) * 0.4,
    })
    g1_roi = df_roi[GENE1_DENSITY].to_numpy()
    g2_roi = df_roi[GENE2_DENSITY].to_numpy()
    g1_ctx = df_ctx[GENE1_DENSITY].to_numpy()
    g2_ctx = df_ctx[GENE2_DENSITY].to_numpy()

    q_hi_g1, q_hi_g2 = np.percentile(g1_ctx, 75), np.percentile(g2_ctx, 75)
    q_lo_g1, q_lo_g2 = np.percentile(g1_ctx, 25), np.percentile(g2_ctx, 25)

    roi_stats = core.sufficient_stats(
        g1_roi, g2_roi,
        q_hi_g1=q_hi_g1, q_hi_g2=q_hi_g2,
        q_lo_g1=q_lo_g1, q_lo_g2=q_lo_g2
    )

    assert roi_stats.n_obs == 30

    log2fc_g1 = core.log2fc(g1_roi, g1_ctx, gene_min_value=0.0)
    cles_g1 = core.cles(g1_roi, g1_ctx)

    assert isinstance(log2fc_g1, (float, np.float64))
    assert isinstance(cles_g1, (float, np.float64))
    assert 0.0 <= cles_g1 <= 1.0
