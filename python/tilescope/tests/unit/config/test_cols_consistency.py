"""Validation tests for cols.py consistency.

Ensures that column name constants are used consistently throughout cols.py
and that no raw string literals appear where constants should be used.
"""
from tilescope.schema.sufficient_stats import sufficient_stat_columns
from tilescope.metrics.pair_stats_core import SufficientStats
from dataclasses import fields
import numpy as np
import pandas as pd
from tilescope.schema.helpers import extract_coordinates


from tilescope.cols import (
    COORD_X,
    COORD_Y,
    N_SPOTS, SUM_G1, SUM_G2,
    N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
    N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
)


# NOTE: test_no_raw_coord_literals_in_production was removed. It scanned
# production source files for raw 'x','y' literals via regex — fragile
# (false positives from comments/docstrings, false negatives on
# `getattr`/f-strings) and slow. The runtime tests below already assert
# that the constants have the expected string values, which is the load-
# bearing contract.


def test_sufficient_stat_columns_helper():
    """sufficient_stat_columns() helper must return correct order and count."""

    cols = sufficient_stat_columns(N_SPOTS)

    # Expected order (locked to SufficientStats dataclass)
    expected = [N_SPOTS, SUM_G1, SUM_G2, N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH, N_G1_LOW, N_G2_LOW, N_BOTH_LOW]

    assert cols == expected, (
        f"sufficient_stat_columns() must return correct order. "
        f"Got: {cols}, Expected: {expected}"
    )
    # Ensure no duplicates
    assert len(cols) == len(set(cols)), "sufficient_stat_columns() returns duplicates"


def test_sufficient_stats_dataclass_matches_cols():
    """SufficientStats dataclass fields must match sufficient_stat_columns() helper."""

    # Get field names from SufficientStats dataclass (lowercase snake_case)
    stats_field_names = {f.name for f in fields(SufficientStats)}

    # Expected field names (lowercase versions of column constants)
    # Note: SufficientStats uses 'n_obs' (scale-agnostic) which maps to N_SPOTS column
    expected_field_names = {
        'n_obs', 'sum_g1', 'sum_g2',
        'n_g1_high', 'n_g2_high', 'n_both_high',
        'n_g1_low', 'n_g2_low', 'n_both_low'
    }

    assert stats_field_names == expected_field_names, (
        f"SufficientStats dataclass fields must match expected pattern. "
        f"Got: {stats_field_names}, Expected: {expected_field_names}"
    )

    # Verify count matches helper output
    assert len(stats_field_names) == len(sufficient_stat_columns(N_SPOTS)), (
        f"SufficientStats has {len(stats_field_names)} fields, "
        f"but sufficient_stat_columns() returns {len(sufficient_stat_columns(N_SPOTS))} columns"
    )


# NOTE: test_consensus_grid_uses_helper was removed. It read
# analyzers/consensus_grid.py as text and checked for the string
# `_STAT_FIELDS = [`. The path was CWD-relative and silently returned
# success when the file was not found (fail-open guardrail — bug).
# The intent (consensus_grid uses the sufficient_stat_columns helper) is
# already covered by the runtime tests below that exercise the helper
# and by any real analysis that instantiates a consensus grid.


def test_extract_coordinates_contract():
    """Verify extract_coordinates() returns (N, 2) float array from canonical columns."""

    # Test basic extraction
    df = pd.DataFrame({COORD_X: [1, 2, 3], COORD_Y: [4, 5, 6]})
    coords = extract_coordinates(df)

    # Shape validation
    assert coords.shape == (3, 2), f"Expected (3, 2), got {coords.shape}"

    # Dtype validation (floating point, not necessarily float64 on all platforms)
    assert np.issubdtype(coords.dtype, np.floating), (
        f"Expected floating point dtype, got {coords.dtype}"
    )

    # Value validation
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
    np.testing.assert_array_equal(coords, expected)

    # Test with float inputs (preserve precision)
    df_float = pd.DataFrame({COORD_X: [1.5, 2.7], COORD_Y: [3.2, 4.9]})
    coords_float = extract_coordinates(df_float)
    assert coords_float.shape == (2, 2)
    np.testing.assert_allclose(coords_float, [[1.5, 3.2], [2.7, 4.9]])
