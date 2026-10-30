"""Contract tests for the canonical median-NN spacing primitive.

Six invariants:

- known spacing: uniform grid → known NN;
- duplicates: shared coordinates give 1-NN = 0 for those rows;
- fewer than two points → NaN (array) / None (DataFrame);
- DataFrame wrapper matches array primitive on the same coords;
- non-finite coordinates raise ValueError;
- malformed shape raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.spatial.spacing import (
    median_nn_distance,
    median_nn_distance_from_df,
)


def test_uniform_grid_returns_grid_step():
    """A 4x4 unit-step grid has 1-NN = 1 for every point → median = 1."""
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    coords = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    assert median_nn_distance(coords) == pytest.approx(1.0)


def test_duplicates_produce_zero_nn_and_shift_median():
    """Duplicated coordinates give 1-NN = 0 for BOTH rows in the pair; the
    dedup semantic is a caller concern, not the primitive's."""
    # Three distinct points spaced by 10, plus one duplicate of the middle.
    coords = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 0.0],  # duplicate of index 1
        [20.0, 0.0],
    ])
    # 1-NN distances: [10, 0, 0, 10] → median = 5.
    assert median_nn_distance(coords) == pytest.approx(5.0)


def test_fewer_than_two_points_returns_nan():
    assert np.isnan(median_nn_distance(np.zeros((0, 2))))
    assert np.isnan(median_nn_distance(np.array([[1.5, 2.5]])))


def test_dataframe_wrapper_matches_array_primitive():
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    assert median_nn_distance_from_df(df, "x", "y") == median_nn_distance(coords)


def test_dataframe_wrapper_returns_none_on_short_input():
    assert median_nn_distance_from_df(None, "x", "y") is None
    assert median_nn_distance_from_df(pd.DataFrame({"x": [], "y": []}), "x", "y") is None
    assert median_nn_distance_from_df(pd.DataFrame({"x": [1.0], "y": [2.0]}), "x", "y") is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_are_rejected(bad):
    coords = np.array([[0.0, 0.0], [1.0, bad]])
    with pytest.raises(ValueError, match="non-finite"):
        median_nn_distance(coords)


@pytest.mark.parametrize("bad_shape", [
    np.zeros((5,)),          # 1D
    np.zeros((5, 3)),        # (N, 3)
    np.zeros((2, 2, 2)),     # 3D
])
def test_malformed_shape_is_rejected(bad_shape):
    with pytest.raises(ValueError, match="shape"):
        median_nn_distance(bad_shape)
