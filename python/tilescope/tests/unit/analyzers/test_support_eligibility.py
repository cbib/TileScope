"""Sample-support classifier + four-class status derivation.

Focused contract tests:

- `classify_sample_support`: deduplicates coordinates for scale; row
  order in input does not affect per-ROI component sizes; empty and
  degenerate inputs handled.

- `derive_roi_support_status`: exercises all four classes exactly;
  input validation on `min_component_spots`; empty-input schema
  stability.

Does not attempt to characterise numerical properties of specific
sample-support radii — the algorithm's scale definition is
covered by the enumeration on production data (Greenwald +
mouse smoke), not synthetic fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.support_eligibility import (
    _STATUS_COLUMNS,
    classify_sample_support,
    derive_roi_support_status,
)
from tilescope.cols import COORD_X, COORD_Y, ROI_ID


def _points(coords):
    xs, ys = zip(*coords)
    return pd.DataFrame({COORD_X: list(xs), COORD_Y: list(ys)})


def _labelled(labels, coords) -> pd.DataFrame:
    df = _points(coords)
    df[ROI_ID] = list(labels)
    return df


# ---------------------------------------------------------------------
# classify_sample_support
# ---------------------------------------------------------------------

def test_classify_empty_input_returns_empty_dict():
    sizes, median_nn = classify_sample_support(pd.DataFrame({COORD_X: [], COORD_Y: [], ROI_ID: []}))
    assert sizes == {}
    assert np.isnan(median_nn)


def test_classify_rejects_all_duplicate_positions():
    df = _labelled(["A", "A"], [(0.0, 0.0), (0.0, 0.0)])
    with pytest.raises(ValueError, match="fewer than two unique"):
        classify_sample_support(df)


def test_classify_rejects_nonpositive_factor():
    df = _labelled(["A", "A"], [(0.0, 0.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="positive and finite"):
        classify_sample_support(df, factor=0.0)


@pytest.mark.parametrize("coords,labels", [
    pytest.param([], [], id="empty"),
    pytest.param([(0.0, 0.0)], ["A"], id="singleton"),
])
def test_classify_rejects_nonpositive_factor_on_degenerate_shapes(coords, labels):
    df = _labelled(labels, coords) if coords else pd.DataFrame(
        {COORD_X: [], COORD_Y: [], ROI_ID: []},
    )
    with pytest.raises(ValueError, match="positive and finite"):
        classify_sample_support(df, factor=0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        classify_sample_support(df, factor=-1.0)


def test_classify_row_order_invariance():
    """Shuffling input rows must yield the same per-ROI component sizes."""
    coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (10.0, 0.0), (11.0, 0.0)]
    labels = ["A", "A", "A", "A", "A"]
    df = _labelled(labels, coords)
    sizes_a, _ = classify_sample_support(df, factor=1.5)

    perm = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    sizes_b, _ = classify_sample_support(perm, factor=1.5)

    assert sizes_a == sizes_b  # both keyed by ROI_ID, values sorted DESC


def test_classify_duplicate_coordinates_dont_collapse_scale():
    """Duplicate spots share edges (distance 0) but do NOT shrink the
    median-NN scale toward zero — the scale is derived from unique
    positions only."""
    # 10 spots at (0,0) as duplicates + 3 spots on a unit line.
    coords = [(0.0, 0.0)] * 10 + [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    labels = ["A"] * 13
    df = _labelled(labels, coords)
    sizes, median_nn = classify_sample_support(df, factor=1.5)
    # Unique NN distance on the four unique positions {(0,0),(1,0),(2,0),(3,0)} is 1.0.
    assert median_nn == pytest.approx(1.0)
    # All 13 spots connect via the fixed-radius graph at 1.5 → one component.
    assert sizes == {"A": [13]}


# ---------------------------------------------------------------------
# derive_roi_support_status — four-class partition
# ---------------------------------------------------------------------

@pytest.mark.parametrize("sizes, expected_class", [
    pytest.param([12, 8, 3], "scattered", id="scattered_all_below"),
    pytest.param([15, 10, 5], "mixed", id="mixed_qualified_and_subthreshold"),
    pytest.param([25], "fully_connected", id="fully_connected_single_qualified"),
    pytest.param([20, 18, 16], "fully_supported_multicomponent", id="fully_supported_multicomponent"),
])
def test_derive_status_four_class_partition(sizes, expected_class):
    df = derive_roi_support_status({"R": sizes}, min_component_spots=15)
    assert len(df) == 1
    assert df.iloc[0]["spatial_support_class"] == expected_class


def test_derive_status_rejects_zero_min():
    with pytest.raises(ValueError, match="≥ 1"):
        derive_roi_support_status({"R": [5]}, min_component_spots=0)


def test_derive_status_empty_input_stable_schema():
    df = derive_roi_support_status({}, min_component_spots=15)
    assert list(df.columns) == list(_STATUS_COLUMNS)
    assert len(df) == 0


def test_derive_status_column_values_on_mixed_roi():
    """One mixed ROI: 30 spots, sizes [20, 6, 4] → qualified only the 20."""
    df = derive_roi_support_status({"R": [20, 6, 4]}, min_component_spots=15)
    row = df.iloc[0]
    assert row["source_spots"] == 30
    assert row["support_component_count"] == 3
    assert row["support_qualified_component_count"] == 1
    assert row["support_qualified_spots"] == 20
    assert row["support_qualified_fraction"] == pytest.approx(20 / 30)
    assert row["largest_support_component_spots"] == 20
    assert row["largest_support_component_fraction"] == pytest.approx(20 / 30)
    assert row["spatial_support_class"] == "mixed"
