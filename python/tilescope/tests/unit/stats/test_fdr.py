"""Unit tests for :func:`tilescope.stats.fdr.benjamini_hochberg`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.stats.fdr import apply_bh_within_families, benjamini_hochberg


def test_bh_hand_computed_family_of_four():
    """For p = [0.01, 0.02, 0.04, 0.5] with n = 4, BH gives
    q = [min(0.04, 0.04, 0.0533, 0.5), min(0.04, 0.0533, 0.5),
         min(0.0533, 0.5), 0.5] = [0.04, 0.04, 0.0533..., 0.5]."""
    p = np.array([0.01, 0.02, 0.04, 0.5])
    q = benjamini_hochberg(p)
    assert q[0] == pytest.approx(0.04)
    assert q[1] == pytest.approx(0.04)
    assert q[2] == pytest.approx(0.0533333, rel=1e-4)
    assert q[3] == pytest.approx(0.5)


def test_bh_nan_pass_through_and_family_size():
    """NaN inputs pass through as NaN; BH runs over the finite subset
    only (family size excludes NaNs)."""
    p = np.array([np.nan, 0.01, np.nan, 0.02, 0.04, 0.5])
    q = benjamini_hochberg(p)
    assert np.isnan(q[0])
    assert np.isnan(q[2])
    # Finite subset in-place is [0.01, 0.02, 0.04, 0.5] with n = 4;
    # same q's as the hand-computed test.
    assert q[1] == pytest.approx(0.04)
    assert q[3] == pytest.approx(0.04)
    assert q[4] == pytest.approx(0.0533333, rel=1e-4)
    assert q[5] == pytest.approx(0.5)


def test_bh_all_nan_returns_all_nan():
    p = np.array([np.nan, np.nan, np.nan])
    q = benjamini_hochberg(p)
    assert np.all(np.isnan(q))


def test_bh_empty_returns_empty():
    q = benjamini_hochberg(np.array([]))
    assert q.shape == (0,)


def test_bh_rejects_non_1d_input():
    with pytest.raises(ValueError, match="requires 1D"):
        benjamini_hochberg(np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_bh_rejects_infinite_input():
    with pytest.raises(ValueError, match="non-finite non-NaN"):
        benjamini_hochberg(np.array([0.1, np.inf, 0.3]))
    with pytest.raises(ValueError, match="non-finite non-NaN"):
        benjamini_hochberg(np.array([0.1, -np.inf, 0.3]))


def test_bh_rejects_negative_input():
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        benjamini_hochberg(np.array([0.1, -0.01, 0.3]))


def test_bh_rejects_greater_than_one_input():
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        benjamini_hochberg(np.array([0.1, 1.5, 0.3]))


def test_bh_permits_zero_and_one_endpoints():
    q = benjamini_hochberg(np.array([0.0, 0.5, 1.0]))
    assert not np.isnan(q).any()


def test_grouped_bh_keeps_families_separate_and_preserves_order():
    frame = pd.DataFrame({
        "family": ["B", "A", "B", "A"],
        "p": [0.01, 0.2, 0.04, 0.4],
        "identity": [3, 1, 4, 2],
    })
    out = apply_bh_within_families(
        frame, family_keys=("family",), p_to_q_columns=(("p", "q"),),
        populated_mask=pd.Series(True, index=frame.index),
    )
    assert out["identity"].tolist() == [3, 1, 4, 2]
    assert out["fdr_family_size"].tolist() == [2, 2, 2, 2]
    np.testing.assert_allclose(out["q"], [0.02, 0.4, 0.04, 0.4])


def test_grouped_bh_excludes_unpopulated_rows():
    frame = pd.DataFrame({"family": ["A"] * 3, "p": [0.01, np.nan, 0.2]})
    out = apply_bh_within_families(
        frame, family_keys=("family",), p_to_q_columns=(("p", "q"),),
        populated_mask=pd.Series([True, False, True]),
    )
    assert out["fdr_family_size"].tolist() == [2, 2, 2]
    np.testing.assert_allclose(out.loc[[0, 2], "q"], [0.02, 0.2])
    assert pd.isna(out.loc[1, "q"])
