"""Semantic pin for the standard-v1 seed-count formula.

Locks the scale-faithful rule `n_seeds = max(1, round(fraction * n_cells))`
that replaced the legacy `max(5, ceil(fraction * n_cells))` floor. The
value 5 in the legacy rule had no scientific role beyond over-seeding
small regions; the new rule tracks region size directly.

Structure: covers the small-region cases (where the two rules diverge
most dramatically), the boundary at the old 5-seed pin, and the large-
region regime (where both rules agree).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tilescope.tessellation.voronoi import select_seeds


def _n_unique_seeds(coords: np.ndarray, fraction: float, seed: int) -> int:
    rng = np.random.default_rng(seed)
    seeds = select_seeds(coords, fraction=fraction, rng=rng)
    return seeds.shape[0]


@pytest.mark.parametrize(
    "n_spots, fraction, expected",
    [
        # Small regions where legacy would have pinned n_seeds = 5.
        # Scale-faithful returns max(1, round(f * n)):
        (3,  0.20, 1),   # round(0.6)  = 1
        (4,  0.20, 1),   # round(0.8)  = 1
        (5,  0.20, 1),   # round(1.0)  = 1
        (6,  0.20, 1),   # round(1.2)  = 1
        (7,  0.20, 1),   # round(1.4)  = 1
        (8,  0.20, 2),   # round(1.6)  = 2
        (10, 0.20, 2),   # round(2.0)  = 2
        (12, 0.20, 2),   # round(2.4)  = 2
        (13, 0.20, 3),   # round(2.6)  = 3
        # Boundary at old five-seed floor: n * f = 5 exactly at n=25.
        (24, 0.20, 5),   # round(4.8)  = 5
        (25, 0.20, 5),   # round(5.0)  = 5
        (26, 0.20, 5),   # round(5.2)  = 5
        # Large-region regime — both rules would agree.
        (100, 0.20, 20),
        (500, 0.20, 100),
        # Single-spot region: floor of 1, not 5.
        (1, 0.20, 1),
    ],
)
def test_select_seeds_scale_faithful_count(n_spots: int, fraction: float, expected: int):
    """`n_seeds = max(1, round(fraction * n_unique_coords))` — no 5-floor."""
    coords = np.stack([np.arange(n_spots, dtype=float), np.zeros(n_spots)], axis=1)
    # jitter should not change unique-coord count for these integer x-values
    assert _n_unique_seeds(coords, fraction, seed=0) == expected


def test_select_seeds_never_returns_five_by_default_on_tiny_regions():
    """Regression: legacy `max(5, ceil(...))` would pin n_seeds=5 for any
    region with `ceil(fraction * n) < 5`; scale-faithful must not."""
    for n_spots in range(1, 8):
        coords = np.stack(
            [np.arange(n_spots, dtype=float), np.zeros(n_spots)], axis=1,
        )
        n_seeds = _n_unique_seeds(coords, fraction=0.20, seed=0)
        # At fraction=0.20 and n_spots ∈ [1, 7]: round(fraction * n) ∈ {0, 1};
        # max(1, ...) → 1. Legacy would have returned 5 (or min(5, n)).
        assert n_seeds == 1, (
            f"n_spots={n_spots}: expected 1 seed under scale-faithful, "
            f"got {n_seeds}"
        )


def test_select_seeds_empty_input_returns_empty():
    coords = np.zeros((0, 2), dtype=float)
    assert _n_unique_seeds(coords, fraction=0.20, seed=0) == 0


def test_select_seeds_is_deterministic_under_same_seed():
    coords = np.stack([np.arange(30, dtype=float), np.zeros(30)], axis=1)
    a = select_seeds(coords, fraction=0.20, rng=np.random.default_rng(42))
    b = select_seeds(coords, fraction=0.20, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_select_seeds_seed_count_matches_formula_across_fractions():
    """Sanity: for randomly-generated coords, count must equal the formula
    for a range of fractions."""
    rng = np.random.default_rng(2026)
    coords = rng.uniform(0, 100, size=(200, 2))
    n_unique = np.unique(coords, axis=0).shape[0]
    assert n_unique == 200  # extremely likely for uniform floats
    for fraction in (0.05, 0.10, 0.20, 0.33, 0.50):
        expected = max(1, int(round(fraction * n_unique)))
        got = _n_unique_seeds(coords, fraction=fraction, seed=1)
        assert got == expected, (
            f"fraction={fraction}: expected {expected}, got {got}"
        )
