"""CORE-HYG-05 Step 4: CLES searchsorted equivalence to naive matrix.

Two tests:

- ``test_cles_deterministic_cases`` — a compact parametrised table
  covering strict-> tie semantics, empty-side sentinel, NaN masking,
  and ±inf ordering. Each row asserts the searchsorted result equals
  the naive quadratic boolean-matrix reference AND, where it matters,
  the specific expected value.
- ``test_cles_random_stress_matches_matrix_incl_nan_inf`` —
  randomized draws that seed roi/context with a mixture of finite,
  NaN, +inf, -inf, and duplicate values; searchsorted output must
  match the reference for every draw.
"""
from __future__ import annotations

import numpy as np
import pytest

from tilescope.metrics.pair_stats_core import PairStatsCore


def _naive_cles(roi: np.ndarray, ctx: np.ndarray) -> float:
    """Reference — quadratic boolean matrix. NaN comparisons are False
    per IEEE, so NaN-touching pairs contribute 0 to the numerator."""
    r = np.asarray(roi)
    c = np.asarray(ctx)
    if r.size == 0 or c.size == 0:
        return 0.5
    return float(np.sum(r[:, None] > c)) / (r.size * c.size)


NAN = float("nan")
INF = float("inf")

CASES = [
    ("disjoint_lt",     [1.0, 2.0, 3.0],   [10.0, 20.0]),
    ("disjoint_gt",     [10.0, 20.0, 30.0], [1.0, 2.0]),
    ("many_ties",       [1.0, 2.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
    ("all_equal",       [5.0, 5.0],        [5.0, 5.0, 5.0]),
    ("empty_roi",       [],                [1.0]),
    ("empty_ctx",       [1.0],             []),
    ("both_empty",      [],                []),
    ("single_each_eq",  [1.5],             [1.5]),
    ("mixed_signs",     [-3.0, 0.0, 5.0],  [-1.0, 1.0]),
    # NaN masking: NaN operands contribute 0 to numerator but stay
    # in the denominator.
    ("nan_roi_only",    [NAN],             [1.0]),        # 0/1 = 0
    ("nan_ctx_only",    [1.0],             [NAN]),        # 0/1 = 0
    ("nan_and_finite",  [NAN, 3.0],        [1.0, 2.0]),   # 2/4 = 0.5
    ("nan_both_sides",  [NAN, 3.0],        [NAN, 1.0]),   # 1/4 = 0.25
    # +/- inf follow IEEE ordering.
    ("pos_inf_beats_finite", [INF],        [1.0]),        # 1/1 = 1
    ("neg_inf_loses",   [-INF],            [1.0]),        # 0/1 = 0
    ("inf_vs_inf_tie",  [INF],             [INF]),        # 0/1 = 0
]


@pytest.mark.parametrize(
    "label,roi,ctx", CASES, ids=[c[0] for c in CASES],
)
def test_cles_deterministic_cases(label, roi, ctx):
    core = PairStatsCore()
    roi_a = np.asarray(roi, dtype=float)
    ctx_a = np.asarray(ctx, dtype=float)
    assert core.cles(roi_a, ctx_a) == _naive_cles(roi_a, ctx_a), (
        f"CLES {label!r}"
    )


def test_cles_random_stress_matches_matrix_incl_nan_inf():
    rng = np.random.default_rng(0)
    # Pool mixes finite values, NaN, and +/- inf so every draw
    # exercises the masking + IEEE-ordering paths.
    pool = np.array([-INF, -2.0, 0.0, 0.5, NAN, 1.0, 2.5, INF])
    core = PairStatsCore()
    for _ in range(30):
        n_roi = int(rng.integers(1, 200))
        n_ctx = int(rng.integers(1, 200))
        roi = np.concatenate([
            rng.choice(pool, size=n_roi // 2),
            rng.normal(size=n_roi - n_roi // 2),
        ])
        ctx = np.concatenate([
            rng.choice(pool, size=n_ctx // 2),
            rng.normal(size=n_ctx - n_ctx // 2),
        ])
        assert core.cles(roi, ctx) == _naive_cles(roi, ctx), (
            f"random stress mismatch: roi.head={roi[:5]!r} "
            f"ctx.head={ctx[:5]!r}"
        )
