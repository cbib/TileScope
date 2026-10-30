"""Contract pins for the bivariate-Moran typed-undefined + region-blocked
adjacency contract.

The Moran path represents undefined outcomes structurally: constant
operands, zero-mass weights, and — under region-blocked adjacency —
singleton canonical regions each raise :class:`UndefinedMoranError`
with a distinct :class:`UndefinedMoranReason`. Callers translate to
NaN at the emission boundary; no numeric fallback (0.0, 1.0) is
manufactured.

Region-blocked adjacency: hex adjacency is built per canonical
``domain_region_id`` block. Off-block entries are exactly zero;
per-block ``k_region = min(k, n_region - 1)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tilescope.stats.spatial_autocorr import (
    bivariate_morans_i,
    build_knn_adjacency,
    build_region_blocked_knn_adjacency,
    UndefinedMoranError,
    UndefinedMoranReason,
)


# ---------------------------------------------------------------------------
# 1. Validation + shape errors on the region-blocked primitive.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "xy, region_ids, k, msg",
    [
        (np.zeros((5, 3)), np.zeros(5, dtype=int), 2, "shape"),
        (np.zeros((5, 2)), np.zeros((5, 1), dtype=int), 2, "1-D"),
        (np.zeros((5, 2)), np.zeros(4, dtype=int), 2, "match"),
        (np.zeros((5, 2)), np.zeros(5, dtype=float), 2, "integer"),
        (np.zeros((5, 2)), np.zeros(5, dtype=int), 0, ">= 1"),
    ],
    ids=["bad_xy_shape", "region_ids_2d", "length_mismatch",
         "non_integer_region_ids", "k_zero"],
)
def test_region_blocked_adjacency_validation(xy, region_ids, k, msg):
    with pytest.raises(ValueError, match=msg):
        build_region_blocked_knn_adjacency(xy, region_ids, k=k)


# ---------------------------------------------------------------------------
# 2. Single-region equivalence with the underlying block primitive.
# ---------------------------------------------------------------------------


def test_region_blocked_matches_unblocked_when_single_region():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(30, 2))
    region_ids = np.zeros(30, dtype=np.int64)
    cache = {}
    W_blocked = build_region_blocked_knn_adjacency(
        xy, region_ids, k=6, cache=cache,
    )
    assert build_region_blocked_knn_adjacency(
        xy.copy(), region_ids.copy(), k=6, cache=cache,
    ) is W_blocked
    W_direct = build_knn_adjacency(xy, k=6)
    np.testing.assert_allclose(W_blocked.toarray(), W_direct.toarray())


# ---------------------------------------------------------------------------
# 3. Cross-region entries exactly zero; nonzero rows sum to one.
# ---------------------------------------------------------------------------


def test_region_blocked_off_block_entries_are_zero_rowsum_one():
    rng = np.random.default_rng(1)
    # Two disjoint clusters, well separated so cross-region kNN edges
    # would exist if the primitive were unblocked.
    xy_a = rng.uniform(0, 5, size=(20, 2))
    xy_b = rng.uniform(100, 105, size=(20, 2))
    xy = np.vstack([xy_a, xy_b])
    region_ids = np.concatenate([
        np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64),
    ])
    W = build_region_blocked_knn_adjacency(xy, region_ids, k=6)
    # Off-block quadrants strictly zero.
    assert W[:20, 20:].nnz == 0
    assert W[20:, :20].nnz == 0
    # Every non-zero row is row-standardized (sums to 1).
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    np.testing.assert_allclose(row_sums[nonzero_rows], 1.0)


# ---------------------------------------------------------------------------
# 4. Joint row-permutation covariance of the adjacency.
# ---------------------------------------------------------------------------


def test_region_blocked_row_permutation_covariance():
    """Permuting rows of (xy, region_ids) permutes W by the same order."""
    rng = np.random.default_rng(2)
    xy_a = rng.uniform(0, 5, size=(15, 2))
    xy_b = rng.uniform(100, 105, size=(15, 2))
    xy = np.vstack([xy_a, xy_b])
    region_ids = np.concatenate([
        np.zeros(15, dtype=np.int64), np.ones(15, dtype=np.int64),
    ])
    W_orig = build_region_blocked_knn_adjacency(xy, region_ids, k=4)
    perm = np.random.default_rng(3).permutation(len(xy))
    W_perm = build_region_blocked_knn_adjacency(
        xy[perm], region_ids[perm], k=4,
    )
    # Undo the permutation on W_perm and compare to the original.
    inv = np.argsort(perm)
    W_undone = W_perm[inv][:, inv]
    np.testing.assert_allclose(W_undone.toarray(), W_orig.toarray())


# ---------------------------------------------------------------------------
# 5. Adaptive k_region + singleton-region undefined reason.
# ---------------------------------------------------------------------------


def test_adaptive_k_region_under_small_region_and_singleton_undefined():
    rng = np.random.default_rng(4)
    # Region A: 15 hexes (uses full k=6). Region B: 3 hexes (k_region=2).
    xy_a = rng.uniform(0, 5, size=(15, 2))
    xy_b = rng.uniform(100, 105, size=(3, 2))
    xy = np.vstack([xy_a, xy_b])
    region_ids = np.concatenate([
        np.zeros(15, dtype=np.int64), np.full(3, 7, dtype=np.int64),
    ])
    W = build_region_blocked_knn_adjacency(xy, region_ids, k=6)
    # Region B block: 3 hexes, k_region=2 → each row has degree 2 after
    # symmetrization + row-standardisation → still sums to 1.
    np.testing.assert_allclose(
        np.asarray(W[15:, 15:].sum(axis=1)).ravel(), 1.0,
    )
    # Cross-block still exactly zero.
    assert W[:15, 15:].nnz == 0

    # Singleton region → UndefinedMoranError with the new reason.
    xy_c = np.vstack([rng.uniform(0, 5, size=(10, 2)), [[200.0, 200.0]]])
    region_ids_c = np.concatenate([
        np.zeros(10, dtype=np.int64), np.array([42], dtype=np.int64),
    ])
    with pytest.raises(UndefinedMoranError) as exc:
        build_region_blocked_knn_adjacency(xy_c, region_ids_c, k=6)
    assert exc.value.reason is UndefinedMoranReason.INSUFFICIENT_WEIGHTS_IN_REGION

# ---------------------------------------------------------------------------
# Retained undefined-status contract on the low-level primitive
# (variance + zero-weights) — unchanged by region blocking.
# ---------------------------------------------------------------------------


def _hex_grid_weights(n: int = 20, k: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(n, 2))
    return build_knn_adjacency(xy, k=k)


@pytest.mark.parametrize(
    "make_x, make_y, expected_reason",
    [
        (lambda r, n: np.full(n, 3.14),
         lambda r, n: np.full(n, -1.0),
         UndefinedMoranReason.BOTH_ZERO_VARIANCE),
        (lambda r, n: np.zeros(n),
         lambda r, n: r.normal(size=n),
         UndefinedMoranReason.ZERO_VARIANCE_X),
        (lambda r, n: r.normal(size=n),
         lambda r, n: np.full(n, 42.0),
         UndefinedMoranReason.ZERO_VARIANCE_Y),
    ],
    ids=["both_constant", "x_constant", "y_constant"],
)
def test_bivariate_morans_i_raises_structured_reason_on_constant(
    make_x, make_y, expected_reason,
):
    rng = np.random.default_rng(0)
    n = 20
    W = _hex_grid_weights(n=n)
    x = make_x(rng, n)
    y = make_y(rng, n)
    with pytest.raises(UndefinedMoranError) as exc_info:
        bivariate_morans_i(x, y, W, permutations=0)
    assert exc_info.value.reason is expected_reason


def test_bivariate_morans_i_raises_on_degenerate_weights():
    n = 20
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    W = np.zeros((n, n), dtype=np.float64)
    with pytest.raises(UndefinedMoranError) as exc_info:
        bivariate_morans_i(x, y, W, permutations=0)
    assert exc_info.value.reason is UndefinedMoranReason.INSUFFICIENT_WEIGHTS


def test_bivariate_morans_i_finite_path_still_works():
    n = 50
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = 0.5 * x + 0.5 * rng.normal(size=n)
    W = _hex_grid_weights(n=n)
    if W.sum() == 0.0:
        pytest.skip("Weights degenerate under test-fixture geometry")
    I, p = bivariate_morans_i(x, y, W, permutations=50, seed=0)
    assert np.isfinite(I)
    assert 0.0 <= p <= 1.0
