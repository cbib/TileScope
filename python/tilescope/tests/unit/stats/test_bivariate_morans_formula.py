"""Deterministic formula pins for `bivariate_morans_i`.

Guards the canonical bivariate Moran definition used by GeoDa /
PySAL `Moran_BV`:

                   n     Σᵢ Σⱼ wᵢⱼ zₓᵢ z_yⱼ
    I(x ← y)  =  ─── · ────────────────────
                   S₀        Σᵢ zₓᵢ²

Permutation null: hold ``x`` fixed, permute ``y``, recompute the
spatial lag, preserve the ``x`` denominator.

Regression against a prior implementation that omitted the
``Σ zₓᵢ²`` denominator and permuted ``x`` instead of ``y``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import block_diag

from tilescope.stats.spatial_autocorr import (
    bivariate_morans_i,
    build_knn_adjacency,
)


def test_fixed_input_matches_hand_computed_I():
    """Canonical formula on a closed-form n=4 example.

    Row-standardized path adjacency:

        W = [[0, 1,   0,   0],
             [1/2, 0, 1/2, 0],
             [0, 1/2, 0, 1/2],
             [0, 0,   1,   0]]

    with S₀ = 4 (every row sums to 1). ``x = [1,2,3,4]``,
    ``y = [4,3,2,1]``. Working out the numerator and denominator
    by hand (see docstring of `bivariate_morans_i`) gives
    ``I = -2/5 = -0.4`` exactly.
    """
    W = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])

    I_actual, _ = bivariate_morans_i(x, y, W, permutations=0)
    assert I_actual == pytest.approx(-0.4)


def _univariate_morans_i(z_std: np.ndarray, W: np.ndarray) -> float:
    n = len(z_std)
    S0 = W.sum()
    numerator = float(z_std @ (W @ z_std))
    denom = float(np.sum(z_std ** 2))
    return (n / S0) * numerator / denom


def test_bivariate_reduces_to_univariate_when_y_equals_x():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(30, 2))
    W = build_knn_adjacency(xy, k=6)
    if W.sum() == 0.0:
        pytest.skip("Weights degenerate under test fixture geometry")
    x = rng.normal(size=30)
    I_bv, _ = bivariate_morans_i(x, x, W, permutations=0)
    x_std = (x - x.mean()) / x.std(ddof=1)
    I_uv = _univariate_morans_i(x_std, W)
    assert I_bv == pytest.approx(I_uv, rel=1e-12)


def test_permutation_holds_x_fixed_and_permutes_y():
    """Reproduce the primitive's permutation loop step-for-step.

    Permutes ``y`` (not ``x``), recomputes the spatial lag from the
    permuted ``y``, evaluates against the ``x`` denominator.
    """
    n = 20
    rng_data = np.random.default_rng(1)
    xy = rng_data.uniform(0, 10, size=(n, 2))
    W = build_knn_adjacency(xy, k=4)
    if W.sum() == 0.0:
        pytest.skip("Weights degenerate under test fixture geometry")
    x = rng_data.normal(size=n)
    y = rng_data.normal(size=n)

    x_std = (x - x.mean()) / x.std(ddof=1)
    y_std = (y - y.mean()) / y.std(ddof=1)
    x_denom = float(np.sum(x_std ** 2))
    S0 = W.sum()

    n_perm = 25
    seed = 7
    rng_ref = np.random.default_rng(seed)
    ref_I = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = rng_ref.permutation(y_std)
        y_lag_perm = W @ y_perm
        ref_I[i] = (n / S0) * float(np.sum(x_std * y_lag_perm)) / x_denom
    ref_I_obs = (n / S0) * float(np.sum(x_std * (W @ y_std))) / x_denom
    ref_p = (int(np.sum(np.abs(ref_I) >= np.abs(ref_I_obs))) + 1) / (n_perm + 1)

    I_obs, p = bivariate_morans_i(x, y, W, permutations=n_perm, seed=seed)
    assert I_obs == pytest.approx(ref_I_obs)
    assert p == pytest.approx(ref_p)


def test_exact_duplication_leaves_I_unchanged():
    """Regression against the pre-fix bug that omitted the ``Σ zₓ²``
    denominator (so I scaled with n − 1).

    Under canonical normalization, replicating a system (block-
    diagonal ``W ⊕ W`` with ``x``/``y`` concatenated to themselves)
    is a mathematically equivalent system and must yield the
    identical I. The pre-fix code inflated the duplicate by
    approximately the size ratio.
    """
    rng = np.random.default_rng(42)
    n = 12
    xy = rng.uniform(0, 10, size=(n, 2))
    W = build_knn_adjacency(xy, k=4)
    if W.sum() == 0.0:
        pytest.skip("Weights degenerate under test fixture geometry")
    x = rng.normal(size=n)
    y = 0.6 * x + 0.4 * rng.normal(size=n)

    W_dup = block_diag((W, W), format="csr")
    x_dup = np.concatenate([x, x])
    y_dup = np.concatenate([y, y])

    I_single, _ = bivariate_morans_i(x, y, W, permutations=0)
    I_dup, _ = bivariate_morans_i(x_dup, y_dup, W_dup, permutations=0)

    assert I_dup == pytest.approx(I_single, rel=1e-12)
