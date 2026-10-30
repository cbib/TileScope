"""
Spatial autocorrelation statistics for consensus hexagonal grids.

Provides bivariate Moran's I for directional spatial correlation analysis.
"""

from enum import Enum
import hashlib
import numpy as np
from typing import MutableMapping, Optional, Tuple
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix
from scipy.spatial import cKDTree


class UndefinedMoranReason(str, Enum):
    """Structured reason for an undefined bivariate Moran result."""
    ZERO_VARIANCE_X = "zero_variance_x"
    ZERO_VARIANCE_Y = "zero_variance_y"
    BOTH_ZERO_VARIANCE = "both_zero_variance"
    INSUFFICIENT_WEIGHTS = "insufficient_weights"
    # Distinct from INSUFFICIENT_WEIGHTS: fires when a canonical region
    # contributes exactly one hex under region-blocked adjacency.
    INSUFFICIENT_WEIGHTS_IN_REGION = "insufficient_weights_in_region"


class UndefinedMoranError(ValueError):
    """Raised when bivariate Moran's I is mathematically or topologically undefined.

    Subclasses ``ValueError``; the ``reason`` attribute carries a
    :class:`UndefinedMoranReason` distinguishing which condition fired.
    """

    def __init__(self, message: str, reason: UndefinedMoranReason):
        super().__init__(message)
        self.reason = reason


def build_knn_adjacency(xy: np.ndarray, k: int = 6) -> csr_matrix:
    """Build row-standardized k-NN spatial weights matrix.

    Args:
        xy: (n, 2) array of hexagon centroids
        k: Number of nearest neighbors (6 for hex grid)

    Returns:
        W: sparse (n, n) row-standardized mutual-kNN weights. The binary
            neighbourhood is symmetric before row scaling; unequal node
            degrees generally make the returned matrix asymmetric.
    """
    n = xy.shape[0]
    if n < k + 1:
        return csr_matrix((n, n), dtype=np.float64)

    tree = cKDTree(xy)
    # Query k+1 because first neighbor is self.
    distances, indices = tree.query(xy, k=k+1)

    # Skip first column (self) when building the binary adjacency.
    rows = np.repeat(np.arange(n, dtype=int), k)
    cols = np.asarray(indices[:, 1:], dtype=int).reshape(-1)
    W = csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(n, n),
    )

    # Symmetrize to mutual neighbors only, then row-standardize.
    W = W.minimum(W.T).tocsr()
    W.eliminate_zeros()

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums, dtype=float)
    positive = row_sums > 0
    inverse[positive] = 1.0 / row_sums[positive]
    return (sparse.diags(inverse) @ W).tocsr()


def build_region_blocked_knn_adjacency(
    xy: np.ndarray,
    region_ids: np.ndarray,
    k: int = 6,
    cache: Optional[MutableMapping[tuple, csr_matrix]] = None,
) -> csr_matrix:
    """Build a block-diagonal row-standardized kNN adjacency by region.

    Region ids are treated categorically. Cross-region entries are exactly zero;
    row order of ``xy`` is preserved so callers can compose ``W`` with
    per-row value vectors indexed in the same order.
    """
    xy = np.asarray(xy, dtype=float)
    region_ids = np.asarray(region_ids)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"xy must have shape (n, 2); got {xy.shape!r}"
        )
    if region_ids.ndim != 1:
        raise ValueError(
            f"region_ids must be 1-D; got shape {region_ids.shape!r}"
        )
    if region_ids.shape[0] != xy.shape[0]:
        raise ValueError(
            f"xy length ({xy.shape[0]}) and region_ids length "
            f"({region_ids.shape[0]}) must match"
        )
    if not np.issubdtype(region_ids.dtype, np.integer):
        raise ValueError(
            f"region_ids must be an integer dtype; got {region_ids.dtype!r}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    cache_key = None
    if cache is not None:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(xy).tobytes())
        digest.update(np.ascontiguousarray(region_ids).tobytes())
        cache_key = ("region_blocked_knn", int(k), xy.shape, digest.digest())
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    n = xy.shape[0]
    rows = []
    cols = []
    data = []
    for region_id in np.unique(region_ids):
        row_idx = np.where(region_ids == region_id)[0]
        n_region = row_idx.size
        if n_region == 1:
            raise UndefinedMoranError(
                f"Region {int(region_id)} contributes exactly one hex; "
                f"cannot form a within-region kNN edge.",
                reason=UndefinedMoranReason.INSUFFICIENT_WEIGHTS_IN_REGION,
            )
        k_region = min(k, n_region - 1)
        block = build_knn_adjacency(xy[row_idx], k=k_region)
        block_coo = block.tocoo()
        rows.extend(row_idx[block_coo.row].tolist())
        cols.extend(row_idx[block_coo.col].tolist())
        data.extend(block_coo.data.tolist())
    weights = csr_matrix(
        (data, (rows, cols)), shape=(n, n), dtype=np.float64,
    )
    if cache is not None and cache_key is not None:
        cache[cache_key] = weights
    return weights


def bivariate_morans_i(x: np.ndarray,
                       y: np.ndarray,
                       W: np.ndarray | spmatrix,
                       permutations: int = 499,
                       seed: int = 42) -> Tuple[float, float]:
    """Compute directional bivariate Moran's I: I(x <- y).

    Canonical bivariate Moran (GeoDa / PySAL ``Moran_BV``): measures spatial
    autocorrelation of x with the spatial lag of y. Permutation null holds x
    fixed and permutes y (PySAL's canonical directional null).

    Raises:
        UndefinedMoranError: if either operand has exactly zero variance
            (``np.ptp == 0``) or ``W`` has effectively no mass
            (``W.sum() < 1e-10``).
    """
    n = len(x)

    # Exact-constant detection via peak-to-peak: np.var can return ~1e-16 for
    # IEEE-754 non-representable constants like np.full(N, 3.14).
    x_constant = np.ptp(x) == 0
    y_constant = np.ptp(y) == 0
    if x_constant and y_constant:
        raise UndefinedMoranError(
            f"Both operands are exactly constant "
            f"(x = {x[0]:.6f}, y = {y[0]:.6f}, n = {n}); "
            "bivariate Moran's I is undefined.",
            reason=UndefinedMoranReason.BOTH_ZERO_VARIANCE,
        )
    if x_constant:
        raise UndefinedMoranError(
            f"Operand x is exactly constant "
            f"(value = {x[0]:.6f}, n = {n}); "
            "bivariate Moran's I is undefined.",
            reason=UndefinedMoranReason.ZERO_VARIANCE_X,
        )
    if y_constant:
        raise UndefinedMoranError(
            f"Operand y is exactly constant "
            f"(value = {y[0]:.6f}, n = {n}); "
            "bivariate Moran's I is undefined.",
            reason=UndefinedMoranReason.ZERO_VARIANCE_Y,
        )

    # Fail-explicit on an all-zero W rather than manufacture a value.
    S0 = float(W.sum())
    if S0 < 1e-10:
        raise UndefinedMoranError(
            f"Spatial weights matrix has effectively zero mass "
            f"(S0 = {S0!r}, n = {n}); bivariate Moran's I is undefined.",
            reason=UndefinedMoranReason.INSUFFICIENT_WEIGHTS,
        )

    x_std = (x - x.mean()) / x.std(ddof=1)
    y_std = (y - y.mean()) / y.std(ddof=1)

    # Explicit sum-of-squares denominator stays correct under future
    # standardization choices (equals n-1 for sample-std).
    x_denom = float(np.sum(x_std ** 2))

    y_lag = W @ y_std
    # Under row-standardized W with no isolate rows, (n / S0) == 1.
    I_obs = (n / S0) * float(np.sum(x_std * y_lag)) / x_denom

    if permutations == 0:
        return I_obs, 1.0

    rng = np.random.default_rng(seed)
    I_perm = np.zeros(permutations, dtype=np.float64)
    for p in range(permutations):
        y_perm = rng.permutation(y_std)
        y_lag_perm = W @ y_perm
        I_perm[p] = (n / S0) * float(np.sum(x_std * y_lag_perm)) / x_denom

    # Two-sided bounded estimator: p = (b + 1) / (m + 1).
    count = np.sum(np.abs(I_perm) >= np.abs(I_obs))
    p_value = (count + 1) / (permutations + 1)

    return I_obs, p_value


__all__ = [
    "build_knn_adjacency",
    "build_region_blocked_knn_adjacency",
    "bivariate_morans_i",
    "UndefinedMoranError",
    "UndefinedMoranReason",
]
