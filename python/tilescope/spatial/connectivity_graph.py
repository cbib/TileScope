"""
k-NN graph-based connectivity for scale-invariant component detection.

Replaces raster-based connectivity with a resolution-free approach that works
consistently across coordinate scales and types (grid vs world).
"""

from typing import Tuple, Literal, Optional, TYPE_CHECKING
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

if TYPE_CHECKING:
    from tilescope.spatial import Canvas

logger = logging.getLogger(__name__)


def adaptive_gamma(
    P: np.ndarray,
    canvas: Optional["Canvas"] = None,
    q_hi: float = 0.99,
    tail_frac: float = 0.01,
    cv_boost_low: float = 1.15,
    cv_boost_high: float = 1.25,
    cv_lo: float = 0.5,
    cv_hi: float = 0.7,
    gamma_min: float = 1.45,
    gamma_max: float = 4.5
) -> float:
    """Tail-aware adaptive gamma with robust quantile + CV-based boosting."""
    # Canvas-based: tail-aware adaptive gamma
    N = len(P)
    if N < 2:
        return gamma_min  # Fallback for small datasets

    # Compute 1-NN distances (local spacing radii)
    k_eff = min(2, N)  # Need at least 2 neighbors (self + 1-NN)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(P)
    d, _ = nn.kneighbors(P)
    radii = d[:, 1] if d.shape[1] > 1 else d[:, 0]

    # Compute base statistics
    r = radii.astype(float)
    m = float(np.median(r))

    if not np.isfinite(m) or m <= 0:
        return gamma_min

    # Quantile-based estimate
    q = float(np.quantile(r, q_hi))

    # Tail-mean fallback: average top k radii (robust to single outliers)
    k = max(3, int(np.ceil(tail_frac * r.size)))
    tail_mean = float(np.mean(np.partition(r, -k)[-k:]))

    # Base gamma: take larger of quantile and tail-mean ratios
    gamma = max(q / m, tail_mean / m)

    # CV-aware boost for highly heterogeneous spacing
    cv = float(np.std(r)) / (m + 1e-12)
    if cv >= cv_hi:
        gamma *= cv_boost_high  # e.g., 1.25 for CV≥0.7
    elif cv >= cv_lo:
        gamma *= cv_boost_low   # e.g., 1.15 for CV≥0.5

    # Clamp to safe range
    gamma_final = float(np.clip(gamma, gamma_min, gamma_max))

    # Debug output for high-variance cases
    if cv > 0.4:
        logger.debug(
            "[adaptive_gamma] median=%.3f, q%d=%.3f, tail_mean=%.3f, CV=%.2f, γ_raw=%.2f, γ_final=%.2f",
            m, int(q_hi*100), q, tail_mean, cv, gamma, gamma_final
        )

    return gamma_final


def knn_components(
    P: np.ndarray,
    k: int = 10,
    gamma: Optional[float] = None,
    threshold: Literal["local", "global"] = "local",
    symmetrize: Literal["mutual", "union"] = "mutual",
    canvas: Optional["Canvas"] = None,
) -> Tuple[np.ndarray, int]:
    """Find connected components using scale-aware k-NN graph."""
    # Adaptive gamma selection
    if gamma is None:
        gamma = adaptive_gamma(P, canvas)
    N = len(P)
    if N == 0:
        return np.array([], dtype=int), 0
    if N == 1:
        return np.array([0], dtype=int), 1

    # Compute k-NN (k_eff accounts for self-neighbor)
    k_eff = min(k, N - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(P)
    d, idx = nn.kneighbors(P)  # d[:,0]=0 (self), idx[:,0]=i

    # 1-NN distances (skip self) → local scale per spot
    d1 = d[:, 1]

    # Build edge list: (i → j) with distances
    I = np.repeat(np.arange(N), k_eff)
    J = idx[:, 1:].ravel()
    D = d[:, 1:].ravel()

    # Apply distance threshold
    if threshold == "global":
        thr = gamma * np.median(d1)
        keep = D <= thr
    else:  # local
        # dist(i,j) <= gamma * 0.5*(d1[i]+d1[j])
        thr_local = 0.5 * (d1[I] + d1[J])
        keep = D <= gamma * thr_local

    I, J, D = I[keep], J[keep], D[keep]

    # Symmetrize edges
    if symmetrize == "mutual":
        # Keep only pairs where both i→j and j→i exist
        key = I.astype(np.int64) << 32 | J.astype(np.int64)
        rev = J.astype(np.int64) << 32 | I.astype(np.int64)
        s = set(key.tolist())
        m = np.array([rv in s for rv in rev], dtype=bool)
        I, J = I[m], J[m]
    # union: keep all edges (no filtering)

    # Build adjacency matrix (unweighted boolean)
    A = coo_matrix((np.ones_like(I, dtype=np.uint8), (I, J)), shape=(N, N))
    # Ensure symmetry for connected_components
    A = (A + A.T).astype(bool).tocsr()

    # Find connected components
    n_cc, labels = connected_components(A, directed=False, return_labels=True)

    return labels.astype(int), int(n_cc)
