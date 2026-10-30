"""
Pure computational functions for correlation analysis.

Stateless mathematical functions that return dictionaries keyed by constants.
Only imports constants.py and scipy - no schema dependencies.
"""

import logging
from enum import Enum
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats

from tilescope.cols import *  # Column names
from tilescope.enums import CorrelationMode

logger = logging.getLogger(__name__)


class UndefinedCorrelationReason(str, Enum):
    """Structured reason for an undefined correlation, attached to
    ``UndefinedCorrelationError.reason``. Callers pattern-match on the
    enum instead of parsing exception messages.
    """
    GENE1_CONSTANT = "gene1_constant"
    GENE2_CONSTANT = "gene2_constant"
    BOTH_CONSTANT = "both_constant"


class UndefinedCorrelationError(ValueError):
    """Raised when correlation is mathematically undefined because at least
    one operand has exactly zero variance. Subclasses ValueError so existing
    ``except ValueError`` handlers continue to catch it; callers may match on
    the subclass for finer control. The ``reason`` attribute carries a
    :class:`UndefinedCorrelationReason` distinguishing which operand(s) were
    constant.
    """

    def __init__(self, message: str, reason: UndefinedCorrelationReason):
        super().__init__(message)
        self.reason = reason


def compute_correlation_from_arrays(g1_array: np.ndarray,
                                    g2_array: np.ndarray,
                                    mode: str = CorrelationMode.PEARSON.value,
                                    n_perms: int = 100,
                                    rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Pure correlation computation on two arrays (vectors).

    Raises:
        UndefinedCorrelationError: if either operand has exactly zero variance
            (constant array). Applies to both Pearson and Spearman. The
            observed correlation is mathematically undefined in these cases;
            no permutation test is run and no numeric fallback is manufactured.
        ValueError: for length mismatch or fewer than 3 observations.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(g1_array) != len(g2_array):
        raise ValueError(f"Arrays must have same length: {len(g1_array)} != {len(g2_array)}")

    if len(g1_array) < 3:
        raise ValueError(f"Need at least 3 observations for correlation, got {len(g1_array)}")

    # Fail-explicit on exact-constant operand. Applies to both Pearson and
    # Spearman: rank correlation is also undefined when one operand has zero
    # variance (all ranks tied). Never manufacture r=1 or r=0.
    #
    # Detection uses peak-to-peak (max - min): mathematically exact for
    # constant arrays under IEEE 754, unlike np.std which accumulates
    # rounding error and can return ~1e-16 for e.g. np.full(50, 3.14).
    g1_constant = np.ptp(g1_array) == 0
    g2_constant = np.ptp(g2_array) == 0

    if g1_constant and g2_constant:
        raise UndefinedCorrelationError(
            f"Both arrays are exactly constant "
            f"(gene1 = {g1_array[0]:.6f}, gene2 = {g2_array[0]:.6f}, "
            f"n = {len(g1_array)}); correlation is undefined.",
            reason=UndefinedCorrelationReason.BOTH_CONSTANT,
        )
    if g1_constant:
        raise UndefinedCorrelationError(
            f"Gene 1 is exactly constant "
            f"(value = {g1_array[0]:.6f}, n = {len(g1_array)}); "
            f"correlation is undefined.",
            reason=UndefinedCorrelationReason.GENE1_CONSTANT,
        )
    if g2_constant:
        raise UndefinedCorrelationError(
            f"Gene 2 is exactly constant "
            f"(value = {g2_array[0]:.6f}, n = {len(g2_array)}); "
            f"correlation is undefined.",
            reason=UndefinedCorrelationReason.GENE2_CONSTANT,
        )

    # Choose correlation method
    if mode == CorrelationMode.PEARSON.value:
        corr_func = stats.pearsonr
    elif mode == CorrelationMode.SPEARMAN.value:
        corr_func = stats.spearmanr
    else:
        raise ValueError(f"Unknown correlation mode: {mode}. Use CorrelationMode.PEARSON.value or CorrelationMode.SPEARMAN.value")

    # Observed correlation (both arrays have variance > 0)
    r_obs, p_r = corr_func(g1_array, g2_array)

    # Permutation test for correlation
    if n_perms > 0:
        perm_correlations = []
        for _ in range(n_perms):
            perm_g2 = rng.permutation(g2_array)
            perm_r, _ = corr_func(g1_array, perm_g2)
            perm_correlations.append(perm_r)

        perm_correlations = np.array(perm_correlations)
        r_perm_mean = np.mean(perm_correlations)
        r_perm_sd = np.std(perm_correlations, ddof=1)

        # Empirical p-value (two-tailed) using bounded estimator (b+1)/(n_perm+1)
        # This ensures p-value is never exactly zero (minimum: 1/(n_perm+1))
        n_more_extreme = np.sum(np.abs(perm_correlations) >= np.abs(r_obs))
        p_r_perm = (n_more_extreme + 1) / (n_perms + 1)
    else:
        # No permutation test requested
        r_perm_mean = np.nan
        r_perm_sd = np.nan
        p_r_perm = np.nan

    return {
        CORR_OBS: r_obs,
        CORR_P_ANALYTIC: p_r,
        CORR_PERM_MEAN: r_perm_mean,
        CORR_PERM_SD: r_perm_sd,
        CORR_P_PERM: p_r_perm,
        SAMPLE_SIZE: len(g1_array)
    }
