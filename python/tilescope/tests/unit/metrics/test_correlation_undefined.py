"""Tests for exact-constant-operand handling in compute_correlation_from_arrays.

Prior behaviour manufactured r=1.0 (both constant) or r=0.0 (one constant).
Current behaviour raises UndefinedCorrelationError instead, for both Pearson
and Spearman. Permutation is never executed after the raise.

For near-constant (but not exactly constant) Pearson inputs, SciPy's own
NearConstantInputWarning still fires; this module asserts the warning is
preserved rather than swallowed.
"""
from tilescope.cols import CORR_OBS


import numpy as np
import pytest
import scipy.stats

from tilescope.metrics.correlation import (
    UndefinedCorrelationError,
    UndefinedCorrelationReason,
    compute_correlation_from_arrays,
)
from tilescope.enums import CorrelationMode


PEARSON = CorrelationMode.PEARSON.value
SPEARMAN = CorrelationMode.SPEARMAN.value


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("mode", [PEARSON, SPEARMAN])
def test_both_constant_raises(mode, rng):
    g1 = np.full(50, 3.14)
    g2 = np.full(50, -1.0)
    with pytest.raises(UndefinedCorrelationError, match="Both arrays are exactly constant"):
        compute_correlation_from_arrays(g1, g2, mode=mode, n_perms=100, rng=rng)


@pytest.mark.parametrize("mode", [PEARSON, SPEARMAN])
def test_g1_constant_raises(mode, rng):
    g1 = np.zeros(50)
    g2 = rng.normal(size=50)
    with pytest.raises(UndefinedCorrelationError, match="Gene 1 is exactly constant"):
        compute_correlation_from_arrays(g1, g2, mode=mode, n_perms=100, rng=rng)


@pytest.mark.parametrize("mode", [PEARSON, SPEARMAN])
def test_g2_constant_raises(mode, rng):
    g1 = rng.normal(size=50)
    g2 = np.full(50, 42.0)
    with pytest.raises(UndefinedCorrelationError, match="Gene 2 is exactly constant"):
        compute_correlation_from_arrays(g1, g2, mode=mode, n_perms=100, rng=rng)


def test_undefined_error_is_valueerror_subclass():
    # Existing except-ValueError handlers keep catching.
    assert issubclass(UndefinedCorrelationError, ValueError)


@pytest.mark.parametrize(
    "make_g1, make_g2, expected_reason",
    [
        (lambda r: np.full(50, 3.14),
         lambda r: np.full(50, -1.0),
         UndefinedCorrelationReason.BOTH_CONSTANT),
        (lambda r: np.zeros(50),
         lambda r: r.normal(size=50),
         UndefinedCorrelationReason.GENE1_CONSTANT),
        (lambda r: r.normal(size=50),
         lambda r: np.full(50, 42.0),
         UndefinedCorrelationReason.GENE2_CONSTANT),
    ],
)
def test_undefined_error_reason_attribute(make_g1, make_g2, expected_reason, rng):
    # Callers pattern-match on the structured reason enum instead of
    # parsing exception messages.
    g1, g2 = make_g1(rng), make_g2(rng)
    with pytest.raises(UndefinedCorrelationError) as exc_info:
        compute_correlation_from_arrays(g1, g2, mode=PEARSON, n_perms=0, rng=rng)
    assert exc_info.value.reason is expected_reason


@pytest.mark.parametrize("mode", [PEARSON, SPEARMAN])
def test_permutation_not_run_on_undefined(mode, rng):
    # Permutation is skipped on undefined correlation. If it ran, calling
    # with a large n_perms on 50-element constant arrays would still raise
    # UndefinedCorrelationError but would first pointlessly permute.
    # Verified structurally by the raise occurring before the perm loop;
    # this test asserts the raise happens (behavioural anchor).
    g1 = np.zeros(50)
    g2 = rng.normal(size=50)
    with pytest.raises(UndefinedCorrelationError):
        compute_correlation_from_arrays(g1, g2, mode=mode, n_perms=1_000_000, rng=rng)


def test_near_constant_pearson_still_warns(rng):
    # A tiny perturbation is not exactly constant, so we do NOT raise; instead
    # SciPy's own NearConstantInputWarning must still surface. No invented
    # variance-tolerance policy inside TileScope.
    g1 = np.ones(50) + rng.normal(scale=1e-15, size=50)
    g2 = rng.normal(size=50)
    with pytest.warns(scipy.stats.NearConstantInputWarning):
        result = compute_correlation_from_arrays(g1, g2, mode=PEARSON, n_perms=0, rng=rng)
    assert np.isfinite(result[CORR_OBS])


def test_valid_input_still_works(rng):
    g1 = rng.normal(size=100)
    g2 = 0.5 * g1 + 0.5 * rng.normal(size=100)
    result = compute_correlation_from_arrays(g1, g2, mode=PEARSON, n_perms=50, rng=rng)
    assert np.isfinite(result[CORR_OBS])
