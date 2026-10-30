"""Focused contract for the grain-neutral partition enrichment primitive."""
import numpy as np
import pytest

from tilescope.analyzers.partition_analysis import (
    PartitionEffectSizeStatus,
    PartitionEnrichmentResult,
    PartitionEnrichmentStatus,
    PartitionValueSurface,
    PartitionValues,
    analyze_partition_enrichment,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus
from tilescope.metrics.enrichment_thresholds import PercentileThresholds
from tilescope.metrics.normalization import ScaleNormalizationParams


THRESHOLDS = PercentileThresholds(
    q_g1_threshold=2.5,
    q_g2_threshold=2.5,
    q_percentile=0.75,
    q_g1_low_threshold=1.5,
    q_g2_low_threshold=1.5,
    q_low_percentile=0.25,
)
NORMALIZATION = ScaleNormalizationParams(
    gene1_min_spot=0.0,
    gene2_min_spot=0.0,
    gene1_min_density=0.0,
    gene2_min_density=0.0,
)


def _values(g1, g2, surface=PartitionValueSurface.VORONOI_TILE_MEAN):
    return PartitionValues(
        value_surface=surface,
        gene1=np.asarray(g1, dtype=float),
        gene2=np.asarray(g2, dtype=float),
    )


def _analyze(target, context):
    return analyze_partition_enrichment(
        target_values=target,
        context_values=context,
        thresholds=THRESHOLDS,
        normalization_params=NORMALIZATION,
    )


def test_surface_mismatch_rejected():
    with pytest.raises(ValueError, match="value surfaces differ"):
        _analyze(
            _values([1], [1]),
            _values(
                [2], [2],
                surface=PartitionValueSurface.CONSENSUS_ENSEMBLE,
            ),
        )


def test_finite_result_carries_ratios_effects_and_target_stats():
    result = _analyze(
        _values([1, 2, 3, 4], [1, 2, 4, 5]),
        _values([1, 1.2, 3, 3.2], [1, 1.1, 3, 3.1]),
    )
    assert result.enrichment_status is PartitionEnrichmentStatus.FINITE
    assert result.effect_size_status is PartitionEffectSizeStatus.FINITE
    assert result.target_stats is not None
    assert result.target_stats.n_obs == result.n_units_finite == 4
    assert result.enrichment_g1_high is not None
    assert result.log2fc_g1 is not None


def test_zero_context_proportion_preserves_effects_and_target_stats():
    result = _analyze(
        _values([1, 2, 3], [1, 2, 3]),
        _values([2, 2, 2], [2, 2, 2]),
    )
    assert (
        result.enrichment_status
        is PartitionEnrichmentStatus.CONTEXT_PROPORTION_ZERO
    )
    assert result.enrichment_g1_high is None
    assert result.effect_size_status is PartitionEffectSizeStatus.FINITE
    assert result.log2fc_g1 is not None
    assert result.target_stats is not None


def test_pairwise_finite_filter_is_single_authority():
    result = _analyze(
        _values([1, np.nan, 3], [1, 2, 3]),
        _values([1, 3], [1, 3]),
    )
    assert result.n_units_total == 3
    assert result.n_units_finite == 2
    assert result.target_stats.n_obs == 2


def test_empty_target_and_context_have_independent_statuses():
    result = _analyze(_values([], []), _values([], []))
    assert result.enrichment_status is PartitionEnrichmentStatus.INSUFFICIENT_UNITS
    assert (
        result.effect_size_status
        is PartitionEffectSizeStatus.INSUFFICIENT_TARGET_UNITS
    )
    assert result.target_stats is None


def test_empty_context_is_structural_not_zero_context():
    # Empty context ⇒ no observations to measure a proportion from,
    # so the aggregate status must be structural INSUFFICIENT_CONTEXT_UNITS
    # (not CONTEXT_PROPORTION_ZERO, which requires ctx_n_total > 0).
    # All per-ratio statuses and values must be None; target stats +
    # effect-size availability are unaffected.
    result = _analyze(_values([1, 2], [1, 2]), _values([], []))
    assert result.target_stats.n_obs == 2
    assert (
        result.enrichment_status
        is PartitionEnrichmentStatus.INSUFFICIENT_CONTEXT_UNITS
    )
    for value in (
        result.enrichment_g1_high, result.enrichment_g2_high,
        result.enrichment_joint_high_fc, result.enrichment_joint_low_fc,
    ):
        assert value is None
    for status in (
        result.enrichment_g1_high_status,
        result.enrichment_g2_high_status,
        result.enrichment_joint_high_fc_status,
        result.enrichment_joint_low_fc_status,
    ):
        assert status is None
    assert (
        result.effect_size_status
        is PartitionEffectSizeStatus.INSUFFICIENT_CONTEXT_UNITS
    )


def test_partition_result_rejects_non_enum_per_ratio_status():
    # An arbitrary string that happens to equal "finite" must not slip
    # through the dataclass invariant; only EnrichmentRatioStatus enum
    # members (or None) are accepted.
    with pytest.raises(ValueError, match="EnrichmentRatioStatus"):
        PartitionEnrichmentResult(
            n_units_total=1, n_units_finite=1,
            enrichment_g1_high=1.5, enrichment_g2_high=1.5,
            enrichment_joint_high_fc=1.5, enrichment_joint_low_fc=1.5,
            enrichment_g1_high_status="finite",
            enrichment_g2_high_status="finite",
            enrichment_joint_high_fc_status="finite",
            enrichment_joint_low_fc_status="finite",
            enrichment_status=PartitionEnrichmentStatus.FINITE,
            log2fc_g1=None, log2fc_g2=None, cles_g1=None, cles_g2=None,
            effect_size_status=(
                PartitionEffectSizeStatus.INSUFFICIENT_CONTEXT_UNITS
            ),
            target_stats=None,
        )


def test_partial_finite_zero_context_on_one_marginal_leaves_other_finite():
    # g2 context is constant while g1 context varies → context proportion
    # for g2-high is 0 but g1-high is positive. The g1 marginal ratio
    # must remain finite; only the g2 ratio (and joint HIGH) become
    # non-finite. This is the sister-side "opposed_gradient" bug the
    # per-ratio status contract fixes.
    result = _analyze(
        _values([0, 1, 2, 4, 5], [1, 1, 1, 4, 4]),
        _values([0, 0, 1, 3, 3], [1, 1, 1, 1, 1]),
    )
    assert (
        result.enrichment_status
        is PartitionEnrichmentStatus.CONTEXT_PROPORTION_ZERO
    )
    assert result.enrichment_g1_high_status == (
        EnrichmentRatioStatus.FINITE.value
    )
    assert result.enrichment_g1_high is not None
    assert np.isfinite(result.enrichment_g1_high)
    assert result.enrichment_g2_high_status != (
        EnrichmentRatioStatus.FINITE.value
    )
    assert result.enrichment_g2_high is None
    assert result.enrichment_joint_high_fc is None
