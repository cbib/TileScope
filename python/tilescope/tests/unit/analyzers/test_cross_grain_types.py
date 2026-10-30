"""Wire-vocabulary coverage + cross-authority alignment for
cross_grain_types. Producer behavior is tested elsewhere."""
from __future__ import annotations

from tilescope.analyzers.cross_grain_types import (
    Channel,
    ComparisonStatus,
    EffectScale,
    Sign,
    Statistic,
    Surface,
    UnitKind,
)
from tilescope.analyzers.parent_roi_moran_common import MORAN_DIRECTIONS
from tilescope.analyzers.partition_analysis import PartitionValueSurface


def test_wire_vocabulary_is_locked():
    """The wire-format strings are load-bearing — the CSV producer keys
    on them and downstream consumers pattern-match them."""
    expected = {
        Statistic: {"correlation", "enrichment", "moran"},
        Channel: {"", "g1", "g2", "joint_high", "joint_low"},
        Surface: {
            "source_native", "tessellation_median", "consensus_ensemble",
        },
        EffectScale: {"identity", "ratio_null_centred"},
        Sign: {"positive", "negative", "zero", "undefined"},
        ComparisonStatus: {
            "both_comparable", "reference_unavailable",
            "target_unavailable", "both_unavailable",
        },
        UnitKind: {"source_spot", "tile", "hex"},
    }
    for enum_type, wire_values in expected.items():
        assert {member.value for member in enum_type} == wire_values


def test_vocabulary_aligns_with_upstream_authorities():
    assert (
        Surface.CONSENSUS_ENSEMBLE.value
        == PartitionValueSurface.CONSENSUS_ENSEMBLE.value
    )
    assert tuple(MORAN_DIRECTIONS) == ("g1_lag_g2", "g2_lag_g1")
