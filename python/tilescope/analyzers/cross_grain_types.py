"""Typed vocabulary for the cross-grain sensitivity producer.

Schema semantics only. Producer logic and the exported table land
together with the pipeline wiring; this module is imported by both.

Scientific scope
----------------
The producer emits an empirical cross-grain / support-sensitivity table
related to the Modifiable Areal Unit Problem. The CSV row set covers
two comparisons per estimand:

- ``source_native`` -> ``tessellation_median``: random-zoning
  sensitivity captured by the tessellation replicate IQR on the effect
  scale.
- ``source_native`` -> ``consensus_ensemble``: headline
  source-to-constructed-support sensitivity.

Direct-versus-ensemble consensus sensitivity is intentionally not a row
in this table; downstream code obtains it by joining the existing
``roi_consensus_*.csv`` surfaces. Interface response (ordered-edge
grain) and radial cross-correlogram (spot/cell component grain)
are also out of scope.

This is not a complete zoning x scale decomposition. The production
configuration uses a single tessellation scale, and moving from source
to constructed surfaces simultaneously changes aggregation, support
coverage, and graph construction; those effects are not disentangled.

The CSV carries continuous values only. The HTML report's compact
``Stable`` / ``Sensitive to zoning`` / ``Direction changes`` reading
is a separate rendering concern and lives under
``tilescope.reporters``, not in this schema vocabulary.

Effect scale
------------
Sign and delta computation always operate on the null-centred effect
value, not the raw value. For correlation and Moran, the effect equals
the raw value (identity). For enrichment,
``E(r) = r / (1 + r) - 0.5``, with ``positive_over_zero_context``
mapped to ``+0.5``. Sharing a signed delta column across families
without this transform would silently invert sign whenever an
enrichment ratio crosses 1, which is common in real data.

Vocabulary reuse
----------------
Moran direction values are reused verbatim from the canonical
``MORAN_DIRECTIONS`` authority (``parent_roi_moran_common``); the
consensus-ensemble surface string is aligned to
``PartitionValueSurface.CONSENSUS_ENSEMBLE`` (``partition_analysis``).
Alignment is asserted at import time so a rename in either authority
fails loud rather than producing rows keyed on a stale label.
"""
from __future__ import annotations

from enum import Enum

from tilescope.analyzers.parent_roi_moran_common import MORAN_DIRECTIONS
from tilescope.analyzers.partition_analysis import PartitionValueSurface


class Statistic(str, Enum):
    """The three estimator families comparable across reference and
    target surfaces."""
    CORRELATION = "correlation"
    ENRICHMENT = "enrichment"
    MORAN = "moran"


class Channel(str, Enum):
    """Enrichment channel enumeration. Empty for correlation and Moran
    rows so the ``(statistic, channel, moran_direction)`` triple is a
    total row-key partition without collision."""
    NONE = ""
    G1 = "g1"
    G2 = "g2"
    JOINT_HIGH = "joint_high"
    JOINT_LOW = "joint_low"


class Surface(str, Enum):
    """The reference and target surfaces referenced by CSV rows.
    Consensus-ensemble aligns to
    :class:`PartitionValueSurface.CONSENSUS_ENSEMBLE`."""
    SOURCE_NATIVE = "source_native"
    TESSELLATION_MEDIAN = "tessellation_median"
    CONSENSUS_ENSEMBLE = "consensus_ensemble"


class EffectScale(str, Enum):
    """Names the transform relating raw value to effect value.
    ``identity`` is used when raw and effect coincide (correlation,
    Moran); ``ratio_null_centred`` names the enrichment transform
    E(r) = r / (1 + r) - 0.5."""
    IDENTITY = "identity"
    RATIO_NULL_CENTRED = "ratio_null_centred"


class Sign(str, Enum):
    """Sign of an effect value. ``ZERO`` corresponds to an exact effect
    of zero (E(1)=0 for enrichment; raw zero for correlation and
    Moran). ``UNDEFINED`` is emitted when the effect value is
    undefined."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ZERO = "zero"
    UNDEFINED = "undefined"


class ComparisonStatus(str, Enum):
    """Availability of the comparison. ``both_comparable`` accepts
    typed saturation states (e.g. ``positive_over_zero_context``) that
    lack a finite raw ratio but carry a defined bounded effect. Missing
    sides always map to their explicit unavailable state; no
    substitution."""
    BOTH_COMPARABLE = "both_comparable"
    REFERENCE_UNAVAILABLE = "reference_unavailable"
    TARGET_UNAVAILABLE = "target_unavailable"
    BOTH_UNAVAILABLE = "both_unavailable"


class UnitKind(str, Enum):
    """Support unit population. Distinguishes surfaces so paired
    ``n_units`` columns cannot be misread as directly comparable
    integers."""
    SOURCE_SPOT = "source_spot"
    TILE = "tile"
    HEX = "hex"


# Load-bearing invariants: producer rows must key on strings that
# match the canonical authorities. Failing loud at import time
# prevents a stale rename in either authority from producing rows
# with silently divergent vocabulary.

_EXPECTED_MORAN_DIRECTIONS = ("g1_lag_g2", "g2_lag_g1")
if tuple(MORAN_DIRECTIONS) != _EXPECTED_MORAN_DIRECTIONS:
    raise AssertionError(
        "MORAN_DIRECTIONS drifted from the cross-grain sensitivity "
        f"contract: expected {_EXPECTED_MORAN_DIRECTIONS!r}, got "
        f"{tuple(MORAN_DIRECTIONS)!r}. Update this module together with "
        "the producer + any consumers before merging."
    )

if Surface.CONSENSUS_ENSEMBLE.value != PartitionValueSurface.CONSENSUS_ENSEMBLE.value:
    raise AssertionError(
        "Surface.CONSENSUS_ENSEMBLE drifted from "
        "PartitionValueSurface.CONSENSUS_ENSEMBLE: "
        f"{Surface.CONSENSUS_ENSEMBLE.value!r} != "
        f"{PartitionValueSurface.CONSENSUS_ENSEMBLE.value!r}."
    )
