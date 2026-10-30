"""Pure-computation producer for ``roi_cross_grain_sensitivity.csv``.

Consumes typed per-surface observation records and emits the canonical
DataFrame documented in :mod:`tilescope.analyzers.cross_grain_types`.
No I/O; no policy decisions; no categorical interpretation.

Scope: one pair-run at a time. The caller supplies ``sample_id``,
``g1_id``, ``g2_id``, and the enumerated ``roi_ids``; the producer
emits exactly 14 rows per ROI (7 estimand triples x 2 target surfaces:
source_native -> tessellation_median and source_native ->
consensus_ensemble). A missing observation on any surface for a
declared ROI is a producer / adapter bug and raises. Canonical
unavailability is encoded upstream as a typed row with an unavailable
``raw_status``, never as row absence.

Sign, delta, and sign-preservation are computed on the null-centred
effect scale, never on raw values. Run provenance (core version,
config hash) lives on the pair-run's ``run_metadata.json`` and is not
repeated per row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
from tilescope.metrics.enrichment import EnrichmentRatioStatus

# Row + column contract version of the roi_cross_grain_sensitivity
# frame. Downstream aggregators pin against it via the backfill
# manifest. Bump on schema-breaking changes only.
PRODUCER_CONTRACT_VERSION = "1.0.0"

# Enumeration of the (statistic, channel, moran_direction) triples the
# CSV covers. Row set is closed per declared ROI.
_ROW_TRIPLES: Tuple[Tuple[Statistic, Channel, str], ...] = (
    (Statistic.CORRELATION, Channel.NONE, ""),
    (Statistic.ENRICHMENT, Channel.G1, ""),
    (Statistic.ENRICHMENT, Channel.G2, ""),
    (Statistic.ENRICHMENT, Channel.JOINT_HIGH, ""),
    (Statistic.ENRICHMENT, Channel.JOINT_LOW, ""),
    (Statistic.MORAN, Channel.NONE, MORAN_DIRECTIONS[0]),
    (Statistic.MORAN, Channel.NONE, MORAN_DIRECTIONS[1]),
)
_LEGAL_TRIPLES: frozenset = frozenset(_ROW_TRIPLES)

_TARGET_SURFACES: Tuple[Surface, ...] = (
    Surface.TESSELLATION_MEDIAN,
    Surface.CONSENSUS_ENSEMBLE,
)


# ---------------------------------------------------------------------
# Typed input records
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PointSurfaceObservation:
    """One estimand value on one point surface (source_native or
    consensus_ensemble) for one ROI.

    ``n_context_units`` is ``None`` for correlation and Moran (no
    context population); ``context_unit_kind`` mirrors that. Support
    counts are ``float`` because construction-surface support can be a
    replicate median.
    """
    roi_id: str
    surface: Surface
    statistic: Statistic
    channel: Channel
    moran_direction: str
    raw_value: Optional[float]
    raw_status: str
    undefined_reason: str
    n_target_units: Optional[float]
    n_context_units: Optional[float]
    target_unit_kind: UnitKind
    context_unit_kind: Optional[UnitKind]


@dataclass(frozen=True)
class TessellationSummaryObservation:
    """Tessellation-surface summary for one estimand on one ROI.

    Adapters compute ``effect_median`` and ``effect_iqr`` from the same
    replicate population so the point estimate and spread describe the
    same set of comparable draws. The producer wires these through
    without recomputation.
    """
    roi_id: str
    statistic: Statistic
    channel: Channel
    moran_direction: str
    raw_summary: Optional[float]
    raw_status: str
    undefined_reason: str
    effect_median: Optional[float]
    effect_iqr: Optional[float]
    n_comparable_replicates: int
    n_planned_replicates: int
    n_target_units: Optional[float]
    n_context_units: Optional[float]
    target_unit_kind: UnitKind
    context_unit_kind: Optional[UnitKind]


@dataclass(frozen=True)
class CrossGrainSensitivityInputs:
    """One pair-run's worth of inputs. Populated by the per-pair
    adapter from the pair's PairStatisticResult."""
    sample_id: str
    g1_id: str
    g2_id: str
    roi_ids: Sequence[str]
    source_observations: Sequence[PointSurfaceObservation]
    tessellation_observations: Sequence[TessellationSummaryObservation]
    consensus_observations: Sequence[PointSurfaceObservation]


# ---------------------------------------------------------------------
# Semantic helpers (pure functions)
# ---------------------------------------------------------------------

def effect_scale_for(statistic: Statistic) -> EffectScale:
    if statistic == Statistic.ENRICHMENT:
        return EffectScale.RATIO_NULL_CENTRED
    return EffectScale.IDENTITY


def compute_effect(
    raw_value: Optional[float],
    raw_status: str,
    statistic: Statistic,
) -> Optional[float]:
    """Validated null-centred effect derivation.

    Correlation / Moran: identity on finite raws. Enrichment: r/(1+r)-0.5
    on finite non-negative raws; POSITIVE_OVER_ZERO_CONTEXT typed
    saturation maps to +0.5. All other statuses yield ``None``.

    Raises ``ValueError`` on malformed combinations:
    - POSITIVE_OVER_ZERO_CONTEXT status for a non-enrichment estimand;
    - FINITE status with a non-finite raw value;
    - FINITE enrichment with a negative raw value.
    """
    finite = EnrichmentRatioStatus.FINITE.value
    saturation = EnrichmentRatioStatus.POSITIVE_OVER_ZERO_CONTEXT.value

    if statistic == Statistic.ENRICHMENT:
        if raw_status == finite:
            if raw_value is None or not np.isfinite(raw_value):
                raise ValueError(
                    f"finite enrichment status requires a finite raw value; "
                    f"got raw_value={raw_value!r}."
                )
            if raw_value < 0:
                raise ValueError(
                    f"finite enrichment raw value must be non-negative; "
                    f"got {raw_value!r}."
                )
            return float(raw_value) / (1.0 + float(raw_value)) - 0.5
        if raw_status == saturation:
            return 0.5
        return None

    # correlation / moran
    if raw_status == saturation:
        raise ValueError(
            f"{saturation!r} is an enrichment-only typed saturation; "
            f"{statistic.value!r} rows must not carry it."
        )
    if raw_status == finite:
        if raw_value is None or not np.isfinite(raw_value):
            raise ValueError(
                f"finite {statistic.value!r} status requires a finite raw "
                f"value; got raw_value={raw_value!r}."
            )
        return float(raw_value)
    return None


def derive_sign(effect: Optional[float]) -> Sign:
    if effect is None:
        return Sign.UNDEFINED
    if effect > 0:
        return Sign.POSITIVE
    if effect < 0:
        return Sign.NEGATIVE
    return Sign.ZERO


def derive_comparison_status(
    reference_effect: Optional[float],
    target_effect: Optional[float],
) -> ComparisonStatus:
    ref_ok = reference_effect is not None
    tgt_ok = target_effect is not None
    if ref_ok and tgt_ok:
        return ComparisonStatus.BOTH_COMPARABLE
    if not ref_ok and not tgt_ok:
        return ComparisonStatus.BOTH_UNAVAILABLE
    if not ref_ok:
        return ComparisonStatus.REFERENCE_UNAVAILABLE
    return ComparisonStatus.TARGET_UNAVAILABLE


# ---------------------------------------------------------------------
# Canonical output schema
# ---------------------------------------------------------------------

CANONICAL_COLUMNS: Tuple[str, ...] = (
    "sample_id", "roi_id",
    "g1_id", "g2_id",
    "statistic", "channel", "moran_direction",
    "reference_surface", "target_surface",
    "effect_scale",
    "reference_raw_value", "reference_raw_status", "reference_undefined_reason",
    "reference_effect_value", "reference_sign",
    "reference_n_target_units", "reference_n_context_units",
    "reference_target_unit_kind", "reference_context_unit_kind",
    "target_raw_value", "target_raw_status", "target_undefined_reason",
    "target_effect_value", "target_sign",
    "target_n_target_units", "target_n_context_units",
    "target_target_unit_kind", "target_context_unit_kind",
    "comparison_status",
    "delta_effect", "abs_delta_effect", "sign_preserved",
    "tessellation_effect_iqr", "n_comparable_replicates", "n_planned_replicates",
)


# ---------------------------------------------------------------------
# Indexing + assembly
# ---------------------------------------------------------------------

_ObsKey = Tuple[str, str, str, str]  # (roi_id, statistic, channel, moran_direction)


def _point_key(obs: PointSurfaceObservation) -> _ObsKey:
    return (obs.roi_id, obs.statistic.value, obs.channel.value, obs.moran_direction)


def _tess_key(obs: TessellationSummaryObservation) -> _ObsKey:
    return (obs.roi_id, obs.statistic.value, obs.channel.value, obs.moran_direction)


def _triple_key(roi_id: str, triple) -> _ObsKey:
    statistic, channel, moran_direction = triple
    return (roi_id, statistic.value, channel.value, moran_direction)


def _index_points(
    observations: Sequence[PointSurfaceObservation],
    surface: Surface,
) -> Mapping[_ObsKey, PointSurfaceObservation]:
    index: dict = {}
    for obs in observations:
        if obs.surface != surface:
            raise ValueError(
                f"observation surface {obs.surface!r} does not match "
                f"container surface {surface!r} for key {_point_key(obs)!r}."
            )
        key = _point_key(obs)
        if key in index:
            raise ValueError(f"duplicate {surface.value!r} observation for {key!r}.")
        index[key] = obs
    return index


def _index_tess(
    observations: Sequence[TessellationSummaryObservation],
) -> Mapping[_ObsKey, TessellationSummaryObservation]:
    index: dict = {}
    for obs in observations:
        key = _tess_key(obs)
        if key in index:
            raise ValueError(f"duplicate tessellation-summary observation for {key!r}.")
        index[key] = obs
    return index


def _unit_kind_str(kind: Optional[UnitKind]) -> str:
    return kind.value if kind is not None else ""


def _build_row(
    sample_id: str,
    g1_id: str,
    g2_id: str,
    roi_id: str,
    triple: Tuple[Statistic, Channel, str],
    target_surface: Surface,
    ref_obs: PointSurfaceObservation,
    target_point: Optional[PointSurfaceObservation],
    target_tess: Optional[TessellationSummaryObservation],
) -> dict:
    statistic, channel, moran_direction = triple

    ref_effect = compute_effect(ref_obs.raw_value, ref_obs.raw_status, statistic)
    ref_sign = derive_sign(ref_effect)

    if target_surface == Surface.TESSELLATION_MEDIAN:
        assert target_tess is not None
        tgt_raw = target_tess.raw_summary
        tgt_status = target_tess.raw_status
        tgt_reason = target_tess.undefined_reason
        tgt_effect = target_tess.effect_median
        tgt_n_target = target_tess.n_target_units
        tgt_n_context = target_tess.n_context_units
        tgt_target_kind = target_tess.target_unit_kind
        tgt_context_kind = target_tess.context_unit_kind
        tess_iqr = target_tess.effect_iqr
        n_comparable = target_tess.n_comparable_replicates
        n_planned = target_tess.n_planned_replicates
    else:
        assert target_point is not None
        tgt_raw = target_point.raw_value
        tgt_status = target_point.raw_status
        tgt_reason = target_point.undefined_reason
        tgt_effect = compute_effect(tgt_raw, tgt_status, statistic)
        tgt_n_target = target_point.n_target_units
        tgt_n_context = target_point.n_context_units
        tgt_target_kind = target_point.target_unit_kind
        tgt_context_kind = target_point.context_unit_kind
        tess_iqr = None
        n_comparable = None
        n_planned = None

    tgt_sign = derive_sign(tgt_effect)
    comparison = derive_comparison_status(ref_effect, tgt_effect)

    if comparison == ComparisonStatus.BOTH_COMPARABLE:
        delta = float(tgt_effect - ref_effect)
        abs_delta = float(abs(delta))
        sign_preserved: Optional[bool] = ref_sign == tgt_sign
    else:
        delta = None
        abs_delta = None
        sign_preserved = None

    return {
        "sample_id": sample_id,
        "roi_id": roi_id,
        "g1_id": g1_id,
        "g2_id": g2_id,
        "statistic": statistic.value,
        "channel": channel.value,
        "moran_direction": moran_direction,
        "reference_surface": Surface.SOURCE_NATIVE.value,
        "target_surface": target_surface.value,
        "effect_scale": effect_scale_for(statistic).value,
        "reference_raw_value": ref_obs.raw_value,
        "reference_raw_status": ref_obs.raw_status,
        "reference_undefined_reason": ref_obs.undefined_reason,
        "reference_effect_value": ref_effect,
        "reference_sign": ref_sign.value,
        "reference_n_target_units": ref_obs.n_target_units,
        "reference_n_context_units": ref_obs.n_context_units,
        "reference_target_unit_kind": _unit_kind_str(ref_obs.target_unit_kind),
        "reference_context_unit_kind": _unit_kind_str(ref_obs.context_unit_kind),
        "target_raw_value": tgt_raw,
        "target_raw_status": tgt_status,
        "target_undefined_reason": tgt_reason,
        "target_effect_value": tgt_effect,
        "target_sign": tgt_sign.value,
        "target_n_target_units": tgt_n_target,
        "target_n_context_units": tgt_n_context,
        "target_target_unit_kind": _unit_kind_str(tgt_target_kind),
        "target_context_unit_kind": _unit_kind_str(tgt_context_kind),
        "comparison_status": comparison.value,
        "delta_effect": delta,
        "abs_delta_effect": abs_delta,
        "sign_preserved": sign_preserved,
        "tessellation_effect_iqr": tess_iqr,
        "n_comparable_replicates": n_comparable,
        "n_planned_replicates": n_planned,
    }


def validate_cross_grain_sensitivity_frame(df: pd.DataFrame, expected_rows: int) -> None:
    """Post-build invariants. Producer bugs raise loud rather than
    escaping into an emitted CSV."""
    if list(df.columns) != list(CANONICAL_COLUMNS):
        raise AssertionError(
            "cross_grain_sensitivity column order drifted from canonical."
        )
    if len(df) != expected_rows:
        raise AssertionError(
            f"expected {expected_rows} rows, produced {len(df)}."
        )
    if expected_rows == 0:
        return
    key_cols = ["sample_id", "roi_id", "g1_id", "g2_id",
                "statistic", "channel", "moran_direction", "target_surface"]
    if df.duplicated(key_cols).any():
        raise AssertionError("duplicate row keys in cross_grain_sensitivity frame.")
    for triple in zip(df["statistic"], df["channel"], df["moran_direction"]):
        stat, chan, direction = triple
        # CSV round-trip parses empty string cells as NaN; normalise to
        # "" so the enum lookup matches the producer's empty-string
        # sentinel for non-enrichment / non-Moran rows.
        chan_str = "" if pd.isna(chan) else str(chan)
        direction_str = "" if pd.isna(direction) else str(direction)
        triple_typed = (Statistic(stat), Channel(chan_str), direction_str)
        if triple_typed not in _LEGAL_TRIPLES:
            raise AssertionError(f"illegal (statistic, channel, direction) triple: {triple!r}.")

    both_comparable = df["comparison_status"] == ComparisonStatus.BOTH_COMPARABLE.value
    if both_comparable.any():
        if df.loc[both_comparable, "delta_effect"].isna().any():
            raise AssertionError("both_comparable rows must carry finite delta_effect.")
        if df.loc[both_comparable, "abs_delta_effect"].isna().any():
            raise AssertionError("both_comparable rows must carry finite abs_delta_effect.")
        if df.loc[both_comparable, "sign_preserved"].isna().any():
            raise AssertionError("both_comparable rows must carry sign_preserved.")
    non_comparable = ~both_comparable
    if non_comparable.any():
        for col in ("delta_effect", "abs_delta_effect", "sign_preserved"):
            if not df.loc[non_comparable, col].isna().all():
                raise AssertionError(
                    f"non-comparable rows must carry null {col!r}."
                )

    consensus_rows = df["target_surface"] == Surface.CONSENSUS_ENSEMBLE.value
    if consensus_rows.any():
        for col in ("tessellation_effect_iqr", "n_comparable_replicates", "n_planned_replicates"):
            if not df.loc[consensus_rows, col].isna().all():
                raise AssertionError(
                    f"consensus_ensemble rows must carry null {col!r}."
                )

    undefined_effect_ref = df["reference_effect_value"].isna()
    undefined_sign_ref = df["reference_sign"] == Sign.UNDEFINED.value
    if not (undefined_effect_ref == undefined_sign_ref).all():
        raise AssertionError(
            "reference_sign undefined-state does not match reference_effect_value nullness."
        )
    undefined_effect_tgt = df["target_effect_value"].isna()
    undefined_sign_tgt = df["target_sign"] == Sign.UNDEFINED.value
    if not (undefined_effect_tgt == undefined_sign_tgt).all():
        raise AssertionError(
            "target_sign undefined-state does not match target_effect_value nullness."
        )


def build_roi_cross_grain_sensitivity_df(
    inputs: CrossGrainSensitivityInputs,
) -> pd.DataFrame:
    """Assemble the canonical DataFrame from typed per-surface inputs.

    Every declared ROI contributes exactly 14 rows. Missing observations
    raise; canonical unavailability is encoded upstream as a typed row
    with an unavailable ``raw_status``.
    """
    source_idx = _index_points(inputs.source_observations, Surface.SOURCE_NATIVE)
    consensus_idx = _index_points(inputs.consensus_observations, Surface.CONSENSUS_ENSEMBLE)
    tess_idx = _index_tess(inputs.tessellation_observations)

    rows = []
    for roi_id in inputs.roi_ids:
        for triple in _ROW_TRIPLES:
            key = _triple_key(roi_id, triple)
            ref_obs = source_idx.get(key)
            if ref_obs is None:
                raise KeyError(
                    f"missing source_native observation for {key!r}. Every "
                    "declared ROI must supply all 7 estimand triples on every "
                    "surface; canonical unavailability is a typed row, not "
                    "row absence."
                )
            for target_surface in _TARGET_SURFACES:
                if target_surface == Surface.TESSELLATION_MEDIAN:
                    target_tess = tess_idx.get(key)
                    if target_tess is None:
                        raise KeyError(
                            f"missing tessellation observation for {key!r}."
                        )
                    target_point = None
                else:
                    target_point = consensus_idx.get(key)
                    if target_point is None:
                        raise KeyError(
                            f"missing consensus_ensemble observation for {key!r}."
                        )
                    target_tess = None
                rows.append(_build_row(
                    sample_id=inputs.sample_id,
                    g1_id=inputs.g1_id, g2_id=inputs.g2_id,
                    roi_id=roi_id, triple=triple,
                    target_surface=target_surface,
                    ref_obs=ref_obs,
                    target_point=target_point,
                    target_tess=target_tess,
                ))

    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    expected_rows = len(inputs.roi_ids) * len(_ROW_TRIPLES) * len(_TARGET_SURFACES)
    validate_cross_grain_sensitivity_frame(df, expected_rows)
    return df
