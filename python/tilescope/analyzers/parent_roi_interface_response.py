"""ROI-pair interface-conditioned response at parent-ROI grain.

Directional between-ROI single-mark statistic: per ordered
`(anchor_roi_id, responder_roi_id)` pair and each responder feature, emits
one row per surface with `interface_response = mean(x_near) - mean(x_far)`.
Near / far are source-defined hop bands on the acquisition graph and are
never re-derived from tile or hex adjacency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from tilescope.cols import ROI_ID
from tilescope.schema.helpers import natural_sort_key
from tilescope.spatial.consensus_bank import ConsensusBank, TileHexOverlap
from tilescope.spatial.feature_projection import FeatureSpatialProjection
from tilescope.spatial.sample_support_bank import SampleSupportBank
from tilescope.spatial.source_spot_adjacency import (
    SourceSpotAdjacency, UNREACHABLE_HOP,
)


NEAR_HOP_MAX = 1
BUFFER_HOP = 2
FAR_HOP_MIN = 3
MIN_REACHABLE_RESPONDER_FRACTION = 0.5
MIN_NEAR_COVERAGE = 5.0
MIN_FAR_COVERAGE = 5.0
UNASSIGNED_HEX_ID = -1


SURFACE_SOURCE_NATIVE = "source_native"
SURFACE_TESSELLATION = "tessellation"
SURFACE_CONSENSUS_DIRECT = "consensus_direct"
SURFACE_CONSENSUS_ENSEMBLE = "consensus_ensemble"
SURFACES: Tuple[str, ...] = (
    SURFACE_SOURCE_NATIVE, SURFACE_TESSELLATION,
    SURFACE_CONSENSUS_DIRECT, SURFACE_CONSENSUS_ENSEMBLE,
)
CONSTRUCTED_SURFACES: Tuple[str, ...] = (
    SURFACE_TESSELLATION, SURFACE_CONSENSUS_DIRECT,
    SURFACE_CONSENSUS_ENSEMBLE,
)

STATUS_FINITE = "finite"
STATUS_NO_INTERFACE = "no_anchor_responder_interface"
STATUS_INSUFFICIENT_REACHABLE = "insufficient_reachable_responder"
STATUS_NO_ELIGIBLE_COMPONENT = "no_tessellation_eligible_component"
STATUS_INSUFFICIENT_NEAR = "insufficient_near_coverage"
STATUS_INSUFFICIENT_FAR = "insufficient_far_coverage"
STATUS_NO_VALID_REPLICATES = "no_valid_replicates"


SAMPLE_ID = "sample_id"
G1_ID = "g1_id"
G2_ID = "g2_id"
ANCHOR_ROI_ID = "anchor_roi_id"
RESPONDER_ROI_ID = "responder_roi_id"
RESPONDER_FEATURE_ID = "responder_feature_id"
SURFACE_COL = "surface"
AVAILABILITY_STATUS = "availability_status"
INTERFACE_RESPONSE = "interface_response"
SOURCE_NEAR_MASS = "source_near_mass"
SOURCE_FAR_MASS = "source_far_mass"
REPRESENTED_NEAR_MASS = "represented_near_mass"
REPRESENTED_FAR_MASS = "represented_far_mass"
REACHABLE_RESPONDER_FRACTION = "reachable_responder_fraction"
N_NEAR_FAR_MIXED_UNITS = "n_near_far_mixed_units"
N_REPLICATES_PLANNED = "n_replicates_planned"
N_VALID_REPLICATES = "n_valid_replicates"

CANONICAL_COLUMNS: Tuple[str, ...] = (
    SAMPLE_ID, G1_ID, G2_ID,
    ANCHOR_ROI_ID, RESPONDER_ROI_ID, RESPONDER_FEATURE_ID,
    SURFACE_COL, AVAILABILITY_STATUS, INTERFACE_RESPONSE,
    SOURCE_NEAR_MASS, SOURCE_FAR_MASS,
    REPRESENTED_NEAR_MASS, REPRESENTED_FAR_MASS,
    REACHABLE_RESPONDER_FRACTION,
    N_NEAR_FAR_MIXED_UNITS,
    N_REPLICATES_PLANNED, N_VALID_REPLICATES,
)


def empty_roi_pair_interface_response_df() -> pd.DataFrame:
    """Contract-shaped empty DataFrame with canonical column order + dtypes."""
    return pd.DataFrame({
        SAMPLE_ID: pd.Series([], dtype="string"),
        G1_ID: pd.Series([], dtype="string"),
        G2_ID: pd.Series([], dtype="string"),
        ANCHOR_ROI_ID: pd.Series([], dtype="string"),
        RESPONDER_ROI_ID: pd.Series([], dtype="string"),
        RESPONDER_FEATURE_ID: pd.Series([], dtype="string"),
        SURFACE_COL: pd.Series([], dtype="string"),
        AVAILABILITY_STATUS: pd.Series([], dtype="string"),
        INTERFACE_RESPONSE: pd.Series([], dtype="Float64"),
        SOURCE_NEAR_MASS: pd.Series([], dtype="Float64"),
        SOURCE_FAR_MASS: pd.Series([], dtype="Float64"),
        REPRESENTED_NEAR_MASS: pd.Series([], dtype="Float64"),
        REPRESENTED_FAR_MASS: pd.Series([], dtype="Float64"),
        REACHABLE_RESPONDER_FRACTION: pd.Series([], dtype="Float64"),
        N_NEAR_FAR_MIXED_UNITS: pd.Series([], dtype="Int64"),
        N_REPLICATES_PLANNED: pd.Series([], dtype="Int64"),
        N_VALID_REPLICATES: pd.Series([], dtype="Int64"),
    })


# ---------------------------------------------------------------------------
# Small internal records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PairField:
    """Source-side field for one ordered `(anchor, responder)` pair.

    Feature-independent: computed once per pair and reused across features.
    """
    # Band tag per source spot: "near" / "far" / "" (buffer, unreachable, or
    # out-of-responder; all excluded from the scalar).
    bands: np.ndarray
    responder_indices: np.ndarray                # global spot/cell indices
    responder_component_ids: Tuple[str, ...]
    source_near_mass: float
    source_far_mass: float
    reachable_fraction: float                    # 1 - unreachable/responder_size


@dataclass(frozen=True)
class _ProjectionResult:
    """One surface's finalised row (excluding row-identity columns)."""
    status: str
    value: float
    represented_near_mass: float
    represented_far_mass: float
    n_near_far_mixed_units: int


@dataclass(frozen=True)
class _WeightedUnitSupport:
    """Feature-independent near/far weights on one constructed surface."""
    unit_keys: Tuple[Tuple, ...]
    denominators: np.ndarray
    near_weights: np.ndarray
    far_weights: np.ndarray


@dataclass(frozen=True)
class _PairProjectionSupport:
    """All reusable supports for one ordered ROI interface."""
    source_positions: np.ndarray
    source_near_weights: np.ndarray
    source_far_weights: np.ndarray
    tessellation_by_replicate: Tuple[_WeightedUnitSupport, ...]
    consensus_direct: _WeightedUnitSupport
    consensus_ensemble: _WeightedUnitSupport


# ---------------------------------------------------------------------------
# Source-side field per ordered pair
# ---------------------------------------------------------------------------


def _pairs_with_cross_roi_edges(
    adjacency: SourceSpotAdjacency,
) -> set[Tuple[str, str]]:
    """Ordered `(anchor, responder)` set for pairs with >= 1 cross-ROI edge.

    Interface presence is symmetric under ordered enumeration, so each
    unordered edge contributes both directions.
    """
    if adjacency.edges.shape[0] == 0:
        return set()
    roi = np.asarray(adjacency.source_spot_table.roi_id)
    i, j = adjacency.edges[:, 0], adjacency.edges[:, 1]
    mask = roi[i] != roi[j]
    out: set[Tuple[str, str]] = set()
    for a, b in zip(roi[i][mask], roi[j][mask]):
        out.add((str(a), str(b)))
        out.add((str(b), str(a)))
    return out


def _build_pair_field(
    *,
    support_bank: SampleSupportBank,
    anchor_roi_id: str,
    responder_roi_id: str,
) -> _PairField:
    """Compute the source-side field for one ordered pair (feature-independent)."""
    adjacency = support_bank.source_spot_adjacency
    roi = np.asarray(adjacency.source_spot_table.roi_id)
    responder_indices = np.where(roi == responder_roi_id)[0]
    n_responder = int(responder_indices.size)

    _, seeds = adjacency.interface_endpoints(anchor_roi_id, responder_roi_id)
    hops = adjacency.hop_distance_within_roi(responder_roi_id, seeds)

    bands = np.full(adjacency.n_spots, "", dtype=object)
    n_near = n_far = n_unreachable = 0
    for idx in responder_indices:
        h = int(hops[idx])
        if h == UNREACHABLE_HOP:
            n_unreachable += 1
            continue
        if h <= NEAR_HOP_MAX:
            bands[idx] = "near"
            n_near += 1
        elif h >= FAR_HOP_MIN:
            bands[idx] = "far"
            n_far += 1
        # buffer (h == BUFFER_HOP) excluded from the scalar; band left "".

    reachable_fraction = (
        (n_responder - n_unreachable) / n_responder if n_responder else 0.0
    )
    responder_cids = tuple(sorted(
        (cid for cid, cs in support_bank.components.items()
         if cs.parent_roi_id == responder_roi_id),
        key=natural_sort_key,
    ))
    return _PairField(
        bands=bands,
        responder_indices=responder_indices,
        responder_component_ids=responder_cids,
        source_near_mass=float(n_near),
        source_far_mass=float(n_far),
        reachable_fraction=float(reachable_fraction),
    )


# ---------------------------------------------------------------------------
# Shared reducer + per-surface weight+carrier assembly
# ---------------------------------------------------------------------------


def _reduce_weighted_units(
    carriers: np.ndarray,
    near_weights: np.ndarray,
    far_weights: np.ndarray,
) -> _ProjectionResult:
    """Compute (S, represented_near, represented_far, n_near_far_mixed) + status.

    Availability is decided ONLY on the surface's represented mass vs the
    frozen coverage floors. `carriers` may contain NaN entries (a unit with
    an unknown carrier); such units are dropped from BOTH sums together.
    """
    valid = np.isfinite(carriers)
    c = carriers[valid]
    w_n = near_weights[valid]
    w_f = far_weights[valid]
    rep_n = float(w_n.sum())
    rep_f = float(w_f.sum())
    if rep_n < MIN_NEAR_COVERAGE:
        return _ProjectionResult(
            status=STATUS_INSUFFICIENT_NEAR, value=float("nan"),
            represented_near_mass=rep_n, represented_far_mass=rep_f,
            n_near_far_mixed_units=int(((w_n > 0) & (w_f > 0)).sum()),
        )
    if rep_f < MIN_FAR_COVERAGE:
        return _ProjectionResult(
            status=STATUS_INSUFFICIENT_FAR, value=float("nan"),
            represented_near_mass=rep_n, represented_far_mass=rep_f,
            n_near_far_mixed_units=int(((w_n > 0) & (w_f > 0)).sum()),
        )
    value = (float((c * w_n).sum()) / rep_n) - (float((c * w_f).sum()) / rep_f)
    return _ProjectionResult(
        status=STATUS_FINITE, value=value,
        represented_near_mass=rep_n, represented_far_mass=rep_f,
        n_near_far_mixed_units=int(((w_n > 0) & (w_f > 0)).sum()),
    )


def _source_native_projection(
    support: _PairProjectionSupport, responder_values: np.ndarray,
) -> _ProjectionResult:
    carriers = responder_values[support.source_positions].astype(float)
    return _reduce_weighted_units(
        carriers, support.source_near_weights, support.source_far_weights,
    )


def _summarise_replicates(
    per_replicate: Sequence["_ProjectionResult"],
) -> Tuple["_ProjectionResult", int]:
    """Median-scalar over FINITE replicates, coverage / mixed-unit medians
    over ALL planned replicates. Emits `STATUS_NO_VALID_REPLICATES` when no
    replicate is finite (diagnostics still summarise every planned replicate).
    """
    if not per_replicate:
        return (_ProjectionResult(
            status=STATUS_NO_VALID_REPLICATES, value=float("nan"),
            represented_near_mass=0.0, represented_far_mass=0.0,
            n_near_far_mixed_units=0,
        ), 0)
    # Coverage + mixed-unit diagnostics summarise every planned replicate:
    # a partial-coverage replicate's mass is still information about MAUP.
    rep_n = float(np.median([r.represented_near_mass for r in per_replicate]))
    rep_f = float(np.median([r.represented_far_mass for r in per_replicate]))
    mixed = int(np.median([r.n_near_far_mixed_units for r in per_replicate]))
    finite = [r for r in per_replicate if r.status == STATUS_FINITE]
    if not finite:
        return (_ProjectionResult(
            status=STATUS_NO_VALID_REPLICATES, value=float("nan"),
            represented_near_mass=rep_n, represented_far_mass=rep_f,
            n_near_far_mixed_units=mixed,
        ), 0)
    scalars = np.array([r.value for r in finite], dtype=float)
    return (_ProjectionResult(
        status=STATUS_FINITE, value=float(np.median(scalars)),
        represented_near_mass=rep_n, represented_far_mass=rep_f,
        n_near_far_mixed_units=mixed,
    ), len(finite))


def _tessellation_projection(
    *,
    supports: Sequence[_WeightedUnitSupport],
    projection: FeatureSpatialProjection,
) -> Tuple[_ProjectionResult, int]:
    """Median-over-finite-replicates. Returns (result, n_valid_replicates)."""
    per_rep: List[_ProjectionResult] = []
    for support in supports:
        carriers = np.array([
            (
                float(record.sum_g) / float(denominator)
                if (record := projection.tile_marginal_sums.get(key)) is not None
                else np.nan
            )
            for key, denominator in zip(
                support.unit_keys, support.denominators,
            )
        ], dtype=float)
        per_rep.append(_reduce_weighted_units(
            carriers, support.near_weights, support.far_weights,
        ))
    return _summarise_replicates(per_rep)


def _prepare_tessellation_replicate_support(
    *,
    field: _PairField, rep_index: int,
    support_bank: SampleSupportBank,
    spot_id_to_index: Mapping[int, int],
) -> _WeightedUnitSupport:
    """Prepare tile keys and near/far weights for one replicate."""
    keys: List[Tuple] = []
    denominators: List[float] = []
    nears: List[float] = []
    fars: List[float] = []
    for cid in field.responder_component_ids:
        component = support_bank.components[cid]
        source_row_ids = component.source_row_ids
        tessellations = support_bank.tessellation_bank.tessellations.get(
            cid, {},
        )
        for _region_id, replicate_map in tessellations.items():
            tess = replicate_map.get(rep_index)
            if tess is None:
                continue
            for tile in tess.tiles:
                if tile.n_spots <= 0:
                    continue
                positions = tile.point_indices
                w_near = w_far = 0.0
                for p in positions:
                    srid = int(source_row_ids[int(p)])
                    g_idx = spot_id_to_index.get(srid)
                    if g_idx is None:
                        continue
                    b = field.bands[g_idx]
                    if b == "near":
                        w_near += 1.0
                    elif b == "far":
                        w_far += 1.0
                if w_near == 0.0 and w_far == 0.0:
                    continue
                keys.append((
                    cid, int(tile.domain_region_id), int(rep_index),
                    int(tile.tile_id),
                ))
                denominators.append(float(tile.n_spots))
                nears.append(w_near)
                fars.append(w_far)
    return _WeightedUnitSupport(
        unit_keys=tuple(keys),
        denominators=np.asarray(denominators, dtype=float),
        near_weights=np.asarray(nears, dtype=float),
        far_weights=np.asarray(fars, dtype=float),
    )


def _consensus_direct_projection(
    *,
    support: _WeightedUnitSupport,
    projection: FeatureSpatialProjection,
) -> _ProjectionResult:
    carriers = np.array([
        (
            float(record.sum_g_direct) / float(denominator)
            if (record := projection.hex_direct_projection.get(key)) is not None
            else np.nan
        )
        for key, denominator in zip(support.unit_keys, support.denominators)
    ], dtype=float)
    return _reduce_weighted_units(
        carriers, support.near_weights, support.far_weights,
    )


def _prepare_consensus_direct_support(
    *,
    field: _PairField,
    consensus_bank: ConsensusBank,
    spot_id_to_index: Mapping[int, int],
) -> _WeightedUnitSupport:
    # Accumulate per-hex near / far masses across responder components.
    per_hex: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for cid in field.responder_component_ids:
        if cid not in consensus_bank.components:
            continue
        for m in consensus_bank.components[cid].direct_hex_membership:
            if m.hex_id == UNASSIGNED_HEX_ID:
                continue
            g_idx = spot_id_to_index.get(int(m.source_row_id))
            if g_idx is None:
                continue
            b = field.bands[g_idx]
            if b != "near" and b != "far":
                continue
            key = (cid, int(m.hex_id))
            w_n, w_f = per_hex.get(key, (0.0, 0.0))
            if b == "near":
                w_n += 1.0
            else:
                w_f += 1.0
            per_hex[key] = (w_n, w_f)

    return _prepare_hex_support(
        per_hex=per_hex,
        consensus_bank=consensus_bank,
        denominator=lambda geom: float(geom.n_spots_direct),
    )


def _consensus_ensemble_projection(
    *,
    support: _WeightedUnitSupport,
    projection: FeatureSpatialProjection,
) -> _ProjectionResult:
    carriers = np.array([
        (
            float(record.sum_g_ensemble_weighted) / float(denominator)
            if (record := projection.hex_ensemble_projection.get(key)) is not None
            else np.nan
        )
        for key, denominator in zip(support.unit_keys, support.denominators)
    ], dtype=float)
    return _reduce_weighted_units(
        carriers, support.near_weights, support.far_weights,
    )


def _prepare_consensus_ensemble_support(
    *,
    field: _PairField,
    support_bank: SampleSupportBank,
    consensus_bank: ConsensusBank,
    spot_id_to_index: Mapping[int, int],
) -> _WeightedUnitSupport:
    # Per-tile near / far masses feed into hex-level overlap weights;
    # then divide by n_tessellations_covered per hex (OBS_EQUIV_SPOT_MASS).
    per_hex: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for cid in field.responder_component_ids:
        cs = support_bank.components.get(cid)
        if cs is None or cid not in consensus_bank.components:
            continue
        srids = cs.source_row_ids
        overlaps_by_tile: Dict[Tuple[int, int, int], List[TileHexOverlap]] = {}
        for edge in consensus_bank.components[cid].tile_hex_overlaps:
            overlaps_by_tile.setdefault(
                (int(edge.domain_region_id), int(edge.replicate_index),
                 int(edge.tile_id)), [],
            ).append(edge)
        tess_map = support_bank.tessellation_bank.tessellations.get(cid, {})
        for _region_id, rep_map in tess_map.items():
            for rep_idx, tess in rep_map.items():
                for tile in tess.tiles:
                    if tile.n_spots <= 0:
                        continue
                    positions = tile.point_indices
                    tile_near = tile_far = 0.0
                    for p in positions:
                        srid = int(srids[int(p)])
                        g_idx = spot_id_to_index.get(srid)
                        if g_idx is None:
                            continue
                        b = field.bands[g_idx]
                        if b == "near":
                            tile_near += 1.0
                        elif b == "far":
                            tile_far += 1.0
                    if tile_near == 0.0 and tile_far == 0.0:
                        continue
                    edges = overlaps_by_tile.get(
                        (int(tile.domain_region_id), int(rep_idx),
                         int(tile.tile_id)), [],
                    )
                    for edge in edges:
                        frac = float(edge.overlap_area) / float(edge.tile_area)
                        key = (cid, int(edge.hex_id))
                        w_n, w_f = per_hex.get(key, (0.0, 0.0))
                        w_n += tile_near * frac
                        w_f += tile_far * frac
                        per_hex[key] = (w_n, w_f)

    # Normalise projected band mass by n_tessellations_covered per hex.
    normalised: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for (cid, hex_id), (w_n, w_f) in per_hex.items():
        geom = consensus_bank.components[cid].hexes[hex_id]
        n_cov = int(geom.n_tessellations_covered)
        if n_cov <= 0:
            continue
        normalised[(cid, hex_id)] = (w_n / n_cov, w_f / n_cov)

    return _prepare_hex_support(
        per_hex=normalised,
        consensus_bank=consensus_bank,
        denominator=lambda geom: float(geom.n_spots_ensemble_weighted),
    )


def _prepare_hex_support(
    *,
    per_hex: Mapping[Tuple[str, int], Tuple[float, float]],
    consensus_bank: ConsensusBank,
    denominator,
) -> _WeightedUnitSupport:
    """Turn hex band masses into feature-independent aligned support."""
    keys: List[Tuple] = []
    denominators: List[float] = []
    nears: List[float] = []
    fars: List[float] = []
    for (cid, hex_id), (w_n, w_f) in per_hex.items():
        geom = consensus_bank.components[cid].hexes[hex_id]
        unit_denominator = denominator(geom)
        if unit_denominator <= 0:
            continue
        keys.append((cid, hex_id))
        denominators.append(unit_denominator)
        nears.append(w_n)
        fars.append(w_f)
    return _WeightedUnitSupport(
        unit_keys=tuple(keys),
        denominators=np.asarray(denominators, dtype=float),
        near_weights=np.asarray(nears, dtype=float),
        far_weights=np.asarray(fars, dtype=float),
    )


def _prepare_pair_projection_support(
    *,
    field: _PairField,
    support_bank: SampleSupportBank,
    consensus_bank: ConsensusBank,
    spot_id_to_index: Mapping[int, int],
    n_replicates_planned: int,
) -> _PairProjectionSupport:
    """Prepare all feature-independent supports for one ordered edge."""
    source_bands = field.bands[field.responder_indices]
    return _PairProjectionSupport(
        source_positions=field.responder_indices,
        source_near_weights=(source_bands == "near").astype(float),
        source_far_weights=(source_bands == "far").astype(float),
        tessellation_by_replicate=tuple(
            _prepare_tessellation_replicate_support(
                field=field,
                rep_index=replicate_index,
                support_bank=support_bank,
                spot_id_to_index=spot_id_to_index,
            )
            for replicate_index in range(n_replicates_planned)
        ),
        consensus_direct=_prepare_consensus_direct_support(
            field=field,
            consensus_bank=consensus_bank,
            spot_id_to_index=spot_id_to_index,
        ),
        consensus_ensemble=_prepare_consensus_ensemble_support(
            field=field,
            support_bank=support_bank,
            consensus_bank=consensus_bank,
            spot_id_to_index=spot_id_to_index,
        ),
    )


# ---------------------------------------------------------------------------
# Top-level enumeration + row assembly
# ---------------------------------------------------------------------------


def _row_index_by_source_row_id(
    support_bank: SampleSupportBank,
) -> Dict[int, int]:
    """Reverse map source_row_id (int) → positional index in SourceSpotTable."""
    return {int(sid): idx
            for idx, sid in enumerate(support_bank.source_spots.spot_id)}


def _base_row(
    *, sample_id: str, g1_id: str, g2_id: str,
    anchor: str, responder: str, feature_id: str, surface: str,
    status: str, value: float,
    source_near_mass: float, source_far_mass: float,
    represented_near_mass: float, represented_far_mass: float,
    reachable_fraction: float,
    n_near_far_mixed_units: int,
    n_replicates_planned,
    n_valid_replicates,
) -> Dict[str, object]:
    return {
        SAMPLE_ID: sample_id, G1_ID: g1_id, G2_ID: g2_id,
        ANCHOR_ROI_ID: anchor, RESPONDER_ROI_ID: responder,
        RESPONDER_FEATURE_ID: feature_id, SURFACE_COL: surface,
        AVAILABILITY_STATUS: status, INTERFACE_RESPONSE: value,
        SOURCE_NEAR_MASS: source_near_mass,
        SOURCE_FAR_MASS: source_far_mass,
        REPRESENTED_NEAR_MASS: represented_near_mass,
        REPRESENTED_FAR_MASS: represented_far_mass,
        REACHABLE_RESPONDER_FRACTION: reachable_fraction,
        N_NEAR_FAR_MIXED_UNITS: int(n_near_far_mixed_units),
        N_REPLICATES_PLANNED: n_replicates_planned,
        N_VALID_REPLICATES: n_valid_replicates,
    }


def _emit_unavailable_all_surfaces(
    *, sample_id: str, g1_id: str, g2_id: str,
    anchor: str, responder: str, feature_id: str, status: str,
    source_near_mass: float, source_far_mass: float,
    reachable_fraction: float,
    n_replicates_planned: int,
) -> List[Dict[str, object]]:
    rows = []
    for surface in SURFACES:
        planned = n_replicates_planned if surface == SURFACE_TESSELLATION else pd.NA
        valid = 0 if surface == SURFACE_TESSELLATION else pd.NA
        rows.append(_base_row(
            sample_id=sample_id, g1_id=g1_id, g2_id=g2_id,
            anchor=anchor, responder=responder, feature_id=feature_id,
            surface=surface, status=status, value=float("nan"),
            source_near_mass=source_near_mass,
            source_far_mass=source_far_mass,
            represented_near_mass=0.0, represented_far_mass=0.0,
            reachable_fraction=reachable_fraction,
            n_near_far_mixed_units=0,
            n_replicates_planned=planned, n_valid_replicates=valid,
        ))
    return rows


def build_sample_feature_interface_response_df(
    *,
    sample_id: str,
    spatial_bank,  # SampleSpatialBank
    feature_projections: Mapping[str, FeatureSpatialProjection],
    support_status_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute each feature's response once per ordered ROI interface.

    Ordered pairs of distinct biological ROIs are exhaustively enumerated
    from `support_status_df`; infeasible pairs / surfaces emit typed
    unavailability. Source-side and constructed-support weights are
    computed once per ordered pair and reused across all requested features.
    """
    support_bank: SampleSupportBank = spatial_bank.support_bank
    consensus_bank: ConsensusBank = spatial_bank.consensus_bank
    adjacency: SourceSpotAdjacency = support_bank.source_spot_adjacency
    n_replicates_planned = int(
        support_bank.tessellation_bank.n_replicates_planned,
    )
    spot_id_to_index = _row_index_by_source_row_id(support_bank)

    roi_ids = sorted(
        support_status_df[ROI_ID].astype(str).unique(),
        key=natural_sort_key,
    )
    if len(roi_ids) < 2:
        return empty_roi_pair_interface_response_df()

    interfaced = _pairs_with_cross_roi_edges(adjacency)
    feature_axes = tuple(feature_projections.items())
    if not feature_axes:
        return empty_roi_pair_interface_response_df()
    rows: List[Dict[str, object]] = []

    for anchor in roi_ids:
        for responder in roi_ids:
            if anchor == responder:
                continue
            has_interface = (anchor, responder) in interfaced

            if not has_interface:
                for feat_id, _proj in feature_axes:
                    rows.extend(_emit_unavailable_all_surfaces(
                        sample_id=sample_id,
                        g1_id="", g2_id="",
                        anchor=anchor, responder=responder,
                        feature_id=feat_id,
                        status=STATUS_NO_INTERFACE,
                        source_near_mass=0.0, source_far_mass=0.0,
                        reachable_fraction=float("nan"),
                        n_replicates_planned=n_replicates_planned,
                    ))
                continue

            # Source-side field, computed once per ordered pair.
            field = _build_pair_field(
                support_bank=support_bank,
                anchor_roi_id=anchor, responder_roi_id=responder,
            )

            reachable_ok = (
                field.reachable_fraction >= MIN_REACHABLE_RESPONDER_FRACTION
            )
            no_component = len(field.responder_component_ids) == 0
            projection_support = (
                _prepare_pair_projection_support(
                    field=field,
                    support_bank=support_bank,
                    consensus_bank=consensus_bank,
                    spot_id_to_index=spot_id_to_index,
                    n_replicates_planned=n_replicates_planned,
                )
                if reachable_ok else None
            )

            for feat_id, projection in feature_axes:
                if not reachable_ok:
                    rows.extend(_emit_unavailable_all_surfaces(
                        sample_id=sample_id, g1_id="", g2_id="",
                        anchor=anchor, responder=responder,
                        feature_id=feat_id,
                        status=STATUS_INSUFFICIENT_REACHABLE,
                        source_near_mass=field.source_near_mass,
                        source_far_mass=field.source_far_mass,
                        reachable_fraction=field.reachable_fraction,
                        n_replicates_planned=n_replicates_planned,
                    ))
                    continue
                assert projection_support is not None

                # source_native is always attempted when the pair is reachable.
                src_res = _source_native_projection(
                    projection_support, projection.source_values,
                )
                rows.append(_base_row(
                    sample_id=sample_id,
                    g1_id="", g2_id="",
                    anchor=anchor, responder=responder,
                    feature_id=feat_id, surface=SURFACE_SOURCE_NATIVE,
                    status=src_res.status, value=src_res.value,
                    source_near_mass=field.source_near_mass,
                    source_far_mass=field.source_far_mass,
                    represented_near_mass=src_res.represented_near_mass,
                    represented_far_mass=src_res.represented_far_mass,
                    reachable_fraction=field.reachable_fraction,
                    n_near_far_mixed_units=src_res.n_near_far_mixed_units,
                    n_replicates_planned=pd.NA, n_valid_replicates=pd.NA,
                ))

                if no_component:
                    for surface in CONSTRUCTED_SURFACES:
                        planned = (
                            n_replicates_planned
                            if surface == SURFACE_TESSELLATION else pd.NA
                        )
                        valid = (
                            0 if surface == SURFACE_TESSELLATION else pd.NA
                        )
                        rows.append(_base_row(
                            sample_id=sample_id,
                            g1_id="", g2_id="",
                            anchor=anchor, responder=responder,
                            feature_id=feat_id, surface=surface,
                            status=STATUS_NO_ELIGIBLE_COMPONENT,
                            value=float("nan"),
                            source_near_mass=field.source_near_mass,
                            source_far_mass=field.source_far_mass,
                            represented_near_mass=0.0,
                            represented_far_mass=0.0,
                            reachable_fraction=field.reachable_fraction,
                            n_near_far_mixed_units=0,
                            n_replicates_planned=planned,
                            n_valid_replicates=valid,
                        ))
                    continue

                # tessellation
                tess_res, n_valid = _tessellation_projection(
                    supports=projection_support.tessellation_by_replicate,
                    projection=projection,
                )
                rows.append(_base_row(
                    sample_id=sample_id,
                    g1_id="", g2_id="",
                    anchor=anchor, responder=responder,
                    feature_id=feat_id, surface=SURFACE_TESSELLATION,
                    status=tess_res.status, value=tess_res.value,
                    source_near_mass=field.source_near_mass,
                    source_far_mass=field.source_far_mass,
                    represented_near_mass=tess_res.represented_near_mass,
                    represented_far_mass=tess_res.represented_far_mass,
                    reachable_fraction=field.reachable_fraction,
                    n_near_far_mixed_units=tess_res.n_near_far_mixed_units,
                    n_replicates_planned=n_replicates_planned,
                    n_valid_replicates=n_valid,
                ))

                # consensus_direct
                cd_res = _consensus_direct_projection(
                    support=projection_support.consensus_direct,
                    projection=projection,
                )
                rows.append(_base_row(
                    sample_id=sample_id,
                    g1_id="", g2_id="",
                    anchor=anchor, responder=responder,
                    feature_id=feat_id, surface=SURFACE_CONSENSUS_DIRECT,
                    status=cd_res.status, value=cd_res.value,
                    source_near_mass=field.source_near_mass,
                    source_far_mass=field.source_far_mass,
                    represented_near_mass=cd_res.represented_near_mass,
                    represented_far_mass=cd_res.represented_far_mass,
                    reachable_fraction=field.reachable_fraction,
                    n_near_far_mixed_units=cd_res.n_near_far_mixed_units,
                    n_replicates_planned=pd.NA, n_valid_replicates=pd.NA,
                ))

                # consensus_ensemble
                ce_res = _consensus_ensemble_projection(
                    support=projection_support.consensus_ensemble,
                    projection=projection,
                )
                rows.append(_base_row(
                    sample_id=sample_id,
                    g1_id="", g2_id="",
                    anchor=anchor, responder=responder,
                    feature_id=feat_id, surface=SURFACE_CONSENSUS_ENSEMBLE,
                    status=ce_res.status, value=ce_res.value,
                    source_near_mass=field.source_near_mass,
                    source_far_mass=field.source_far_mass,
                    represented_near_mass=ce_res.represented_near_mass,
                    represented_far_mass=ce_res.represented_far_mass,
                    reachable_fraction=field.reachable_fraction,
                    n_near_far_mixed_units=ce_res.n_near_far_mixed_units,
                    n_replicates_planned=pd.NA, n_valid_replicates=pd.NA,
                ))

    if not rows:
        return empty_roi_pair_interface_response_df()
    return _cast_dtypes(
        pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS)),
    )


def select_pair_interface_response_df(
    sample_feature_df: pd.DataFrame,
    *,
    feature_g1_id: str,
    feature_g2_id: str,
) -> pd.DataFrame:
    """Select two precomputed feature lanes and stamp pair provenance."""
    if sample_feature_df.empty:
        return empty_roi_pair_interface_response_df()
    selected = sample_feature_df[
        sample_feature_df[RESPONDER_FEATURE_ID].isin(
            (feature_g1_id, feature_g2_id),
        )
    ].copy()
    selected[G1_ID] = feature_g1_id
    selected[G2_ID] = feature_g2_id
    return _cast_dtypes(selected.loc[:, list(CANONICAL_COLUMNS)])


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    string_cols = (SAMPLE_ID, G1_ID, G2_ID,
                   ANCHOR_ROI_ID, RESPONDER_ROI_ID,
                   RESPONDER_FEATURE_ID, SURFACE_COL, AVAILABILITY_STATUS)
    int_cols = (N_NEAR_FAR_MIXED_UNITS, N_REPLICATES_PLANNED,
                N_VALID_REPLICATES)
    for c in string_cols:
        df[c] = df[c].astype("string")
    for c in int_cols:
        df[c] = df[c].astype("Int64")
    for c in df.columns:
        if c not in string_cols and c not in int_cols:
            df[c] = df[c].astype("Float64")
    return df
