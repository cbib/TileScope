"""Per-feature spatial projection builder.

Runs after the sample spatial bank exists; consumes only bank-owned
substrate (component supports, tessellation ensemble, consensus
edges + direct membership) plus the sample-wide `source_values`
array for one feature. Joint / bivariate statistics live in the
pair layer, not here.
"""

from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType
from typing import Dict, List, Tuple

import numpy as np

from tilescope.spatial.consensus_bank import HexKey, TileKey
from tilescope.spatial.errors import FeatureAttachmentError
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
    HexDirectMarginalProjection, HexEnsembleMarginalProjection,
    TileMarginalSums,
)
from tilescope.spatial.consensus_bank import SampleSpatialBank

_UNASSIGNED_HEX_ID = -1


def _density_stability(densities: List[float]) -> float:
    """SD/IQR: scale-free variability across per-tess densities.

    Matches the pair-layer stability used before the projection split;
    returns NaN if fewer than two samples or IQR == 0.
    """
    if len(densities) < 2:
        return float("nan")
    std = float(np.std(densities, ddof=1))
    q75, q25 = np.percentile(densities, [75, 25])
    iqr = float(q75 - q25)
    return float(std / iqr) if iqr > 0 else float("nan")


def build_feature_spatial_projection(
    *,
    feature_name: str,
    source_values: np.ndarray,
    thresholds: FeatureThresholds,
    spatial_bank: SampleSpatialBank,
) -> FeatureSpatialProjection:
    """Assemble marginal projection over the spatial bank for one feature."""
    support_bank = spatial_bank.support_bank
    consensus_bank = spatial_bank.consensus_bank

    vals = np.asarray(source_values, dtype=float)
    if vals.ndim != 1:
        raise FeatureAttachmentError(
            f"build_feature_spatial_projection({feature_name!r}): "
            f"source_values must be 1-D; got shape {vals.shape!r}."
        )
    expected_n = support_bank.source_spots.n_spots
    if vals.shape[0] != expected_n:
        raise FeatureAttachmentError(
            f"build_feature_spatial_projection({feature_name!r}): "
            f"source_values length {vals.shape[0]} != SourceSpotTable.n_spots "
            f"{expected_n}."
        )

    tile_marginal_sums: Dict[TileKey, TileMarginalSums] = {}
    hex_ensemble: Dict[HexKey, HexEnsembleMarginalProjection] = {}
    hex_direct: Dict[HexKey, HexDirectMarginalProjection] = {}

    tess_by_cid = support_bank.tessellation_bank.tessellations

    for cid in sorted(support_bank.components.keys()):
        comp_support = support_bank.components[cid]
        # `source_values` is sample-scope (indexed by canonical SOURCE_ROW_ID).
        # `point_indices` on Tile is component-local. Component-local values:
        comp_values = vals[comp_support.source_row_ids]

        tile_marginals_local: Dict[Tuple[int, int, int], TileMarginalSums] = {}
        tile_n_spots_local: Dict[Tuple[int, int, int], int] = {}
        regions_map = tess_by_cid.get(cid, {})
        for region_id in sorted(regions_map.keys()):
            rep_map = regions_map[region_id]
            for rep_idx in sorted(rep_map.keys()):
                tess = rep_map[rep_idx]
                for tile in tess.tiles:
                    tile_vals = comp_values[tile.point_indices]
                    sum_g = float(tile_vals.sum()) if tile_vals.size else 0.0
                    if tile_vals.size:
                        n_high = float((tile_vals > thresholds.resolved_high).sum())
                        n_low = float((tile_vals < thresholds.resolved_low).sum())
                    else:
                        n_high = 0.0
                        n_low = 0.0
                    tms = TileMarginalSums(
                        sum_g=sum_g, n_high=n_high, n_low=n_low,
                    )
                    tk: TileKey = (
                        cid, int(region_id), int(rep_idx), int(tile.tile_id),
                    )
                    tile_marginal_sums[tk] = tms
                    tile_marginals_local[
                        (int(region_id), int(rep_idx), int(tile.tile_id))
                    ] = tms
                    tile_n_spots_local[
                        (int(region_id), int(rep_idx), int(tile.tile_id))
                    ] = int(tile.n_spots)

        comp_consensus = consensus_bank.components.get(cid)
        if comp_consensus is None:
            continue

        hex_ensemble_acc: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"sum_g": 0.0, "n_high": 0.0, "n_low": 0.0},
        )
        per_tess_by_hex: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"sum_g": 0.0, "n_spots": 0.0}),
        )
        for edge in comp_consensus.tile_hex_overlaps:
            weight = edge.overlap_area / edge.tile_area
            local_key = (
                int(edge.domain_region_id), int(edge.replicate_index),
                int(edge.tile_id),
            )
            tm = tile_marginals_local[local_key]
            hex_ensemble_acc[edge.hex_id]["sum_g"] += weight * tm.sum_g
            hex_ensemble_acc[edge.hex_id]["n_high"] += weight * tm.n_high
            hex_ensemble_acc[edge.hex_id]["n_low"] += weight * tm.n_low
            per_tess = per_tess_by_hex[edge.hex_id][int(edge.replicate_index)]
            per_tess["sum_g"] += weight * tm.sum_g
            per_tess["n_spots"] += weight * tile_n_spots_local[local_key]

        for hx in comp_consensus.hexes:
            hex_id = int(hx.hex_id)
            hk: HexKey = (cid, hex_id)
            acc = hex_ensemble_acc.get(
                hex_id, {"sum_g": 0.0, "n_high": 0.0, "n_low": 0.0},
            )
            per_tess_map = per_tess_by_hex.get(hex_id, {})
            densities = [
                (s["sum_g"] / s["n_spots"]) if s["n_spots"] > 0 else float("nan")
                for _, s in sorted(per_tess_map.items())
            ]
            hex_ensemble[hk] = HexEnsembleMarginalProjection(
                hex_key=hk,
                sum_g_ensemble_weighted=float(acc["sum_g"]),
                n_high_ensemble_weighted=float(acc["n_high"]),
                n_low_ensemble_weighted=float(acc["n_low"]),
                density_ensemble_stability=_density_stability(densities),
            )

        src_to_local: Dict[int, int] = {
            int(sid): i for i, sid in enumerate(comp_support.source_row_ids)
        }
        hex_direct_sums: Dict[int, float] = defaultdict(float)
        for m in comp_consensus.direct_hex_membership:
            if m.hex_id == _UNASSIGNED_HEX_ID:
                continue
            local = src_to_local[int(m.source_row_id)]
            hex_direct_sums[int(m.hex_id)] += float(comp_values[local])
        for hx in comp_consensus.hexes:
            hex_id = int(hx.hex_id)
            hk = (cid, hex_id)
            hex_direct[hk] = HexDirectMarginalProjection(
                hex_key=hk,
                sum_g_direct=float(hex_direct_sums.get(hex_id, 0.0)),
            )

    return FeatureSpatialProjection(
        feature_name=feature_name,
        bank_fingerprint=support_bank.fingerprint,
        source_values=vals,
        thresholds=thresholds,
        tile_marginal_sums=MappingProxyType(
            dict(sorted(tile_marginal_sums.items())),
        ),
        hex_ensemble_projection=MappingProxyType(
            dict(sorted(hex_ensemble.items())),
        ),
        hex_direct_projection=MappingProxyType(
            dict(sorted(hex_direct.items())),
        ),
    )
