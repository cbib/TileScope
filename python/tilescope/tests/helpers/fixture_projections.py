"""Test-only helper: build FeatureSpatialProjection pairs from `_hex` fixtures.

Legacy hex fixtures declared per-feature sums directly on ConsensusHex.
Phase 3a moves that state to the pair-layer projection. This helper
lets fixture-heavy suites keep their existing `_hex(...)` signatures by
capturing the declared per-feature values at fixture-build time and
replaying them into two FeatureSpatialProjection records at frame-build
time. Not for production use.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, Tuple

import numpy as np

from tilescope.spatial.consensus_bank import HexKey
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
    HexDirectMarginalProjection, HexEnsembleMarginalProjection,
)
from tilescope.spatial.sample_support_bank import SampleSupportFingerprint


_HEX_FEATURES: Dict[Tuple[str, int], Dict[str, float]] = {}


def register_hex_features(
    *,
    component_id: str,
    hex_id: int,
    sum_g1_ens: float = 0.0,
    sum_g2_ens: float = 0.0,
    g1_high: float = 0.0,
    g2_high: float = 0.0,
    g1_low: float = 0.0,
    g2_low: float = 0.0,
    sum_g1_direct: float = 0.0,
    sum_g2_direct: float = 0.0,
    stability_g1: float = float("nan"),
    stability_g2: float = float("nan"),
) -> None:
    _HEX_FEATURES[(str(component_id), int(hex_id))] = {
        "sum_g1_ens": float(sum_g1_ens),
        "sum_g2_ens": float(sum_g2_ens),
        "g1_high": float(g1_high),
        "g2_high": float(g2_high),
        "g1_low": float(g1_low),
        "g2_low": float(g2_low),
        "sum_g1_direct": float(sum_g1_direct),
        "sum_g2_direct": float(sum_g2_direct),
        "stability_g1": float(stability_g1),
        "stability_g2": float(stability_g2),
    }


def _fake_fingerprint(sample_id: str) -> SampleSupportFingerprint:
    return SampleSupportFingerprint(
        sample_id=str(sample_id),
        source_frame_hash="fixture",
        support_policy_version="fixture",
        source_adjacency_config_hash="fixture",
        scattered_support_policy_hash="fixture",
        component_geometry_contract_hash="fixture",
        n_replicates_planned=1,
        base_seed=0,
        tessellation_config_hash="fixture",
    )


def projection_from_hex_features(
    *, sample_consensus_result,
) -> Tuple[FeatureSpatialProjection, FeatureSpatialProjection]:
    """Build the two per-feature projections from the registered hex data."""
    ens_g1: Dict[HexKey, HexEnsembleMarginalProjection] = {}
    ens_g2: Dict[HexKey, HexEnsembleMarginalProjection] = {}
    dir_g1: Dict[HexKey, HexDirectMarginalProjection] = {}
    dir_g2: Dict[HexKey, HexDirectMarginalProjection] = {}
    for cid, component in sample_consensus_result.components.items():
        for hx in component.hexes:
            key: HexKey = (str(cid), int(hx.hex_id))
            data = _HEX_FEATURES.get(key, {})
            ens_g1[key] = HexEnsembleMarginalProjection(
                hex_key=key,
                sum_g_ensemble_weighted=data.get("sum_g1_ens", 0.0),
                n_high_ensemble_weighted=data.get("g1_high", 0.0),
                n_low_ensemble_weighted=data.get("g1_low", 0.0),
                density_ensemble_stability=data.get("stability_g1", float("nan")),
            )
            ens_g2[key] = HexEnsembleMarginalProjection(
                hex_key=key,
                sum_g_ensemble_weighted=data.get("sum_g2_ens", 0.0),
                n_high_ensemble_weighted=data.get("g2_high", 0.0),
                n_low_ensemble_weighted=data.get("g2_low", 0.0),
                density_ensemble_stability=data.get("stability_g2", float("nan")),
            )
            dir_g1[key] = HexDirectMarginalProjection(
                hex_key=key, sum_g_direct=data.get("sum_g1_direct", 0.0),
            )
            dir_g2[key] = HexDirectMarginalProjection(
                hex_key=key, sum_g_direct=data.get("sum_g2_direct", 0.0),
            )
    fp = _fake_fingerprint(sample_consensus_result.sample_id)
    thresholds = FeatureThresholds(
        q_high=0.9, q_low=0.1,
        resolved_high=0.0, resolved_low=0.0, n_source_values_used=0,
    )
    p_g1 = FeatureSpatialProjection(
        feature_name="gene1", bank_fingerprint=fp,
        source_values=np.zeros(0),
        thresholds=thresholds,
        tile_marginal_sums=MappingProxyType({}),
        hex_ensemble_projection=MappingProxyType(ens_g1),
        hex_direct_projection=MappingProxyType(dir_g1),
    )
    p_g2 = FeatureSpatialProjection(
        feature_name="gene2", bank_fingerprint=fp,
        source_values=np.zeros(0),
        thresholds=thresholds,
        tile_marginal_sums=MappingProxyType({}),
        hex_ensemble_projection=MappingProxyType(ens_g2),
        hex_direct_projection=MappingProxyType(dir_g2),
    )
    return p_g1, p_g2
