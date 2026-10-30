"""Frozen records for the per-feature (single-gene) spatial projection.

Marginal-only surface. Joint counts, correlation, CLES, Moran, and joint
enrichment fold changes belong in the pair layer, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np

from tilescope.spatial.consensus_bank import HexKey, TileKey
from tilescope.spatial.errors import FeatureAttachmentError
from tilescope.spatial.sample_support_bank import SampleSupportFingerprint


@dataclass(frozen=True)
class FeatureThresholds:
    """Per-feature high / low thresholds resolved against sample spot/cell values."""
    q_high: float
    q_low: float
    resolved_high: float
    resolved_low: float
    n_source_values_used: int


@dataclass(frozen=True)
class TileMarginalSums:
    """Per-tile per-feature marginal accumulators (occupancy is bank-owned)."""
    sum_g: float
    n_high: float
    n_low: float


@dataclass(frozen=True)
class HexEnsembleMarginalProjection:
    """Per-hex per-feature ensemble-weighted marginals (spot count is bank-owned)."""
    hex_key: HexKey
    sum_g_ensemble_weighted: float
    n_high_ensemble_weighted: float
    n_low_ensemble_weighted: float
    density_ensemble_stability: float


@dataclass(frozen=True)
class HexDirectMarginalProjection:
    """Per-hex per-feature direct marginals (direct spot count is bank-owned)."""
    hex_key: HexKey
    sum_g_direct: float


@dataclass(frozen=True)
class FeatureSpatialProjection:
    """Sample-scope per-feature marginal authority; joint statistics stay in the pair layer."""
    feature_name: str
    bank_fingerprint: SampleSupportFingerprint
    source_values: np.ndarray
    thresholds: FeatureThresholds
    tile_marginal_sums: Mapping[TileKey, TileMarginalSums]
    hex_ensemble_projection: Mapping[HexKey, HexEnsembleMarginalProjection]
    hex_direct_projection: Mapping[HexKey, HexDirectMarginalProjection]

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("FeatureSpatialProjection: feature_name must be non-empty.")

        vals = np.array(self.source_values, dtype=float, copy=True)
        if vals.ndim != 1:
            raise ValueError(
                f"FeatureSpatialProjection({self.feature_name!r}): "
                f"source_values must be 1-D; got shape {vals.shape!r}."
            )
        vals.flags.writeable = False
        object.__setattr__(self, "source_values", vals)

        for name in (
            "tile_marginal_sums", "hex_ensemble_projection", "hex_direct_projection",
        ):
            m = getattr(self, name)
            if not isinstance(m, MappingProxyType):
                object.__setattr__(self, name, MappingProxyType(dict(m)))

        for tk, tms in self.tile_marginal_sums.items():
            for fname, fval in (
                ("sum_g", tms.sum_g), ("n_high", tms.n_high), ("n_low", tms.n_low),
            ):
                if not math.isfinite(fval):
                    raise ValueError(
                        f"FeatureSpatialProjection({self.feature_name!r}): "
                        f"tile {tk} {fname}={fval!r} is not finite."
                    )

        for hk, hep in self.hex_ensemble_projection.items():
            if hep.hex_key != hk:
                raise ValueError(
                    f"FeatureSpatialProjection({self.feature_name!r}): "
                    f"hex_ensemble_projection key {hk!r} disagrees with "
                    f"record.hex_key={hep.hex_key!r}."
                )
            for fname, fval in (
                ("sum_g_ensemble_weighted", hep.sum_g_ensemble_weighted),
                ("n_high_ensemble_weighted", hep.n_high_ensemble_weighted),
                ("n_low_ensemble_weighted", hep.n_low_ensemble_weighted),
            ):
                if not math.isfinite(fval):
                    raise ValueError(
                        f"FeatureSpatialProjection({self.feature_name!r}): "
                        f"hex {hk} {fname}={fval!r} is not finite."
                    )

        for hk, hdp in self.hex_direct_projection.items():
            if hdp.hex_key != hk:
                raise ValueError(
                    f"FeatureSpatialProjection({self.feature_name!r}): "
                    f"hex_direct_projection key {hk!r} disagrees with "
                    f"record.hex_key={hdp.hex_key!r}."
                )
            if not math.isfinite(hdp.sum_g_direct):
                raise ValueError(
                    f"FeatureSpatialProjection({self.feature_name!r}): "
                    f"hex {hk} sum_g_direct={hdp.sum_g_direct!r} is not finite."
                )


def canonical_tile_key_order(keys) -> Tuple[TileKey, ...]:
    """Canonical order: (component_id, domain_region_id, replicate_index, tile_id) ascending."""
    return tuple(sorted(keys))


def canonical_hex_key_order(keys) -> Tuple[HexKey, ...]:
    """Canonical order: (component_id, hex_id) ascending."""
    return tuple(sorted(keys))
