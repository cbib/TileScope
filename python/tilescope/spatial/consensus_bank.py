"""Frozen records for the consensus geometry authority + composed spatial bank.

Feature-independent: hex geometry, tile↔hex overlap weights, source→hex
membership, coverage provenance. Gene-valued fields do not live here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Set, Tuple, Union

from shapely.geometry import MultiPolygon, Polygon

from tilescope.spatial.sample_support_bank import SampleSupportBank


HexKey = Tuple[str, int]                     # (component_id, hex_id)
TileKey = Tuple[str, int, int, int]          # (component_id, region_id, replicate_index, tile_id)

_UNASSIGNED_HEX_ID = -1


@dataclass(frozen=True)
class ConsensusHexGeometry:
    """Feature-independent geometry + occupancy for one consensus hex."""
    component_id: str
    hex_id: int
    domain_region_id: int
    polygon: Union[Polygon, MultiPolygon]
    lattice_col: int
    lattice_row: int
    center_x: float
    center_y: float
    n_tessellations_covered: int
    n_spots_direct: int
    n_spots_ensemble_weighted: float


@dataclass(frozen=True)
class TileHexOverlap:
    """One (tile, hex) overlap edge with area weight."""
    component_id: str
    domain_region_id: int
    replicate_index: int
    tile_id: int
    hex_id: int
    overlap_area: float
    tile_area: float


@dataclass(frozen=True)
class DirectHexMembership:
    """Canonical spot/cell → hex assignment (component-scope); -1 = UNASSIGNED."""
    source_row_id: int
    hex_id: int


@dataclass(frozen=True)
class ComponentConsensusSupport:
    """Per-component consensus geometry bundle."""
    component_id: str
    parent_roi_id: str
    hex_radius: float
    n_tessellation_eligible_regions: int
    hexes: Tuple[ConsensusHexGeometry, ...]
    tile_hex_overlaps: Tuple[TileHexOverlap, ...]
    direct_hex_membership: Tuple[DirectHexMembership, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "hexes", tuple(self.hexes))
        object.__setattr__(self, "tile_hex_overlaps", tuple(self.tile_hex_overlaps))
        object.__setattr__(
            self, "direct_hex_membership", tuple(self.direct_hex_membership),
        )

        canonical_ids: Set[int] = set()
        prev_key = None
        for i, hx in enumerate(self.hexes):
            if hx.component_id != self.component_id:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"hex[{i}].component_id={hx.component_id!r} mismatch."
                )
            if hx.hex_id != i:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"hex[{i}].hex_id={hx.hex_id} must equal canonical position {i}."
                )
            canonical_ids.add(int(hx.hex_id))
            key = (int(hx.domain_region_id), int(hx.lattice_col), int(hx.lattice_row))
            if prev_key is not None and key < prev_key:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): hexes "
                    "not in canonical (domain_region_id, lattice_col, "
                    "lattice_row) order."
                )
            prev_key = key

        seen_edges: Set[Tuple[int, int, int, int]] = set()
        prev_edge = None
        for edge in self.tile_hex_overlaps:
            if edge.component_id != self.component_id:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"overlap.component_id={edge.component_id!r} mismatch."
                )
            if not (math.isfinite(edge.overlap_area) and edge.overlap_area > 0):
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"overlap_area={edge.overlap_area!r} must be finite and positive."
                )
            if not (math.isfinite(edge.tile_area) and edge.tile_area > 0):
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"tile_area={edge.tile_area!r} must be finite and positive."
                )
            if edge.hex_id not in canonical_ids:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"overlap.hex_id={edge.hex_id} references unknown hex."
                )
            key = (
                int(edge.domain_region_id), int(edge.replicate_index),
                int(edge.tile_id), int(edge.hex_id),
            )
            if key in seen_edges:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"duplicate overlap edge {key}."
                )
            seen_edges.add(key)
            if prev_edge is not None and key < prev_edge:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    "tile_hex_overlaps not in canonical order (domain_region_id, "
                    "replicate_index, tile_id, hex_id)."
                )
            prev_edge = key

        seen_srids: Set[int] = set()
        for m in self.direct_hex_membership:
            if m.source_row_id in seen_srids:
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"duplicate direct membership for source_row_id={m.source_row_id}."
                )
            seen_srids.add(int(m.source_row_id))
            if (
                m.hex_id != _UNASSIGNED_HEX_ID
                and m.hex_id not in canonical_ids
            ):
                raise ValueError(
                    f"ComponentConsensusSupport({self.component_id!r}): "
                    f"direct membership hex_id={m.hex_id} is neither UNASSIGNED "
                    "nor a canonical hex."
                )


@dataclass(frozen=True)
class ConsensusBank:
    """Sample-scope consensus geometry authority."""
    sample_id: str
    sample_anchor_x: float
    sample_anchor_y: float
    n_replicates_planned: int
    components: Mapping[str, ComponentConsensusSupport]

    def __post_init__(self) -> None:
        if not isinstance(self.components, MappingProxyType):
            object.__setattr__(
                self, "components", MappingProxyType(dict(self.components)),
            )
        if not math.isfinite(self.sample_anchor_x):
            raise ValueError(
                f"ConsensusBank({self.sample_id!r}): sample_anchor_x must be finite."
            )
        if not math.isfinite(self.sample_anchor_y):
            raise ValueError(
                f"ConsensusBank({self.sample_id!r}): sample_anchor_y must be finite."
            )
        if self.n_replicates_planned <= 0:
            raise ValueError(
                f"ConsensusBank({self.sample_id!r}): n_replicates_planned="
                f"{self.n_replicates_planned} must be positive."
            )
        for cid, support in self.components.items():
            if cid != support.component_id:
                raise ValueError(
                    f"ConsensusBank({self.sample_id!r}): mapping key {cid!r} "
                    f"disagrees with support.component_id={support.component_id!r}."
                )
            for edge in support.tile_hex_overlaps:
                if edge.replicate_index >= self.n_replicates_planned:
                    raise ValueError(
                        f"ConsensusBank({self.sample_id!r}): overlap "
                        f"replicate_index={edge.replicate_index} exceeds "
                        f"n_replicates_planned={self.n_replicates_planned}."
                    )


@dataclass(frozen=True)
class SampleSpatialBank:
    """Sample-scope spatial substrate: support + consensus, composed by reference."""
    support_bank: SampleSupportBank
    consensus_bank: ConsensusBank

    def __post_init__(self) -> None:
        if self.support_bank.sample_id != self.consensus_bank.sample_id:
            raise ValueError(
                "SampleSpatialBank: sample_id mismatch between support_bank "
                f"({self.support_bank.sample_id!r}) and consensus_bank "
                f"({self.consensus_bank.sample_id!r})."
            )
