"""Frozen records + fingerprint helpers for the sample-scope support bank.

The builder is in `tilescope.analyzers.sample_support_stage`;
consensus fields are intentionally absent so consensus geometry
changes reuse this bank and rebuild only consensus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple, Union

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from tilescope.component_data import Tessellation
from tilescope.defaults import SEED_COUNT_FORMULA, SUPPORT_POLICY_VERSION
from tilescope.spatial.component_domains import ComponentDomainPartition
from tilescope.spatial.source_spot_adjacency import SourceSpotAdjacency
from tilescope.spatial.source_spot_table import (
    SourceSpotTable, hash_source_frame_rows,
)


SPATIAL_BANK_CONTRACT_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSupportFingerprint:
    """Cache key for `SampleSupportBank` reuse; support-affecting fields only."""
    sample_id: str
    source_frame_hash: str
    support_policy_version: str
    source_adjacency_config_hash: str
    scattered_support_policy_hash: str
    component_geometry_contract_hash: str
    n_replicates_planned: int
    base_seed: int
    tessellation_config_hash: str
    spatial_bank_contract_version: str = SPATIAL_BANK_CONTRACT_VERSION


@dataclass(frozen=True)
class ComponentSupport:
    """Spatial-only component slice: coords + source_row_ids + polygon (arrays defensively frozen)."""
    component_id: str
    parent_roi_id: str
    points: np.ndarray
    source_row_ids: np.ndarray
    polygon: Union[Polygon, MultiPolygon]

    def __post_init__(self) -> None:
        pts = np.array(self.points, dtype=float, copy=True)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"ComponentSupport.points must be (N, 2); got {pts.shape!r}."
            )
        pts.flags.writeable = False
        object.__setattr__(self, "points", pts)

        ids = np.array(self.source_row_ids, dtype=np.int64, copy=True)
        if ids.ndim != 1 or ids.shape[0] != pts.shape[0]:
            raise ValueError(
                f"ComponentSupport.source_row_ids length ({ids.shape}) "
                f"must match points ({pts.shape})."
            )
        ids.flags.writeable = False
        object.__setattr__(self, "source_row_ids", ids)


@dataclass(frozen=True)
class TessellationBank:
    """Feature-independent tessellation ensemble; mappings self-wrap to MappingProxyType."""
    n_replicates_planned: int
    base_seed: int
    tessellations: Mapping[str, Mapping[int, Mapping[int, Tessellation]]]
    region_diagnostics: Mapping[str, Tuple]
    region_replicate_diagnostics: Mapping[str, Tuple]

    def __post_init__(self) -> None:
        if not isinstance(self.tessellations, MappingProxyType):
            wrapped_outer: dict = {}
            for cid, region_map in self.tessellations.items():
                wrapped_inner: dict = {}
                for rid, rep_map in region_map.items():
                    wrapped_inner[rid] = (
                        rep_map if isinstance(rep_map, MappingProxyType)
                        else MappingProxyType(dict(rep_map))
                    )
                wrapped_outer[cid] = (
                    region_map if isinstance(region_map, MappingProxyType)
                    else MappingProxyType(wrapped_inner)
                )
            object.__setattr__(
                self, "tessellations",
                MappingProxyType(wrapped_outer),
            )
        if not isinstance(self.region_diagnostics, MappingProxyType):
            object.__setattr__(
                self, "region_diagnostics",
                MappingProxyType(dict(self.region_diagnostics)),
            )
        if not isinstance(self.region_replicate_diagnostics, MappingProxyType):
            object.__setattr__(
                self, "region_replicate_diagnostics",
                MappingProxyType(dict(self.region_replicate_diagnostics)),
            )


@dataclass(frozen=True)
class SampleSupportBank:
    """Single authority for the sample-scope, feature-independent spatial substrate."""
    sample_id: str
    fingerprint: SampleSupportFingerprint
    source_spots: SourceSpotTable
    source_spot_adjacency: SourceSpotAdjacency
    components: Mapping[str, ComponentSupport]
    component_domains: Mapping[str, ComponentDomainPartition]
    tessellation_bank: TessellationBank

    def __post_init__(self) -> None:
        if self.source_spot_adjacency.source_spot_table is not self.source_spots:
            raise ValueError(
                "SampleSupportBank.source_spot_adjacency must be built from "
                "the same SourceSpotTable held in `source_spots`."
            )
        if not isinstance(self.components, MappingProxyType):
            object.__setattr__(
                self, "components", MappingProxyType(dict(self.components)),
            )
        if not isinstance(self.component_domains, MappingProxyType):
            object.__setattr__(
                self, "component_domains",
                MappingProxyType(dict(self.component_domains)),
            )


# ---------------------------------------------------------------------------
# Canonical-hash helpers
# ---------------------------------------------------------------------------


def _canonical_json_hash(payload) -> str:
    """SHA-256 of `json.dumps(payload, sort_keys=True, separators=(",", ":"),
    default=str)` UTF-8 bytes."""
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_source_adjacency_config_hash(config) -> str:
    """SHA-256 of technology plus its resolved graph policy."""
    return _canonical_json_hash({
        "technology": config.run_config.technology,
        "policy": config.resolved_policy.source_adjacency.model_dump(),
    })


def build_scattered_support_policy_hash(config) -> str:
    """SHA-256 of the scattered / sample-support policy inputs."""
    return _canonical_json_hash({
        "min_component_observations": (
            config.run_config.tessellation.min_component_observations
        ),
    })


def build_component_geometry_contract_hash(config) -> str:
    """SHA-256 of the three bounded_spot_union radius factors."""
    radii = config.resolved_policy.component_domain_radii
    return _canonical_json_hash({
        "name": "bounded_spot_union",
        "version": "v1",
        "params": {
            "component_support_radius_factor": (
                radii.component_support_radius_factor
            ),
            "acquisition_support_radius_factor": (
                radii.acquisition_support_radius_factor
            ),
            "ineligible_spot_exclusion_radius_factor": (
                radii.ineligible_spot_exclusion_radius_factor
            ),
        },
    })


def build_tessellation_config_hash(config) -> str:
    """SHA-256 of tessellation ensemble and observation-scale parameters."""
    tessellation = config.run_config.tessellation
    return _canonical_json_hash({
        "n": tessellation.n,
        "target_observations_per_tile": tessellation.target_observations_per_tile,
        "resolved_voronoi_fraction": tessellation.voronoi_fraction,
        "min_observations_per_tile": tessellation.min_observations_per_tile,
        "min_component_observations": tessellation.min_component_observations,
        "tile_value_aggregation": config.resolved_policy.tile_value_aggregation,
        "seed_count_formula": SEED_COUNT_FORMULA,
        "support_policy_version": SUPPORT_POLICY_VERSION,
    })


def build_sample_support_fingerprint(
    *,
    sample_id: str,
    source_spot_table: SourceSpotTable,
    config,
    n_replicates: int,
    base_seed: int,
) -> SampleSupportFingerprint:
    """Assemble the fingerprint. `config` is required in production."""
    return SampleSupportFingerprint(
        sample_id=str(sample_id),
        source_frame_hash=hash_source_frame_rows(source_spot_table),
        support_policy_version=SUPPORT_POLICY_VERSION,
        source_adjacency_config_hash=(
            build_source_adjacency_config_hash(config)
        ),
        scattered_support_policy_hash=build_scattered_support_policy_hash(config),
        component_geometry_contract_hash=(
            build_component_geometry_contract_hash(config)
        ),
        n_replicates_planned=n_replicates,
        base_seed=base_seed,
        tessellation_config_hash=build_tessellation_config_hash(config),
    )
