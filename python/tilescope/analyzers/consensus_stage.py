"""Typed consensus stage.

`run_consensus_stage` consumes a `SampleTessellationResult` and emits
a `SampleConsensusResult` with the sample-anchored hexagonal lattice,
per-hex ensemble + direct sufficient stats, and (when thresholds are
supplied) per-hex HIGH / LOW / BOTH ensemble-weighted counters.
Scientific statistics at ROI grain live in the ROI-grain analyzers
downstream.
"""
from __future__ import annotations

from tilescope.tessellation.geometry import assign_points_to_first_covering_polygon
from tilescope.cols import N_BOTH_HIGH as COL_N_BOTH_HIGH, N_BOTH_LOW as COL_N_BOTH_LOW
from shapely.ops import unary_union


from dataclasses import dataclass, field, replace as dataclass_replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.prepared import prep

from tilescope.analyzers.consensus_grid import (
    _aggregate_hex_from_ensemble,
    _aggregate_per_tess_stats,
    _clip_hex_to_roi,
    _generate_hex_grid,
    compute_optimal_hex_radius,
)
from tilescope.analyzers.tessellation_analysis import (
    SampleTessellationResult,
)
from tilescope.cols import (
    COMPONENT_ID, DOMAIN_REGION_ID,
    N_BOTH_HIGH, N_BOTH_LOW, N_G1_HIGH, N_G1_LOW,
    N_G2_HIGH, N_G2_LOW, N_SPOTS, SOURCE_ROW_ID, SUM_G1, SUM_G2,
    TESSELLATION_ID, TILE_ID,
)
from tilescope.component_data import ComponentData, Tessellation
from tilescope.spatial import Canvas
from tilescope.spatial.component_domains import ComponentDomainPartition
from tilescope.spatial.consensus_bank import (
    ComponentConsensusSupport, ConsensusBank, ConsensusHexGeometry,
    DirectHexMembership, SampleSpatialBank, TileHexOverlap,
)
from tilescope.spatial.sample_support_bank import SampleSupportBank
from tilescope.analyzers.consensus_grid import OverlapIndex

if TYPE_CHECKING:
    from tilescope.analyzers.tessellation_outcomes import (
        RegionTessellationEligibilityDiagnostic,
    )
    from tilescope.metrics.enrichment_thresholds import PercentileThresholds

# Absolute tolerance for coherence invariants on ensemble-weighted
# accumulators; absorbs overlap-weighted rounding.
_MASS_COHERENCE_TOL = 1e-9


@dataclass(frozen=True)
class ConsensusHex:
    """Bank geometry plus pair-owned joint counts for one hex."""
    geometry: ConsensusHexGeometry
    # Joint (bivariate) ensemble-weighted counts. All-or-none per hex;
    # populated when run_consensus_stage runs with thresholds.
    n_both_high_ensemble_weighted: Optional[float] = None
    n_both_low_ensemble_weighted: Optional[float] = None

    @classmethod
    def build(
        cls, *,
        component_id: str,
        hex_id: int, domain_region_id: int,
        polygon: Union[Polygon, MultiPolygon],
        lattice_col: int, lattice_row: int,
        center_x: float, center_y: float,
        n_tessellations_covered: int,
        n_spots_ensemble_weighted: float,
        n_spots_direct: int,
        n_both_high_ensemble_weighted: Optional[float] = None,
        n_both_low_ensemble_weighted: Optional[float] = None,
    ) -> "ConsensusHex":
        """Build a hex and its matching bank geometry; identity holds by construction."""
        geometry = ConsensusHexGeometry(
            component_id=component_id, hex_id=hex_id,
            domain_region_id=domain_region_id, polygon=polygon,
            lattice_col=lattice_col, lattice_row=lattice_row,
            center_x=center_x, center_y=center_y,
            n_tessellations_covered=n_tessellations_covered,
            n_spots_direct=n_spots_direct,
            n_spots_ensemble_weighted=n_spots_ensemble_weighted,
        )
        return cls(
            geometry=geometry,
            n_both_high_ensemble_weighted=n_both_high_ensemble_weighted,
            n_both_low_ensemble_weighted=n_both_low_ensemble_weighted,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ConsensusHexGeometry):
            raise TypeError("ConsensusHex.geometry must be ConsensusHexGeometry.")
        if self.n_tessellations_covered <= 0:
            raise ValueError(
                f"ConsensusHex(hex_id={self.hex_id}) has "
                f"n_tessellations_covered={self.n_tessellations_covered}; "
                "a retained hex must have positive exposure."
            )
        if self.n_spots_ensemble_weighted < -_MASS_COHERENCE_TOL:
            raise ValueError(
                f"ConsensusHex(hex_id={self.hex_id}) has negative "
                f"n_spots_ensemble_weighted={self.n_spots_ensemble_weighted}"
            )
        # Joint fields are all-or-none.
        both_fields = (
            ("n_both_high_ensemble_weighted", self.n_both_high_ensemble_weighted),
            ("n_both_low_ensemble_weighted", self.n_both_low_ensemble_weighted),
        )
        populated = [n for n, v in both_fields if v is not None]
        if 0 < len(populated) < 2:
            missing = [n for n, v in both_fields if v is None]
            raise ValueError(
                f"ConsensusHex(hex_id={self.hex_id}): joint accumulators "
                f"must be all-present or all-absent; populated={populated!r} "
                f"missing={missing!r}"
            )
        spot = float(self.n_spots_ensemble_weighted)
        for name, val in both_fields:
            if val is None:
                continue
            if val < -_MASS_COHERENCE_TOL:
                raise ValueError(
                    f"ConsensusHex(hex_id={self.hex_id}): {name}={val} is negative"
                )
            if val > spot + _MASS_COHERENCE_TOL:
                raise ValueError(
                    f"ConsensusHex(hex_id={self.hex_id}): {name}={val} > "
                    f"n_spots_ensemble_weighted={spot}"
                )

    @property
    def hex_id(self) -> int:
        return self.geometry.hex_id

    @property
    def domain_region_id(self) -> int:
        return self.geometry.domain_region_id

    @property
    def polygon(self) -> Union[Polygon, MultiPolygon]:
        return self.geometry.polygon

    @property
    def lattice_col(self) -> int:
        return self.geometry.lattice_col

    @property
    def lattice_row(self) -> int:
        return self.geometry.lattice_row

    @property
    def center_x(self) -> float:
        return self.geometry.center_x

    @property
    def center_y(self) -> float:
        return self.geometry.center_y

    @property
    def n_tessellations_covered(self) -> int:
        return self.geometry.n_tessellations_covered

    @property
    def n_spots_ensemble_weighted(self) -> float:
        return self.geometry.n_spots_ensemble_weighted

    @property
    def n_spots_direct(self) -> int:
        return self.geometry.n_spots_direct

    @property
    def hex_area(self) -> float:
        return float(self.polygon.area)

    @property
    def obs_equiv_spot_mass(self) -> float:
        return self.n_spots_ensemble_weighted / self.n_tessellations_covered

    @property
    def obs_equiv_both_high_mass(self) -> Optional[float]:
        if self.n_both_high_ensemble_weighted is None:
            return None
        return self.n_both_high_ensemble_weighted / self.n_tessellations_covered

    @property
    def obs_equiv_both_low_mass(self) -> Optional[float]:
        if self.n_both_low_ensemble_weighted is None:
            return None
        return self.n_both_low_ensemble_weighted / self.n_tessellations_covered


@dataclass(frozen=True)
class ComponentConsensusResult:
    """Per-component consensus outcome: primary records only.

    Scientific statistics live at ROI grain in the parent-ROI consensus
    modules; this record carries geometry, per-hex sufficient stats,
    and identity.
    """
    parent_roi_id: str
    component_id: str
    hex_radius: float
    hexes: Tuple[ConsensusHex, ...]
    n_tessellation_eligible_regions: int
    # Each hex.geometry must be `is`-equal to the support's hex at the
    # same canonical position.
    support: ComponentConsensusSupport = field(kw_only=True)

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError(
                "ComponentConsensusResult: component_id must be non-empty."
            )
        if self.n_tessellation_eligible_regions < 0:
            raise ValueError(
                "ComponentConsensusResult: "
                "n_tessellation_eligible_regions must be non-negative."
            )
        if self.n_tessellation_eligible_regions == 0 and self.hexes:
            raise ValueError(
                "ComponentConsensusResult: a component with no "
                "tessellation-eligible region cannot carry consensus hexes."
            )
        if self.support is None:
            raise ValueError(
                f"ComponentConsensusResult({self.component_id!r}): "
                "support is required; use ComponentConsensusResult.from_hexes(...)."
            )
        if len(self.support.hexes) != len(self.hexes):
            raise ValueError(
                f"ComponentConsensusResult({self.component_id!r}): "
                f"support has {len(self.support.hexes)} geometries but "
                f"view has {len(self.hexes)} hexes."
            )
        for i, hex_view in enumerate(self.hexes):
            if hex_view.geometry is not self.support.hexes[i]:
                raise ValueError(
                    f"ComponentConsensusResult({self.component_id!r}): "
                    f"hex[{i}].geometry is not `is`-identical to the "
                    "support's geometry at the same position."
                )

    @classmethod
    def from_hexes(
        cls, *,
        parent_roi_id: str,
        component_id: str,
        hex_radius: float,
        hexes: Tuple[ConsensusHex, ...],
        n_tessellation_eligible_regions: int,
        tile_hex_overlaps: Tuple[TileHexOverlap, ...] = (),
        direct_hex_membership: Tuple[DirectHexMembership, ...] = (),
    ) -> "ComponentConsensusResult":
        """Derive support from hexes' bank-owned geometries."""
        support = ComponentConsensusSupport(
            component_id=component_id,
            parent_roi_id=parent_roi_id,
            hex_radius=float(hex_radius),
            n_tessellation_eligible_regions=n_tessellation_eligible_regions,
            hexes=tuple(h.geometry for h in hexes),
            tile_hex_overlaps=tuple(tile_hex_overlaps),
            direct_hex_membership=tuple(direct_hex_membership),
        )
        return cls(
            parent_roi_id=parent_roi_id,
            component_id=component_id,
            hex_radius=float(hex_radius),
            hexes=tuple(hexes),
            n_tessellation_eligible_regions=n_tessellation_eligible_regions,
            support=support,
        )

    @property
    def n_hexes(self) -> int:
        return len(self.hexes)


@dataclass(frozen=True)
class SampleConsensusResult:
    """Sample-wide typed consensus result.

    `sample_anchor_(x|y)` are the source-frame minimum coordinates
    before any component filtering: the sample-level lattice origin.
    """
    sample_id: str
    components: Mapping[str, ComponentConsensusResult]
    n_replicates_planned: int
    sample_anchor_x: float
    sample_anchor_y: float
    # consensus_bank component supports must be `is`-identical to the
    # view's per-component supports.
    spatial_bank: SampleSpatialBank = field(kw_only=True)

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_anchor_x):
            raise ValueError(
                f"SampleConsensusResult: sample_anchor_x must be finite, "
                f"got {self.sample_anchor_x!r}."
            )
        if not np.isfinite(self.sample_anchor_y):
            raise ValueError(
                f"SampleConsensusResult: sample_anchor_y must be finite, "
                f"got {self.sample_anchor_y!r}."
            )
        for cid, result in self.components.items():
            if cid != result.component_id:
                raise ValueError(
                    f"SampleConsensusResult: mapping key {cid!r} "
                    f"disagrees with component_id={result.component_id!r}."
                )
        object.__setattr__(
            self, "components", MappingProxyType(dict(self.components)),
        )
        if self.spatial_bank is None:
            raise ValueError(
                f"SampleConsensusResult({self.sample_id!r}): spatial_bank "
                "is required; use SampleConsensusResult.from_components(...)."
            )
        cb = self.spatial_bank.consensus_bank
        if cb.sample_id != self.sample_id:
            raise ValueError(
                f"SampleConsensusResult: sample_id={self.sample_id!r} "
                f"disagrees with consensus_bank ({cb.sample_id!r})."
            )
        if set(cb.components.keys()) != set(self.components.keys()):
            raise ValueError(
                "SampleConsensusResult: component key sets disagree "
                "between view and consensus_bank."
            )
        for cid, result in self.components.items():
            if result.support is not cb.components.get(cid):
                raise ValueError(
                    f"SampleConsensusResult: component {cid!r} support "
                    "is not `is`-identical to the bank's copy."
                )

    @classmethod
    def from_components(
        cls, *,
        sample_id: str,
        components: Mapping[str, ComponentConsensusResult],
        n_replicates_planned: int,
        sample_anchor_x: float,
        sample_anchor_y: float,
        support_bank: SampleSupportBank,
    ) -> "SampleConsensusResult":
        """Compose SampleSpatialBank around a shared support_bank."""
        consensus_bank = ConsensusBank(
            sample_id=str(sample_id),
            sample_anchor_x=float(sample_anchor_x),
            sample_anchor_y=float(sample_anchor_y),
            n_replicates_planned=int(n_replicates_planned),
            components={
                cid: cr.support for cid, cr in components.items()
            },
        )
        spatial_bank = SampleSpatialBank(
            support_bank=support_bank, consensus_bank=consensus_bank,
        )
        return cls(
            sample_id=str(sample_id),
            components=components,
            n_replicates_planned=int(n_replicates_planned),
            sample_anchor_x=float(sample_anchor_x),
            sample_anchor_y=float(sample_anchor_y),
            spatial_bank=spatial_bank,
        )


def run_consensus_stage(
    *,
    sample_tessellation_result: SampleTessellationResult,
    canvas: Canvas,
    sample_anchor: Tuple[float, float],
    consensus_config,
    thresholds: Optional["PercentileThresholds"] = None,
    tessellation_tiles_df: Optional[pd.DataFrame] = None,
) -> SampleConsensusResult:
    """Run the sample-anchored consensus stage.

    `sample_anchor = (anchor_x, anchor_y)` must be the source-frame
    minimum coordinates, both finite.
    """
    if not isinstance(sample_anchor, (tuple, list)) or len(sample_anchor) != 2:
        raise ValueError(
            f"run_consensus_stage: sample_anchor must be a 2-tuple, "
            f"got {sample_anchor!r}."
        )
    ax, ay = float(sample_anchor[0]), float(sample_anchor[1])
    if not (np.isfinite(ax) and np.isfinite(ay)):
        raise ValueError(
            f"run_consensus_stage: sample_anchor components must be "
            f"finite; got ({sample_anchor[0]!r}, {sample_anchor[1]!r})."
        )
    if thresholds is not None and tessellation_tiles_df is None:
        raise ValueError(
            "run_consensus_stage requires tessellation_tiles_df when "
            "thresholds are supplied; consensus must consume the canonical "
            "per-tile sufficient statistics instead of recomputing them."
        )

    components_out: Dict[str, ComponentConsensusResult] = {}
    for cid in sorted(sample_tessellation_result.components):
        comp_result = sample_tessellation_result.components[cid]
        components_out[cid] = _run_component(
            parent_roi_id=comp_result.parent_roi_id,
            component_id=cid,
            component_data=comp_result.component_data,
            eligible_region_diagnostics=(
                comp_result.tessellation_eligible_region_diagnostics
            ),
            tessellations_by_region=comp_result.tessellations,
            domain_partition=comp_result.domain_partition,
            canvas=canvas,
            anchor_x=ax,
            anchor_y=ay,
            consensus_config=consensus_config,
            thresholds=thresholds,
            tessellation_tiles_df=tessellation_tiles_df,
        )

    consensus_bank = ConsensusBank(
        sample_id=sample_tessellation_result.sample_id,
        sample_anchor_x=ax, sample_anchor_y=ay,
        n_replicates_planned=sample_tessellation_result.n_replicates_planned,
        components={cid: cr.support for cid, cr in components_out.items()},
    )
    spatial_bank = SampleSpatialBank(
        support_bank=sample_tessellation_result.support_bank,
        consensus_bank=consensus_bank,
    )
    return SampleConsensusResult(
        sample_id=sample_tessellation_result.sample_id,
        components=components_out,
        n_replicates_planned=sample_tessellation_result.n_replicates_planned,
        sample_anchor_x=ax,
        sample_anchor_y=ay,
        spatial_bank=spatial_bank,
    )


def _run_component(
    *,
    parent_roi_id: str,
    component_id: str,
    component_data: ComponentData,
    eligible_region_diagnostics: Tuple[
        RegionTessellationEligibilityDiagnostic, ...
    ],
    tessellations_by_region: Mapping[int, Mapping[int, Tessellation]],
    domain_partition: ComponentDomainPartition,
    canvas: Canvas,
    anchor_x: float,
    anchor_y: float,
    consensus_config,
    thresholds: Optional["PercentileThresholds"] = None,
    tessellation_tiles_df: Optional[pd.DataFrame] = None,
) -> ComponentConsensusResult:
    eligible_region_ids = tuple(
        diagnostic.domain_region_id
        for diagnostic in eligible_region_diagnostics
    )

    if not eligible_region_ids:
        return ComponentConsensusResult(
            parent_roi_id=parent_roi_id,
            component_id=component_id,
            hex_radius=0.0,
            hexes=(),
            n_tessellation_eligible_regions=0,
            support=ComponentConsensusSupport(
                component_id=component_id, parent_roi_id=parent_roi_id,
                hex_radius=0.0, n_tessellation_eligible_regions=0,
                hexes=(), tile_hex_overlaps=(), direct_hex_membership=(),
            ),
        )

    # Reuse the domain partition already computed by
    # run_tessellation_stage; do NOT re-partition here. The two
    # partitions must be byte-identical or downstream identity checks
    # fail. Verification lives between the geometry builder and the
    # tessellation stage; consensus consumes the verified result.
    partition_regions = {
        rid: domain_partition.canonical_regions[rid]
        for rid in eligible_region_ids
    }
    pooled_ensemble = _pool_replicates_across_regions(
        tessellations_by_region=tessellations_by_region,
        eligible_region_ids=eligible_region_ids,
    )

    n_component_spots_eligible = sum(
        diagnostic.n_component_spots
        for diagnostic in eligible_region_diagnostics
    )
    hex_radius = compute_optimal_hex_radius(
        tessellation_ensemble=pooled_ensemble,
        region_polygon=_union_of(partition_regions.values()),
        n_roi_spots=n_component_spots_eligible,
        config=consensus_config,
    )
    min_tess_per_hex = int(consensus_config.resolved_policy.consensus.min_tess_per_hex)
    min_overlap = _resolve_min_hex_overlap(
        consensus_config=consensus_config,
        n_component_spots=n_component_spots_eligible,
    )

    # Filter hexes below the minimum tessellation coverage threshold
    # BEFORE canonical sort + enumeration so hex_id stays a contiguous
    # 0..N-1 over retained hexes only.
    candidate_hex_records: List[dict] = []

    for region_id in eligible_region_ids:
        region_polygon = partition_regions[region_id]
        rep_map = tessellations_by_region.get(region_id, {})
        if not rep_map:
            continue
        tiles_by_tess = _tiles_by_tess_from_ensemble(
            component_data=component_data,
            component_id=component_id,
            domain_region_id=region_id,
            ensemble=rep_map,
            thresholds=thresholds,
            tessellation_tiles_df=tessellation_tiles_df,
        )
        overlap_indexes = {
            rep_idx: OverlapIndex(tess.tiles)
            for rep_idx, tess in rep_map.items()
        }
        region_hex_grid = _generate_hex_grid(
            region_polygon, hex_radius,
            anchor_x=anchor_x, anchor_y=anchor_y,
        )
        region_prep = prep(region_polygon)
        for lattice_col, lattice_row, cx, cy, hex_poly in region_hex_grid:
            clipped = _clip_hex_to_roi(
                hex_poly, region_prep, region_polygon,
                min_area_fraction=min_overlap,
            )
            if clipped is None:
                continue
            per_tess_stats, edges = _aggregate_hex_from_ensemble(
                clipped, overlap_indexes, rep_map, tiles_by_tess,
            )
            # Replicate-coverage gate: hexes covered by fewer than
            # min_tess_per_hex replicates are excluded from the
            # retained set.
            if len(per_tess_stats) < min_tess_per_hex:
                continue
            candidate_hex_records.append({
                "domain_region_id": region_id,
                "lattice_col": lattice_col,
                "lattice_row": lattice_row,
                "center_x": cx,
                "center_y": cy,
                "polygon": clipped,
                "per_tess_stats": per_tess_stats,
                "edges": edges,
            })

    if not candidate_hex_records:
        return ComponentConsensusResult(
            parent_roi_id=parent_roi_id,
            component_id=component_id,
            hex_radius=float(hex_radius),
            hexes=(),
            n_tessellation_eligible_regions=len(eligible_region_ids),
            support=ComponentConsensusSupport(
                component_id=component_id, parent_roi_id=parent_roi_id,
                hex_radius=float(hex_radius),
                n_tessellation_eligible_regions=len(eligible_region_ids),
                hexes=(), tile_hex_overlaps=(), direct_hex_membership=(),
            ),
        )

    # Canonical sort: (domain_region_id, lattice_col, lattice_row).
    candidate_hex_records.sort(
        key=lambda h: (h["domain_region_id"], h["lattice_col"], h["lattice_row"]),
    )

    # Build ONE canonical spot -> hex membership over the sorted
    # retained-hex sequence. First-canonical-hex wins on shared
    # boundaries under covers(); uncovered spots stay UNASSIGNED.
    direct_coords = component_data.points
    canonical_polygons = [rec["polygon"] for rec in candidate_hex_records]
    spot_to_hex = assign_points_to_first_covering_polygon(
        canonical_polygons, direct_coords,
    )

    hexes: List[ConsensusHex] = []
    hex_geometries: List[ConsensusHexGeometry] = []
    all_tile_hex_overlaps: List[TileHexOverlap] = []
    for hex_id, rec in enumerate(candidate_hex_records):
        per_tess_stats = rec["per_tess_stats"]
        aggregated = _aggregate_per_tess_stats(per_tess_stats)
        n_tess_covered = len(per_tess_stats)
        hex_mask = spot_to_hex == hex_id
        n_spots_direct = int(hex_mask.sum()) if hex_mask.any() else 0
        # Joint counters populate only under a threshold-bearing pair
        # call; ConsensusHex carries only joint statistics, not
        # per-feature marginals.
        if thresholds is not None:
            n_both_high_ew: Optional[float] = float(aggregated[N_BOTH_HIGH])
            n_both_low_ew: Optional[float] = float(aggregated[N_BOTH_LOW])
        else:
            n_both_high_ew = None
            n_both_low_ew = None
        # Materialise captured overlap edges as bank-owned
        # TileHexOverlap tuples. `n_spots_ensemble_weighted` uses the
        # existing summation order (aggregated[N_SPOTS]) for byte
        # identity; reconstruction from edges matches up to float
        # summation order.
        region_id_int = int(rec["domain_region_id"])
        edge_records: List[TileHexOverlap] = []
        for tess_idx_e, tile_id_e, oa_e, ta_e in rec["edges"]:
            edge_records.append(TileHexOverlap(
                component_id=component_id,
                domain_region_id=region_id_int,
                replicate_index=int(tess_idx_e),
                tile_id=int(tile_id_e),
                hex_id=int(hex_id),
                overlap_area=float(oa_e),
                tile_area=float(ta_e),
            ))
        edge_records.sort(key=lambda e: (
            e.domain_region_id, e.replicate_index, e.tile_id, e.hex_id,
        ))
        all_tile_hex_overlaps.extend(edge_records)
        geometry = ConsensusHexGeometry(
            component_id=component_id,
            hex_id=hex_id,
            domain_region_id=region_id_int,
            polygon=rec["polygon"],
            lattice_col=int(rec["lattice_col"]),
            lattice_row=int(rec["lattice_row"]),
            center_x=float(rec["center_x"]),
            center_y=float(rec["center_y"]),
            n_tessellations_covered=int(n_tess_covered),
            n_spots_direct=int(n_spots_direct),
            n_spots_ensemble_weighted=float(aggregated[N_SPOTS]),
        )
        hex_geometries.append(geometry)
        hexes.append(ConsensusHex(
            geometry=geometry,
            n_both_high_ensemble_weighted=n_both_high_ew,
            n_both_low_ensemble_weighted=n_both_low_ew,
        ))

    source_row_ids_arr = (
        component_data.expression_data[SOURCE_ROW_ID].to_numpy(dtype=np.int64)
    )
    if source_row_ids_arr.shape[0] != spot_to_hex.shape[0]:
        raise ValueError(
            "DirectHexMembership: SOURCE_ROW_ID length "
            f"({source_row_ids_arr.shape[0]}) != spot_to_hex length "
            f"({spot_to_hex.shape[0]}) for component {component_id!r}."
        )
    direct_membership = tuple(
        DirectHexMembership(source_row_id=int(sid), hex_id=int(hid))
        for sid, hid in zip(
            source_row_ids_arr.tolist(), spot_to_hex.tolist(),
        )
    )
    all_tile_hex_overlaps.sort(
        key=lambda e: (
            e.domain_region_id, e.replicate_index, e.tile_id, e.hex_id,
        ),
    )
    support = ComponentConsensusSupport(
        component_id=component_id,
        parent_roi_id=parent_roi_id,
        hex_radius=float(hex_radius),
        n_tessellation_eligible_regions=len(eligible_region_ids),
        hexes=tuple(hex_geometries),
        tile_hex_overlaps=tuple(all_tile_hex_overlaps),
        direct_hex_membership=direct_membership,
    )
    return ComponentConsensusResult(
        parent_roi_id=parent_roi_id,
        component_id=component_id,
        hex_radius=float(hex_radius),
        hexes=tuple(hexes),
        n_tessellation_eligible_regions=len(eligible_region_ids),
        support=support,
    )


def build_consensus_spatial_bank(
    *,
    sample_tessellation_result: SampleTessellationResult,
    canvas: Canvas,
    sample_anchor: Tuple[float, float],
    consensus_config,
) -> SampleSpatialBank:
    """Bank-scope consensus geometry. Called EXACTLY once per run_sample.

    Feature-independent: no thresholds, no per-tile marginals, no joint
    counts. Per-pair joint accumulation is `run_pair_consensus_from_bank`.
    """
    dummy_result = run_consensus_stage(
        sample_tessellation_result=sample_tessellation_result,
        canvas=canvas,
        sample_anchor=sample_anchor,
        consensus_config=consensus_config,
        thresholds=None,
        tessellation_tiles_df=None,
    )
    return dummy_result.spatial_bank


def run_pair_consensus_from_bank(
    *,
    spatial_bank: SampleSpatialBank,
    sample_tessellation_result: SampleTessellationResult,
    tessellation_tiles_df: pd.DataFrame,
    thresholds: "PercentileThresholds",
) -> SampleConsensusResult:
    """Pair-scope joint-count reconstruction over an already-built bank.

    Consumes bank-owned hex geometry and overlap edges; computes only
    the pair's joint counters. No re-tessellation, no re-hex-grid, no
    re-membership.
    """
    consensus_bank = spatial_bank.consensus_bank

    # Group tile-scope joint stats by (component, region, replicate, tile).
    tile_joint_by_key: Dict[Tuple[str, int, int, int], Tuple[float, float]] = {}
    if not tessellation_tiles_df.empty:
        for row in tessellation_tiles_df.itertuples(index=False):
            key = (
                str(getattr(row, COMPONENT_ID)),
                int(getattr(row, DOMAIN_REGION_ID)),
                int(getattr(row, "replicate_index")),
                int(getattr(row, TILE_ID)),
            )
            tile_joint_by_key[key] = (
                float(getattr(row, COL_N_BOTH_HIGH)),
                float(getattr(row, COL_N_BOTH_LOW)),
            )

    components_out: Dict[str, ComponentConsensusResult] = {}
    for cid in sorted(consensus_bank.components.keys()):
        support = consensus_bank.components[cid]
        parent_roi_id = support.parent_roi_id
        hex_radius = support.hex_radius
        n_eligible = support.n_tessellation_eligible_regions

        # Per-hex-per-replicate accumulator: per_tess totals are
        # summed per replicate (tile-inner) then across replicates.
        # A single-accumulator sum over all edges produces ULP drift.
        per_tess_by_hex: Dict[int, Dict[int, Tuple[float, float]]] = {
            int(g.hex_id): {} for g in support.hexes
        }
        for edge in support.tile_hex_overlaps:
            key = (
                cid, int(edge.domain_region_id),
                int(edge.replicate_index), int(edge.tile_id),
            )
            joint = tile_joint_by_key.get(key)
            if joint is None:
                continue
            n_both_high, n_both_low = joint
            weight = edge.overlap_area / edge.tile_area
            rep_idx = int(edge.replicate_index)
            cur_high, cur_low = per_tess_by_hex[int(edge.hex_id)].get(
                rep_idx, (0.0, 0.0),
            )
            per_tess_by_hex[int(edge.hex_id)][rep_idx] = (
                cur_high + weight * n_both_high,
                cur_low + weight * n_both_low,
            )

        hexes: List[ConsensusHex] = []
        for geom in support.hexes:
            per_tess = per_tess_by_hex[int(geom.hex_id)]
            n_both_high_ew = sum(
                v[0] for _, v in sorted(per_tess.items())
            )
            n_both_low_ew = sum(
                v[1] for _, v in sorted(per_tess.items())
            )
            hexes.append(ConsensusHex(
                geometry=geom,
                n_both_high_ensemble_weighted=float(n_both_high_ew),
                n_both_low_ensemble_weighted=float(n_both_low_ew),
            ))
        components_out[cid] = ComponentConsensusResult(
            parent_roi_id=parent_roi_id,
            component_id=cid,
            hex_radius=hex_radius,
            hexes=tuple(hexes),
            n_tessellation_eligible_regions=n_eligible,
            support=support,
        )

    return SampleConsensusResult(
        sample_id=spatial_bank.support_bank.sample_id,
        components=components_out,
        n_replicates_planned=consensus_bank.n_replicates_planned,
        sample_anchor_x=consensus_bank.sample_anchor_x,
        sample_anchor_y=consensus_bank.sample_anchor_y,
        spatial_bank=spatial_bank,
    )


def _pool_replicates_across_regions(
    *,
    tessellations_by_region: Mapping[int, Mapping[int, Tessellation]],
    eligible_region_ids: Tuple[int, ...],
) -> Dict[int, Tessellation]:
    """Build a per-replicate ensemble whose tiles pool every eligible
    region's tiles for that replicate."""
    # Accumulate tiles per replicate in lists then materialise once;
    # Tessellation is frozen and cannot be mutated post-construction.
    tiles_by_rep: Dict[int, list] = {}
    template_by_rep: Dict[int, Tessellation] = {}
    for region_id in eligible_region_ids:
        for rep_idx, tess in tessellations_by_region.get(region_id, {}).items():
            tiles_by_rep.setdefault(rep_idx, []).extend(tess.tiles)
            template_by_rep.setdefault(rep_idx, tess)
    pooled: Dict[int, Tessellation] = {
        rep_idx: dataclass_replace(template_by_rep[rep_idx], tiles=tuple(tiles))
        for rep_idx, tiles in tiles_by_rep.items()
    }
    return pooled


def _union_of(polygons) -> object:
    polys = [p for p in polygons if p is not None and not p.is_empty]
    if not polys:
        return polys[0] if polys else None
    return unary_union(polys)


def _tiles_by_tess_from_ensemble(
    *,
    component_data: ComponentData,
    component_id: str,
    domain_region_id: int,
    ensemble: Mapping[int, Tessellation],
    thresholds: Optional["PercentileThresholds"] = None,
    tessellation_tiles_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[int, dict]]:
    """Project canonical tile observations for overlap aggregation.

    Production supplies ``tessellation_tiles_df`` and reuses those
    sufficient statistics verbatim. Threshold-less bank construction
    omits the frame and derives only feature-independent occupancy.
    """
    canonical_rows = None
    if tessellation_tiles_df is not None:
        canonical_rows = tessellation_tiles_df.loc[
            (tessellation_tiles_df[COMPONENT_ID].astype(str) == str(component_id))
            & (tessellation_tiles_df[DOMAIN_REGION_ID].astype(int)
               == int(domain_region_id))
        ]

    out: Dict[str, Dict[int, dict]] = {}
    for tess in ensemble.values():
        by_tile: Dict[int, dict] = {}
        if canonical_rows is not None:
            tess_rows = canonical_rows.loc[
                canonical_rows[TESSELLATION_ID].astype(str)
                == str(tess.tessellation_id)
            ]
            row_by_tile = {
                int(row[TILE_ID]): row for _, row in tess_rows.iterrows()
            }
            expected_ids = {int(tile.tile_id) for tile in tess.tiles}
            if set(row_by_tile) != expected_ids:
                raise ValueError(
                    "canonical tessellation tile rows disagree with typed "
                    f"tiles for component={component_id!r}, "
                    f"region={domain_region_id}, "
                    f"tessellation={tess.tessellation_id!r}: "
                    f"expected={sorted(expected_ids)!r}, "
                    f"observed={sorted(row_by_tile)!r}"
                )
            for tile in tess.tiles:
                row = row_by_tile[int(tile.tile_id)]
                by_tile[int(tile.tile_id)] = {
                    'tessellation_id': tess.tessellation_id,
                    'tile_id': int(tile.tile_id),
                    'area': float(tile.area),
                    N_SPOTS: int(row[N_SPOTS]),
                    SUM_G1: float(row[SUM_G1]),
                    SUM_G2: float(row[SUM_G2]),
                    N_G1_HIGH: int(row[N_G1_HIGH]),
                    N_G2_HIGH: int(row[N_G2_HIGH]),
                    N_BOTH_HIGH: int(row[N_BOTH_HIGH]),
                    N_G1_LOW: int(row[N_G1_LOW]),
                    N_G2_LOW: int(row[N_G2_LOW]),
                    N_BOTH_LOW: int(row[N_BOTH_LOW]),
                }
        else:
            for tile in tess.tiles:
                n_spots = int(tile.point_indices.size)
                by_tile[int(tile.tile_id)] = {
                    'tessellation_id': tess.tessellation_id,
                    'tile_id': int(tile.tile_id),
                    'area': float(tile.area),
                    N_SPOTS: n_spots,
                    SUM_G1: 0.0,
                    SUM_G2: 0.0,
                    N_G1_HIGH: 0, N_G2_HIGH: 0, N_BOTH_HIGH: 0,
                    N_G1_LOW: 0, N_G2_LOW: 0, N_BOTH_LOW: 0,
                }
        out[tess.tessellation_id] = by_tile
    return out


def _resolve_min_hex_overlap(
    *, consensus_config, n_component_spots: int,
) -> float:
    """Adaptive minimum hex overlap fraction. Explicit config value
    overrides; otherwise use the small/medium/large-ROI schedule keyed
    on spot count."""
    explicit = consensus_config.resolved_policy.consensus.min_hex_roi_overlap_fraction
    if explicit is not None:
        return float(explicit)
    ao = consensus_config.resolved_policy.consensus.adaptive_overlap
    if n_component_spots < ao.small_roi_max_spots:
        return float(ao.small_roi_overlap)
    if n_component_spots < ao.medium_roi_max_spots:
        return float(ao.medium_roi_overlap)
    return float(ao.large_roi_overlap)
