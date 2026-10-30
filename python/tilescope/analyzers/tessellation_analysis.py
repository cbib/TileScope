"""Region-aware tessellation stage entry.

`run_tessellation_stage` is the sample-wide tessellation driver: it partitions each component's domain into canonical regions, applies the per-region construction-floor eligibility gate (spot count >= `tile_min_points`), invokes `run_region_tessellation` per (component, region, replicate) with a stable seed, and emits per-triple diagnostics. Output is primary records only; reporter materialisation is a separate layer.
"""
from __future__ import annotations

import logging
from time import perf_counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import List, Mapping, Optional, Tuple
import numpy as np
from tilescope.analyzers import sample_support_stage

from tilescope.analyzers.region_tessellation import (
    prepare_component, prepare_region, run_region_tessellation,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus,
    RegionIneligibilityReason,
    RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.cols import SOURCE_ROW_ID
from tilescope.component_data import ComponentData, Tessellation
from tilescope.spatial.component_domains import ComponentDomainPartition
from tilescope.spatial.sample_support_bank import SampleSupportBank
from tilescope.stable_seed import derive_stable_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentTessellationResult:
    """Single-component slice of the tessellation stage: primary records only.

    `tessellations[region_id][replicate_index].tiles` is the single tile authority (hole-reclipped, component-local-indexed, with `Tile.domain_region_id` set); consensus and reporters read tiles from here.
    """
    parent_roi_id: str
    component_data: ComponentData
    tessellations: Mapping[int, Mapping[int, Tessellation]]
    region_diagnostics: Tuple[RegionTessellationEligibilityDiagnostic, ...]
    region_replicate_diagnostics: Tuple[RegionReplicateTessellationDiagnostic, ...]
    # Single authority for region membership and source-row fingerprint;
    # consensus reuses `canonical_regions`, reporters read
    # `canonical_ids_by_source_row` and `source_row_fingerprint_by_canonical_id`.
    domain_partition: ComponentDomainPartition
    tessellation_eligible_region_diagnostics: Tuple[
        RegionTessellationEligibilityDiagnostic, ...
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        eligible_diagnostics = tuple(
            diagnostic for diagnostic in self.region_diagnostics
            if diagnostic.is_tessellation_eligible
        )
        object.__setattr__(
            self,
            "tessellation_eligible_region_diagnostics",
            eligible_diagnostics,
        )
        # Prevent attaching an unrelated partition: region_diagnostics IDs
        # must 1:1 the canonical-region IDs and spot counts must match.
        cid = self.component_data.component_id
        canonical_ids = set(range(len(self.domain_partition.canonical_regions)))
        diagnostic_ids = {d.domain_region_id for d in self.region_diagnostics}
        if diagnostic_ids != canonical_ids:
            raise ValueError(
                f"ComponentTessellationResult (component_id={cid!r}): "
                f"region_diagnostics IDs {sorted(diagnostic_ids)} do not "
                f"match domain_partition canonical IDs "
                f"{sorted(canonical_ids)}."
            )
        for d in self.region_diagnostics:
            if d.component_id != cid:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"region diagnostic component_id={d.component_id!r} "
                    f"disagrees with owning component_id={cid!r}."
                )
            expected = len(
                self.domain_partition.source_row_ids_by_canonical_id.get(
                    d.domain_region_id, ()
                )
            )
            if d.n_component_spots != expected:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"region {d.domain_region_id} n_component_spots="
                    f"{d.n_component_spots} disagrees with "
                    f"domain_partition source_row count={expected}."
                )

        # Region-replicate diagnostics are the completeness authority; missing
        # ones would silently reduce `n_contributing_regions` at the ROI grain.
        eligible_ids = {
            diagnostic.domain_region_id
            for diagnostic in self.tessellation_eligible_region_diagnostics
        }
        rr_keys: set[tuple[int, int]] = set()
        for rr in self.region_replicate_diagnostics:
            if rr.component_id != cid:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"region-replicate diagnostic component_id="
                    f"{rr.component_id!r} disagrees with owning "
                    f"component_id={cid!r}."
                )
            if rr.domain_region_id not in eligible_ids:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"region-replicate diagnostic references "
                    f"domain_region_id={rr.domain_region_id} which is not "
                    f"tessellation-eligible (eligible={sorted(eligible_ids)})."
                )
            key = (rr.domain_region_id, rr.replicate_index)
            if key in rr_keys:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"duplicate region-replicate diagnostic key "
                    f"(domain_region_id={rr.domain_region_id}, "
                    f"replicate_index={rr.replicate_index})."
                )
            rr_keys.add(key)

            has_tess = (
                rr.domain_region_id in self.tessellations
                and rr.replicate_index in self.tessellations[rr.domain_region_id]
            )
            contributed = (
                rr.region_status is RegionExecutionStatus.CONTRIBUTED
            )
            if contributed != has_tess:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"region-replicate (domain_region_id="
                    f"{rr.domain_region_id}, replicate_index="
                    f"{rr.replicate_index}) region_status="
                    f"{rr.region_status.name} but "
                    f"tessellation {'present' if has_tess else 'absent'}; "
                    f"CONTRIBUTED must coincide with an entry in "
                    f"`tessellations`."
                )
            if contributed:
                tile_count = len(
                    self.tessellations[rr.domain_region_id][rr.replicate_index].tiles
                )
                if rr.n_tiles != tile_count:
                    raise ValueError(
                        f"ComponentTessellationResult (component_id={cid!r}): "
                        f"region-replicate (domain_region_id="
                        f"{rr.domain_region_id}, replicate_index="
                        f"{rr.replicate_index}) diagnostic n_tiles="
                        f"{rr.n_tiles} disagrees with tessellation tile "
                        f"count={tile_count}."
                    )

        # Tessellations must not carry entries missing from the diagnostic
        # set (would hide a region from the ROI-grain reducer).
        for region_id, rep_map in self.tessellations.items():
            if region_id not in eligible_ids:
                raise ValueError(
                    f"ComponentTessellationResult (component_id={cid!r}): "
                    f"tessellations carry domain_region_id={region_id} "
                    f"which is not tessellation-eligible."
                )
            for rep in rep_map:
                if (region_id, rep) not in rr_keys:
                    raise ValueError(
                        f"ComponentTessellationResult (component_id={cid!r}): "
                        f"tessellation (domain_region_id={region_id}, "
                        f"replicate_index={rep}) has no matching "
                        f"region-replicate diagnostic."
                    )

    @property
    def component_id(self) -> str:
        return self.component_data.component_id


@dataclass(frozen=True)
class SampleTessellationResult:
    """Sample-wide typed tessellation result.

    `component_to_roi` and `diagnostics` are derived from `components`. Adds the `n_replicates_planned` invariant that no single component can enforce: region-replicate diagnostics form the exact Cartesian (eligible region x planned replicate) key set per component.
    """
    sample_id: str
    components: Mapping[str, ComponentTessellationResult]
    n_replicates_planned: int
    # Substrate fields on every ComponentTessellationResult must be
    # `is`-equal to bank copies (derived-view identity contract).
    support_bank: SampleSupportBank

    def __post_init__(self) -> None:
        if self.n_replicates_planned < 1:
            raise ValueError(
                f"SampleTessellationResult: n_replicates_planned must be "
                f">= 1, got {self.n_replicates_planned}."
            )
        expected_replicate_set = set(range(self.n_replicates_planned))
        for cid, result in self.components.items():
            if cid != result.component_id:
                raise ValueError(
                    f"SampleTessellationResult: mapping key {cid!r} "
                    f"disagrees with component_data.component_id="
                    f"{result.component_id!r}."
                )
            component_id = result.component_id
            eligible_ids = sorted(
                diagnostic.domain_region_id
                for diagnostic in
                result.tessellation_eligible_region_diagnostics
            )
            expected_keys = {
                (rid, t) for rid in eligible_ids
                for t in expected_replicate_set
            }
            actual_keys = {
                (rr.domain_region_id, rr.replicate_index)
                for rr in result.region_replicate_diagnostics
            }
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise ValueError(
                    f"SampleTessellationResult: component {component_id!r} "
                    f"region-replicate diagnostic key set does not equal "
                    f"the Cartesian (eligible_region × planned_replicate) "
                    f"set for n_replicates_planned="
                    f"{self.n_replicates_planned}. missing={missing[:10]!r}, "
                    f"unexpected={extra[:10]!r}."
                )
        # Derived-view identity contract: substrate fields must be
        # `is`-identical to bank copies; keys / sample_id / n_replicates agree.
        bank = self.support_bank
        if bank.sample_id != self.sample_id:
            raise ValueError(
                f"SampleTessellationResult: sample_id={self.sample_id!r} "
                f"disagrees with support_bank.sample_id={bank.sample_id!r}."
            )
        if bank.tessellation_bank.n_replicates_planned != self.n_replicates_planned:
            raise ValueError(
                f"SampleTessellationResult: n_replicates_planned="
                f"{self.n_replicates_planned} disagrees with "
                f"tessellation_bank="
                f"{bank.tessellation_bank.n_replicates_planned}."
            )
        view_cids = set(self.components.keys())
        bank_cids = set(bank.components.keys())
        if view_cids != bank_cids:
            missing = sorted(bank_cids - view_cids)
            extra = sorted(view_cids - bank_cids)
            raise ValueError(
                f"SampleTessellationResult: component key set "
                f"{sorted(view_cids)!r} disagrees with bank "
                f"{sorted(bank_cids)!r}. missing={missing!r}, "
                f"extra={extra!r}."
            )
        tb = bank.tessellation_bank
        for cid, result in self.components.items():
            if result.tessellations is not tb.tessellations.get(cid):
                raise ValueError(
                    f"SampleTessellationResult: component {cid!r} "
                    "tessellations is not `is`-identical to the bank's copy; "
                    "derived-view identity contract violated."
                )
            if result.region_diagnostics is not tb.region_diagnostics.get(cid):
                raise ValueError(
                    f"SampleTessellationResult: component {cid!r} "
                    "region_diagnostics is not `is`-identical to the "
                    "bank's copy."
                )
            if result.region_replicate_diagnostics is not tb.region_replicate_diagnostics.get(cid):
                raise ValueError(
                    f"SampleTessellationResult: component {cid!r} "
                    "region_replicate_diagnostics is not `is`-identical "
                    "to the bank's copy."
                )
            if result.domain_partition is not bank.component_domains.get(cid):
                raise ValueError(
                    f"SampleTessellationResult: component {cid!r} "
                    "domain_partition is not `is`-identical to the "
                    "bank's copy."
                )
            # ComponentData.points / polygon must be identity-sourced from
            # the bank's ComponentSupport, or the view wraps genes over
            # foreign geometry.
            support = bank.components[cid]
            if result.component_data.points is not support.points:
                # Defensive-copy inside ComponentSupport means the bank
                # holds a distinct ndarray object; compare by value.
                if not np.array_equal(result.component_data.points, support.points):
                    raise ValueError(
                        f"SampleTessellationResult: component {cid!r} "
                        "component_data.points disagrees with the bank's "
                        "ComponentSupport.points."
                    )
            if result.component_data.polygon is not support.polygon:
                raise ValueError(
                    f"SampleTessellationResult: component {cid!r} "
                    "component_data.polygon is not the bank's polygon."
                )

    @property
    def component_to_roi(self) -> Mapping[str, str]:
        return MappingProxyType({
            cid: r.parent_roi_id for cid, r in self.components.items()
        })

    @classmethod
    def from_component_results(
        cls,
        *, sample_id: str,
        components: Mapping[str, "ComponentTessellationResult"],
        n_replicates_planned: int,
        base_seed: int = 0,
    ) -> "SampleTessellationResult":
        """Test helper: builds a matching bank so fixtures don't hand-construct one."""
        bank = sample_support_stage.sample_support_bank_from_component_results(
            sample_id=sample_id,
            component_results=components,
            n_replicates=n_replicates_planned,
            base_seed=base_seed,
        )
        return cls(
            sample_id=sample_id,
            components=components,
            n_replicates_planned=n_replicates_planned,
            support_bank=bank,
        )

def _stable_region_seed(
    *,
    base_seed: int,
    sample_id: str,
    region_source_row_fingerprint: str,
    replicate_index: int,
) -> int:
    """Per-region tessellation seed keyed on (sample, region spot/cell fingerprint, replicate).

    The fingerprint digests spot/cell membership so spot-identical regions receive the same seed even if ordinal `component_id` renumbered elsewhere in the sample; changing any region spot changes the fingerprint and redraws that region's tessellation.
    """
    return derive_stable_seed(
        base_seed,
        "tessellation_region",
        sample_id,
        region_source_row_fingerprint,
        replicate_index,
    )


def run_tessellation_stage(
    *,
    sample_id: str,
    spatial_components: Mapping[str, ComponentData],
    domain_partitions: Mapping[str, ComponentDomainPartition],
    component_to_roi: Mapping[str, str],
    n_replicates: int,
    tile_min_points: int,
    voronoi_fraction: float,
    base_seed: int,
    source_spot_adjacency,
    config,
) -> SampleTessellationResult:
    """Run the region-aware tessellation stage for a whole sample.

    Delegates substrate construction to `build_sample_support_bank`; the returned view's substrate fields are `is`-identical to bank copies, and the bank is exposed via `.support_bank`.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}.")
    if tile_min_points < 1:
        raise ValueError(
            f"tile_min_points must be >= 1, got {tile_min_points}."
        )
    for cid in sorted(spatial_components):
        if cid not in component_to_roi:
            raise ValueError(
                f"run_tessellation_stage: component {cid!r} missing from "
                f"component_to_roi mapping."
            )
        if cid not in domain_partitions:
            raise ValueError(
                f"run_tessellation_stage: component {cid!r} missing from "
                f"domain_partitions."
            )

    bank = sample_support_stage.build_sample_support_bank(
        sample_id=sample_id,
        source_spot_adjacency=source_spot_adjacency,
        spatial_components=spatial_components,
        domain_partitions=domain_partitions,
        component_to_roi=component_to_roi,
        config=config,
        n_replicates=n_replicates,
        tile_min_points=tile_min_points,
        voronoi_fraction=voronoi_fraction,
        base_seed=base_seed,
    )

    components_view: dict[str, ComponentTessellationResult] = {}
    for cid in sorted(spatial_components):
        components_view[cid] = ComponentTessellationResult(
            parent_roi_id=component_to_roi[cid],
            component_data=spatial_components[cid],
            tessellations=bank.tessellation_bank.tessellations[cid],
            region_diagnostics=bank.tessellation_bank.region_diagnostics[cid],
            region_replicate_diagnostics=(
                bank.tessellation_bank.region_replicate_diagnostics[cid]
            ),
            domain_partition=bank.component_domains[cid],
        )

    return SampleTessellationResult(
        sample_id=sample_id,
        components=MappingProxyType(dict(components_view)),
        n_replicates_planned=n_replicates,
        support_bank=bank,
    )


def _run_component(
    *,
    sample_id: str,
    component_id: str,
    component_data: ComponentData,
    parent_roi_id: str,
    domain_partition: ComponentDomainPartition,
    n_replicates: int,
    tile_min_points: int,
    voronoi_fraction: float,
    base_seed: int,
) -> ComponentTessellationResult:
    if SOURCE_ROW_ID not in component_data.expression_data.columns:
        raise ValueError(
            f"run_tessellation_stage: component {component_id!r} "
            f"expression_data missing SOURCE_ROW_ID."
        )
    region_diagnostics = _build_region_diagnostics(
        component_id=component_id,
        partition=domain_partition,
        tile_min_points=tile_min_points,
    )
    eligible_region_ids: Tuple[int, ...] = tuple(
        d.domain_region_id for d in region_diagnostics if d.is_tessellation_eligible
    )

    if not eligible_region_ids:
        return ComponentTessellationResult(
            parent_roi_id=parent_roi_id,
            component_data=component_data,
            tessellations=MappingProxyType({}),
            region_diagnostics=tuple(region_diagnostics),
            region_replicate_diagnostics=(),
            domain_partition=domain_partition,
        )

    tessellations: dict[int, dict[int, Tessellation]] = {rid: {} for rid in eligible_region_ids}
    region_replicate_diagnostics: List[RegionReplicateTessellationDiagnostic] = []

    # Computed once per component and reused by every region via `prepare_region`.
    prepared_component = prepare_component(component_data)

    for region_id in eligible_region_ids:
        source_ids = domain_partition.source_row_ids_by_canonical_id[region_id]
        # Prepared once before the K-replicate loop; caches the local index
        # remap, coord array, unique-coord reduction, and median-NN distance.
        prepared = prepare_region(
            component_id=component_id,
            domain_region_id=region_id,
            region_polygon=domain_partition.canonical_regions[region_id],
            component_data=component_data,
            source_row_ids_in_region=source_ids,
            prepared_component=prepared_component,
        )
        region_fingerprint = (
            domain_partition.source_row_fingerprint_by_canonical_id[region_id]
        )
        for replicate_index in range(n_replicates):
            started = perf_counter()
            logger.info(
                "[TESSELLATION] component=%s region=%s rows=%d replicate=%d/%d start",
                component_id, region_id, len(source_ids), replicate_index + 1, n_replicates,
            )
            seed = _stable_region_seed(
                base_seed=base_seed,
                sample_id=sample_id,
                region_source_row_fingerprint=region_fingerprint,
                replicate_index=replicate_index,
            )
            outcome = run_region_tessellation(
                prepared=prepared,
                replicate_index=replicate_index,
                tile_min_points=tile_min_points,
                voronoi_fraction=voronoi_fraction,
                rng_seed=seed,
            )
            region_replicate_diagnostics.append(outcome.diagnostic)
            logger.info(
                "[TESSELLATION] component=%s region=%s replicate=%d/%d tiles=%d elapsed=%.1fs",
                component_id, region_id, replicate_index + 1, n_replicates,
                outcome.diagnostic.n_tiles, perf_counter() - started,
            )
            if outcome.tessellation is not None:
                tessellations[region_id][replicate_index] = outcome.tessellation

    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=component_data,
        tessellations=MappingProxyType(
            {rid: MappingProxyType(dict(rmap)) for rid, rmap in tessellations.items()}
        ),
        region_diagnostics=tuple(region_diagnostics),
        region_replicate_diagnostics=tuple(region_replicate_diagnostics),
        domain_partition=domain_partition,
    )


def _build_region_diagnostics(
    *,
    component_id: str,
    partition,
    tile_min_points: int,
) -> List[RegionTessellationEligibilityDiagnostic]:
    out: List[RegionTessellationEligibilityDiagnostic] = []
    for region_id, region in enumerate(partition.canonical_regions):
        n_spots = len(partition.source_row_ids_by_canonical_id.get(region_id, ()))
        reason: Optional[RegionIneligibilityReason]
        if n_spots < tile_min_points:
            reason = RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN
        else:
            reason = None
        out.append(RegionTessellationEligibilityDiagnostic(
            component_id=component_id,
            domain_region_id=region_id,
            n_component_spots=n_spots,
            region_area=float(region.area),
            ineligibility_reason=reason,
        ))
    return out
