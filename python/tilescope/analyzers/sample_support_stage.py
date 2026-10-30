"""Single sample-scope authority for `SampleSupportBank` construction.

Records + fingerprint helpers live in
`tilescope.spatial.sample_support_bank`; this module coordinates
the analyzer + engine machinery and is called once per sample.
"""

from __future__ import annotations

from typing import Mapping
import numpy as np
from tilescope.cols import COORD_X, COORD_Y, SOURCE_ROW_ID
from tilescope.analyzers import tessellation_analysis

from tilescope.spatial.component_domains import ComponentDomainPartition
from tilescope.spatial.sample_support_bank import (
    ComponentSupport, SampleSupportBank, SampleSupportFingerprint,
    TessellationBank, build_sample_support_fingerprint,
)
from tilescope.spatial.source_spot_adjacency import (
    DEFAULT_SAMPLE_SUPPORT_FACTOR, SourceSpotAdjacency,
    build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import SourceSpotTable


def build_component_support(
    *, component_id: str, parent_roi_id: str, component_data,
) -> ComponentSupport:
    """Extract spatial-only fields from a `ComponentData`."""
    expr = component_data.expression_data
    if SOURCE_ROW_ID not in expr.columns:
        raise ValueError(
            f"build_component_support: component {component_id!r} "
            f"expression_data missing SOURCE_ROW_ID."
        )
    return ComponentSupport(
        component_id=str(component_id),
        parent_roi_id=str(parent_roi_id),
        points=component_data.points,
        source_row_ids=expr[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
        polygon=component_data.polygon,
    )


def build_sample_support_bank(
    *,
    sample_id: str,
    source_spot_adjacency: SourceSpotAdjacency,
    spatial_components: Mapping[str, object],
    domain_partitions: Mapping[str, ComponentDomainPartition],
    component_to_roi: Mapping[str, str],
    config,
    n_replicates: int,
    tile_min_points: int,
    voronoi_fraction: float,
    base_seed: int,
) -> SampleSupportBank:
    """Single sample-scope authority; delegates per-component tessellation to `_run_component`."""
    if config is None:
        raise ValueError(
            "build_sample_support_bank: config is required — the "
            "production fingerprint depends on splitter / geometry / "
            "tessellation / scattered-policy hashes."
        )
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}.")
    if tile_min_points < 1:
        raise ValueError(
            f"tile_min_points must be >= 1, got {tile_min_points}."
        )

    components_support: dict[str, ComponentSupport] = {}
    tessellations_by_cid: dict = {}
    region_diags_by_cid: dict = {}
    region_rep_diags_by_cid: dict = {}

    for cid in sorted(spatial_components):
        if cid not in component_to_roi:
            raise ValueError(
                f"build_sample_support_bank: component {cid!r} missing from "
                f"component_to_roi mapping."
            )
        if cid not in domain_partitions:
            raise ValueError(
                f"build_sample_support_bank: component {cid!r} missing from "
                f"domain_partitions."
            )
        component_data = spatial_components[cid]
        parent_roi_id = component_to_roi[cid]

        components_support[cid] = build_component_support(
            component_id=cid, parent_roi_id=parent_roi_id,
            component_data=component_data,
        )

        cres = tessellation_analysis._run_component(
            sample_id=sample_id, component_id=cid,
            component_data=component_data,
            parent_roi_id=parent_roi_id,
            domain_partition=domain_partitions[cid],
            n_replicates=n_replicates,
            tile_min_points=tile_min_points,
            voronoi_fraction=voronoi_fraction,
            base_seed=base_seed,
        )
        tessellations_by_cid[cid] = cres.tessellations
        region_diags_by_cid[cid] = cres.region_diagnostics
        region_rep_diags_by_cid[cid] = cres.region_replicate_diagnostics

    tessellation_bank = TessellationBank(
        n_replicates_planned=n_replicates,
        base_seed=base_seed,
        tessellations=tessellations_by_cid,
        region_diagnostics=region_diags_by_cid,
        region_replicate_diagnostics=region_rep_diags_by_cid,
    )

    fingerprint = build_sample_support_fingerprint(
        sample_id=sample_id,
        source_spot_table=source_spot_adjacency.source_spot_table,
        config=config,
        n_replicates=n_replicates,
        base_seed=base_seed,
    )

    return SampleSupportBank(
        sample_id=str(sample_id),
        fingerprint=fingerprint,
        source_spots=source_spot_adjacency.source_spot_table,
        source_spot_adjacency=source_spot_adjacency,
        components=components_support,
        component_domains=domain_partitions,
        tessellation_bank=tessellation_bank,
    )


def build_empty_sample_support_bank(
    *, sample_id: str, source_spot_adjacency: SourceSpotAdjacency,
    config, n_replicates: int, base_seed: int,
) -> SampleSupportBank:
    """Empty bank for all-scattered samples; same single-authority path."""
    return SampleSupportBank(
        sample_id=str(sample_id),
        fingerprint=build_sample_support_fingerprint(
            sample_id=sample_id,
            source_spot_table=source_spot_adjacency.source_spot_table,
            config=config,
            n_replicates=n_replicates,
            base_seed=base_seed,
        ),
        source_spots=source_spot_adjacency.source_spot_table,
        source_spot_adjacency=source_spot_adjacency,
        components={},
        component_domains={},
        tessellation_bank=TessellationBank(
            n_replicates_planned=n_replicates,
            base_seed=base_seed,
            tessellations={},
            region_diagnostics={},
            region_replicate_diagnostics={},
        ),
    )


def sample_support_bank_from_component_results(
    *, sample_id: str, component_results, n_replicates: int, base_seed: int = 0,
) -> SampleSupportBank:
    """Test-scoped: build a bank whose substrate is `is`-identical to the supplied results."""
    components_support: dict[str, ComponentSupport] = {}
    tessellations_by_cid: dict = {}
    region_diags_by_cid: dict = {}
    region_rep_diags_by_cid: dict = {}
    partitions_by_cid: dict = {}
    all_rows: list[tuple] = []
    for cid, cres in component_results.items():
        cd = cres.component_data
        expr = cd.expression_data
        components_support[cid] = build_component_support(
            component_id=cid, parent_roi_id=cres.parent_roi_id,
            component_data=cd,
        )
        partitions_by_cid[cid] = cres.domain_partition
        tessellations_by_cid[cid] = cres.tessellations
        region_diags_by_cid[cid] = cres.region_diagnostics
        region_rep_diags_by_cid[cid] = cres.region_replicate_diagnostics
        # Coord columns may be absent in stub fixtures; fall back to
        # ComponentData.points positions.
        if COORD_X in expr.columns and COORD_Y in expr.columns:
            xs = expr[COORD_X].to_numpy(dtype=float)
            ys = expr[COORD_Y].to_numpy(dtype=float)
        else:
            xs = np.asarray(cd.points[:, 0], dtype=float)
            ys = np.asarray(cd.points[:, 1], dtype=float)
        for sid, x, y in zip(expr[SOURCE_ROW_ID].to_numpy(), xs, ys):
            all_rows.append((str(sid), float(x), float(y),
                             cres.parent_roi_id))
    # De-dupe by spot_id (fixtures may share spots across components).
    seen: dict[str, tuple] = {}
    for row in all_rows:
        seen.setdefault(row[0], row)
    rows = list(seen.values())
    source_spots = SourceSpotTable(
        spot_id=tuple(r[0] for r in rows),
        x=np.asarray([r[1] for r in rows], dtype=float),
        y=np.asarray([r[2] for r in rows], dtype=float),
        roi_id=tuple(r[3] for r in rows),
    )
    tessellation_bank = TessellationBank(
        n_replicates_planned=n_replicates,
        base_seed=base_seed,
        tessellations=tessellations_by_cid,
        region_diagnostics=region_diags_by_cid,
        region_replicate_diagnostics=region_rep_diags_by_cid,
    )
    fingerprint = SampleSupportFingerprint(
        sample_id=str(sample_id),
        source_frame_hash="test-fixture",
        support_policy_version="test-fixture",
        source_adjacency_config_hash="",
        scattered_support_policy_hash="",
        component_geometry_contract_hash="",
        n_replicates_planned=n_replicates,
        base_seed=base_seed,
        tessellation_config_hash="",
    )
    return SampleSupportBank(
        sample_id=str(sample_id),
        fingerprint=fingerprint,
        source_spots=source_spots,
        source_spot_adjacency=build_source_spot_adjacency(
            source_spots, radius_factor=DEFAULT_SAMPLE_SUPPORT_FACTOR,
        ),
        components=components_support,
        component_domains=partitions_by_cid,
        tessellation_bank=tessellation_bank,
    )
