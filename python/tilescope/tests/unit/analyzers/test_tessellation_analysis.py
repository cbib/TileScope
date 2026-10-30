"""Contract tests for `run_tessellation_stage`.

Semantic scenarios end-to-end at the stage boundary:

- Undersized region excluded → correct region diagnostic.
- Component with no eligible region → empty tessellations mapping.
- Reproducibility: two runs with the same base seed produce
  bit-identical tessellation ensembles.
- base_seed changes → tessellations diverge (the base seed is
  actually threaded into the region-seed derivation).
- The component-polygon partition is consumed directly.
"""
from __future__ import annotations

from tilescope.cols import ROI_ID
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
from tilescope.settings import load_run_config


from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from tilescope.analyzers.tessellation_analysis import (
    SampleTessellationResult,
    run_tessellation_stage,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionIneligibilityReason,
)
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, SOURCE_ROW_ID
from tilescope.component_data import ComponentData
from tilescope.spatial.component_domains import (
    DomainRegionDiagnostic,
    partition_component_domain,
)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _two_region_component(
    component_id: str = "A",
    n_left: int = 20,
    n_right: int = 20,
    right_offset: float = 8.0,
    seed: int = 0,
) -> Tuple[ComponentData, List[DomainRegionDiagnostic]]:
    rng = np.random.default_rng(seed)
    left_xy = np.column_stack([
        rng.uniform(0.1, 0.9, n_left),
        rng.uniform(0.1, 0.9, n_left),
    ])
    right_xy = np.column_stack([
        right_offset + rng.uniform(0.1, 0.9, n_right),
        rng.uniform(0.1, 0.9, n_right),
    ])
    coords = np.vstack([left_xy, right_xy])
    src_ids = np.arange(n_left + n_right, dtype=np.int64)
    expr = pd.DataFrame({
        SOURCE_ROW_ID: src_ids,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: rng.uniform(0.0, 1.0, n_left + n_right),
        GENE2: rng.uniform(0.0, 1.0, n_left + n_right),
    })
    geometry = MultiPolygon([
        box(0.0, 0.0, 1.0, 1.0),
        box(right_offset, 0.0, right_offset + 1.0, 1.0),
    ])
    component = ComponentData(
        component_id=component_id, points=coords, canvas=None,
        polygon=geometry, expression_data=expr,
    )
    partition = partition_component_domain(geometry, coords, src_ids)
    region_diags = [
        DomainRegionDiagnostic(
            component_id=component_id,
            domain_region_id=rid,
            n_component_spots=len(
                partition.source_row_ids_by_canonical_id.get(rid, ())
            ),
            area=float(partition.canonical_regions[rid].area),
            n_holes=0,
            geometry_status="ok",
            source_row_fingerprint=partition.source_row_fingerprint_by_canonical_id[rid],
        )
        for rid in range(len(partition.canonical_regions))
    ]
    return component, region_diags


def _run(
    component: ComponentData,
    region_diags: List[DomainRegionDiagnostic],
    *,
    n_replicates: int = 2,
    tile_min_points: int = 3,
    voronoi_fraction: float = 0.4,
    base_seed: int = 100,
) -> SampleTessellationResult:
    partition = partition_component_domain(
        component.polygon,
        component.points,
        component.expression_data[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    frame = component.expression_data.copy()
    frame[ROI_ID] = "A"
    return run_tessellation_stage(
        sample_id="sample1",
        spatial_components={component.component_id: component},
        domain_partitions={component.component_id: partition},
        component_to_roi={component.component_id: "A"},
        n_replicates=n_replicates,
        tile_min_points=tile_min_points,
        voronoi_fraction=voronoi_fraction,
        base_seed=base_seed,
        source_spot_adjacency=build_source_spot_adjacency(
            build_source_spot_table_from_frame(frame),
        ),
        config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
        },
        ),
    )


# ---------------------------------------------------------------------------
# Diagnostic emission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_left, expect_left_eligible",
    [(3, False), (5, True)],
    ids=["below_floor_ineligible", "at_floor_eligible"],
)
def test_construction_floor_gates_region_eligibility_at_tile_min_points(
    n_left, expect_left_eligible,
):
    # Pins both sides of the construction-floor boundary at
    # tile_min_points=5: n_left=3 ineligible, n_left=5 eligible.
    component, region_diags = _two_region_component(n_left=n_left, n_right=20)
    result = _run(component, region_diags, tile_min_points=5)

    comp_result = result.components["A"]
    left_diag = next(
        d for d in comp_result.region_diagnostics if d.n_component_spots == n_left
    )
    if expect_left_eligible:
        assert left_diag.is_tessellation_eligible is True
        assert left_diag.ineligibility_reason is None
    else:
        assert left_diag.is_tessellation_eligible is False
        assert (
            left_diag.ineligibility_reason
            is RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN
        )
        # No region-replicate rows for the ineligible region.
        rr = comp_result.region_replicate_diagnostics
        assert not any(
            d.domain_region_id == left_diag.domain_region_id for d in rr
        )


def test_component_with_no_eligible_region_produces_empty_tessellations():
    # 30/30 split, construction floor 40 → neither region eligible.
    component, region_diags = _two_region_component(n_left=30, n_right=30)
    result = _run(component, region_diags, tile_min_points=40)

    comp = result.components["A"]
    assert dict(comp.tessellations) == {}
    assert comp.region_replicate_diagnostics == ()
    n_eligible = sum(
        1 for d in comp.region_diagnostics if d.is_tessellation_eligible
    )
    assert n_eligible == 0


# ---------------------------------------------------------------------------
# Reproducibility + base-seed threading
# ---------------------------------------------------------------------------


def _tile_signature(result):
    sig = []
    for cid, comp in sorted(result.components.items()):
        for rid, rep_map in sorted(comp.tessellations.items()):
            for rep, tess in sorted(rep_map.items()):
                for tile in tess.tiles:
                    sig.append((
                        cid, rid, rep, tile.tile_id, tile.n_spots,
                        float(tile.area),
                        tile.polygon.wkb,
                        tile.point_indices.tobytes(),
                    ))
    return sig


def test_same_base_seed_bit_identical():
    component, region_diags = _two_region_component()
    a = _run(component, region_diags, base_seed=13)
    b = _run(component, region_diags, base_seed=13)
    assert _tile_signature(a) == _tile_signature(b)


def test_different_base_seed_diverges():
    component, region_diags = _two_region_component()
    a = _run(component, region_diags, base_seed=13)
    b = _run(component, region_diags, base_seed=999)
    # If base_seed weren't threaded, these would be equal.
    assert _tile_signature(a) != _tile_signature(b)


def test_tessellation_seed_stable_under_component_id_relabelling():
    """Regression: tessellation seed derives from the region's source-
    row fingerprint (stable digest of the spot membership), NOT from
    the ordinal component_id. Relabelling component_id must therefore
    leave the tessellation ensemble bit-identical."""
    comp_a, _ = _two_region_component(component_id="c_ordinal_007")
    comp_b, _ = _two_region_component(component_id="c_ordinal_042")
    # Same coords, same source_row_ids, same regions -- only the
    # label differs. Under the old ordinal-in-seed derivation the two
    # runs would produce different tile positions.
    a = _run(comp_a, [], base_seed=13)
    b = _run(comp_b, [], base_seed=13)
    sig_a = _tile_signature(a)
    sig_b = _tile_signature(b)
    # Strip the component_id from the signature so we compare tile
    # geometry only, not the label.
    strip_a = [t[1:] for t in sig_a]
    strip_b = [t[1:] for t in sig_b]
    assert strip_a == strip_b, (
        "Tessellation drifted when only component_id relabelled; "
        "seed still depends on ordinal identity."
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _prod_kwargs(component):
    """Production-shaped source_spot_adjacency + config for run_tessellation_stage."""
    frame = component.expression_data.copy()
    frame[ROI_ID] = "A"
    return {
        "source_spot_adjacency": build_source_spot_adjacency(
            build_source_spot_table_from_frame(frame),
        ),
        "config": load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
        },
        ),
    }


def test_missing_component_to_roi_key_raises():
    component, region_diags = _two_region_component()
    partition = partition_component_domain(
        component.polygon, component.points,
        component.expression_data[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    with pytest.raises(ValueError, match="component_to_roi"):
        run_tessellation_stage(
            sample_id="s", spatial_components={"A": component},
            domain_partitions={"A": partition},
            component_to_roi={},   # missing 'A'
            n_replicates=1,
            tile_min_points=3, voronoi_fraction=0.4, base_seed=0,
            **_prod_kwargs(component),
        )


def test_missing_domain_partition_key_raises():
    component, region_diags = _two_region_component()
    with pytest.raises(ValueError, match="domain_partitions"):
        run_tessellation_stage(
            sample_id="s", spatial_components={"A": component},
            domain_partitions={},   # missing 'A'
            component_to_roi={"A": "A"},
            n_replicates=1,
            tile_min_points=3, voronoi_fraction=0.4, base_seed=0,
            **_prod_kwargs(component),
        )


@pytest.mark.parametrize(
    "kwargs, error_pattern",
    [
        ({"n_replicates": 0}, "n_replicates"),
        ({"n_replicates": -1}, "n_replicates"),
        ({"tile_min_points": 0}, "tile_min_points"),
    ],
    ids=["n_replicates_zero", "n_replicates_negative", "tile_min_points_zero"],
)
def test_invalid_construction_input_raises(kwargs, error_pattern):
    component, region_diags = _two_region_component()
    with pytest.raises(ValueError, match=error_pattern):
        _run(component, region_diags, **kwargs)


# ---------------------------------------------------------------------------
# SampleTessellationResult identity invariants
# ---------------------------------------------------------------------------


def test_component_id_property_derived_from_component_data():
    component, region_diags = _two_region_component(component_id="Q_c1")
    partition = partition_component_domain(
        component.polygon, component.points,
        component.expression_data[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    result = run_tessellation_stage(
        sample_id="s", spatial_components={"Q_c1": component},
        domain_partitions={"Q_c1": partition},
        component_to_roi={"Q_c1": "Q"},
        n_replicates=1,
        tile_min_points=3, voronoi_fraction=0.4, base_seed=0,
        **_prod_kwargs(component),
    )
    assert result.components["Q_c1"].component_id == "Q_c1"
    assert result.component_to_roi == {"Q_c1": "Q"}


def test_components_are_iterated_in_canonical_sorted_order():
    # Caller passes components in reverse-alphabetical insertion
    # order. Result must expose them in sorted order so downstream
    # tables are stable regardless of caller's dict construction.
    a, a_diags = _two_region_component(component_id="Z_c1")
    b, b_diags = _two_region_component(component_id="A_c1")
    c, c_diags = _two_region_component(component_id="M_c1")
    partitions = {
        component.component_id: partition_component_domain(
            component.polygon, component.points,
            component.expression_data[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
        )
        for component in (a, b, c)
    }
    result = run_tessellation_stage(
        sample_id="sample1",
        spatial_components={"Z_c1": a, "A_c1": b, "M_c1": c},
        domain_partitions=partitions,
        component_to_roi={"Z_c1": "Z", "A_c1": "A", "M_c1": "M"},
        **_prod_kwargs(a),
        n_replicates=1,
        tile_min_points=3, voronoi_fraction=0.4, base_seed=0,
    )
    keys = list(result.components)
    assert keys == sorted(keys) == ["A_c1", "M_c1", "Z_c1"]
