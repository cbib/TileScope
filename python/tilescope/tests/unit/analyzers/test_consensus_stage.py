"""Contract + stage-boundary tests for the typed consensus stage.

Six semantic scenarios (some parametrised):

1. `sample_anchor` finite-value validation.
2. `SampleConsensusResult` mapping-key ↔ `component_id` validation
   and immutability of `components`.
3. Zero-eligible-region component yields empty hexes.
4. Stage-boundary single-region: replicate-coverage filter honours
   `config.resolved_policy.consensus.min_tess_per_hex`, and stored hex
   centres come from the lattice-cell centres returned by
   `_generate_hex_grid`.
5. Stage-boundary multi-region: two-region component produces
   unique `hex_id`s across the component with correct
   `domain_region_id` labelling and same-anchor lattice alignment.
6. ConsensusHex derived-density + obs_equiv_mass round-trip +
   __post_init__ coherence invariants.
"""
from __future__ import annotations

from tilescope.analyzers.sample_support_stage import sample_support_bank_from_component_results
from tilescope.spatial.component_domains import ComponentDomainPartition
from types import MappingProxyType as _MPT
from tilescope.spatial.component_domains import partition_component_domain
from tilescope.analyzers.consensus_grid import _generate_hex_grid


from dataclasses import fields
from types import MappingProxyType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from tilescope.analyzers.consensus_stage import (
    ComponentConsensusResult,
    ConsensusHex,
    SampleConsensusResult,
    _run_component,
    run_consensus_stage,
)
from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult,
    SampleTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionExecutionStatus,
    RegionReplicateTessellationDiagnostic,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.cols import GENE1, GENE2, SOURCE_ROW_ID
from tilescope.component_data import ComponentData, Tessellation
from tilescope.spatial.canvas import make_canvas
from tilescope.settings import load_run_config
from tilescope.component_data import Tile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hex(
    *, component_id: str = "C", **overrides,
) -> ConsensusHex:
    """Test-fixture helper."""
    defaults = dict(
        component_id=component_id,
        hex_id=0, domain_region_id=0,
        polygon=box(0.0, 0.0, 1.0, 1.0),
        lattice_col=0, lattice_row=0,
        center_x=0.5, center_y=0.5,
        n_tessellations_covered=1,
        n_spots_ensemble_weighted=1.0,
        n_spots_direct=1,
    )
    defaults.update(overrides)
    return ConsensusHex.build(**defaults)


def _stub_sample_tessellation_result(components) -> SampleTessellationResult:
    return SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType(components),
        n_replicates_planned=1,
    )


def _sample_from_components(
    components, sample_id: str = "S", n_replicates_planned: int = 1,
    sample_anchor_x: float = 0.0, sample_anchor_y: float = 0.0,
) -> SampleConsensusResult:
    """Fixture helper: compose SampleConsensusResult with an empty support bank."""
    support_bank = sample_support_bank_from_component_results(
        sample_id=sample_id,
        component_results={},
        n_replicates=n_replicates_planned,
        base_seed=0,
    )
    return SampleConsensusResult.from_components(
        sample_id=sample_id, components=components,
        n_replicates_planned=n_replicates_planned,
        sample_anchor_x=sample_anchor_x, sample_anchor_y=sample_anchor_y,
        support_bank=support_bank,
    )


# ---------------------------------------------------------------------------
# 1. sample_anchor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_anchor,pattern",
    [
        ((float("nan"), 0.0), "finite"),
        ((0.0, float("nan")), "finite"),
        ((float("inf"), 0.0), "finite"),
        ((0.0,), "2-tuple"),
        ("not a tuple", "2-tuple"),
    ],
    ids=["nan_x", "nan_y", "inf_x", "one_element", "not_a_tuple"],
)
def test_sample_anchor_validation_rejects_bad_input(bad_anchor, pattern):
    result = _stub_sample_tessellation_result({})
    with pytest.raises(ValueError, match=pattern):
        run_consensus_stage(
            sample_tessellation_result=result,
            canvas=Mock(spec=[]),
            sample_anchor=bad_anchor,
            consensus_config=Mock(spec=[]),
        )


# ---------------------------------------------------------------------------
# 2. SampleConsensusResult mapping / immutability
# ---------------------------------------------------------------------------


def test_sample_consensus_result_rejects_mapping_key_component_id_mismatch():
    comp = ComponentConsensusResult.from_hexes(
        parent_roi_id="R", component_id="C_real",
        hex_radius=1.0, hexes=(), n_tessellation_eligible_regions=1,
    )
    with pytest.raises(ValueError, match="disagrees with"):
        _sample_from_components({"WRONG_KEY": comp})


def test_sample_consensus_result_freezes_components_mapping():
    comp = ComponentConsensusResult.from_hexes(
        parent_roi_id="R", component_id="C",
        hex_radius=1.0, hexes=(), n_tessellation_eligible_regions=1,
    )
    result = _sample_from_components({"C": comp})
    with pytest.raises(TypeError):
        result.components["C2"] = comp  # type: ignore[index]


# ---------------------------------------------------------------------------
# ConsensusHex joint (bivariate) invariants + bank-delegated accessors
# ---------------------------------------------------------------------------


def _make_hex(
    *,
    n_tess=2, n_spots_ew=4.0,
    threshold_kwargs=None,
):
    kwargs = dict(
        component_id="C",
        hex_id=0, domain_region_id=0,
        polygon=box(0.0, 0.0, 1.0, 1.0),
        lattice_col=0, lattice_row=0, center_x=0.5, center_y=0.5,
        n_tessellations_covered=n_tess,
        n_spots_ensemble_weighted=n_spots_ew,
        n_spots_direct=int(n_spots_ew),
    )
    if threshold_kwargs:
        kwargs.update(threshold_kwargs)
    return ConsensusHex.build(**kwargs)


def test_joint_accumulators_are_all_or_none():
    complete = dict(
        n_both_high_ensemble_weighted=0.5,
        n_both_low_ensemble_weighted=0.5,
    )
    _make_hex(threshold_kwargs=complete)
    _make_hex(threshold_kwargs=None)
    for missing in complete:
        partial = {k: v for k, v in complete.items() if k != missing}
        with pytest.raises(ValueError, match="all-present or all-absent"):
            _make_hex(threshold_kwargs=partial)


def test_zero_exposure_hex_rejected_by_post_init():
    with pytest.raises(ValueError, match="a retained hex must have positive"):
        _make_hex(n_tess=0)


@pytest.mark.parametrize(
    "bad_kwargs,pattern",
    [
        (dict(n_both_high_ensemble_weighted=10.0,
              n_both_low_ensemble_weighted=0.5),
         "n_both_high_ensemble_weighted=10.0 > n_spots_ensemble_weighted"),
        (dict(n_both_high_ensemble_weighted=-0.5,
              n_both_low_ensemble_weighted=0.5),
         "is negative"),
    ],
    ids=["joint_exceeds_spot", "joint_negative"],
)
def test_post_init_rejects_incoherent_joint_accumulators(bad_kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        _make_hex(threshold_kwargs=bad_kwargs)


def test_obs_equiv_both_mass_round_trip():
    h = _make_hex(
        n_tess=3, n_spots_ew=7.0,
        threshold_kwargs=dict(
            n_both_high_ensemble_weighted=1.25,
            n_both_low_ensemble_weighted=0.75,
        ),
    )
    n = h.n_tessellations_covered
    assert h.obs_equiv_spot_mass * n == pytest.approx(
        h.n_spots_ensemble_weighted,
    )
    assert h.obs_equiv_both_high_mass * n == pytest.approx(1.25)
    assert h.obs_equiv_both_low_mass * n == pytest.approx(0.75)


def test_consensus_hex_stores_geometry_once_and_delegates_access():
    h = _make_hex()
    assert tuple(field.name for field in fields(h)) == (
        "geometry",
        "n_both_high_ensemble_weighted",
        "n_both_low_ensemble_weighted",
    )
    for name in (
        "hex_id", "domain_region_id", "polygon", "lattice_col",
        "lattice_row", "center_x", "center_y",
        "n_tessellations_covered", "n_spots_ensemble_weighted",
        "n_spots_direct",
    ):
        assert getattr(h, name) is getattr(h.geometry, name)


# ---------------------------------------------------------------------------
# 3. Zero-eligible-region component → empty hexes
# ---------------------------------------------------------------------------


def test_no_eligible_regions_yields_no_hexes():
    result = _run_component(
        parent_roi_id="R", component_id="C",
        component_data=Mock(spec=[]),
        eligible_region_diagnostics=(),
        tessellations_by_region={},
        domain_partition=ComponentDomainPartition(
            canonical_regions=(),
            canonical_ids_by_source_row=_MPT({}),
            source_row_ids_by_canonical_id=_MPT({}),
            source_row_fingerprint_by_canonical_id=_MPT({}),
        ),
        canvas=Mock(spec=[]),
        anchor_x=0.0, anchor_y=0.0,
        consensus_config=Mock(spec=[]),
    )
    assert result.hexes == ()
    assert result.n_hexes == 0


# ---------------------------------------------------------------------------
# 4. Stage-boundary: coverage-fraction filtering + lattice-cell centres
# ---------------------------------------------------------------------------


def _build_partial_coverage_sample_result():
    region_polygon = box(0.0, 0.0, 20.0, 8.0)
    xs = np.arange(0.5, 20.0, 1.0)
    ys = np.arange(0.5, 8.0, 1.0)
    XX, YY = np.meshgrid(xs, ys)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    n = len(coords)

    canvas_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    canvas = make_canvas(canvas_df)
    expr = pd.DataFrame({
        GENE1: np.ones(n),
        GENE2: np.arange(n, dtype=float),
        SOURCE_ROW_ID: np.arange(n, dtype=np.int64),
        "x": coords[:, 0],
        "y": coords[:, 1],
    })
    comp_data = ComponentData(
        component_id="C", points=coords.astype(float),
        canvas=canvas, polygon=region_polygon, expression_data=expr,
    )

    def _mk_tess(rep_idx: int, x_high: float) -> Tessellation:
        tid = f"C_rep{rep_idx}"
        mask = coords[:, 0] < x_high
        tile = Tile(
            tile_id=0, polygon=box(0.0, 0.0, x_high, 8.0),
            point_indices=np.flatnonzero(mask), n_spots=int(mask.sum()),
            area=x_high * 8.0,
            tessellation_id=tid, domain_region_id=0,
        )
        return Tessellation(
            tessellation_id=tid, component_id="C", tiles=[tile],
            method="voronoi", tile_min_points=1, fraction=1.0, rng_seed=42,
        )

    tessellations = {0: {r: _mk_tess(r, high) for r, high in
                         enumerate([20.0, 15.0, 10.0])}}
    region_diag = RegionTessellationEligibilityDiagnostic(
        component_id="C", domain_region_id=0,
        n_component_spots=n, region_area=float(region_polygon.area),
        ineligibility_reason=None,
    )
    partition = partition_component_domain(
        comp_data.polygon, coords.astype(float),
        expr[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    rr_diags = tuple(
        RegionReplicateTessellationDiagnostic(
            component_id="C", domain_region_id=0, replicate_index=r,
            n_tiles=1, region_status=RegionExecutionStatus.CONTRIBUTED,
        )
        for r in range(3)
    )
    comp_result = ComponentTessellationResult(
        parent_roi_id="R", component_data=comp_data,
        tessellations=MappingProxyType(tessellations),
        region_diagnostics=(region_diag,),
        region_replicate_diagnostics=rr_diags,
        domain_partition=partition,
    )
    sample_result = SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({"C": comp_result}),
        n_replicates_planned=3,
    )
    return sample_result, canvas


def test_stage_boundary_min_tess_filter_and_lattice_cell_centres():
    sample_result, canvas = _build_partial_coverage_sample_result()
    anchor = (0.0, 0.0)

    strict = run_consensus_stage(
        sample_tessellation_result=sample_result, canvas=canvas,
        sample_anchor=anchor,
        consensus_config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
            'consensus': {
                'min_tess_coverage_fraction': 0.3,
            },
        },
        ),
    ).components["C"]
    lax = run_consensus_stage(
        sample_tessellation_result=sample_result, canvas=canvas,
        sample_anchor=anchor,
        consensus_config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
            'consensus': {
                'min_tess_coverage_fraction': 0.1,
            },
        },
        ),
    ).components["C"]

    assert len(strict.hexes) > 0
    assert len(strict.hexes) < len(lax.hexes)
    assert all(h.n_tessellations_covered >= 3 for h in strict.hexes)
    assert any(h.n_tessellations_covered < 3 for h in lax.hexes)

    assert [h.hex_id for h in strict.hexes] == list(range(len(strict.hexes)))
    canonical_keys = [(h.lattice_col, h.lattice_row) for h in strict.hexes]
    assert canonical_keys == sorted(canonical_keys)

    grid_by_key = {(c, r): (cx, cy) for c, r, cx, cy, _ in _generate_hex_grid(
        sample_result.components["C"].component_data.polygon,
        strict.hex_radius,
        anchor_x=anchor[0], anchor_y=anchor[1],
    )}
    for h in strict.hexes:
        expected = grid_by_key[(h.lattice_col, h.lattice_row)]
        assert h.center_x == pytest.approx(expected[0])
        assert h.center_y == pytest.approx(expected[1])


# ---------------------------------------------------------------------------
# 5. Stage-boundary multi-region: two-region component
# ---------------------------------------------------------------------------


def _build_two_region_sample_result():
    left = box(0.0, 0.0, 6.0, 6.0)
    right = box(12.0, 0.0, 18.0, 6.0)
    comp_polygon = MultiPolygon([left, right])
    xs_l = np.arange(0.5, 6.0, 1.0)
    xs_r = np.arange(12.5, 18.0, 1.0)
    ys = np.arange(0.5, 6.0, 1.0)
    XL, YL = np.meshgrid(xs_l, ys)
    XR, YR = np.meshgrid(xs_r, ys)
    coords = np.column_stack([
        np.concatenate([XL.ravel(), XR.ravel()]),
        np.concatenate([YL.ravel(), YR.ravel()]),
    ])
    n = len(coords)
    canvas = make_canvas(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}))
    expr = pd.DataFrame({
        GENE1: np.ones(n),
        GENE2: np.arange(n, dtype=float),
        SOURCE_ROW_ID: np.arange(n, dtype=np.int64),
        "x": coords[:, 0], "y": coords[:, 1],
    })
    comp_data = ComponentData(
        component_id="C", points=coords.astype(float),
        canvas=canvas, polygon=comp_polygon, expression_data=expr,
    )

    def _tess(region_id: int, rep_idx: int, tile_box) -> Tessellation:
        tid = f"C_r{region_id}_rep{rep_idx}"
        minx, miny, maxx, maxy = tile_box.bounds
        mask = (
            (coords[:, 0] >= minx) & (coords[:, 0] < maxx)
            & (coords[:, 1] >= miny) & (coords[:, 1] < maxy)
        )
        return Tessellation(
            tessellation_id=tid, component_id="C",
            tiles=[Tile(
                tile_id=0, polygon=tile_box, point_indices=np.flatnonzero(mask),
                n_spots=int(mask.sum()), area=float(tile_box.area),
                tessellation_id=tid, domain_region_id=region_id,
            )],
            method="voronoi", tile_min_points=1, fraction=1.0, rng_seed=42,
        )

    tessellations = {
        0: {r: _tess(0, r, left) for r in range(2)},
        1: {r: _tess(1, r, right) for r in range(2)},
    }
    region_diags = tuple(
        RegionTessellationEligibilityDiagnostic(
            component_id="C", domain_region_id=rid,
            n_component_spots=int(len(coords) // 2),
            region_area=float(left.area),
            ineligibility_reason=None,
        )
        for rid in (0, 1)
    )
    partition = partition_component_domain(
        comp_data.polygon, coords.astype(float),
        expr[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    rr_diags = tuple(
        RegionReplicateTessellationDiagnostic(
            component_id="C", domain_region_id=rid, replicate_index=r,
            n_tiles=1, region_status=RegionExecutionStatus.CONTRIBUTED,
        )
        for rid in (0, 1) for r in range(2)
    )
    comp_result = ComponentTessellationResult(
        parent_roi_id="R", component_data=comp_data,
        tessellations=MappingProxyType(tessellations),
        region_diagnostics=region_diags,
        region_replicate_diagnostics=rr_diags,
        domain_partition=partition,
    )
    return SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({"C": comp_result}),
        n_replicates_planned=2,
    ), canvas


def test_stage_boundary_multi_region_unique_ids_and_anchor_alignment():
    sample_result, canvas = _build_two_region_sample_result()
    anchor = (0.0, 0.0)
    result = run_consensus_stage(
        sample_tessellation_result=sample_result, canvas=canvas,
        sample_anchor=anchor,
        consensus_config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
            'consensus': {
                'min_tess_coverage_fraction': 0.2,
            },
        },
        ),
    ).components["C"]

    region_ids = {h.domain_region_id for h in result.hexes}
    assert region_ids == {0, 1}

    hex_ids = [h.hex_id for h in result.hexes]
    assert hex_ids == list(range(len(result.hexes)))
    assert len(set(hex_ids)) == len(hex_ids)

    canonical = [(h.domain_region_id, h.lattice_col, h.lattice_row)
                 for h in result.hexes]
    assert canonical == sorted(canonical)

    grid = {(c, rw): (cx, cy) for c, rw, cx, cy, _ in _generate_hex_grid(
        sample_result.components["C"].component_data.polygon,
        result.hex_radius,
        anchor_x=anchor[0], anchor_y=anchor[1],
    )}
    for h in result.hexes:
        expected = grid[(h.lattice_col, h.lattice_row)]
        assert h.center_x == pytest.approx(expected[0])
        assert h.center_y == pytest.approx(expected[1])
