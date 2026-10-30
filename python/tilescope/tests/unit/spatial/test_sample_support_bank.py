"""Architectural tests for the sample-scope support bank.

Covers recursive immutability, NaN rejection at ingest, build-once,
and derived-view identity contracts.
"""
from __future__ import annotations

from tilescope.component_data import Tile
from shapely.geometry import box
from tilescope.component_data import Tessellation
from tilescope.settings import load_run_config
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
from tilescope.analyzers import sample_support_stage as sss_mod
from types import MappingProxyType
from tilescope.analyzers.sample_support_stage import build_empty_sample_support_bank


from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from tilescope.analyzers.tessellation_analysis import (
    SampleTessellationResult, ComponentTessellationResult,
    run_tessellation_stage,
)
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, ROI_ID, SOURCE_ROW_ID
from tilescope.component_data import ComponentData
from tilescope.spatial.component_domains import partition_component_domain
from tilescope.spatial.canvas import make_canvas
from tilescope.spatial.source_spot_table import (
    FeatureNonfiniteRejectedError,
    SourceSpotTable,
    reject_feature_nonfinite,
)


# ---------------------------------------------------------------------------
# Recursive immutability
# ---------------------------------------------------------------------------


def test_source_spot_table_arrays_are_read_only():
    t = SourceSpotTable(
        spot_id=("a", "b", "c"),
        x=np.asarray([0.0, 1.0, 2.0]),
        y=np.asarray([0.0, 1.0, 2.0]),
        roi_id=("R", "R", "R"),
    )
    with pytest.raises(ValueError):
        t.x[0] = 99.0
    with pytest.raises(ValueError):
        t.y[0] = 99.0


def test_source_spot_table_frozen_attribute_rebinding_rejected():
    t = SourceSpotTable(
        spot_id=("a",), x=np.asarray([0.0]), y=np.asarray([0.0]), roi_id=("R",),
    )
    with pytest.raises(Exception):
        t.x = np.asarray([1.0])  # type: ignore[misc]


def test_source_spot_table_rejects_malformed_shape():
    with pytest.raises(ValueError, match="1-D"):
        SourceSpotTable(
            spot_id=("a",), x=np.zeros((1, 2)), y=np.zeros(1), roi_id=("R",),
        )
    with pytest.raises(ValueError, match="length mismatch"):
        SourceSpotTable(
            spot_id=("a", "b"), x=np.zeros(1), y=np.zeros(1), roi_id=("R", "R"),
        )


def test_tile_point_indices_are_read_only_after_construction():
    tile = Tile(
        tile_id=0, polygon=box(0, 0, 1, 1),
        point_indices=np.asarray([0, 1, 2]),
        n_spots=3, area=1.0,
    )
    with pytest.raises(ValueError):
        tile.point_indices[0] = 999


def test_tessellation_tiles_is_tuple_not_list():
    tile = Tile(
        tile_id=0, polygon=box(0, 0, 1, 1),
        point_indices=np.asarray([0]), n_spots=1, area=1.0,
    )
    tess = Tessellation(
        tessellation_id="t", component_id="c",
        tiles=[tile], method="voronoi",
        tile_min_points=1, fraction=0.2, rng_seed=0,
    )
    assert isinstance(tess.tiles, tuple)


def test_component_data_requires_source_identity_not_legacy_gene_names():
    """Spatial attachment accepts arbitrary features and pins row identity."""
    points = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    component = ComponentData(
        component_id="C", points=points, canvas=object(),
        polygon=Polygon([(-1, -1), (2, -1), (2, 1), (-1, 1)]),
        expression_data=pd.DataFrame({
            SOURCE_ROW_ID: [0, 1], "G3": [1.0, 2.0], "G4": [3.0, 4.0],
        }),
    )
    assert list(component.expression_data.columns) == [SOURCE_ROW_ID, "G3", "G4"]

    with pytest.raises(ValueError, match="missing required 'source_row_id'"):
        ComponentData(
            component_id="C", points=points, canvas=object(),
            polygon=component.polygon,
            expression_data=pd.DataFrame({"G3": [1.0, 2.0]}),
        )
    with pytest.raises(ValueError, match=r"rows \(1\) != points \(2\)"):
        ComponentData(
            component_id="C", points=points, canvas=object(),
            polygon=component.polygon,
            expression_data=pd.DataFrame({SOURCE_ROW_ID: [0], "G3": [1.0]}),
        )


# ---------------------------------------------------------------------------
# NaN rejection at ingest
# ---------------------------------------------------------------------------


def test_source_frame_ingest_rejects_nan_coordinates():
    df = pd.DataFrame({
        SOURCE_ROW_ID: [0, 1, 2],
        COORD_X: [0.0, np.nan, 2.0],
        COORD_Y: [0.0, 1.0, 2.0],
        ROI_ID: ["R", "R", "R"],
    })
    with pytest.raises(ValueError, match="COORD_X|x"):
        build_source_spot_table_from_frame(df)


def test_source_frame_ingest_rejects_nan_roi():
    df = pd.DataFrame({
        SOURCE_ROW_ID: [0, 1],
        COORD_X: [0.0, 1.0], COORD_Y: [0.0, 1.0],
        ROI_ID: ["R", None],
    })
    with pytest.raises(ValueError, match="ROI_ID|roi_id"):
        build_source_spot_table_from_frame(df)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_reject_feature_nonfinite_raises_typed_error(bad_value):
    df = pd.DataFrame({
        SOURCE_ROW_ID: [0, 1, 2],
        GENE1: [1.0, bad_value, 3.0],
        GENE2: [1.0, 2.0, 3.0],
    })
    with pytest.raises(FeatureNonfiniteRejectedError, match=r"gene1|non-finite"):
        reject_feature_nonfinite(df, (GENE1, GENE2))


def test_reject_feature_nonfinite_passes_clean_columns():
    df = pd.DataFrame({
        GENE1: [1.0, 2.0], GENE2: [3.0, 4.0],
    })
    reject_feature_nonfinite(df, (GENE1, GENE2))  # must not raise


# ---------------------------------------------------------------------------
# Build-once + identity contract
# ---------------------------------------------------------------------------


def _minimal_component_dataset():
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    coords = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
    n = len(coords)
    expr = pd.DataFrame({
        SOURCE_ROW_ID: np.arange(n, dtype=np.int64),
        GENE1: np.ones(n), GENE2: np.arange(n, dtype=float),
        COORD_X: coords[:, 0], COORD_Y: coords[:, 1],
        ROI_ID: ["R"] * n,
    })
    canvas = make_canvas(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}))
    poly = Polygon([(-1, -1), (5, -1), (5, 5), (-1, 5)])
    cd = ComponentData(
        component_id="C", points=coords.astype(float),
        canvas=canvas, polygon=poly, expression_data=expr,
    )
    partition = partition_component_domain(
        poly, coords, expr[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    return {"C": cd}, {"C": partition}, {"C": "R"}


def _minimal_config():
    return load_run_config(profile="standard", set_overrides={"output_dir": "tests/tests_output"})


def _run_stage(comps, partitions, mapping):
    # Build a source_expression_df from the concatenated expression_data.
    concat = pd.concat([cd.expression_data for cd in comps.values()], ignore_index=True)
    sst = build_source_spot_table_from_frame(concat)
    adj = build_source_spot_adjacency(sst)
    return run_tessellation_stage(
        sample_id="S", spatial_components=comps,
        domain_partitions=partitions, component_to_roi=mapping,
        n_replicates=2, tile_min_points=1, voronoi_fraction=0.4,
        base_seed=0, source_spot_adjacency=adj, config=_minimal_config(),
    )


def test_run_tessellation_stage_calls_bank_builder_exactly_once():
    """Build-once contract: bank builder invoked exactly once per stage call."""

    comps, partitions, mapping = _minimal_component_dataset()
    counter = {"n": 0}
    real_builder = sss_mod.build_sample_support_bank

    def _spy(*args, **kwargs):
        counter["n"] += 1
        return real_builder(*args, **kwargs)

    with patch.object(sss_mod, "build_sample_support_bank", _spy):
        result = _run_stage(comps, partitions, mapping)
    assert counter["n"] == 1
    assert result.support_bank is not None


def test_sample_tessellation_result_substrate_fields_are_bank_owned_by_identity():
    comps, partitions, mapping = _minimal_component_dataset()
    result = _run_stage(comps, partitions, mapping)
    bank = result.support_bank
    for cid, cres in result.components.items():
        assert cres.tessellations is bank.tessellation_bank.tessellations[cid]
        assert cres.region_diagnostics is bank.tessellation_bank.region_diagnostics[cid]
        assert cres.region_replicate_diagnostics is (
            bank.tessellation_bank.region_replicate_diagnostics[cid]
        )
        assert cres.domain_partition is bank.component_domains[cid]


def test_sample_tessellation_result_rejects_independent_substrate():
    """Constructor rejects substrate state not `is`-equal to the attached bank's."""
    comps, partitions, mapping = _minimal_component_dataset()
    result = _run_stage(comps, partitions, mapping)
    bank = result.support_bank
    cid = next(iter(result.components))
    cres = result.components[cid]
    forged = ComponentTessellationResult(
        parent_roi_id=cres.parent_roi_id,
        component_data=cres.component_data,
        tessellations=MappingProxyType(dict(cres.tessellations)),  # NEW object
        region_diagnostics=cres.region_diagnostics,
        region_replicate_diagnostics=cres.region_replicate_diagnostics,
        domain_partition=cres.domain_partition,
    )
    with pytest.raises(ValueError, match="is`-identical|is-identical"):
        SampleTessellationResult(
            sample_id="S",
            components=MappingProxyType({cid: forged}),
            n_replicates_planned=2,
            support_bank=bank,
        )


def test_alias_mutation_after_freeze_is_rejected():
    """Mutating the source array must not leak into the frozen record."""
    original_x = np.asarray([1.0, 2.0, 3.0])
    t = SourceSpotTable(
        spot_id=("a", "b", "c"), x=original_x, y=np.zeros(3),
        roi_id=("R", "R", "R"),
    )
    original_x[0] = 999.0
    assert t.x[0] == 1.0, "alias mutation leaked into the frozen table"


def test_source_spot_table_rejects_duplicate_spot_ids():
    with pytest.raises(ValueError, match="unique"):
        SourceSpotTable(
            spot_id=("a", "b", "a"),
            x=np.zeros(3), y=np.zeros(3),
            roi_id=("R", "R", "R"),
        )


def test_empty_scattered_bank_uses_shared_builder():
    """All-scattered path builds an empty bank through the same builder."""
    df = pd.DataFrame({
        SOURCE_ROW_ID: [0, 1], COORD_X: [0.0, 1.0], COORD_Y: [0.0, 1.0],
        ROI_ID: ["R", "R"], GENE1: [1.0, 2.0], GENE2: [1.0, 2.0],
    })
    sst = build_source_spot_table_from_frame(df)
    adj = build_source_spot_adjacency(sst)
    bank = build_empty_sample_support_bank(
        sample_id="S", source_spot_adjacency=adj, config=_minimal_config(),
        n_replicates=3, base_seed=0,
    )
    assert bank.components == {}
    assert bank.component_domains == {}
    assert bank.tessellation_bank.n_replicates_planned == 3
    assert bank.tessellation_bank.tessellations == {}
