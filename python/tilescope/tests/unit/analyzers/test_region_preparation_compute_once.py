"""Compute-once contract for region preparation.

Counts how many times region-static work is done across a
multi-region multi-replicate run. If a future edit moves
``prepare_component`` / ``prepare_region`` inside the K-replicate
loop these tests fail.

Contracts:

- ``median_nn_distance`` is called EXACTLY once per region
  (not per replicate).
- ``sorted(set(map(tuple, roi_coords)))`` — the unique-coord
  reduction inside ``select_seeds`` — runs zero times when
  ``PreparedRegion`` supplies ``precomputed_unique_coords``.
- ``prepare_component`` runs once per component.
- ``prepare_region`` runs once per (component, region).
- Every replicate reuses the same ``PreparedRegion`` object.
- Generic ``compute_tessellation(precomputed_d1=None)`` computes
  ``d1`` exactly once (engine no longer recomputes).
"""
from __future__ import annotations

from tilescope.tessellation.voronoi import select_seeds
from tilescope.tessellation import builder as builder_mod


from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from tilescope.analyzers import region_tessellation as rt_mod
from tilescope.analyzers import tessellation_analysis as ta_mod
from tilescope.analyzers.tessellation_analysis import _run_component
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, SOURCE_ROW_ID
from tilescope.component_data import ComponentData
from tilescope.spatial.canvas import make_canvas
from tilescope.spatial.component_domains import partition_component_domain
from tilescope.tessellation.engine_geos import GeosVoronoiEngine


def _two_region_component() -> ComponentData:
    """A two-region component: left half [0,6]x[0,6] + right half
    [12,18]x[0,6] joined by MultiPolygon. Region 0 has 30 spots,
    region 1 has 24 spots."""
    left = box(0.0, 0.0, 6.0, 6.0)
    right = box(12.0, 0.0, 18.0, 6.0)
    poly = MultiPolygon([left, right])
    xs_l = np.arange(0.5, 6.0, 1.0)
    xs_r = np.arange(12.5, 18.0, 1.0)
    ys = np.arange(0.5, 6.0, 1.0)
    xl, yl = np.meshgrid(xs_l, ys)
    xr, yr = np.meshgrid(xs_r, ys)
    coords = np.column_stack([
        np.concatenate([xl.ravel(), xr.ravel()]),
        np.concatenate([yl.ravel(), yr.ravel()]),
    ])
    n = len(coords)
    canvas = make_canvas(pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}))
    expr = pd.DataFrame({
        GENE1: np.ones(n),
        GENE2: np.arange(n, dtype=float),
        SOURCE_ROW_ID: np.arange(n, dtype=np.int64),
        COORD_X: coords[:, 0], COORD_Y: coords[:, 1],
    })
    return ComponentData(
        component_id="C", points=coords.astype(float),
        canvas=canvas, polygon=poly, expression_data=expr,
    )


class _Counter:
    """Callable wrapper that delegates to a real function and counts
    invocations."""
    def __init__(self, real):
        self.real = real
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self.real(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. Region-static work is done once per region (not per replicate)
# ---------------------------------------------------------------------------


def test_region_static_work_is_done_once_per_region():
    component = _two_region_component()
    partition = partition_component_domain(
        component.polygon,
        component.points,
        component.expression_data[SOURCE_ROW_ID].to_numpy(dtype=np.int64),
    )
    n_replicates = 5

    # Wrap the module-level functions to count calls.
    prep_component_counter = _Counter(rt_mod.prepare_component)
    prep_region_counter = _Counter(rt_mod.prepare_region)
    d1_counter = _Counter(rt_mod.median_nn_distance)
    geometry_preparations = []
    real_prepare_geometry = GeosVoronoiEngine.prepare_region

    def prepare_geometry(engine, polygon, *, d1):
        geometry_preparations.append(polygon)
        return real_prepare_geometry(engine, polygon, d1=d1)

    # Also verify each PreparedRegion is a stable object reused across
    # replicates: monkey-patch run_region_tessellation to record the
    # `prepared` object identities per (region, replicate).
    real_run = rt_mod.run_region_tessellation
    seen_ids: list[tuple[int, int, int]] = []

    def spy_run(*, prepared, replicate_index, tile_min_points,
                voronoi_fraction, rng_seed):
        seen_ids.append((
            prepared.domain_region_id, replicate_index, id(prepared),
        ))
        return real_run(
            prepared=prepared, replicate_index=replicate_index,
            tile_min_points=tile_min_points,
            voronoi_fraction=voronoi_fraction, rng_seed=rng_seed,
        )

    with patch.object(ta_mod, "prepare_component", prep_component_counter), \
         patch.object(ta_mod, "prepare_region", prep_region_counter), \
         patch.object(rt_mod, "median_nn_distance", d1_counter), \
         patch.object(GeosVoronoiEngine, "prepare_region", prepare_geometry), \
         patch.object(ta_mod, "run_region_tessellation", spy_run):
        _run_component(
            sample_id="S", component_id="C", component_data=component,
            parent_roi_id="R", domain_partition=partition,
            n_replicates=n_replicates,
            tile_min_points=1, voronoi_fraction=0.4, base_seed=42,
        )

    # 1 component -> prepare_component invoked once.
    assert prep_component_counter.n == 1, (
        f"prepare_component called {prep_component_counter.n} times "
        "for a single-component stage; expected exactly 1"
    )
    # 2 eligible regions -> prepare_region invoked twice.
    assert prep_region_counter.n == 2, (
        f"prepare_region called {prep_region_counter.n} times "
        "across 2 regions; expected exactly 2 (once per region, "
        "not per replicate)"
    )
    # median_nn_distance: called once per region during
    # prepare_region; NOT called again inside compute_tessellation
    # for the K replicates (unique_coords + d1 cached).
    assert d1_counter.n == 2, (
        f"median_nn_distance called {d1_counter.n} times "
        f"for 2 regions x {n_replicates} replicates; expected 2 "
        "(once per region)"
    )
    assert len(geometry_preparations) == 2
    # Each region's PreparedRegion is reused across all its replicates.
    per_region_ids: dict[int, set[int]] = {}
    per_region_replicates: dict[int, set[int]] = {}
    for region_id, rep_idx, obj_id in seen_ids:
        per_region_ids.setdefault(region_id, set()).add(obj_id)
        per_region_replicates.setdefault(region_id, set()).add(rep_idx)
    for region_id, ids in per_region_ids.items():
        assert len(ids) == 1, (
            f"region {region_id}: {len(ids)} distinct PreparedRegion "
            "instances observed across its replicates; expected 1 "
            "(reuse violated)"
        )
        assert per_region_replicates[region_id] == set(range(n_replicates))


# ---------------------------------------------------------------------------
# 2. select_seeds skips unique-coord reduction when cache supplied
# ---------------------------------------------------------------------------


def test_select_seeds_reuses_precomputed_unique_coords():
    """When ``precomputed_unique_coords`` is supplied ``select_seeds``
    must not re-run ``sorted(set(map(tuple, coords)))`` — verified by
    supplying a distinguishable cached array and asserting it's what
    the sampler drew from."""
    rng = np.random.default_rng(0)
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    # Deliberately unusual "unique_coords" cache (single point) to
    # prove the function draws from IT, not from `coords`.
    cache = np.asarray([[99.0, 99.0]], dtype=float)
    seeds = select_seeds(
        coords, fraction=1.0, rng=rng,
        precomputed_unique_coords=cache,
    )
    assert seeds.shape == (1, 2)
    # Seed must be the cache's single point plus jitter — the |x| >>
    # any true coord confirms the cache path fired.
    assert abs(seeds[0, 0]) > 50.0
    assert abs(seeds[0, 1]) > 50.0


# ---------------------------------------------------------------------------
# 3. Generic compute_tessellation(precomputed_d1=None) computes d1 once
# ---------------------------------------------------------------------------


def test_compute_tessellation_without_cache_computes_d1_once():
    """The builder is the single d1 authority. When called without
    ``precomputed_d1`` it must compute d1 exactly once."""

    counter = _Counter(builder_mod.median_nn_distance)
    coords = np.column_stack([
        np.arange(0.0, 10.0, 1.0), np.zeros(10),
    ])
    with patch.object(builder_mod, "median_nn_distance", counter):
        builder_mod.compute_tessellation(
            component_id="X",
            roi_coords=coords,
            region_polygon=box(-1.0, -1.0, 11.0, 1.0),
            tile_min_points=1,
            fraction=0.5,
            rng_seed=17,
        )
    assert counter.n == 1, (
        f"median_nn_distance called {counter.n} times inside "
        "compute_tessellation(precomputed_d1=None); expected 1 "
        "(single-authority contract)"
    )


def test_compute_tessellation_rejects_nonfinite_precomputed_d1():

    coords = np.column_stack([
        np.arange(0.0, 10.0, 1.0), np.zeros(10),
    ])
    with pytest.raises(ValueError, match="precomputed_d1 must be finite"):
        builder_mod.compute_tessellation(
            component_id="X",
            roi_coords=coords,
            region_polygon=box(-1.0, -1.0, 11.0, 1.0),
            tile_min_points=1,
            fraction=0.5,
            rng_seed=17,
            precomputed_d1=float("nan"),
        )
