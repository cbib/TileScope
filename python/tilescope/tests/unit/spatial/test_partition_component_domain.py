"""Determinism of `partition_component_domain` under storage-order changes.

`_regions_of(geometry)` observes whatever order the underlying
Shapely object stored — which is not durable across `make_valid`,
`set_precision`, or a WKB round-trip. The partition helper's job is
to produce a canonical result driven by geometry + coords + source-
row IDs, not by that incoming storage order.

Coverage:

- shared-boundary tie-break is stable under `MultiPolygon.geoms`
  permutation and under WKB round-trip;
- canonical `domain_region_id` matches between calls on
  permuted-vs-baseline geometry, given the same coords/src_ids;
- per-region source-row fingerprints agree across those calls.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest
from shapely import from_wkb, to_wkb
from shapely.geometry import MultiPolygon, Polygon, box

from tilescope.spatial.component_domains import (
    ComponentDomainPartition,
    partition_component_domain,
)


def _baseline_multipolygon() -> MultiPolygon:
    # Two unit squares sharing the edge x == 1: a spot exactly on that
    # boundary is covered by both regions and exercises the tie-break.
    left = box(0.0, 0.0, 1.0, 1.0)
    right = box(1.0, 0.0, 2.0, 1.0)
    return MultiPolygon([left, right])


def _reversed_multipolygon() -> MultiPolygon:
    left = box(0.0, 0.0, 1.0, 1.0)
    right = box(1.0, 0.0, 2.0, 1.0)
    return MultiPolygon([right, left])  # geoms permuted


def _wkb_roundtripped(mp: MultiPolygon) -> MultiPolygon:
    return from_wkb(to_wkb(mp))


def _coords_and_ids():
    # Three spots: one deep in each region + one on the shared edge.
    coords = np.array([
        [0.5, 0.5],   # deep in the left region
        [1.5, 0.5],   # deep in the right region
        [1.0, 0.5],   # exactly on the shared boundary
    ])
    src_ids = np.array([10, 20, 30], dtype=np.int64)
    return coords, src_ids


def _by_source(result: ComponentDomainPartition) -> dict:
    return dict(result.canonical_ids_by_source_row)


def _fp(result: ComponentDomainPartition) -> dict:
    return dict(result.source_row_fingerprint_by_canonical_id)


def test_partition_baseline_shape():
    coords, src_ids = _coords_and_ids()
    result = partition_component_domain(_baseline_multipolygon(), coords, src_ids)
    assert len(result.canonical_regions) == 2
    assigned = set(result.canonical_ids_by_source_row.keys())
    assert assigned == {10, 20, 30}
    # Every region has a stable fingerprint.
    for cid, ids in result.source_row_ids_by_canonical_id.items():
        assert result.source_row_fingerprint_by_canonical_id[cid] != ""
        assert tuple(sorted(ids)) == ids


def test_partition_stable_under_geoms_permutation():
    coords, src_ids = _coords_and_ids()
    a = partition_component_domain(_baseline_multipolygon(), coords, src_ids)
    b = partition_component_domain(_reversed_multipolygon(), coords, src_ids)
    assert _by_source(a) == _by_source(b), (
        "Canonical domain_region_id assignment differs when the "
        "MultiPolygon.geoms input order changes — the tie-break is "
        "leaking storage order."
    )
    assert _fp(a) == _fp(b)


def test_partition_stable_under_wkb_roundtrip():
    coords, src_ids = _coords_and_ids()
    a = partition_component_domain(_baseline_multipolygon(), coords, src_ids)
    b = partition_component_domain(
        _wkb_roundtripped(_baseline_multipolygon()), coords, src_ids,
    )
    assert _by_source(a) == _by_source(b)
    assert _fp(a) == _fp(b)


def test_partition_shared_boundary_spot_assigned_deterministically():
    coords, src_ids = _coords_and_ids()
    result = partition_component_domain(_baseline_multipolygon(), coords, src_ids)
    # src_id 30 is on the boundary; it must land in exactly one region.
    boundary_region = result.canonical_ids_by_source_row[30]
    # Whichever region gets it, the SAME region must get it under a
    # storage-order permutation.
    result_permuted = partition_component_domain(
        _reversed_multipolygon(), coords, src_ids,
    )
    assert result_permuted.canonical_ids_by_source_row[30] == boundary_region


def test_partition_empty_geometry_returns_empty_result():
    result = partition_component_domain(Polygon(), np.zeros((0, 2)), np.zeros(0, dtype=np.int64))
    assert result.canonical_regions == ()
    assert dict(result.canonical_ids_by_source_row) == {}
    assert dict(result.source_row_ids_by_canonical_id) == {}
    assert dict(result.source_row_fingerprint_by_canonical_id) == {}


def test_partition_coord_and_src_length_mismatch_raises():
    with pytest.raises(ValueError, match="coords/src_ids length mismatch"):
        partition_component_domain(
            _baseline_multipolygon(),
            np.array([[0.5, 0.5], [1.5, 0.5]]),
            np.array([10], dtype=np.int64),
        )


def test_partition_spots_outside_all_regions_absent_from_mapping():
    coords = np.array([
        [0.5, 0.5],
        [10.0, 10.0],  # far outside both regions
    ])
    src_ids = np.array([10, 99], dtype=np.int64)
    result = partition_component_domain(_baseline_multipolygon(), coords, src_ids)
    assert 10 in result.canonical_ids_by_source_row
    assert 99 not in result.canonical_ids_by_source_row


def test_partition_fingerprint_reflects_membership_change():
    coords_a = np.array([[0.5, 0.5], [1.5, 0.5]])
    src_a = np.array([10, 20], dtype=np.int64)
    coords_b = np.array([[0.5, 0.5], [1.5, 0.5]])
    src_b = np.array([10, 21], dtype=np.int64)  # different member
    a = partition_component_domain(_baseline_multipolygon(), coords_a, src_a)
    b = partition_component_domain(_baseline_multipolygon(), coords_b, src_b)
    # Region holding the changed member has a different fingerprint.
    changed = a.canonical_ids_by_source_row[20]
    assert (
        a.source_row_fingerprint_by_canonical_id[changed]
        != b.source_row_fingerprint_by_canonical_id[changed]
    )
