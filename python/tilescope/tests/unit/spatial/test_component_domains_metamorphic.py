"""Metamorphic properties of the component-domain builder.

Applies rigid motions and uniform scaling to the source coordinates
and asserts the constructed geometry commutes with the transform. Two
comparison paths are used:

- Geometric: inverse-transform the transformed component's `G_c` and
  check equality against the baseline. Translation and uniform scale
  preserve polygon vertices exactly, so `equals_exact` with a
  precision-band tolerance is safe. Rotation and reflection do not:
  `Point.buffer` discretizes disks with a fixed segment count and the
  polygonal boundary is not orientation-equivariant. For those
  transforms a symmetric-difference-to-max-area ratio is used, which
  absorbs the discretization residue while still detecting real drift.
- Membership: the source-row IDs assigned to each component's domain
  and to each non-empty region are transform-invariant.

These are operator checks: they verify that the builder respects the
invariances promised by its inputs (Voronoi partitioning and disk
buffering are E(2)- and scale-equivariant). They are not evidence
that the geometry is scientifically fit for a downstream analysis.

`domain_region_id` values are not compared under rotation or
reflection because the canonical region order sorts by centroid
coordinates, which are not rotation-invariant. The set of non-empty
region-membership sets is invariant and is compared instead. Topology
metrics — number of regions per component and number of holes per
region — are asserted directly.

Estimator-eligibility comparisons are deliberately out of scope for
this file: the geometry builder has no eligibility threshold, so
those checks belong with the estimators that own them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon

from tilescope.cols import COMPONENT_ID, COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID
from tilescope.spatial import ComponentPolygonDict
from tilescope.spatial.component_domains import (
    _assign_component_spots_to_regions,
    _regions_of,
    build_component_polygons,
)


R_C = 0.75
R_A = 0.60
R_E = 0.75


# ---------------------------------------------------------------------------
# Fixture + helpers
# ---------------------------------------------------------------------------


def _baseline_frames() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Two disjoint 4-spot ROIs on a coarse grid; nothing pathological.

    Wide inter-ROI gap keeps the two Voronoi cells cleanly separated so
    the metamorphic properties do not sit on a numerical knife-edge.
    """
    rows = [
        {COORD_X: 0.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 0.0, ROI_ID: "A"},
        {COORD_X: 0.0, COORD_Y: 1.0, ROI_ID: "A"},
        {COORD_X: 1.0, COORD_Y: 1.0, ROI_ID: "A"},
        {COORD_X: 10.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 11.0, COORD_Y: 0.0, ROI_ID: "B"},
        {COORD_X: 10.0, COORD_Y: 1.0, ROI_ID: "B"},
        {COORD_X: 11.0, COORD_Y: 1.0, ROI_ID: "B"},
    ]
    source = pd.DataFrame(rows)
    source[SOURCE_ROW_ID] = np.arange(len(source), dtype=np.int64)
    spatial = source.copy()
    spatial[COMPONENT_ID] = spatial[ROI_ID].astype(str)
    spatial[ROI_ID] = spatial[ROI_ID].astype(str)
    component_to_roi = {"A": "A", "B": "B"}
    return source, spatial, component_to_roi


def _build(source, spatial, component_to_roi, sample_median_nn=1.0):
    result = build_component_polygons(
        source,
        spatial,
        component_to_roi,
        sample_median_nn=sample_median_nn,
        component_support_radius_factor=R_C,
        acquisition_support_radius_factor=R_A,
        ineligible_spot_exclusion_radius_factor=R_E,
    )
    return result.component_polygons, list(result.component_diagnostics), list(result.region_diagnostics)


def _apply_coord_transform(
    source: pd.DataFrame,
    spatial: pd.DataFrame,
    transform,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply `transform(x, y) -> (x', y')` to both frames in place-safe form."""
    src = source.copy()
    spa = spatial.copy()
    for df in (src, spa):
        xy = df[[COORD_X, COORD_Y]].to_numpy()
        transformed = np.array([transform(x, y) for x, y in xy])
        df[COORD_X] = transformed[:, 0]
        df[COORD_Y] = transformed[:, 1]
    return src, spa


def _spots_in_domain_by_component(
    spatial: pd.DataFrame,
    domains: ComponentPolygonDict,
) -> Dict[str, List[int]]:
    """SOURCE_ROW_IDs whose (x, y) lies inside their component's `G_c`.

    Matches the accounting used by the builder's diagnostics.
    """
    out: Dict[str, List[int]] = {cid: [] for cid in domains.keys()}
    for _, row in spatial.iterrows():
        cid = str(row[COMPONENT_ID])
        pt = Point(float(row[COORD_X]), float(row[COORD_Y]))
        if cid in domains and domains[cid].covers(pt):
            out[cid].append(int(row[SOURCE_ROW_ID]))
    for cid in out:
        out[cid].sort()
    return out


def _precision_band(sample_median_nn: float) -> float:
    """Precision-band tolerance scaled to the sample spacing.

    Matches the builder's own snap-boundary allowance: a fixed lower
    bound below which coordinate operations are meaningless in double
    precision, plus a linear scaling with `sample_median_nn` above
    that. Kept in sync with the builder rather than imported so a
    regression on either side produces a visible test failure.
    """
    return max(1e-4 * float(sample_median_nn), 1e-9)


def _assert_geometry_equivalent(
    a,
    b,
    *,
    rel_area_tol: float,
    label: str,
) -> None:
    """Symmetric-difference tolerance rather than `equals_exact`.

    `Point.buffer` discretizes disks with a fixed segment count, so
    the polygonal boundary is not exactly rotation- or reflection-
    equivariant even though the mathematical geometry is. The
    symmetric-difference-to-max-area ratio absorbs that discretization
    residue while still detecting real geometry drift.
    """
    if a.is_empty and b.is_empty:
        return
    denom = max(a.area, b.area)
    if denom == 0:
        assert a.symmetric_difference(b).area == 0.0, label
        return
    residue = a.symmetric_difference(b).area / denom
    assert residue < rel_area_tol, (
        f"{label}: symmetric-difference/max_area={residue:.3e} exceeds "
        f"rel_area_tol={rel_area_tol:.3e}."
    )


def _topology_signature(
    spatial: pd.DataFrame,
    domains: ComponentPolygonDict,
) -> Dict[str, Dict[str, object]]:
    """Coordinate-free topology summary per component.

    - `n_regions`: number of connected polygons in `G_c`.
    - `n_holes_sorted`: sorted per-region interior counts.
    - `region_membership_sets`: sorted list of non-empty SOURCE_ROW_ID
      sets, one per region.

    All three are invariant under rigid motions and uniform scaling.
    Region identity is not compared; only the topology and its
    partition of source IDs.
    """
    out: Dict[str, Dict[str, object]] = {}
    for cid, geom in domains.items():
        regions = _regions_of(geom)
        n_holes_sorted = sorted(
            len(r.interiors) if isinstance(r, Polygon) else 0 for r in regions
        )
        comp_rows = spatial[spatial[COMPONENT_ID].astype(str) == cid]
        coords = comp_rows[[COORD_X, COORD_Y]].to_numpy()
        src_ids = comp_rows[SOURCE_ROW_ID].astype(np.int64).to_numpy()
        assignments = _assign_component_spots_to_regions(regions, coords, src_ids)
        by_region: Dict[int, List[int]] = {}
        for src_id, r_idx in assignments.items():
            by_region.setdefault(r_idx, []).append(int(src_id))
        membership_sets = sorted(
            (frozenset(v) for v in by_region.values() if len(v) > 0),
            key=lambda s: (len(s), min(s)),
        )
        out[cid] = {
            "n_regions": len(regions),
            "n_holes_sorted": n_holes_sorted,
            "region_membership_sets": membership_sets,
        }
    return out


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def test_translation_invariance():
    source0, spatial0, c2r = _baseline_frames()
    domains0, _, _ = _build(source0, spatial0, c2r)

    dx, dy = 5.7, -3.1
    source1, spatial1 = _apply_coord_transform(
        source0, spatial0, lambda x, y: (x + dx, y + dy),
    )
    domains1, _, _ = _build(source1, spatial1, c2r)

    tol = _precision_band(1.0)
    for cid in domains0:
        back = translate(domains1[cid], xoff=-dx, yoff=-dy)
        # Translation preserves polygon vertices exactly (no re-
        # discretization); `equals_exact` is safe here.
        assert back.equals_exact(domains0[cid], tolerance=tol), (
            f"component {cid!r}: inverse-translated geometry differs "
            f"from baseline (tol={tol!r})."
        )

    assert _spots_in_domain_by_component(spatial1, domains1) == \
        _spots_in_domain_by_component(spatial0, domains0)
    assert _topology_signature(spatial1, domains1) == \
        _topology_signature(spatial0, domains0)

# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_invariance():
    source0, spatial0, c2r = _baseline_frames()
    domains0, _, _ = _build(source0, spatial0, c2r)

    angle_deg = 37.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    source1, spatial1 = _apply_coord_transform(
        source0, spatial0,
        lambda x, y: (cos_t * x - sin_t * y, sin_t * x + cos_t * y),
    )
    domains1, _, _ = _build(source1, spatial1, c2r)

    for cid in domains0:
        back = rotate(domains1[cid], -angle_deg, origin=(0, 0), use_radians=False)
        _assert_geometry_equivalent(
            back, domains0[cid], rel_area_tol=1e-2,
            label=f"component {cid!r} rotation",
        )

    # Membership + topology at the component grain are rotation-invariant.
    # `domain_region_id` values are not compared: the canonical region
    # order uses centroid coordinates, which are not rotation-invariant.
    assert _spots_in_domain_by_component(spatial1, domains1) == \
        _spots_in_domain_by_component(spatial0, domains0)
    assert _topology_signature(spatial1, domains1) == \
        _topology_signature(spatial0, domains0)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


def test_reflection_invariance_x_axis():
    source0, spatial0, c2r = _baseline_frames()
    domains0, _, _ = _build(source0, spatial0, c2r)

    source1, spatial1 = _apply_coord_transform(
        source0, spatial0, lambda x, y: (x, -y),
    )
    domains1, _, _ = _build(source1, spatial1, c2r)

    for cid in domains0:
        # Inverse of reflection over x-axis is the same reflection.
        back = scale(domains1[cid], xfact=1.0, yfact=-1.0, origin=(0, 0))
        _assert_geometry_equivalent(
            back, domains0[cid], rel_area_tol=1e-2,
            label=f"component {cid!r} reflection",
        )

    assert _spots_in_domain_by_component(spatial1, domains1) == \
        _spots_in_domain_by_component(spatial0, domains0)
    assert _topology_signature(spatial1, domains1) == \
        _topology_signature(spatial0, domains0)


# ---------------------------------------------------------------------------
# Uniform scaling — extreme range at Greenwald-like and mouse-like base sizes
#
# Audited median nearest-neighbour spacings on the working corpus fall
# around 7 (Greenwald / HCC glioma sections) and 110 (mouse sections);
# we take those as `base_sample_median_nn` values and cover both large
# (×1000) and small (×0.001) rescalings on top of that biological
# baseline. The transformed frame's `sample_median_nn` is scaled
# together with the coordinates so the radii scale by the same
# factor; the geometry must then commute with the transform (up to a
# precision band that scales with the baseline coordinate size) and
# preserve topology + component-grain membership.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_scale", [7.0, 110.0])
@pytest.mark.parametrize("k", [1000.0, 0.001])
def test_uniform_scale_invariance_extreme(base_scale, k):
    source0, spatial0, c2r = _baseline_frames()
    # Establish the baseline at the biological scale, not at 1.0, so
    # the ×0.001 branch does not collapse coordinates below double
    # precision.
    source0, spatial0 = _apply_coord_transform(
        source0, spatial0, lambda x, y: (base_scale * x, base_scale * y),
    )
    domains0, _, _ = _build(
        source0, spatial0, c2r, sample_median_nn=base_scale,
    )

    source1, spatial1 = _apply_coord_transform(
        source0, spatial0, lambda x, y: (k * x, k * y),
    )
    domains1, _, _ = _build(
        source1, spatial1, c2r, sample_median_nn=base_scale * k,
    )

    tol = _precision_band(base_scale)
    for cid in domains0:
        back = scale(domains1[cid], xfact=1.0 / k, yfact=1.0 / k, origin=(0, 0))
        # Boolean union may rotate a ring's starting vertex without
        # changing its geometry; normalize before the exact comparison.
        assert back.equals_exact(
            domains0[cid], tolerance=tol, normalize=True,
        ), (
            f"component {cid!r}: inverse-scaled geometry differs from "
            f"baseline (base={base_scale!r}, k={k!r}, tol={tol!r})."
        )
        # Area scales as k^2 up to relative float error.
        assert domains1[cid].area == pytest.approx(
            domains0[cid].area * (k ** 2), rel=1e-6,
        )

    assert _spots_in_domain_by_component(spatial1, domains1) == \
        _spots_in_domain_by_component(spatial0, domains0)
    assert _topology_signature(spatial1, domains1) == \
        _topology_signature(spatial0, domains0)
