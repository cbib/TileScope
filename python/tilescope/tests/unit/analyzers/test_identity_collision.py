"""Regression: cross-namespace ID collision must not confuse
parent_roi_id with component_id.

Component "1" belongs to parent "2" and component "2" belongs to
parent "1". Both analysis-frame builders and their join on the canonical
identity tuple must preserve these two rows distinctly — code that
derives one identifier from the other, or joins by only one of them,
would collapse or swap them.
"""
from __future__ import annotations

from tilescope.spatial.component_domains import ComponentDomainDiagnostic
from tilescope.analyzers.tessellation_analysis import ComponentTessellationResult, SampleTessellationResult
from tilescope.spatial.component_domains import ComponentDomainPartition
from types import MappingProxyType
import numpy as _np
import pandas as _pd
from shapely.geometry import box as _box
from tilescope.component_data import ComponentData
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, ROI_ID, SOURCE_ROW_ID


import pandas as pd
from shapely.geometry import Polygon

from tilescope.cols import COMPONENT_ID
from tilescope.analyzers.sample_support_frames import (
    build_component_geometry, build_spot_ledger,
)


def _square(cx: float, cy: float, side: float = 1.0) -> Polygon:
    h = side / 2
    return Polygon([(cx - h, cy - h), (cx + h, cy - h),
                    (cx + h, cy + h), (cx - h, cy + h)])


def test_cross_namespace_id_collision_preserved_across_analysis_frames():
    # comp "1" ↦ parent "2"; comp "2" ↦ parent "1".
    component_polygons = {"1": _square(0.0, 0.0), "2": _square(5.0, 0.0)}
    comp_to_parent = {"1": "2", "2": "1"}

    # Minimal ComponentDomainDiagnostic per component so the
    # canonical builder can populate its scalar rows.
    diags = [
        ComponentDomainDiagnostic(
            component_id=cid, parent_roi_id=parent,
            n_component_spots=2, n_spots_in_domain=2,
            n_domain_regions=1, n_regions_with_spots=1,
            component_support_area=1.0,
            ownership_excluded_area=0.0, acquisition_excluded_area=0.0,
            final_domain_area=1.0, geometry_status="ok",
        )
        for cid, parent in comp_to_parent.items()
    ]
    empty_elig = pd.DataFrame(columns=[
        "sample_id", "roi_id", "component_id",
        "n_tessellation_eligible_regions",
        "n_spots_in_tessellation_eligible_regions",
        "n_spots_excluded_by_tessellation_eligibility",
        "tessellated_domain_area",
    ])
    geom = build_component_geometry(
        sample_id="S1",
        component_polygons=component_polygons,
        component_to_roi=comp_to_parent,
        component_domain_diagnostics=diags,
        sample_median_nn=1.0,
        component_support_radius_factor=0.60,
        acquisition_support_radius_factor=0.60,
        ineligible_spot_exclusion_radius_factor=0.60,
        tessellation_eligibility_frame=empty_elig,
    )
    geom_tuples = set(zip(geom[ROI_ID], geom[COMPONENT_ID]))
    assert geom_tuples == {("2", "1"), ("1", "2")}

    original = pd.DataFrame({
        COORD_X: [0.0] * 4, COORD_Y: [0.0] * 4,
        GENE1: [0.0] * 4, GENE2: [0.0] * 4,
        ROI_ID: ["2", "2", "1", "1"],
        SOURCE_ROW_ID: [0, 1, 2, 3],
    })
    post_split = original.copy()
    post_split["comp_id"] = [1, 1, 2, 2]
    post_split["cc_idx"] = 0
    support = pd.DataFrame([
        {"roi_id": "1", "spatial_support_class": "supported",
         "source_spots": 2, "is_scattered": False},
        {"roi_id": "2", "spatial_support_class": "supported",
         "source_spots": 2, "is_scattered": False},
    ])
    # Retained non-scattered components require a matching
    # sample_tess_result entry; supply a minimal stub for comps 1
    # and 2 so the identity check passes without leaking spot IDs
    # across parent labels.

    def _stub(cid: str, parent: str, src_ids: list) -> ComponentTessellationResult:
        # Empty partition — this identity-collision test targets the
        # (roi_id, component_id) join, not tessellation eligibility.
        # An empty partition yields the "retained-outside-domain"
        # truth-table pattern (D) for each retained spot, which is
        # sufficient to exercise the identity join.
        _expr = _pd.DataFrame({
            SOURCE_ROW_ID: _np.asarray(src_ids, dtype=_np.int64),
            COORD_X: [0.0] * len(src_ids), COORD_Y: [0.0] * len(src_ids),
            GENE1: [0.0] * len(src_ids), GENE2: [0.0] * len(src_ids),
            ROI_ID: [parent] * len(src_ids),
        })
        stub = ComponentData(
            component_id=cid, points=_np.zeros((len(src_ids), 2)),
            canvas=None, polygon=_box(0.0, 0.0, 1.0, 1.0),
            expression_data=_expr,
        )
        partition = ComponentDomainPartition(
            canonical_regions=(),
            canonical_ids_by_source_row=MappingProxyType({}),
            source_row_ids_by_canonical_id=MappingProxyType({}),
            source_row_fingerprint_by_canonical_id=MappingProxyType({}),
        )
        return ComponentTessellationResult(
            parent_roi_id=parent, component_data=stub,
            tessellations=MappingProxyType({}),
            region_diagnostics=(), region_replicate_diagnostics=(),
            domain_partition=partition,
        )
    sample_tess_result = SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({
            "1": _stub("1", "2", [0, 1]),
            "2": _stub("2", "1", [2, 3]),
        }),
        n_replicates_planned=1,
    )
    ledger = build_spot_ledger(
        original_df=original,
        post_split_df=post_split,
        post_filter_df=post_split,
        sample_tess_result=sample_tess_result,
        support_status_df=support,
    )
    ledger_tuples = set(zip(ledger["roi_id"], ledger["component_id"]))
    assert ledger_tuples == {("2", "1"), ("1", "2")}

    ledger_keys = (
        ledger[["roi_id", "component_id"]]
        .drop_duplicates()
        .rename(columns={"roi_id": ROI_ID, "component_id": COMPONENT_ID})
    )
    joined = geom.merge(
        ledger_keys, on=[ROI_ID, COMPONENT_ID],
        how="inner", validate="one_to_one",
    )
    assert set(zip(joined[ROI_ID], joined[COMPONENT_ID])) == {
        ("2", "1"), ("1", "2"),
    }
