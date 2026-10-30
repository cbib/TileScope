"""Tests for the canonical production spot-ledger frame.

Semantic contracts:

1. **Conservation** — every input spot appears exactly once; status
   counts sum to input count.
2. **Fate classification** — exactly two ``component_status`` values
   are emitted (``retained_component`` / ``small_connected_component``);
   ``exclusion_reason`` populated only on the suppressed branch.
3. **Export contract** — extended schema is stable (10 columns);
   ``component_size`` is populated wherever ``component_id`` is.
4. **Truth table** — retained-component spots carry populated
   ``domain_region_id`` / ``tessellation_eligible`` /
   ``tessellation_ineligibility_reason`` sourced from typed sample
   result; ineligibility reasons emit verbatim; suppressed spots
   keep those cols null; scattered parent branch keeps typed cols
   null while ``parent_scattered`` is True. Final row-pattern
   assertion runs inside the builder.
5. **Identity contract** — the builder joins on ``SOURCE_ROW_ID``
   value, not positional index. A non-default (shuffled + offset)
   input index still emits correct ``spot_index`` (= SOURCE_ROW_ID)
   and correct domain-region joins.
6. **Contract failures raise** — missing SOURCE_ROW_ID; missing
   support_status_df; duplicated / missing parent rows in
   support_status_df; retained non-scattered component missing from
   sample_tess_result.
"""

from __future__ import annotations

from tilescope.analyzers.tessellation_outcomes import RegionExecutionStatus, RegionReplicateTessellationDiagnostic
import numpy as _np
import pandas as _pd
from tilescope.component_data import ComponentData


from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.analyzers.tessellation_analysis import (
    ComponentTessellationResult,
    SampleTessellationResult,
)
from tilescope.analyzers.tessellation_outcomes import (
    RegionIneligibilityReason,
    RegionTessellationEligibilityDiagnostic,
)
from tilescope.cols import (
    COORD_X, COORD_Y, GENE1, GENE2, ROI_ID, SOURCE_ROW_ID,
)
from tilescope.analyzers.sample_support_frames import build_spot_ledger
from tilescope.spatial.component_domains import ComponentDomainPartition


# ---------------------------------------------------------------------------
# Fixtures — plain retention (no typed extensions)
# ---------------------------------------------------------------------------


def _make_original(n_by_roi: dict, source_row_id_offset: int = 0) -> pd.DataFrame:
    """Build a source frame with an explicit SOURCE_ROW_ID column.
    `source_row_id_offset` shifts the SOURCE_ROW_ID range so tests
    can exercise non-arange identity streams.
    """
    rows = []
    src = source_row_id_offset
    for roi, n in n_by_roi.items():
        for _ in range(n):
            rows.append({
                COORD_X: 0.0, COORD_Y: 0.0,
                GENE1: 0.0, GENE2: 0.0,
                ROI_ID: str(roi),
                SOURCE_ROW_ID: src,
            })
            src += 1
    return pd.DataFrame(rows).reset_index(drop=True)


def _default_support(parents: list) -> pd.DataFrame:
    """A support_status_df covering every parent as non-scattered."""
    return pd.DataFrame([
        {"roi_id": p, "spatial_support_class": "supported",
         "source_spots": 1, "is_scattered": False}
        for p in parents
    ])


def _mixed_fixture():
    """P1: 15 spots survive both gates (two retained components 10+5).
    P2: 5 spots dropped by the splitter."""
    original = _make_original({"P1": 15, "P2": 5})
    # Splitter emits comp_id for the 15 surviving spots (all P1).
    survivors = original.iloc[:15].copy()
    survivors["comp_id"] = [10] * 10 + [11] * 5
    survivors["cc_idx"] = 0
    post_filter = survivors.copy()
    support = _default_support(["P1", "P2"])
    return dict(
        original_df=original,
        post_split_df=survivors,
        post_filter_df=post_filter,
        support_status_df=support,
    )


# ---------------------------------------------------------------------------
# Fixtures — typed extensions
# ---------------------------------------------------------------------------


def _stub_component_result(
    cid: str, parent_roi_id: str,
    src_id_to_region: dict,
    region_eligibility: dict,
    n_replicates_planned: int = 1,
) -> ComponentTessellationResult:
    """Minimal ComponentTessellationResult with a hand-built partition
    + region_diagnostics matching the caller-supplied maps."""
    region_ids = sorted(set(src_id_to_region.values()))
    canonical_regions = tuple(
        box(float(rid), 0.0, float(rid) + 1.0, 1.0) for rid in region_ids
    )
    src_by_region: dict = {rid: [] for rid in region_ids}
    for src, rid in src_id_to_region.items():
        src_by_region[rid].append(int(src))
    partition = ComponentDomainPartition(
        canonical_regions=canonical_regions,
        canonical_ids_by_source_row=MappingProxyType(dict(src_id_to_region)),
        source_row_ids_by_canonical_id=MappingProxyType({
            rid: tuple(sorted(v)) for rid, v in src_by_region.items()
        }),
        source_row_fingerprint_by_canonical_id=MappingProxyType({
            rid: f"stub_fp_r{rid}" for rid in region_ids
        }),
    )
    region_diagnostics = tuple(
        RegionTessellationEligibilityDiagnostic(
            component_id=cid, domain_region_id=rid,
            n_component_spots=len(src_by_region[rid]),
            region_area=1.0,
            ineligibility_reason=region_eligibility[rid],
        )
        for rid in region_ids
    )
    # Every (eligible_region × planned_replicate) needs a matching
    # region-replicate diagnostic — this fixture treats them all as
    # NO_TILES (no tessellation attached) to satisfy the completeness
    # invariant.
    eligible_rids = [
        rid for rid in region_ids if region_eligibility[rid] is None
    ]
    rr_diagnostics = tuple(
        RegionReplicateTessellationDiagnostic(
            component_id=cid, domain_region_id=rid,
            replicate_index=t, n_tiles=0,
            region_status=RegionExecutionStatus.NO_TILES,
        )
        for rid in eligible_rids
        for t in range(n_replicates_planned)
    )
    all_src = sorted(src_id_to_region.keys())
    _expr = _pd.DataFrame({
        SOURCE_ROW_ID: _np.asarray(all_src, dtype=_np.int64),
        COORD_X: [0.0] * len(all_src), COORD_Y: [0.0] * len(all_src),
        GENE1: [0.0] * len(all_src), GENE2: [0.0] * len(all_src),
        ROI_ID: [parent_roi_id] * len(all_src),
    })
    stub = ComponentData(
        component_id=cid, points=_np.zeros((len(all_src), 2)),
        canvas=None, polygon=box(0.0, 0.0, 1.0, 1.0),
        expression_data=_expr,
    )
    return ComponentTessellationResult(
        parent_roi_id=parent_roi_id,
        component_data=stub,
        tessellations=MappingProxyType({}),
        region_diagnostics=region_diagnostics,
        region_replicate_diagnostics=rr_diagnostics,
        domain_partition=partition,
    )


def _typed_fixture():
    """P1 comp 10 (src 0..9): region 0 eligible (5 spots),
    region 1 ineligible (5 spots, N_SPOTS_BELOW_TILE_MIN).
    P1 comp 11 (src 10..14): single region 0 eligible.
    P2 (src 15..19): splitter-dropped.
    P3 (src 20..24): scattered — retained by splitter but no
    tessellation stage runs."""
    original = _make_original({"P1": 15, "P2": 5, "P3": 5})
    survivors_idx = [i for i in range(25) if i not in range(15, 20)]
    post_split = original.iloc[survivors_idx].copy()
    post_split["comp_id"] = [10] * 10 + [11] * 5 + [12] * 5
    post_split["cc_idx"] = 0
    post_filter = post_split.copy()

    comp10 = _stub_component_result(
        cid="10", parent_roi_id="P1",
        src_id_to_region={i: (0 if i < 5 else 1) for i in range(10)},
        region_eligibility={
            0: None,
            1: RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN,
        },
    )
    comp11 = _stub_component_result(
        cid="11", parent_roi_id="P1",
        src_id_to_region={i: 0 for i in range(10, 15)},
        region_eligibility={0: None},
    )
    # P3 is scattered → its component 12 doesn't appear on sample_tess_result.
    sample = SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({"10": comp10, "11": comp11}),
        n_replicates_planned=1,
    )
    support = pd.DataFrame([
        {"roi_id": "P1", "spatial_support_class": "supported",
         "source_spots": 15, "is_scattered": False},
        {"roi_id": "P2", "spatial_support_class": "supported",
         "source_spots": 5, "is_scattered": False},
        {"roi_id": "P3", "spatial_support_class": "scattered",
         "source_spots": 5, "is_scattered": True},
    ])
    return dict(
        original_df=original,
        post_split_df=post_split,
        post_filter_df=post_filter,
        sample_tess_result=sample,
        support_status_df=support,
    )


# ---------------------------------------------------------------------------
# 1. Conservation
# ---------------------------------------------------------------------------


def test_conservation():
    args = _mixed_fixture()
    ledger = build_spot_ledger(sample_tess_result=None, **args)
    assert len(ledger) == len(args["original_df"])
    assert ledger["spot_index"].is_unique
    assert set(ledger["spot_index"]) == set(
        args["original_df"][SOURCE_ROW_ID].astype(int)
    )
    assert ledger["component_status"].value_counts().sum() == len(ledger)


# ---------------------------------------------------------------------------
# 2. Fate classification
# ---------------------------------------------------------------------------


def test_fate_classification_is_purely_geometric():
    args = _mixed_fixture()
    ledger = build_spot_ledger(sample_tess_result=None, **args)
    counts = ledger["component_status"].value_counts()
    assert set(counts.index) == {"retained_component", "small_connected_component"}
    assert counts["retained_component"] == 15
    assert counts["small_connected_component"] == 5
    is_supp = ledger["component_status"] == "small_connected_component"
    assert ledger.loc[is_supp, "exclusion_reason"].notna().all()
    assert ledger.loc[~is_supp, "exclusion_reason"].isna().all()


# ---------------------------------------------------------------------------
# 3. Export contract
# ---------------------------------------------------------------------------


def test_extended_frame_contract():
    args = _mixed_fixture()
    ledger = build_spot_ledger(sample_tess_result=None, **args)
    known_comp = ledger["component_id"].notna()
    assert ledger.loc[known_comp, "component_size"].notna().all()
    assert list(ledger.columns) == [
        "spot_index", "roi_id",
        "component_id", "component_size",
        "component_status", "exclusion_reason",
        "parent_scattered",
        "domain_region_id",
        "tessellation_eligible",
        "tessellation_ineligibility_reason",
    ]


# ---------------------------------------------------------------------------
# 4. Truth-table on retained-component spots
# ---------------------------------------------------------------------------


def test_retained_component_spot_carries_domain_region_and_eligibility():
    args = _typed_fixture()
    ledger = build_spot_ledger(**args)
    row = ledger[ledger["spot_index"] == 0].iloc[0]
    assert row["component_status"] == "retained_component"
    assert row["component_id"] == "10"
    assert int(row["domain_region_id"]) == 0
    assert bool(row["tessellation_eligible"]) is True
    assert pd.isna(row["tessellation_ineligibility_reason"])
    row = ledger[ledger["spot_index"] == 12].iloc[0]
    assert row["component_id"] == "11"
    assert int(row["domain_region_id"]) == 0
    assert bool(row["tessellation_eligible"]) is True


def test_ineligible_region_reason_surfaces_verbatim():
    args = _typed_fixture()
    ledger = build_spot_ledger(**args)
    row = ledger[ledger["spot_index"] == 7].iloc[0]
    assert int(row["domain_region_id"]) == 1
    assert bool(row["tessellation_eligible"]) is False
    assert row["tessellation_ineligibility_reason"] == (
        RegionIneligibilityReason.N_SPOTS_BELOW_TILE_MIN.value
    )


def test_suppressed_spot_keeps_typed_columns_null_but_scattered_populated():
    args = _typed_fixture()
    ledger = build_spot_ledger(**args)
    p2 = ledger[ledger["roi_id"] == "P2"]
    assert len(p2) == 5
    assert (p2["component_status"] == "small_connected_component").all()
    assert p2["domain_region_id"].isna().all()
    assert p2["tessellation_eligible"].isna().all()
    assert p2["tessellation_ineligibility_reason"].isna().all()
    assert (p2["parent_scattered"] == False).all()  # noqa: E712


def test_scattered_parent_populates_flag_regardless_of_component_status():
    args = _typed_fixture()
    ledger = build_spot_ledger(**args)
    p3 = ledger[ledger["roi_id"] == "P3"]
    assert len(p3) == 5
    assert (p3["component_status"] == "retained_component").all()
    assert p3["domain_region_id"].isna().all()
    assert p3["tessellation_eligible"].isna().all()
    assert (p3["parent_scattered"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# 5. Identity contract — non-default input index
# ---------------------------------------------------------------------------


def test_join_on_source_row_id_not_positional_index():
    """Regression: with a non-default DataFrame index (shuffled +
    offset by 100), the builder must still emit spot_index equal to
    SOURCE_ROW_ID and correctly join to sample_tess_result via
    SOURCE_ROW_ID. A builder that joins on positional index would
    silently emit wrong spot IDs and miss every domain-region
    membership lookup."""
    args = _typed_fixture()

    # Reshuffle the input frames + reassign non-default indices.
    rng = np.random.default_rng(2028)
    def _shuffle_with_offset(df: pd.DataFrame, offset: int) -> pd.DataFrame:
        perm = rng.permutation(len(df))
        out = df.iloc[perm].copy()
        out.index = np.arange(len(out)) + offset
        return out

    args["original_df"] = _shuffle_with_offset(args["original_df"], 1000)
    args["post_split_df"] = _shuffle_with_offset(args["post_split_df"], 2000)
    args["post_filter_df"] = _shuffle_with_offset(args["post_filter_df"], 3000)

    ledger = build_spot_ledger(**args)

    # spot_index values equal SOURCE_ROW_ID (0..24 from the fixture),
    # NOT the 1000+ positional index of original_df.
    assert set(ledger["spot_index"]) == set(range(25))

    # Domain-region joins still land correctly: src 0 in region 0,
    # src 7 in region 1 ineligible, src 12 in region 0.
    assert int(
        ledger[ledger["spot_index"] == 0].iloc[0]["domain_region_id"]
    ) == 0
    assert int(
        ledger[ledger["spot_index"] == 7].iloc[0]["domain_region_id"]
    ) == 1
    assert int(
        ledger[ledger["spot_index"] == 12].iloc[0]["domain_region_id"]
    ) == 0


# ---------------------------------------------------------------------------
# 6. Contract failures raise
# ---------------------------------------------------------------------------


def test_missing_source_row_id_raises():
    args = _mixed_fixture()
    bad_original = args["original_df"].drop(columns=[SOURCE_ROW_ID])
    with pytest.raises(ValueError, match="missing required column"):
        build_spot_ledger(
            original_df=bad_original,
            post_split_df=args["post_split_df"],
            post_filter_df=args["post_filter_df"],
            sample_tess_result=None,
            support_status_df=args["support_status_df"],
        )


def test_missing_support_status_df_raises():
    args = _mixed_fixture()
    with pytest.raises(ValueError, match="support_status_df is required"):
        build_spot_ledger(
            original_df=args["original_df"],
            post_split_df=args["post_split_df"],
            post_filter_df=args["post_filter_df"],
            sample_tess_result=None,
            support_status_df=None,
        )


def test_duplicate_support_status_row_raises():
    args = _mixed_fixture()
    dup_support = pd.concat(
        [args["support_status_df"], args["support_status_df"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate roi_id"):
        build_spot_ledger(
            original_df=args["original_df"],
            post_split_df=args["post_split_df"],
            post_filter_df=args["post_filter_df"],
            sample_tess_result=None,
            support_status_df=dup_support,
        )


def test_missing_support_status_parent_raises():
    args = _mixed_fixture()
    partial_support = args["support_status_df"].iloc[:1]  # drop P2
    with pytest.raises(ValueError, match="missing entries for parent ROI"):
        build_spot_ledger(
            original_df=args["original_df"],
            post_split_df=args["post_split_df"],
            post_filter_df=args["post_filter_df"],
            sample_tess_result=None,
            support_status_df=partial_support,
        )


def test_retained_component_missing_from_tess_result_raises():
    args = _typed_fixture()
    # Drop comp 11 from sample_tess_result so retained non-scattered
    # spots (src 10..14) have no matching entry.
    trimmed = SampleTessellationResult.from_component_results(
        sample_id="S",
        components=MappingProxyType({
            "10": args["sample_tess_result"].components["10"],
        }),
        n_replicates_planned=1,
    )
    args["sample_tess_result"] = trimmed
    with pytest.raises(ValueError, match="missing entries for retained"):
        build_spot_ledger(**args)
