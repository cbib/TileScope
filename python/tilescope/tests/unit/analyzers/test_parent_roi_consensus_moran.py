"""Contract tests for :func:`build_roi_consensus_moran_df`.

Bivariate Moran on per-hex densities + region-blocked kNN adjacency.
"""
from __future__ import annotations

from tilescope.tests.helpers.fixture_projections import register_hex_features
from dataclasses import replace as _replace
from tilescope.tests.helpers.fixture_projections import _HEX_FEATURES
from tilescope.tests.helpers.fixture_projections import projection_from_hex_features


import numpy as np
import pandas as pd
from shapely.geometry import box

from tilescope.analyzers.consensus_frames import build_consensus_analysis_frames
from tilescope.analyzers.consensus_stage import (
    ComponentConsensusResult, ConsensusHex, SampleConsensusResult,
)
from tilescope.analyzers.parent_roi_consensus_moran import (
    build_roi_consensus_moran_df as _build_roi_consensus_moran_df,
)
from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.analyzers.sample_support_stage import (
    sample_support_bank_from_component_results,
)
from tilescope.cols import (
    ADJACENCY_SUPPORTED_FRACTION,
    MORAN_VALUE,
    N_ADJACENCY_SUPPORTED_UNITS,
    PARENT_MORAN_STATUS,
    ROI_ID,
    VALUE_SURFACE,
)


def build_roi_consensus_moran_df(**kwargs):
    return _build_roi_consensus_moran_df(
        g1_id="Neuron", g2_id="AC", **kwargs,
    )


def _hex(hid, cx, cy, g1_sum, g2_sum, *, n_spots=4.0, component_id="C",
         region_id=0):
    hx = ConsensusHex.build(
        component_id=component_id, hex_id=hid,
        domain_region_id=region_id,
        polygon=box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5),
        lattice_col=hid, lattice_row=0,
        center_x=cx, center_y=cy,
        n_tessellations_covered=2,
        n_spots_ensemble_weighted=n_spots,
        n_spots_direct=int(n_spots),
        n_both_high_ensemble_weighted=0.0,
        n_both_low_ensemble_weighted=0.0,
    )
    register_hex_features(
        component_id=component_id, hex_id=hid,
        sum_g1_ens=g1_sum, sum_g2_ens=g2_sum,
        g1_high=0.0, g2_high=0.0, g1_low=0.0, g2_low=0.0,
        sum_g1_direct=g1_sum, sum_g2_direct=g2_sum,
    )
    return hx


def _component(cid, roi_id, hexes, *, n_eligible_regions=1):
    rebound = []
    for h in hexes:
        orig_cid = h.geometry.component_id
        rebound_hex = _replace(h, geometry=_replace(h.geometry, component_id=cid))
        rebound.append(rebound_hex)
        data = _HEX_FEATURES.pop((str(orig_cid), int(h.hex_id)), None)
        if data is not None:
            register_hex_features(
                component_id=cid, hex_id=int(h.hex_id),
                sum_g1_ens=data["sum_g1_ens"], sum_g2_ens=data["sum_g2_ens"],
                g1_high=data["g1_high"], g2_high=data["g2_high"],
                g1_low=data["g1_low"], g2_low=data["g2_low"],
                sum_g1_direct=data["sum_g1_direct"],
                sum_g2_direct=data["sum_g2_direct"],
            )
    return ComponentConsensusResult.from_hexes(
        parent_roi_id=roi_id, component_id=cid, hex_radius=1.0,
        hexes=tuple(rebound),
        n_tessellation_eligible_regions=n_eligible_regions,
    )


def _sample(components):
    support_bank = sample_support_bank_from_component_results(
        sample_id="S", component_results={},
        n_replicates=2, base_seed=0,
    )
    return SampleConsensusResult.from_components(
        sample_id="S",
        components={c.component_id: c for c in components},
        n_replicates_planned=2,
        sample_anchor_x=0.0, sample_anchor_y=0.0,
        support_bank=support_bank,
    )


def _frames(sample, support):
    roi_ids = support[ROI_ID].astype(str).tolist()
    p_g1, p_g2 = projection_from_hex_features(sample_consensus_result=sample)
    return build_consensus_analysis_frames(
        sample_consensus_result=sample,
        roi_ids=roi_ids,
        feature_projection_g1=p_g1,
        feature_projection_g2=p_g2,
    )


def _support(rows):
    return pd.DataFrame(
        rows, columns=[ROI_ID, "source_spots", "spatial_support_class"],
    )


def test_positive_spatial_autocorrelation_yields_positive_moran():
    # 12 hexes in a line; k=2 (mutual-kNN) selects nearest 2 real neighbours.
    hexes = [_hex(i, float(i), 0.0, float(i + 1), float(i + 1))
             for i in range(12)]
    a = _component("Ca", "A", hexes)
    b = _component("Cb", "B", [_hex(0, 100.0, 0.0, 1.0, 1.0),
                                _hex(1, 101.0, 0.0, 2.0, 2.0),
                                _hex(2, 102.0, 0.0, 3.0, 3.0)])
    sample = _sample([a, b])
    support = _support([
        [rid, 10, "supported"] for rid in ("A", "B")
    ])
    frames = _frames(sample, support)
    df = build_roi_consensus_moran_df(
        consensus_frames=frames, support_status_df=support,
        k_neighbors=2, moran_enabled=True,
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    )
    a_rows = df[df[ROI_ID] == "A"]
    finite = a_rows[a_rows[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert (finite[MORAN_VALUE] > 0).all()
    assert (finite[N_ADJACENCY_SUPPORTED_UNITS] == 12).all()
    assert (finite[ADJACENCY_SUPPORTED_FRACTION] == 1.0).all()


def test_anti_correlated_hex_densities_yield_negative_moran():
    ramp = np.arange(1.0, 13.0)
    hexes = [_hex(i, float(i), 0.0, ramp[i], 13.0 - ramp[i])
             for i in range(12)]
    a = _component("Ca", "A", hexes)
    b = _component("Cb", "B", [_hex(0, 100.0, 0.0, 1.0, 1.0),
                                _hex(1, 101.0, 0.0, 2.0, 2.0),
                                _hex(2, 102.0, 0.0, 3.0, 3.0)])
    sample = _sample([a, b])
    support = _support([[rid, 10, "supported"] for rid in ("A", "B")])
    frames = _frames(sample, support)
    df = build_roi_consensus_moran_df(
        consensus_frames=frames, support_status_df=support,
        k_neighbors=2, moran_enabled=True,
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    )
    finite = df[
        (df[ROI_ID] == "A")
        & (df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value)
    ]
    assert len(finite) == 2
    assert (finite[MORAN_VALUE] < 0).all()


def test_ensemble_and_direct_surfaces_both_emitted():
    hexes = [_hex(i, float(i), 0.0, float(i + 1), float(i + 1))
             for i in range(4)]
    a = _component("Ca", "A", hexes)
    b = _component("Cb", "B", [_hex(0, 100.0, 0.0, 1.0, 1.0),
                                _hex(1, 101.0, 0.0, 2.0, 2.0),
                                _hex(2, 102.0, 0.0, 3.0, 3.0)])
    sample = _sample([a, b])
    support = _support([[rid, 10, "supported"] for rid in ("A", "B")])
    frames = _frames(sample, support)
    df = build_roi_consensus_moran_df(
        consensus_frames=frames, support_status_df=support,
        k_neighbors=6, moran_enabled=True,
    )
    surfaces = set(df[VALUE_SURFACE].unique())
    assert surfaces == {"consensus_ensemble", "consensus_direct"}
    assert (
        df[PARENT_MORAN_STATUS]
        == ParentMoranStatus.NO_SPATIAL_DISCRIMINATION.value
    ).all()


def test_moran_disabled_emits_total_table_all_disabled():
    a = _component("Ca", "A", [_hex(0, 0.0, 0.0, 1.0, 1.0),
                                _hex(1, 1.0, 0.0, 2.0, 2.0)])
    sample = _sample([a])
    support = _support([["A", 10, "supported"]])
    frames = _frames(sample, support)
    df = build_roi_consensus_moran_df(
        consensus_frames=frames, support_status_df=support,
        k_neighbors=6, moran_enabled=False,
    )
    # 1 ROI × 2 surfaces × 2 directions
    assert len(df) == 4
    assert (
        df[PARENT_MORAN_STATUS] == ParentMoranStatus.MORAN_DISABLED.value
    ).all()


def test_insufficient_units_reports_typed_status():
    a = _component("Ca", "A", [_hex(0, 0.0, 0.0, 1.0, 1.0)])
    sample = _sample([a])
    support = _support([["A", 10, "supported"]])
    frames = _frames(sample, support)
    df = build_roi_consensus_moran_df(
        consensus_frames=frames, support_status_df=support,
        k_neighbors=6, moran_enabled=True,
        surfaces=(PartitionValueSurface.CONSENSUS_ENSEMBLE,),
    )
    assert (
        df[PARENT_MORAN_STATUS]
        == ParentMoranStatus.INSUFFICIENT_UNITS.value
    ).all()
    assert df[MORAN_VALUE].isna().all()
