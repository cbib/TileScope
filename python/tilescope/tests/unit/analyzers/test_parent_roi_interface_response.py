"""Three substantive tests for parent_roi_interface_response.

1. Four-surface identity fixture: one spot or cell per tile / direct-hex
   with a matching overlap edge — all four surfaces reproduce the
   spot/cell scalar exactly. Exercises integer point_indices,
   direct membership, and ensemble overlap
   normalisation.
2. Enumeration + availability: both ordered directions and both
   requested responder features are emitted; no-interface ordered
   pairs are typed; scattered responder keeps spot/cell but
   constructed surfaces get `no_tessellation_eligible_component`.
3. Tessellation heterogeneity: median over finite replicates;
   correct planned / valid counts; `no_valid_replicates` when none
   finite.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tilescope.analyzers.parent_roi_interface_response import (
    AVAILABILITY_STATUS,
    INTERFACE_RESPONSE,
    STATUS_FINITE,
    STATUS_INSUFFICIENT_NEAR,
    STATUS_NO_ELIGIBLE_COMPONENT,
    STATUS_NO_INTERFACE,
    STATUS_NO_VALID_REPLICATES,
    SURFACE_CONSENSUS_DIRECT,
    SURFACE_CONSENSUS_ENSEMBLE,
    SURFACE_SOURCE_NATIVE,
    SURFACE_TESSELLATION,
    _ProjectionResult,
    _summarise_replicates,
    build_sample_feature_interface_response_df,
    select_pair_interface_response_df,
)


from tilescope.cols import ROI_ID
from tilescope.component_data import Tile, Tessellation
from tilescope.spatial.consensus_bank import (
    ComponentConsensusSupport, ConsensusBank, ConsensusHexGeometry,
    DirectHexMembership, SampleSpatialBank, TileHexOverlap,
)
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
    HexDirectMarginalProjection, HexEnsembleMarginalProjection,
    TileMarginalSums,
)
from tilescope.spatial.sample_support_bank import (
    ComponentSupport, SampleSupportBank, SampleSupportFingerprint,
    TessellationBank,
)
from tilescope.spatial.source_spot_adjacency import (
    build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import SourceSpotTable


def _build_pair_response(
    *, spatial_bank, projection, support_status_df,
    feature_g1_id: str, feature_g2_id: str,
):
    sample_features = build_sample_feature_interface_response_df(
        sample_id="S",
        spatial_bank=spatial_bank,
        feature_projections={
            feature_g1_id: projection,
            feature_g2_id: projection,
        },
        support_status_df=support_status_df,
    )
    return select_pair_interface_response_df(
        sample_features,
        feature_g1_id=feature_g1_id,
        feature_g2_id=feature_g2_id,
    )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _fingerprint(sample_id="S") -> SampleSupportFingerprint:
    return SampleSupportFingerprint(
        sample_id=sample_id, source_frame_hash="fixture",
        support_policy_version="fixture", source_adjacency_config_hash="",
        scattered_support_policy_hash="",
        component_geometry_contract_hash="",
        n_replicates_planned=1, base_seed=0,
        tessellation_config_hash="",
    )


def _source_spot_table(spot_ids, coords, roi_ids) -> SourceSpotTable:
    xs = np.asarray([c[0] for c in coords], dtype=float)
    ys = np.asarray([c[1] for c in coords], dtype=float)
    return SourceSpotTable(
        spot_id=tuple(str(s) for s in spot_ids),
        x=xs, y=ys, roi_id=tuple(str(r) for r in roi_ids),
    )


def _feature_projection(
    *, name: str, source_values: np.ndarray,
    tile_marginal_sums: Dict, hex_direct: Dict, hex_ensemble: Dict,
    fingerprint: SampleSupportFingerprint,
) -> FeatureSpatialProjection:
    thresholds = FeatureThresholds(
        q_high=0.9, q_low=0.1, resolved_high=1.0, resolved_low=0.0,
        n_source_values_used=int(len(source_values)),
    )
    return FeatureSpatialProjection(
        feature_name=name, bank_fingerprint=fingerprint,
        source_values=source_values, thresholds=thresholds,
        tile_marginal_sums=tile_marginal_sums,
        hex_ensemble_projection=hex_ensemble,
        hex_direct_projection=hex_direct,
    )


def _identity_four_surface_fixture():
    """One spot or cell per tile / hex; all four surfaces should agree.

    Anchor A: 3 spots at (0..2, 0).
    Responder B chain: 11 spots at (0..10, 1).

    median_nn = 1, threshold = 1.5. Interface (B spots with an A within
    1.5): B[0..3] (h=0, since B[3]=(3,1) is √2 from A[2]=(2,0)). B chain
    then continues: B[4]→h=1, B[5]→h=2, B[6..10]→h∈{3,4,5,6,7}.

    Near band (h ≤ 1): B[0..4] (5 spots) — 5 meets MIN_NEAR_COVERAGE.
    Buffer (h == 2): B[5] (excluded).
    Far (h ≥ 3): B[6..10] (5 spots) — 5 meets MIN_FAR_COVERAGE.

    Feature values: near = 1,2,3,4,5 (mean 3); far = 10,11,12,13,14
    (mean 12). Expected scalar = 3 − 12 = −9 on all four surfaces.
    """
    coords = []
    rois = []
    for x in range(3):
        coords.append((x, 0.0))
        rois.append("A")
    for x in range(11):
        coords.append((x, 1.0))
        rois.append("B")
    spot_ids = list(range(len(coords)))

    sst = _source_spot_table(spot_ids, coords, rois)
    adj = build_source_spot_adjacency(sst)

    fingerprint = _fingerprint()

    # One tile per B spot, one direct-hex per B spot; each hex covers only
    # its own single tile with area 1.0. Hex geometry: matching box; overlap 1.0.
    b_source_row_ids = np.asarray([3 + x for x in range(11)], dtype=np.int64)
    b_positions = np.asarray([(x, 1.0) for x in range(11)], dtype=float)

    # Component "C_B": all 11 B spots. One region_id=0, single replicate.
    tiles = []
    for i in range(11):
        tile_poly = box(i - 0.5, 0.5, i + 0.5, 1.5)
        point_indices = np.asarray([i], dtype=np.int64)
        tiles.append(Tile(
            tile_id=i, polygon=tile_poly,
            point_indices=point_indices,
            n_spots=1, area=float(tile_poly.area),
            domain_region_id=0,
        ))
    tessellation = Tessellation(
        tessellation_id="C_B_r0", component_id="C_B",
        tiles=tiles, method="voronoi",
        tile_min_points=1, fraction=1.0, rng_seed=0,
    )
    tess_map = {"C_B": MappingProxyType({0: MappingProxyType({0: tessellation})})}

    # Hex geometry — one hex per tile, matching poly.
    hexes = []
    for i in range(11):
        hex_poly = box(i - 0.5, 0.5, i + 0.5, 1.5)
        hexes.append(ConsensusHexGeometry(
            component_id="C_B", hex_id=i, domain_region_id=0,
            polygon=hex_poly, lattice_col=i, lattice_row=0,
            center_x=float(i), center_y=1.0,
            n_tessellations_covered=1,
            n_spots_direct=1,
            n_spots_ensemble_weighted=1.0,
        ))
    tile_overlaps = tuple(
        TileHexOverlap(
            component_id="C_B", domain_region_id=0, replicate_index=0,
            tile_id=i, hex_id=i, overlap_area=1.0, tile_area=1.0,
        )
        for i in range(11)
    )
    direct_memberships = tuple(
        DirectHexMembership(source_row_id=int(3 + i), hex_id=i)
        for i in range(11)
    )
    cons_support = ComponentConsensusSupport(
        component_id="C_B", parent_roi_id="B",
        hex_radius=0.5, n_tessellation_eligible_regions=1,
        hexes=tuple(hexes), tile_hex_overlaps=tile_overlaps,
        direct_hex_membership=direct_memberships,
    )
    consensus_bank = ConsensusBank(
        sample_id="S", sample_anchor_x=0.0, sample_anchor_y=0.0,
        n_replicates_planned=1,
        components={"C_B": cons_support},
    )

    # Feature values (indexed into the sample-wide source_values array):
    # Anchor A indices 0..2 unused (values 0). B[0..4] near (idx 3..7)
    # → 1..5. B[5] buffer (idx 8) = 100 (excluded). B[6..10] far
    # (idx 9..13) → 10..14.
    fvals = np.zeros(len(coords), dtype=float)
    for i, v in enumerate([1, 2, 3, 4, 5]):
        fvals[3 + i] = float(v)
    fvals[8] = 100.0
    for i, v in enumerate([10, 11, 12, 13, 14]):
        fvals[9 + i] = float(v)

    # Per-tile / per-hex responder mass = 1 spot; carrier = fvals[3 + tile_id].
    tile_marginal_sums = {
        ("C_B", 0, 0, i): TileMarginalSums(
            sum_g=float(fvals[3 + i]), n_high=0.0, n_low=0.0,
        )
        for i in range(11)
    }
    hex_direct_map = {
        ("C_B", i): HexDirectMarginalProjection(
            hex_key=("C_B", i), sum_g_direct=float(fvals[3 + i]),
        )
        for i in range(11)
    }
    hex_ensemble_map = {
        ("C_B", i): HexEnsembleMarginalProjection(
            hex_key=("C_B", i),
            sum_g_ensemble_weighted=float(fvals[3 + i]),
            n_high_ensemble_weighted=0.0, n_low_ensemble_weighted=0.0,
            density_ensemble_stability=1.0,
        )
        for i in range(11)
    }
    projection = _feature_projection(
        name="fB", source_values=fvals,
        tile_marginal_sums=tile_marginal_sums,
        hex_direct=hex_direct_map,
        hex_ensemble=hex_ensemble_map,
        fingerprint=fingerprint,
    )

    # Support bank
    b_polygon = box(-1.0, 0.5, 11.0, 1.5)
    comp_support = ComponentSupport(
        component_id="C_B", parent_roi_id="B",
        points=b_positions, source_row_ids=b_source_row_ids,
        polygon=b_polygon,
    )
    tess_bank = TessellationBank(
        n_replicates_planned=1, base_seed=0,
        tessellations=tess_map,
        region_diagnostics={"C_B": ()},
        region_replicate_diagnostics={"C_B": ()},
    )
    support_bank = SampleSupportBank(
        sample_id="S", fingerprint=fingerprint,
        source_spots=sst, source_spot_adjacency=adj,
        components={"C_B": comp_support},
        component_domains={},
        tessellation_bank=tess_bank,
    )
    spatial_bank = SampleSpatialBank(
        support_bank=support_bank, consensus_bank=consensus_bank,
    )
    return spatial_bank, projection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_four_surface_identity_agrees_with_source_native():
    spatial_bank, projection = _identity_four_surface_fixture()
    support_status_df = pd.DataFrame({
        ROI_ID: ["A", "B"],
        "spatial_support_class": ["fully_connected", "fully_connected"],
    })
    df = _build_pair_response(
        spatial_bank=spatial_bank, projection=projection,
        feature_g1_id="fA", feature_g2_id="fB",
        support_status_df=support_status_df,
    )
    # Focus on the (A → B, fB) rows.
    sub = df[(df["anchor_roi_id"] == "A") & (df["responder_roi_id"] == "B")
             & (df["responder_feature_id"] == "fB")].set_index("surface")
    # Source-native scalar: mean(1..5) − mean(10..14) = 3 − 12 = −9.
    src = sub.loc[SURFACE_SOURCE_NATIVE]
    assert src[AVAILABILITY_STATUS] == STATUS_FINITE
    assert float(src[INTERFACE_RESPONSE]) == pytest.approx(-9.0)
    # All three constructed surfaces reproduce that scalar (one-source-per-unit).
    for surface in (SURFACE_TESSELLATION, SURFACE_CONSENSUS_DIRECT,
                    SURFACE_CONSENSUS_ENSEMBLE):
        row = sub.loc[surface]
        assert row[AVAILABILITY_STATUS] == STATUS_FINITE, (
            f"{surface} status = {row[AVAILABILITY_STATUS]}"
        )
        assert float(row[INTERFACE_RESPONSE]) == pytest.approx(-9.0), (
            f"{surface} scalar = {row[INTERFACE_RESPONSE]}"
        )
    # No near/far-mixed units when every unit holds a single source spot.
    assert int(sub.loc[SURFACE_TESSELLATION, "n_near_far_mixed_units"]) == 0
    assert int(sub.loc[SURFACE_CONSENSUS_DIRECT, "n_near_far_mixed_units"]) == 0
    assert int(sub.loc[SURFACE_CONSENSUS_ENSEMBLE, "n_near_far_mixed_units"]) == 0


def _scattered_c_fixture():
    """Same A/B geometry as the identity fixture, plus a scattered ROI C.

    C has 6 spots or cells at y=2 (x=0..5) directly above A at y=0. C spots
    are within 1.5·median_nn of A spots (√5 > 1.5, so NOT to A y=0; but
    C at y=2 to B at y=1 IS at distance 1 → C↔B interface). Add spots
    close enough to A that A↔C also has an interface via C's leftmost spots
    to A's rightmost. Simplest: put a single C spot at (2, 2) so C[0] is √5
    from A[2]=(2,0) — no A↔C edge. Then add another C spot at (1, 1.5) so
    it's distance √(1 + 0.25)=1.118 from A[1]=(1,0) → A↔C edge.

    C has NO ComponentSupport in the bank. Result: A↔C interfaced pairs
    hit `no_tessellation_eligible_component` on constructed surfaces but
    source_native fires with the responder-C spot/cell values.
    """
    coords = []
    rois = []
    for x in range(3):
        coords.append((x, 0.0))
        rois.append("A")
    for x in range(11):
        coords.append((x, 1.0))
        rois.append("B")
    # C: 6 spots at y=2, x=0..5 — none within 1.5 of A (min A-C dist = 2).
    # Add one C spot at (1, 1.5): distance to A[1]=(1,0) is 1.5 (edge — but
    # tree.query_pairs uses ≤ threshold in practice). Distance to B[1]=(1,1)
    # is 0.5 → C connects to B.
    # To guarantee A↔C: use (1, 1.4) — 1.4 < 1.5 from A[1]=(1,0).
    coords.append((1.0, 1.4))
    rois.append("C")
    for x in range(5):
        coords.append((x, 2.0))
        rois.append("C")

    spot_ids = list(range(len(coords)))
    sst = _source_spot_table(spot_ids, coords, rois)
    adj = build_source_spot_adjacency(sst)
    fingerprint = _fingerprint()

    # C has NO components in the bank (models "responder ROI without a
    # supported component" on the constructed surfaces).
    #
    # Include the same B geometry as the identity fixture, so A↔B behaves
    # normally (finite constructed surfaces) and A↔C emits
    # no_tessellation_eligible_component for constructed surfaces.
    b_source_row_ids = np.asarray([3 + x for x in range(11)], dtype=np.int64)
    b_positions = np.asarray([(x, 1.0) for x in range(11)], dtype=float)

    tiles = []
    for i in range(11):
        tile_poly = box(i - 0.5, 0.5, i + 0.5, 1.5)
        tiles.append(Tile(
            tile_id=i, polygon=tile_poly,
            point_indices=np.asarray([i], dtype=np.int64),
            n_spots=1, area=float(tile_poly.area),
            domain_region_id=0,
        ))
    tessellation = Tessellation(
        tessellation_id="C_B_r0", component_id="C_B",
        tiles=tiles, method="voronoi",
        tile_min_points=1, fraction=1.0, rng_seed=0,
    )
    tess_map = {"C_B": MappingProxyType({0: MappingProxyType({0: tessellation})})}

    hexes = [
        ConsensusHexGeometry(
            component_id="C_B", hex_id=i, domain_region_id=0,
            polygon=box(i - 0.5, 0.5, i + 0.5, 1.5),
            lattice_col=i, lattice_row=0, center_x=float(i), center_y=1.0,
            n_tessellations_covered=1, n_spots_direct=1,
            n_spots_ensemble_weighted=1.0,
        )
        for i in range(11)
    ]
    cons_support = ComponentConsensusSupport(
        component_id="C_B", parent_roi_id="B",
        hex_radius=0.5, n_tessellation_eligible_regions=1,
        hexes=tuple(hexes),
        tile_hex_overlaps=tuple(
            TileHexOverlap(
                component_id="C_B", domain_region_id=0, replicate_index=0,
                tile_id=i, hex_id=i, overlap_area=1.0, tile_area=1.0,
            ) for i in range(11)
        ),
        direct_hex_membership=tuple(
            DirectHexMembership(source_row_id=int(3 + i), hex_id=i)
            for i in range(11)
        ),
    )
    consensus_bank = ConsensusBank(
        sample_id="S", sample_anchor_x=0.0, sample_anchor_y=0.0,
        n_replicates_planned=1, components={"C_B": cons_support},
    )

    fvals = np.zeros(len(coords), dtype=float)
    for i, v in enumerate([1, 2, 3, 4, 5]):
        fvals[3 + i] = float(v)
    fvals[8] = 100.0
    for i, v in enumerate([10, 11, 12, 13, 14]):
        fvals[9 + i] = float(v)
    # C spot/cell values (indices 14..19): 20..25 — needs to satisfy the
    # min-coverage floor to actually emit a finite source_native scalar
    # on A → C. But C is scattered so we can't fully test finiteness.
    for k in range(6):
        fvals[14 + k] = float(20 + k)

    tile_marginal_sums = {
        ("C_B", 0, 0, i): TileMarginalSums(
            sum_g=float(fvals[3 + i]), n_high=0.0, n_low=0.0,
        )
        for i in range(11)
    }
    hex_direct_map = {
        ("C_B", i): HexDirectMarginalProjection(
            hex_key=("C_B", i), sum_g_direct=float(fvals[3 + i]),
        )
        for i in range(11)
    }
    hex_ensemble_map = {
        ("C_B", i): HexEnsembleMarginalProjection(
            hex_key=("C_B", i),
            sum_g_ensemble_weighted=float(fvals[3 + i]),
            n_high_ensemble_weighted=0.0, n_low_ensemble_weighted=0.0,
            density_ensemble_stability=1.0,
        )
        for i in range(11)
    }
    projection = _feature_projection(
        name="fB", source_values=fvals,
        tile_marginal_sums=tile_marginal_sums,
        hex_direct=hex_direct_map,
        hex_ensemble=hex_ensemble_map,
        fingerprint=fingerprint,
    )
    comp_support = ComponentSupport(
        component_id="C_B", parent_roi_id="B",
        points=b_positions, source_row_ids=b_source_row_ids,
        polygon=box(-1.0, 0.5, 11.0, 1.5),
    )
    support_bank = SampleSupportBank(
        sample_id="S", fingerprint=fingerprint,
        source_spots=sst, source_spot_adjacency=adj,
        components={"C_B": comp_support},
        component_domains={},
        tessellation_bank=TessellationBank(
            n_replicates_planned=1, base_seed=0,
            tessellations=tess_map,
            region_diagnostics={"C_B": ()},
            region_replicate_diagnostics={"C_B": ()},
        ),
    )
    spatial_bank = SampleSpatialBank(
        support_bank=support_bank, consensus_bank=consensus_bank,
    )
    return spatial_bank, projection


def test_enumeration_includes_both_directions_both_features_with_typed_unavailability():
    spatial_bank, projection = _scattered_c_fixture()
    support_status_df = pd.DataFrame({
        ROI_ID: ["A", "B", "C"],
        "spatial_support_class": [
            "fully_connected", "fully_connected", "scattered",
        ],
    })
    df = _build_pair_response(
        spatial_bank=spatial_bank, projection=projection,
        feature_g1_id="g1_id", feature_g2_id="g2_id",
        support_status_df=support_status_df,
    )
    # 3 ROIs × 2 ordered = 6 pairs × 2 features × 4 surfaces = 48 rows.
    assert len(df) == 48
    # Both ordered A/B directions present.
    ab = df[(df["anchor_roi_id"] == "A") & (df["responder_roi_id"] == "B")]
    ba = df[(df["anchor_roi_id"] == "B") & (df["responder_roi_id"] == "A")]
    assert len(ab) == 8 and len(ba) == 8
    # Both features emitted per direction.
    assert set(ab["responder_feature_id"].unique()) == {"g1_id", "g2_id"}
    # Feature-pair provenance is preserved on every row.
    assert (df["g1_id"] == "g1_id").all()
    assert (df["g2_id"] == "g2_id").all()
    # A → C: interfaced (via the (1, 1.4) C spot bridging to A[1]),
    # responder C has NO component in the bank → source_native emits an
    # availability status (finite if coverage OK, else typed insufficient),
    # constructed surfaces all emit `no_tessellation_eligible_component`.
    ac = df[(df["anchor_roi_id"] == "A") & (df["responder_roi_id"] == "C")]
    ac_constructed = ac[ac["surface"].isin([
        SURFACE_TESSELLATION, SURFACE_CONSENSUS_DIRECT,
        SURFACE_CONSENSUS_ENSEMBLE,
    ])]
    assert (ac_constructed[AVAILABILITY_STATUS]
            == STATUS_NO_ELIGIBLE_COMPONENT).all()
    # A → C source_native rows are emitted (not silently dropped) — status
    # is either finite or a typed insufficiency, never no_anchor_responder_interface.
    ac_source = ac[ac["surface"] == SURFACE_SOURCE_NATIVE]
    assert (ac_source[AVAILABILITY_STATUS] != STATUS_NO_INTERFACE).all()
    assert len(ac_source) == 2  # 2 responder features, both emitted.


def test_tessellation_medians_over_finite_replicates_only():
    """Test the production replicate reducer on synthetic outcomes."""
    finite_1 = _ProjectionResult(
        status=STATUS_FINITE, value=1.0,
        represented_near_mass=10.0, represented_far_mass=10.0,
        n_near_far_mixed_units=2,
    )
    finite_2 = _ProjectionResult(
        status=STATUS_FINITE, value=3.0,
        represented_near_mass=10.0, represented_far_mass=10.0,
        n_near_far_mixed_units=2,
    )
    infeasible = _ProjectionResult(
        status=STATUS_INSUFFICIENT_NEAR, value=float("nan"),
        represented_near_mass=2.0, represented_far_mass=10.0,
        n_near_far_mixed_units=0,
    )

    result, n_valid = _summarise_replicates([finite_1, infeasible, finite_2])
    assert result.status == STATUS_FINITE
    assert result.value == pytest.approx(2.0)  # median(1, 3) — finite only
    assert n_valid == 2
    # Coverage / mixed-unit diagnostics summarise ALL 3 replicates:
    # median(10, 2, 10) = 10 for near, median(10, 10, 10) = 10 for far,
    # median(2, 0, 2) = 2 for mixed units.
    assert result.represented_near_mass == pytest.approx(10.0)
    assert result.represented_far_mass == pytest.approx(10.0)
    assert result.n_near_far_mixed_units == 2

    # None finite → status is no_valid_replicates, but diagnostics still
    # come from the median over the two infeasible replicates (not an
    # arbitrary copy of replicate 0).
    result_none, n_valid_none = _summarise_replicates([infeasible, infeasible])
    assert result_none.status == STATUS_NO_VALID_REPLICATES
    assert np.isnan(result_none.value)
    assert n_valid_none == 0
    assert result_none.represented_near_mass == pytest.approx(2.0)
    assert result_none.represented_far_mass == pytest.approx(10.0)
    assert result_none.n_near_far_mixed_units == 0
