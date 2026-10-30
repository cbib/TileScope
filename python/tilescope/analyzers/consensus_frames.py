"""Layered consensus observations and component support.

Consensus geometry/support is materialised once per sample, marginal
values once per feature, and joint masses once per pair. Correlation,
enrichment, Moran, and the public ``consensus_hexes`` projector still
consume one legacy-shaped pair frame, assembled from those authorities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tilescope import cols
from tilescope.analyzers.consensus_stage import SampleConsensusResult
from tilescope.analyzers.partition_analysis import PartitionValueSurface
from tilescope.schema.helpers import natural_sort_key
from tilescope.spatial.consensus_bank import HexKey, SampleSpatialBank
from tilescope.spatial.feature_projection import FeatureSpatialProjection


HEX_FINITE_DENSITY_ENSEMBLE = "hex_finite_density_ensemble"
HEX_FINITE_DENSITY_DIRECT = "hex_finite_density_direct"
HEX_HAS_THRESHOLD_MASSES = "hex_has_threshold_masses"
HEX_OBSERVATION_COLUMNS: Tuple[str, ...] = (
    cols.SAMPLE_ID, cols.ROI_ID, cols.COMPONENT_ID, cols.HEX_ID,
    cols.DOMAIN_REGION_ID, cols.LATTICE_COL, cols.LATTICE_ROW,
    cols.CENTER_X, cols.CENTER_Y, cols.HEX_GEOMETRY_WKT, cols.HEX_AREA,
    cols.N_TESSELLATIONS_COVERED,
    cols.OBS_EQUIV_SPOT_MASS,
    cols.OBS_EQUIV_G1_SUM_MASS, cols.OBS_EQUIV_G2_SUM_MASS,
    cols.OBS_EQUIV_G1_HIGH_MASS, cols.OBS_EQUIV_G2_HIGH_MASS,
    cols.OBS_EQUIV_BOTH_HIGH_MASS,
    cols.OBS_EQUIV_G1_LOW_MASS, cols.OBS_EQUIV_G2_LOW_MASS,
    cols.OBS_EQUIV_BOTH_LOW_MASS,
    cols.GENE1_DENSITY_ENSEMBLE, cols.GENE2_DENSITY_ENSEMBLE,
    cols.GENE1_DENSITY_DIRECT, cols.GENE2_DENSITY_DIRECT,
    cols.GENE1_DENSITY_STABILITY, cols.GENE2_DENSITY_STABILITY,
    cols.N_SPOTS_DIRECT, cols.SUM_G1_DIRECT, cols.SUM_G2_DIRECT,
    HEX_FINITE_DENSITY_ENSEMBLE, HEX_FINITE_DENSITY_DIRECT,
    HEX_HAS_THRESHOLD_MASSES,
)

HEX_OBSERVATION_DTYPES: Dict[str, str] = {
    **{c: "string" for c in (
        cols.SAMPLE_ID, cols.ROI_ID, cols.COMPONENT_ID, cols.HEX_GEOMETRY_WKT,
    )},
    **{c: "Int64" for c in (
        cols.HEX_ID, cols.DOMAIN_REGION_ID, cols.LATTICE_COL, cols.LATTICE_ROW,
        cols.N_TESSELLATIONS_COVERED, cols.N_SPOTS_DIRECT,
    )},
    **{c: "Float64" for c in (
        cols.CENTER_X, cols.CENTER_Y, cols.HEX_AREA,
        cols.OBS_EQUIV_SPOT_MASS,
        cols.OBS_EQUIV_G1_SUM_MASS, cols.OBS_EQUIV_G2_SUM_MASS,
        cols.OBS_EQUIV_G1_HIGH_MASS, cols.OBS_EQUIV_G2_HIGH_MASS,
        cols.OBS_EQUIV_BOTH_HIGH_MASS,
        cols.OBS_EQUIV_G1_LOW_MASS, cols.OBS_EQUIV_G2_LOW_MASS,
        cols.OBS_EQUIV_BOTH_LOW_MASS,
        cols.GENE1_DENSITY_ENSEMBLE, cols.GENE2_DENSITY_ENSEMBLE,
        cols.GENE1_DENSITY_DIRECT, cols.GENE2_DENSITY_DIRECT,
        cols.GENE1_DENSITY_STABILITY, cols.GENE2_DENSITY_STABILITY,
        cols.SUM_G1_DIRECT, cols.SUM_G2_DIRECT,
    )},
    HEX_FINITE_DENSITY_ENSEMBLE: "boolean",
    HEX_FINITE_DENSITY_DIRECT: "boolean",
    HEX_HAS_THRESHOLD_MASSES: "boolean",
}

COUNT_MASS_COLUMNS = (
    (cols.N_G1_HIGH, cols.OBS_EQUIV_G1_HIGH_MASS),
    (cols.N_G2_HIGH, cols.OBS_EQUIV_G2_HIGH_MASS),
    (cols.N_BOTH_HIGH, cols.OBS_EQUIV_BOTH_HIGH_MASS),
    (cols.N_G1_LOW, cols.OBS_EQUIV_G1_LOW_MASS),
    (cols.N_G2_LOW, cols.OBS_EQUIV_G2_LOW_MASS),
    (cols.N_BOTH_LOW, cols.OBS_EQUIV_BOTH_LOW_MASS),
)

CONSENSUS_SPATIAL_COLUMNS: Tuple[str, ...] = (
    cols.SAMPLE_ID, cols.ROI_ID, cols.COMPONENT_ID, cols.HEX_ID,
    cols.DOMAIN_REGION_ID, cols.LATTICE_COL, cols.LATTICE_ROW,
    cols.CENTER_X, cols.CENTER_Y, cols.HEX_GEOMETRY_WKT, cols.HEX_AREA,
    cols.N_TESSELLATIONS_COVERED, cols.OBS_EQUIV_SPOT_MASS,
    cols.N_SPOTS_DIRECT,
    HEX_FINITE_DENSITY_ENSEMBLE, HEX_FINITE_DENSITY_DIRECT,
)
CONSENSUS_SPATIAL_DTYPES = {
    column: HEX_OBSERVATION_DTYPES[column]
    for column in CONSENSUS_SPATIAL_COLUMNS
}


@dataclass(frozen=True)
class ConsensusSurfaceSupport:
    eligible_component_ids: tuple[str, ...]
    contributing_component_ids: tuple[str, ...]
    n_hexes_constructed: int
    n_hexes_with_finite_density: int
    n_incomplete_components: int


@dataclass(frozen=True)
class ConsensusMassAggregate:
    counts: Mapping[str, float]
    spot_mass: float
    g1_sum_mass: float
    g2_sum_mass: float
    n_hexes: int
    threshold_masses_available: bool


@dataclass(frozen=True)
class ConsensusFeatureHexValues:
    """Derived marginal values for one feature on one consensus hex."""

    obs_equiv_sum_mass: float
    obs_equiv_high_mass: float
    obs_equiv_low_mass: float
    density_ensemble: float
    density_direct: float
    density_stability: Optional[float]
    sum_direct: float


@dataclass(frozen=True)
class ConsensusFeatureComponentAggregate:
    """One feature's ensemble marginal totals on a component."""

    sum_mass: float
    high_mass: float
    low_mass: float


@dataclass(frozen=True)
class FeatureConsensusFrame:
    """Sample-scope consensus marginal projection for one feature."""

    feature_name: str
    values_by_hex: Mapping[HexKey, ConsensusFeatureHexValues]
    aggregate_by_component: Mapping[str, ConsensusFeatureComponentAggregate]


@dataclass(frozen=True)
class ConsensusSpatialFrame:
    """Feature-independent consensus rows and ROI/context selections."""

    sample_id: str
    hex_support: pd.DataFrame
    hex_keys: tuple[HexKey, ...]
    ensemble_spot_weight_by_hex: Mapping[HexKey, float]
    ensemble_spot_mass_by_component: Mapping[str, float]
    finite_ensemble_count_by_component: Mapping[str, int]
    component_row_indices: Mapping[str, tuple[int, ...]]
    component_to_roi: Mapping[str, str]
    canonical_roi_ids: tuple[str, ...]
    support_by_roi_surface: Mapping[tuple[str, str], ConsensusSurfaceSupport]
    context_support_by_roi_surface: Mapping[
        tuple[str, str], ConsensusSurfaceSupport
    ]
    finite_hex_indices_by_roi_surface: Mapping[
        tuple[str, str], tuple[int, ...]
    ]
    context_finite_hex_indices_by_roi_surface: Mapping[
        tuple[str, str], tuple[int, ...]
    ]


@dataclass(frozen=True)
class ConsensusAnalysisFrames:
    """Canonical observations plus every reusable ROI projection."""

    sample_id: str
    hex_observations: pd.DataFrame
    support_by_roi_surface: Mapping[tuple[str, str], ConsensusSurfaceSupport]
    context_support_by_roi_surface: Mapping[tuple[str, str], ConsensusSurfaceSupport]
    finite_hex_indices_by_roi_surface: Mapping[tuple[str, str], tuple[int, ...]]
    context_finite_hex_indices_by_roi_surface: Mapping[
        tuple[str, str], tuple[int, ...]
    ]
    ensemble_mass_by_roi: Mapping[str, ConsensusMassAggregate]
    context_ensemble_mass_by_roi: Mapping[str, ConsensusMassAggregate]

    def support(
        self, roi_id: str, surface: PartitionValueSurface,
    ) -> ConsensusSurfaceSupport:
        return self.support_by_roi_surface.get(
            (str(roi_id), surface.value),
            ConsensusSurfaceSupport((), (), 0, 0, 0),
        )

    def finite_hexes(
        self, roi_id: str, surface: PartitionValueSurface,
    ) -> pd.DataFrame:
        indices = self.finite_hex_indices_by_roi_surface.get(
            (str(roi_id), surface.value), (),
        )
        return self.hex_observations.iloc[list(indices)]

    def context_support(
        self, roi_id: str, surface: PartitionValueSurface,
    ) -> ConsensusSurfaceSupport:
        return self.context_support_by_roi_surface.get(
            (str(roi_id), surface.value),
            ConsensusSurfaceSupport((), (), 0, 0, 0),
        )

    def context_finite_hexes(
        self, roi_id: str, surface: PartitionValueSurface,
    ) -> pd.DataFrame:
        indices = self.context_finite_hex_indices_by_roi_surface.get(
            (str(roi_id), surface.value), (),
        )
        return self.hex_observations.iloc[list(indices)]

    def ensemble_mass(self, roi_id: str) -> ConsensusMassAggregate:
        return self.ensemble_mass_by_roi.get(str(roi_id), _empty_mass_aggregate())

    def context_ensemble_mass(self, roi_id: str) -> ConsensusMassAggregate:
        return self.context_ensemble_mass_by_roi.get(
            str(roi_id), _empty_mass_aggregate(),
        )


def build_consensus_analysis_frames(
    *,
    sample_consensus_result: SampleConsensusResult,
    roi_ids: Sequence[str] = (),
    feature_projection_g1: FeatureSpatialProjection = None,
    feature_projection_g2: FeatureSpatialProjection = None,
    spatial_frame: Optional[ConsensusSpatialFrame] = None,
    feature_frame_g1: Optional[FeatureConsensusFrame] = None,
    feature_frame_g2: Optional[FeatureConsensusFrame] = None,
) -> ConsensusAnalysisFrames:
    """Compose a pair frame from sample and per-feature authorities.

    Production supplies the prebuilt frames. Projection arguments remain a
    fixture compatibility path and are routed through the same builders.
    """
    if spatial_frame is None:
        spatial_frame = build_consensus_spatial_frame(
            spatial_bank=sample_consensus_result.spatial_bank,
            roi_ids=roi_ids,
        )
    if feature_frame_g1 is None:
        if feature_projection_g1 is None:
            raise ValueError(
                "consensus pair frame requires feature_frame_g1 or "
                "feature_projection_g1"
            )
        feature_frame_g1 = build_feature_consensus_frame(
            spatial_frame=spatial_frame,
            feature_projection=feature_projection_g1,
        )
    if feature_frame_g2 is None:
        if feature_projection_g2 is None:
            raise ValueError(
                "consensus pair frame requires feature_frame_g2 or "
                "feature_projection_g2"
            )
        feature_frame_g2 = build_feature_consensus_frame(
            spatial_frame=spatial_frame,
            feature_projection=feature_projection_g2,
        )
    sample_id = str(sample_consensus_result.sample_id)
    if sample_id != spatial_frame.sample_id:
        raise ValueError(
            "consensus pair/spatial sample_id mismatch: "
            f"{sample_id!r} != {spatial_frame.sample_id!r}"
        )
    _validate_feature_frame(spatial_frame, feature_frame_g1)
    _validate_feature_frame(spatial_frame, feature_frame_g2)
    joint_by_hex = _joint_masses_by_hex(sample_consensus_result)
    if set(joint_by_hex) != set(spatial_frame.hex_keys):
        raise ValueError(
            "consensus pair joint keys disagree with spatial authority"
        )

    hex_rows = []
    for row, hex_key in zip(
        spatial_frame.hex_support.to_dict(orient="records"),
        spatial_frame.hex_keys,
    ):
        g1 = feature_frame_g1.values_by_hex[hex_key]
        g2 = feature_frame_g2.values_by_hex[hex_key]
        both_high, both_low = joint_by_hex[hex_key]
        row.update({
            cols.OBS_EQUIV_G1_SUM_MASS: g1.obs_equiv_sum_mass,
            cols.OBS_EQUIV_G2_SUM_MASS: g2.obs_equiv_sum_mass,
            cols.OBS_EQUIV_G1_HIGH_MASS: g1.obs_equiv_high_mass,
            cols.OBS_EQUIV_G2_HIGH_MASS: g2.obs_equiv_high_mass,
            cols.OBS_EQUIV_BOTH_HIGH_MASS: _optional_float(both_high),
            cols.OBS_EQUIV_G1_LOW_MASS: g1.obs_equiv_low_mass,
            cols.OBS_EQUIV_G2_LOW_MASS: g2.obs_equiv_low_mass,
            cols.OBS_EQUIV_BOTH_LOW_MASS: _optional_float(both_low),
            cols.GENE1_DENSITY_ENSEMBLE: g1.density_ensemble,
            cols.GENE2_DENSITY_ENSEMBLE: g2.density_ensemble,
            cols.GENE1_DENSITY_DIRECT: g1.density_direct,
            cols.GENE2_DENSITY_DIRECT: g2.density_direct,
            cols.GENE1_DENSITY_STABILITY: g1.density_stability,
            cols.GENE2_DENSITY_STABILITY: g2.density_stability,
            cols.SUM_G1_DIRECT: g1.sum_direct,
            cols.SUM_G2_DIRECT: g2.sum_direct,
            HEX_HAS_THRESHOLD_MASSES: both_high is not None,
        })
        hex_rows.append(row)
    hex_frame = _typed_frame(
        hex_rows, HEX_OBSERVATION_COLUMNS, HEX_OBSERVATION_DTYPES,
    )
    _validate_key(
        hex_frame, (cols.SAMPLE_ID, cols.COMPONENT_ID, cols.HEX_ID),
        "consensus hex observations",
    )
    _validate_component_roi_dependency(hex_frame)

    component_masses: dict[str, list[ConsensusMassAggregate]] = {}
    for component_id in sorted(
        spatial_frame.component_row_indices, key=natural_sort_key,
    ):
        indices = spatial_frame.component_row_indices[component_id]
        joint_high = joint_low = 0.0
        thresholds_available = True
        for index in indices:
            row = hex_rows[index]
            if not bool(row[HEX_FINITE_DENSITY_ENSEMBLE]):
                continue
            if pd.isna(row[cols.OBS_EQUIV_BOTH_HIGH_MASS]):
                thresholds_available = False
            else:
                joint_high += float(row[cols.OBS_EQUIV_BOTH_HIGH_MASS])
            if pd.isna(row[cols.OBS_EQUIV_BOTH_LOW_MASS]):
                thresholds_available = False
            else:
                joint_low += float(row[cols.OBS_EQUIV_BOTH_LOW_MASS])
        g1 = feature_frame_g1.aggregate_by_component[component_id]
        g2 = feature_frame_g2.aggregate_by_component[component_id]
        aggregate = ConsensusMassAggregate(
            counts=MappingProxyType({
                cols.N_G1_HIGH: g1.high_mass,
                cols.N_G2_HIGH: g2.high_mass,
                cols.N_BOTH_HIGH: joint_high,
                cols.N_G1_LOW: g1.low_mass,
                cols.N_G2_LOW: g2.low_mass,
                cols.N_BOTH_LOW: joint_low,
            }),
            spot_mass=(
                spatial_frame.ensemble_spot_mass_by_component[component_id]
            ),
            g1_sum_mass=g1.sum_mass,
            g2_sum_mass=g2.sum_mass,
            n_hexes=(
                spatial_frame.finite_ensemble_count_by_component[component_id]
            ),
            threshold_masses_available=thresholds_available,
        )
        roi_id = spatial_frame.component_to_roi[component_id]
        component_masses.setdefault(roi_id, []).append(aggregate)
    ensemble_mass_by_roi = {
        roi_id: _combine_mass_aggregates(parts)
        for roi_id, parts in component_masses.items()
    }
    context_mass_by_roi = {
        roi_id: _combine_mass_aggregates([
            ensemble_mass_by_roi.get(other, _empty_mass_aggregate())
            for other in spatial_frame.canonical_roi_ids if other != roi_id
        ])
        for roi_id in spatial_frame.canonical_roi_ids
    }
    return ConsensusAnalysisFrames(
        sample_id=sample_id,
        hex_observations=hex_frame,
        support_by_roi_surface=spatial_frame.support_by_roi_surface,
        context_support_by_roi_surface=(
            spatial_frame.context_support_by_roi_surface
        ),
        finite_hex_indices_by_roi_surface=(
            spatial_frame.finite_hex_indices_by_roi_surface
        ),
        context_finite_hex_indices_by_roi_surface=(
            spatial_frame.context_finite_hex_indices_by_roi_surface
        ),
        ensemble_mass_by_roi=MappingProxyType(ensemble_mass_by_roi),
        context_ensemble_mass_by_roi=MappingProxyType(context_mass_by_roi),
    )


def build_consensus_spatial_frame(
    *, spatial_bank: SampleSpatialBank, roi_ids: Sequence[str] = (),
) -> ConsensusSpatialFrame:
    """Materialise geometry, WKT, and ROI/context selections once/sample."""
    sample_id = str(spatial_bank.consensus_bank.sample_id)
    surfaces = (
        PartitionValueSurface.CONSENSUS_ENSEMBLE,
        PartitionValueSurface.CONSENSUS_DIRECT,
    )
    rows = []
    hex_keys = []
    ensemble_spot_weight_by_hex = {}
    ensemble_spot_mass_by_component = {}
    finite_ensemble_count_by_component = {}
    component_row_indices = {}
    component_to_roi = {}
    support_parts = {}
    finite_indices = {}
    for component_id in sorted(
        spatial_bank.consensus_bank.components, key=natural_sort_key,
    ):
        component = spatial_bank.consensus_bank.components[component_id]
        roi_id = str(component.parent_roi_id)
        component_to_roi[str(component_id)] = roi_id
        if component.n_tessellation_eligible_regions == 0:
            continue
        finite_counts = {surface: 0 for surface in surfaces}
        component_indices = []
        component_spot_mass = 0.0
        for geometry in component.hexes:
            n_tess = int(geometry.n_tessellations_covered)
            n_spots_ew = float(geometry.n_spots_ensemble_weighted)
            n_spots_direct = int(geometry.n_spots_direct)
            row = {
                cols.SAMPLE_ID: sample_id,
                cols.ROI_ID: roi_id,
                cols.COMPONENT_ID: str(component_id),
                cols.HEX_ID: int(geometry.hex_id),
                cols.DOMAIN_REGION_ID: int(geometry.domain_region_id),
                cols.LATTICE_COL: int(geometry.lattice_col),
                cols.LATTICE_ROW: int(geometry.lattice_row),
                cols.CENTER_X: float(geometry.center_x),
                cols.CENTER_Y: float(geometry.center_y),
                cols.HEX_GEOMETRY_WKT: geometry.polygon.wkt,
                cols.HEX_AREA: float(geometry.polygon.area),
                cols.N_TESSELLATIONS_COVERED: n_tess,
                cols.OBS_EQUIV_SPOT_MASS: n_spots_ew / n_tess,
                cols.N_SPOTS_DIRECT: n_spots_direct,
                HEX_FINITE_DENSITY_ENSEMBLE: n_spots_ew > 0,
                HEX_FINITE_DENSITY_DIRECT: n_spots_direct > 0,
            }
            row_index = len(rows)
            rows.append(row)
            hex_key = (str(component_id), int(geometry.hex_id))
            hex_keys.append(hex_key)
            ensemble_spot_weight_by_hex[hex_key] = n_spots_ew
            component_indices.append(row_index)
            for surface, finite_col in (
                (PartitionValueSurface.CONSENSUS_ENSEMBLE,
                 HEX_FINITE_DENSITY_ENSEMBLE),
                (PartitionValueSurface.CONSENSUS_DIRECT,
                 HEX_FINITE_DENSITY_DIRECT),
            ):
                if row[finite_col]:
                    finite_counts[surface] += 1
                    finite_indices.setdefault(
                        (roi_id, surface.value), [],
                    ).append(row_index)
            if row[HEX_FINITE_DENSITY_ENSEMBLE]:
                component_spot_mass += float(row[cols.OBS_EQUIV_SPOT_MASS])
        n_hexes = len(component.hexes)
        for surface in surfaces:
            support_parts.setdefault((roi_id, surface.value), []).append(
                (str(component_id), n_hexes, finite_counts[surface])
            )
        component_row_indices[str(component_id)] = tuple(component_indices)
        ensemble_spot_mass_by_component[str(component_id)] = (
            component_spot_mass
        )
        finite_ensemble_count_by_component[str(component_id)] = finite_counts[
            PartitionValueSurface.CONSENSUS_ENSEMBLE
        ]
    support_frame = _typed_frame(
        rows, CONSENSUS_SPATIAL_COLUMNS, CONSENSUS_SPATIAL_DTYPES,
    )
    _validate_key(
        support_frame, (cols.SAMPLE_ID, cols.COMPONENT_ID, cols.HEX_ID),
        "consensus spatial frame",
    )
    _validate_component_roi_dependency(support_frame)
    support_by_roi_surface = {
        key: ConsensusSurfaceSupport(
            eligible_component_ids=tuple(part[0] for part in parts),
            contributing_component_ids=tuple(
                part[0] for part in parts if part[2] > 0
            ),
            n_hexes_constructed=sum(part[1] for part in parts),
            n_hexes_with_finite_density=sum(part[2] for part in parts),
            n_incomplete_components=sum(part[2] == 0 for part in parts),
        )
        for key, parts in support_parts.items()
    }
    canonical_rois = sorted(
        {str(roi_id) for roi_id in roi_ids}
        | {key[0] for key in support_by_roi_surface},
        key=natural_sort_key,
    )
    context_support = {}
    context_indices = {}
    empty_support = ConsensusSurfaceSupport((), (), 0, 0, 0)
    for roi_id in canonical_rois:
        for surface in surfaces:
            context_support[(roi_id, surface.value)] = (
                _combine_surface_support([
                    support_by_roi_surface.get(
                        (other, surface.value), empty_support,
                    )
                    for other in canonical_rois if other != roi_id
                ])
            )
            context_indices[(roi_id, surface.value)] = tuple(
                index
                for other in canonical_rois if other != roi_id
                for index in finite_indices.get((other, surface.value), ())
            )
    return ConsensusSpatialFrame(
        sample_id=sample_id,
        hex_support=support_frame,
        hex_keys=tuple(hex_keys),
        ensemble_spot_weight_by_hex=MappingProxyType(
            ensemble_spot_weight_by_hex,
        ),
        ensemble_spot_mass_by_component=MappingProxyType(
            ensemble_spot_mass_by_component,
        ),
        finite_ensemble_count_by_component=MappingProxyType(
            finite_ensemble_count_by_component,
        ),
        component_row_indices=MappingProxyType(component_row_indices),
        component_to_roi=MappingProxyType(component_to_roi),
        canonical_roi_ids=tuple(canonical_rois),
        support_by_roi_surface=MappingProxyType(support_by_roi_surface),
        context_support_by_roi_surface=MappingProxyType(context_support),
        finite_hex_indices_by_roi_surface=MappingProxyType({
            key: tuple(indices) for key, indices in finite_indices.items()
        }),
        context_finite_hex_indices_by_roi_surface=MappingProxyType(
            context_indices,
        ),
    )


def build_feature_consensus_frame(
    *, spatial_frame: ConsensusSpatialFrame,
    feature_projection: FeatureSpatialProjection,
) -> FeatureConsensusFrame:
    """Derive consensus marginal values exactly once per feature."""
    values_by_hex = {}
    aggregates = {
        component_id: [0.0, 0.0, 0.0]
        for component_id in spatial_frame.component_row_indices
    }
    for row, hex_key in zip(
        spatial_frame.hex_support.to_dict(orient="records"),
        spatial_frame.hex_keys,
    ):
        try:
            ensemble = feature_projection.hex_ensemble_projection[hex_key]
            direct = feature_projection.hex_direct_projection[hex_key]
        except KeyError as exc:
            raise ValueError(
                f"feature {feature_projection.feature_name!r} is missing "
                f"consensus projection for hex {hex_key!r}"
            ) from exc
        n_tess = int(row[cols.N_TESSELLATIONS_COVERED])
        n_spots_ew = spatial_frame.ensemble_spot_weight_by_hex[hex_key]
        n_spots_direct = int(row[cols.N_SPOTS_DIRECT])
        density_ensemble = (
            ensemble.sum_g_ensemble_weighted / n_spots_ew
            if n_spots_ew > 0 else float("nan")
        )
        density_direct = (
            direct.sum_g_direct / n_spots_direct
            if n_spots_direct > 0 else float("nan")
        )
        values = ConsensusFeatureHexValues(
            obs_equiv_sum_mass=ensemble.sum_g_ensemble_weighted / n_tess,
            obs_equiv_high_mass=ensemble.n_high_ensemble_weighted / n_tess,
            obs_equiv_low_mass=ensemble.n_low_ensemble_weighted / n_tess,
            density_ensemble=_finite_or_nan(density_ensemble),
            density_direct=_finite_or_nan(density_direct),
            density_stability=_optional_native_float(
                ensemble.density_ensemble_stability,
            ),
            sum_direct=float(direct.sum_g_direct),
        )
        values_by_hex[hex_key] = values
        if n_spots_ew > 0:
            aggregate = aggregates[hex_key[0]]
            aggregate[0] += values.obs_equiv_sum_mass
            aggregate[1] += values.obs_equiv_high_mass
            aggregate[2] += values.obs_equiv_low_mass
    expected = set(spatial_frame.hex_keys)
    if set(feature_projection.hex_ensemble_projection) != expected:
        raise ValueError(
            f"feature {feature_projection.feature_name!r} ensemble hex keys "
            "disagree with the consensus spatial frame"
        )
    if set(feature_projection.hex_direct_projection) != expected:
        raise ValueError(
            f"feature {feature_projection.feature_name!r} direct hex keys "
            "disagree with the consensus spatial frame"
        )
    return FeatureConsensusFrame(
        feature_name=feature_projection.feature_name,
        values_by_hex=MappingProxyType(values_by_hex),
        aggregate_by_component=MappingProxyType({
            component_id: ConsensusFeatureComponentAggregate(
                sum_mass=values[0], high_mass=values[1], low_mass=values[2],
            )
            for component_id, values in aggregates.items()
        }),
    )


def _combine_surface_support(
    supports: Sequence[ConsensusSurfaceSupport],
) -> ConsensusSurfaceSupport:
    return ConsensusSurfaceSupport(
        eligible_component_ids=tuple(
            component for support in supports
            for component in support.eligible_component_ids
        ),
        contributing_component_ids=tuple(
            component for support in supports
            for component in support.contributing_component_ids
        ),
        n_hexes_constructed=sum(s.n_hexes_constructed for s in supports),
        n_hexes_with_finite_density=sum(
            s.n_hexes_with_finite_density for s in supports
        ),
        n_incomplete_components=sum(s.n_incomplete_components for s in supports),
    )


def _joint_masses_by_hex(
    sample_consensus_result: SampleConsensusResult,
) -> Mapping[HexKey, tuple[object, object]]:
    """Return only the pair-owned joint masses keyed by canonical hex."""
    values = {}
    for component_id in sorted(
        sample_consensus_result.components, key=natural_sort_key,
    ):
        component = sample_consensus_result.components[component_id]
        if component.n_tessellation_eligible_regions == 0:
            continue
        for hex_obj in component.hexes:
            values[(str(component_id), int(hex_obj.hex_id))] = (
                hex_obj.obs_equiv_both_high_mass,
                hex_obj.obs_equiv_both_low_mass,
            )
    return MappingProxyType(values)


def _validate_feature_frame(
    spatial_frame: ConsensusSpatialFrame,
    feature_frame: FeatureConsensusFrame,
) -> None:
    if set(feature_frame.values_by_hex) != set(spatial_frame.hex_keys):
        raise ValueError(
            f"feature {feature_frame.feature_name!r} hex keys disagree with "
            "the consensus spatial frame"
        )
    if set(feature_frame.aggregate_by_component) != set(
        spatial_frame.component_row_indices
    ):
        raise ValueError(
            f"feature {feature_frame.feature_name!r} component aggregates "
            "disagree with the consensus spatial frame"
        )


def _finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else np.nan


def _optional_float(value):
    if value is None:
        return pd.NA
    value = float(value)
    return value if math.isfinite(value) else pd.NA


def _optional_native_float(value) -> Optional[float]:
    """Finite optional float for typed internal records (never pandas NA)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _typed_frame(
    rows: List[dict], columns: Tuple[str, ...], dtypes: Dict[str, str],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(columns))
    return frame.astype(dtypes)


def _validate_key(frame: pd.DataFrame, key: Tuple[str, ...], label: str) -> None:
    if frame[list(key)].isna().any().any():
        raise ValueError(f"{label}: null value in composite key {list(key)!r}")
    duplicate = frame.duplicated(subset=list(key), keep=False)
    if duplicate.any():
        raise ValueError(
            f"{label}: {int(duplicate.sum())} duplicate rows on {list(key)!r}"
        )


def _validate_component_roi_dependency(hex_frame: pd.DataFrame) -> None:
    pairs = hex_frame[
        [cols.SAMPLE_ID, cols.COMPONENT_ID, cols.ROI_ID]
    ].drop_duplicates()
    counts = pairs.groupby(
        [cols.SAMPLE_ID, cols.COMPONENT_ID], sort=False,
    )[cols.ROI_ID].nunique()
    if (counts > 1).any():
        bad = counts[counts > 1].head().to_dict()
        raise ValueError(
            "consensus frames violate (sample_id, component_id) -> roi_id; "
            f"offending pairs: {bad!r}"
        )


def _empty_mass_aggregate() -> ConsensusMassAggregate:
    return ConsensusMassAggregate(
        counts=MappingProxyType({key: 0.0 for key, _ in COUNT_MASS_COLUMNS}),
        spot_mass=0.0,
        g1_sum_mass=0.0,
        g2_sum_mass=0.0,
        n_hexes=0,
        threshold_masses_available=True,
    )


def _combine_mass_aggregates(
    aggregates: List[ConsensusMassAggregate],
) -> ConsensusMassAggregate:
    counts = {key: 0.0 for key, _ in COUNT_MASS_COLUMNS}
    spot_mass = g1_sum_mass = g2_sum_mass = 0.0
    n_hexes = 0
    threshold_masses_available = True
    for aggregate in aggregates:
        for key in counts:
            counts[key] += aggregate.counts[key]
        spot_mass += aggregate.spot_mass
        g1_sum_mass += aggregate.g1_sum_mass
        g2_sum_mass += aggregate.g2_sum_mass
        n_hexes += aggregate.n_hexes
        threshold_masses_available &= aggregate.threshold_masses_available
    return ConsensusMassAggregate(
        counts=MappingProxyType(counts),
        spot_mass=spot_mass,
        g1_sum_mass=g1_sum_mass,
        g2_sum_mass=g2_sum_mass,
        n_hexes=n_hexes,
        threshold_masses_available=threshold_masses_available,
    )
