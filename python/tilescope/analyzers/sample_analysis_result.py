"""Typed sample-scope analysis authority + one-pair assembler.

Strict-subset records: no mixed export frames, no runtime /
visualization artefacts. `assemble_pair_output` is the sole
dict flattening path; every mixed-provenance frame is
materialised there once per pair from bank + projections,
never recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

import pandas as pd

from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.spatial.consensus_bank import SampleSpatialBank
from tilescope.spatial.errors import BankFingerprintMismatchError
from tilescope.spatial.feature_projection import FeatureSpatialProjection
from tilescope.spatial.sample_support_bank import SampleSupportFingerprint


PairKey = Tuple[str, str]


@dataclass(frozen=True)
class SampleSupportOutputs:
    """Feature-independent per-sample tabular outputs."""
    component_geometry_df: pd.DataFrame
    component_domain_regions_df: pd.DataFrame
    component_region_replicate_status_df: pd.DataFrame
    roi_support_status_df: pd.DataFrame
    spot_ledger_df: pd.DataFrame


@dataclass(frozen=True)
class PairStatisticResult:
    """Per-pair statistic tables. Primarily ROI-grain (biological
    reporting grain), plus one deliberately component-grain descriptor
    (`component_radial_shell_correlogram_df`) whose shell edges are
    normalised per-component and cannot be pooled across a ROI's
    disjoint components without changing the estimand.

    `feature_g1_id`/`feature_g2_id` are the SEMANTIC feature identities
    (e.g. `"Neuron"`, `"AC"`). All downstream surfaces (CSV `g1_id`/
    `g2_id` columns, `run_metadata.json`, HTML report, PDF headers) derive
    their labels from these two required fields.
    """
    feature_g1: str
    feature_g2: str
    feature_g1_id: str
    feature_g2_id: str
    bank_fingerprint: SampleSupportFingerprint
    roi_correlations_df: pd.DataFrame
    roi_replicate_correlations_df: pd.DataFrame
    roi_tessellation_correlations_df: pd.DataFrame
    roi_enrichment_df: pd.DataFrame
    roi_replicate_enrichment_df: pd.DataFrame
    roi_tessellation_enrichment_df: pd.DataFrame
    roi_consensus_correlations_df: pd.DataFrame
    roi_consensus_enrichment_df: pd.DataFrame
    roi_source_moran_df: pd.DataFrame
    roi_tessellation_moran_df: pd.DataFrame
    roi_consensus_moran_df: pd.DataFrame
    roi_cross_grain_sensitivity_df: pd.DataFrame
    roi_pair_interface_response_df: pd.DataFrame
    component_radial_shell_correlogram_df: pd.DataFrame
    def __post_init__(self) -> None:
        g1_id, g2_id = require_semantic_feature_ids(
            self.feature_g1_id,
            self.feature_g2_id,
            owner="PairStatisticResult",
        )
        object.__setattr__(self, "feature_g1_id", g1_id)
        object.__setattr__(self, "feature_g2_id", g2_id)

    @property
    def pair_key(self) -> PairKey:
        return (self.feature_g1, self.feature_g2)


@dataclass(frozen=True)
class SampleAnalysisResult:
    """Sample-scope analysis authority: bank + projections + support + pairs."""
    sample_id: str
    spatial_bank: SampleSpatialBank
    feature_projections: Mapping[str, FeatureSpatialProjection]
    sample_support_outputs: SampleSupportOutputs
    pair_results: Mapping[PairKey, PairStatisticResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.sample_id) != self.spatial_bank.support_bank.sample_id:
            raise ValueError(
                f"SampleAnalysisResult({self.sample_id!r}): sample_id "
                f"disagrees with spatial_bank.support_bank.sample_id="
                f"{self.spatial_bank.support_bank.sample_id!r}."
            )
        if not isinstance(self.feature_projections, MappingProxyType):
            object.__setattr__(
                self, "feature_projections",
                MappingProxyType(dict(self.feature_projections)),
            )
        if not isinstance(self.pair_results, MappingProxyType):
            object.__setattr__(
                self, "pair_results",
                MappingProxyType(dict(self.pair_results)),
            )
        bank_fp = self.spatial_bank.support_bank.fingerprint
        for feature_name, proj in self.feature_projections.items():
            if proj.feature_name != feature_name:
                raise ValueError(
                    f"SampleAnalysisResult({self.sample_id!r}): mapping key "
                    f"{feature_name!r} disagrees with projection.feature_name="
                    f"{proj.feature_name!r}."
                )
            if proj.bank_fingerprint != bank_fp:
                raise BankFingerprintMismatchError(
                    f"SampleAnalysisResult({self.sample_id!r}): projection "
                    f"{feature_name!r} was built against a different bank."
                )
        for pair_key, pair_result in self.pair_results.items():
            if pair_result.pair_key != pair_key:
                raise ValueError(
                    f"SampleAnalysisResult({self.sample_id!r}): pair mapping "
                    f"key {pair_key!r} disagrees with pair_result.pair_key="
                    f"{pair_result.pair_key!r}."
                )
            if pair_result.bank_fingerprint != bank_fp:
                raise BankFingerprintMismatchError(
                    f"SampleAnalysisResult({self.sample_id!r}): pair "
                    f"{pair_key!r} was computed against a different bank."
                )
            for feature in pair_key:
                if feature not in self.feature_projections:
                    raise ValueError(
                        f"SampleAnalysisResult({self.sample_id!r}): pair "
                        f"{pair_key!r} references unknown feature {feature!r}."
                    )
