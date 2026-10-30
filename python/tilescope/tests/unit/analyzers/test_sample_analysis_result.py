"""SampleAnalysisResult invariants + `assemble_pair_output` identity contract.

`test_assembler_reads_typed_authority_verbatim` verifies the assembler
reads pair-scope + support-scope frames from the typed authority
without transformation. This is the assembler identity contract, not
A13; A13 (legacy-route equivalence vs typed one-pair route) lands with
the public typed materialiser in Phase 4.
"""
from __future__ import annotations

from dataclasses import replace as _replace
from types import MappingProxyType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.analysis_output import assemble_pair_output
from tilescope.analyzers.sample_analysis_result import (
    PairKey, PairStatisticResult, SampleAnalysisResult, SampleSupportOutputs,
)
from tilescope.analyzers.sample_support_stage import (
    build_empty_sample_support_bank,
)
from tilescope.spatial.consensus_bank import ConsensusBank, SampleSpatialBank
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
)
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame
from tilescope.settings import load_run_config


def _bank(sample_id: str = "S") -> SampleSpatialBank:
    src = build_source_spot_table_from_frame(pd.DataFrame({
        "x": [0.0, 1.0], "y": [0.0, 1.0],
        "gene1": [0.0, 1.0], "gene2": [0.0, 1.0],
        "roi_id": ["R", "R"], "source_row_id": [0, 1],
    }))
    support = build_empty_sample_support_bank(
        sample_id=sample_id, source_spot_adjacency=build_source_spot_adjacency(src),
        config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
        },
        ), n_replicates=1, base_seed=0,
    )
    consensus = ConsensusBank(
        sample_id=sample_id, sample_anchor_x=0.0, sample_anchor_y=0.0,
        n_replicates_planned=1, components={},
    )
    return SampleSpatialBank(support_bank=support, consensus_bank=consensus)


def _projection(feature_name: str, bank: SampleSpatialBank) -> FeatureSpatialProjection:
    return FeatureSpatialProjection(
        feature_name=feature_name,
        bank_fingerprint=bank.support_bank.fingerprint,
        source_values=np.zeros(bank.support_bank.source_spots.n_spots),
        thresholds=FeatureThresholds(
            q_high=0.9, q_low=0.1,
            resolved_high=1.0, resolved_low=0.0, n_source_values_used=0,
        ),
        tile_marginal_sums=MappingProxyType({}),
        hex_ensemble_projection=MappingProxyType({}),
        hex_direct_projection=MappingProxyType({}),
    )


def _pair_result(bank: SampleSpatialBank, g1="gene1", g2="gene2") -> PairStatisticResult:
    tag_df = lambda name: pd.DataFrame({"tag": [name]})  # noqa: E731
    # roi_correlations_df drives the cross-grain adapter's ROI universe;
    # empty roi_id column -> adapter iterates 0 ROIs and never accesses
    # the other stub frames.
    empty_roi_corr = pd.DataFrame({"tag": [], "roi_id": []})
    return PairStatisticResult(
        feature_g1=g1, feature_g2=g2,
        feature_g1_id="Neuron", feature_g2_id="AC",
        bank_fingerprint=bank.support_bank.fingerprint,
        roi_correlations_df=empty_roi_corr,
        roi_replicate_correlations_df=tag_df("replicate_corr"),
        roi_tessellation_correlations_df=tag_df("tess_corr"),
        roi_enrichment_df=tag_df("roi_enr"),
        roi_replicate_enrichment_df=tag_df("replicate_enr"),
        roi_tessellation_enrichment_df=tag_df("tess_enr"),
        roi_consensus_correlations_df=tag_df("consensus_corr"),
        roi_consensus_enrichment_df=tag_df("consensus_enr"),
        roi_source_moran_df=tag_df("source_moran"),
        roi_tessellation_moran_df=tag_df("tess_moran"),
        roi_consensus_moran_df=tag_df("consensus_moran"),
        roi_cross_grain_sensitivity_df=tag_df("cross_grain"),
        roi_pair_interface_response_df=tag_df("interface_response"),
        component_radial_shell_correlogram_df=tag_df("radial_shell"),
    )


def _support_outputs() -> SampleSupportOutputs:
    tag_df = lambda name: pd.DataFrame({"tag": [name]})  # noqa: E731
    return SampleSupportOutputs(
        component_geometry_df=tag_df("geometry"),
        component_domain_regions_df=tag_df("regions"),
        component_region_replicate_status_df=tag_df("status"),
        roi_support_status_df=tag_df("roi_support"),
        spot_ledger_df=tag_df("ledger"),
    )


def _sample_result(bank: SampleSpatialBank) -> SampleAnalysisResult:
    return SampleAnalysisResult(
        sample_id=bank.support_bank.sample_id,
        spatial_bank=bank,
        feature_projections=MappingProxyType({
            "gene1": _projection("gene1", bank),
            "gene2": _projection("gene2", bank),
        }),
        sample_support_outputs=_support_outputs(),
        pair_results=MappingProxyType({
            ("gene1", "gene2"): _pair_result(bank),
        }),
    )


def test_pair_key_derives_from_feature_names():
    assert _pair_result(_bank()).pair_key == ("gene1", "gene2")


@pytest.mark.parametrize("mutate,pattern", [
    (
        lambda bank: dict(sample_id="OTHER"),
        r"disagrees with spatial_bank",
    ),
    (
        lambda bank: dict(
            feature_projections={"WRONG": _projection("gene1", bank)},
        ),
        r"disagrees with projection.feature_name",
    ),
    (
        lambda bank: dict(
            pair_results={("wrong", "pair"): _pair_result(bank)},
        ),
        r"disagrees with pair_result.pair_key",
    ),
    (
        lambda bank: dict(
            feature_projections={"gene1": _projection("gene1", bank)},
            pair_results={("gene1", "gene2"): _pair_result(bank)},
        ),
        r"references unknown feature",
    ),
    (
        lambda bank: dict(
            feature_projections={
                "gene1": _replace(
                    _projection("gene1", bank),
                    bank_fingerprint=_bank("OTHER").support_bank.fingerprint,
                ),
                "gene2": _projection("gene2", bank),
            },
        ),
        r"projection 'gene1' was built against a different bank",
    ),
    (
        lambda bank: dict(
            pair_results={("gene1", "gene2"): _replace(
                _pair_result(bank),
                bank_fingerprint=_bank("OTHER").support_bank.fingerprint,
            )},
        ),
        r"pair \('gene1', 'gene2'\) was computed against a different bank",
    ),
], ids=[
    "sample_id_mismatch", "projection_key_mismatch",
    "pair_key_mismatch", "unknown_feature_in_pair",
    "projection_fingerprint_mismatch", "pair_fingerprint_mismatch",
])
def test_sample_analysis_result_rejects_incoherent_construction(mutate, pattern):
    bank = _bank("S")
    default_kwargs = dict(
        sample_id="S", spatial_bank=bank,
        feature_projections=MappingProxyType({
            "gene1": _projection("gene1", bank),
            "gene2": _projection("gene2", bank),
        }),
        sample_support_outputs=_support_outputs(),
        pair_results=MappingProxyType({("gene1", "gene2"): _pair_result(bank)}),
    )
    default_kwargs.update(mutate(bank))
    with pytest.raises(ValueError, match=pattern):
        SampleAnalysisResult(**default_kwargs)


def test_pair_results_mapping_is_frozen():
    with pytest.raises(TypeError):
        _sample_result(_bank()).pair_results[("x", "y")] = None  # type: ignore[index]


def test_assembler_reads_typed_authority_verbatim():
    """Sole legacy-flattening path reads pair + support frames verbatim.

    Assembler identity contract only — legacy-route vs typed-route
    equivalence (A13) lands with the public typed materialiser in Phase 4.
    """
    bank = _bank("S")
    sr = _sample_result(bank)
    pair_key: PairKey = ("gene1", "gene2")
    canvas = Mock(spec=[])
    component_polygons = {"C": object()}
    component_to_roi = {"C": "R"}
    source_expr = pd.DataFrame({
        "roi_id": ["R", "R"], "x": [0.0, 1.0], "y": [0.0, 1.0],
        "source_row_id": [0, 1], "gene1": [0.0, 1.0], "gene2": [0.0, 1.0],
    })
    component_expr = source_expr.copy()
    tiles_df = pd.DataFrame({
        "component_id": [], "replicate_index": [], "n_spots": [],
    })
    hexes_df = pd.DataFrame({"component_id": []})

    output = assemble_pair_output(
        sample_id=sr.sample_id,
        pair=sr.pair_results[pair_key],
        sample_support=sr.sample_support_outputs,
        canvas=canvas,
        component_polygons=component_polygons,
        component_to_roi=component_to_roi,
        source_expression_df=source_expr,
        component_expression_df=component_expr,
        retained_component_datasets={},
        sample_median_nn=1.0,
        precision=0.01, d1=1.0,
        thresholds=Mock(
            q_g1_threshold=1.0, q_g2_threshold=1.0,
            q_g1_low_threshold=0.0, q_g2_low_threshold=0.0,
        ),
        config=load_run_config(
            profile="standard",
            set_overrides={
            'output_dir': 'tests/tests_output',
        },
        ),
        tessellation_tiles_df=tiles_df,
        consensus_hexes_df=hexes_df,
        tessellation_pooling_index=None,
    )

    pair = sr.pair_results[pair_key]
    support = sr.sample_support_outputs
    pair_verbatim = {
        "roi_enrichment": pair.roi_enrichment_df,
        "roi_correlations": pair.roi_correlations_df,
        "roi_replicate_correlations_df": pair.roi_replicate_correlations_df,
        "roi_replicate_enrichment_df": pair.roi_replicate_enrichment_df,
        "roi_tessellation_correlations": pair.roi_tessellation_correlations_df,
        "roi_tessellation_enrichment_df": pair.roi_tessellation_enrichment_df,
        "roi_consensus_correlations_df": pair.roi_consensus_correlations_df,
        "roi_consensus_enrichment_df": pair.roi_consensus_enrichment_df,
        "roi_source_moran_df": pair.roi_source_moran_df,
        "roi_tessellation_moran_df": pair.roi_tessellation_moran_df,
        "roi_consensus_moran_df": pair.roi_consensus_moran_df,
        "roi_cross_grain_sensitivity_df": pair.roi_cross_grain_sensitivity_df,
        "roi_pair_interface_response_df": pair.roi_pair_interface_response_df,
    }
    support_verbatim = {
        "component_geometry_df": support.component_geometry_df,
        "component_domain_regions_df": support.component_domain_regions_df,
        "component_region_replicate_status_df": (
            support.component_region_replicate_status_df
        ),
        "roi_support_status": support.roi_support_status_df,
        "spot_ledger_df": support.spot_ledger_df,
    }
    for key, expected in {**pair_verbatim, **support_verbatim}.items():
        assert output[key] is expected, f"assembler mutated {key}"

    assert output["dataset_id"] == "S"
    assert output["canvas"] is canvas
    assert output["component_polygons"] is component_polygons
    assert output["component_to_roi"] is component_to_roi
    assert output["source_expression_df"] is source_expr
    assert output["analysis_df"] is component_expr
    assert output["tessellation_tiles_df"] is tiles_df
    assert output["consensus_hexes_df"] is hexes_df
    assert output["precision"] == 0.01
    assert output["d1"] == 1.0
    assert "tilescope_version" in output["run_metadata"]
