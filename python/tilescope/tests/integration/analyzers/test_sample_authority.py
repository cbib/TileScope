"""Multi-pair authority contract — three pipeline executions max.

Pipeline runs:
  1. Baseline (public `run_sample`) under A4/A5/A10 spies. Pairs cover
     forward, reverse, and unrelated axes; also proves feature-name
     identity across ROI-grain frames, natural pair ordering, A12
     orientation symmetry, and A8 fingerprint identity in one pass.
  2. Superset run in a different pair order — proves A6 (ordering
     irrelevance on shared pairs) and A7 (unrelated pair does not
     perturb an existing pair) at once.
  3. A13 legacy-vs-public integration — the sole legacy-adapter
     equivalence contract.

A8, A9a, A9b, A11 are unit-level and covered by:
  - `test_sample_analysis_result.py` (fingerprint mismatch construction)
  - `test_feature_projection.py` (feature length + recursive immutability)
  - This module's `test_pair_only_config_change_preserves_fingerprint`
    (direct fingerprint construction; no pipeline)
"""
from __future__ import annotations

from tilescope.schema.helpers import natural_sort_key
from tilescope.analyzers.pre_analysis import prepare_source_frame
from tilescope.spatial.sample_support_bank import build_sample_support_fingerprint
from tilescope.spatial.source_spot_table import build_source_spot_table_from_frame


import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from tilescope.analyzers.consensus_stage import build_consensus_spatial_bank
from tilescope.analyzers.consensus_frames import (
    build_consensus_spatial_frame, build_feature_consensus_frame,
)
from tilescope.analyzers.feature_projection_stage import (
    build_feature_spatial_projection,
)
from tilescope.analyzers.parent_roi_projection import (
    build_feature_parent_roi_projection, build_parent_roi_support_index,
)
from tilescope.analyzers.parent_roi_interface_response import (
    build_sample_feature_interface_response_df,
)
from tilescope.analyzers.component_radial_shell_correlogram import (
    prepare_component_radial_shell_geometry,
)
from tilescope.analyzers.orchestrator import run_analysis
from tilescope.analyzers.run_sample import run_sample
from tilescope.analyzers.parent_roi_tessellation_common import (
    bind_tessellation_pooling_index, build_tessellation_support_index,
)
from tilescope.analyzers.consensus_stage import run_pair_consensus_from_bank
from tilescope.analyzers.tessellation_tile_frames import (
    build_tessellation_tile_support,
    build_tessellation_tiles_df as _real_build_tiles,
)
from tilescope.analyzers.tessellation_analysis import run_tessellation_stage
from tilescope.cols import GENE1, GENE2
from tilescope.settings import load_run_config
from tilescope.tests.helpers.spatial_spots import (
    make_supported_multifeature_spots,
)


def _config():
    return load_run_config(
        profile="standard",
        set_overrides={
            "output_dir": tempfile.mkdtemp(),
            "tessellation": {"n": 2},
            "permutations": 25,
            "consensus": {"min_tess_coverage_fraction": 1.0},
        },
    )


BASELINE_PAIRS = [
    ("gene1", "gene2"),  # forward
    ("gene2", "gene1"),  # reverse — A12
    ("G3", "G4"),        # unrelated — A7 perturbation control
]
PAIR_IDENTITIES = {
    (GENE1, GENE2): ("Neuron", "AC"),
    (GENE2, GENE1): ("AC", "Neuron"),
    (GENE1, "G3"): ("Neuron", "G3"),
}


@pytest.fixture(scope="module")
def supported_df():
    return make_supported_multifeature_spots()


@pytest.fixture(scope="module")
def base_config():
    return _config()


@pytest.fixture(scope="module")
def baseline_result(supported_df, base_config):
    """Public `run_sample` baseline under A4/A5/A10 spies. This is the
    ONE spied run; it also produces the frames every downstream contract
    compares against."""
    unique_features = sorted({f for pair in BASELINE_PAIRS for f in pair})
    with patch(
        "tilescope.analyzers.sample_authority.run_tessellation_stage",
        wraps=run_tessellation_stage,
    ) as tess_spy, patch(
        "tilescope.analyzers.sample_authority.build_consensus_spatial_bank",
        wraps=build_consensus_spatial_bank,
    ) as bank_spy, patch(
        "tilescope.analyzers.sample_authority.build_feature_spatial_projection",
        wraps=build_feature_spatial_projection,
    ) as proj_spy:
        result = run_sample(
            expression_df=supported_df, config=base_config,
            pair_features=BASELINE_PAIRS, dataset_id="mock",
            pair_identities=PAIR_IDENTITIES,
        )
    assert tess_spy.call_count == 1, f"A4: tessellation ran {tess_spy.call_count}x"
    assert bank_spy.call_count == 1, f"A5: bank built {bank_spy.call_count}x"
    assert proj_spy.call_count == len(unique_features), (
        f"A10: projected {proj_spy.call_count}x; "
        f"expected {len(unique_features)}"
    )
    return result


@pytest.fixture(scope="module")
def arbitrary_only_result(supported_df, base_config):
    """G3/G4 remains a complete authority when legacy columns are absent."""
    captured = {}

    def _sink(_pair_key, pair_dict):
        captured.update(pair_dict)

    result = run_sample(
        expression_df=supported_df.drop(columns=[GENE1, GENE2]),
        config=base_config,
        pair_features=[("G3", "G4")], dataset_id="mock",
        pair_sink=_sink,
    )
    return result, captured


# ---------------------------------------------------------------------------
# Baseline contracts: shape + feature identity + orientation + natural order
# ---------------------------------------------------------------------------


ROI_GRAIN_FIELDS = (
    "roi_correlations_df", "roi_replicate_correlations_df",
    "roi_tessellation_correlations_df", "roi_enrichment_df",
    "roi_replicate_enrichment_df", "roi_tessellation_enrichment_df",
    "roi_consensus_correlations_df", "roi_consensus_enrichment_df",
    "roi_source_moran_df", "roi_tessellation_moran_df",
    "roi_consensus_moran_df",
)


def test_supported_fixture_exercises_spatial_pipeline(baseline_result):
    """Guards the fixture: if this fails, every downstream contract is vacuous."""
    bank = baseline_result.spatial_bank
    assert len(bank.consensus_bank.components) > 0
    pair = baseline_result.pair_results[("gene1", "gene2")]
    assert not pair.roi_correlations_df.empty
    corr_col = next(
        (c for c in ("corr_obs", "r") if c in pair.roi_correlations_df.columns),
        None,
    )
    if corr_col is not None:
        finite_corr = pair.roi_correlations_df.dropna(subset=[corr_col])
        assert len(finite_corr) >= 2


def test_pair_frames_carry_actual_feature_names(baseline_result):
    for pair_key, pair in baseline_result.pair_results.items():
        g1, g2 = PAIR_IDENTITIES.get(pair_key, pair_key)
        for field in ROI_GRAIN_FIELDS:
            frame = getattr(pair, field)
            if frame.empty:
                continue
            if "g1_id" in frame.columns:
                assert (frame["g1_id"].astype(str) == g1).all()
            if "g2_id" in frame.columns:
                assert (frame["g2_id"].astype(str) == g2).all()


def test_arbitrary_only_pair_ignores_legacy_distractor_columns(
    baseline_result, arbitrary_only_result,
):
    """Removing gene1/gene2 cannot perturb any G3/G4 statistic.

    This exercises the reduced permutation null as well as support,
    tessellation, consensus, and ROI-grain calculations.
    """
    result, captured = arbitrary_only_result
    expected = baseline_result.pair_results[("G3", "G4")]
    actual = result.pair_results[("G3", "G4")]
    for field in ROI_GRAIN_FIELDS:
        pd.testing.assert_frame_equal(
            getattr(expected, field).reset_index(drop=True),
            getattr(actual, field).reset_index(drop=True),
        )
    assert captured["feature_g1"] == "G3"
    assert captured["feature_g2"] == "G4"
    assert "gene1" not in captured["source_expression_df"].columns
    assert "gene2" not in captured["source_expression_df"].columns


def test_pair_results_iterate_in_natural_order(baseline_result):
    keys = list(baseline_result.pair_results.keys())
    expected = sorted(
        keys, key=lambda p: (natural_sort_key(p[0]), natural_sort_key(p[1])),
    )
    assert keys == expected


def test_a12_reversed_orientation_agrees_on_symmetric_stats(baseline_result):
    """(gene1,gene2) and (gene2,gene1) share the baseline run."""
    fwd = baseline_result.pair_results[("gene1", "gene2")]
    rev = baseline_result.pair_results[("gene2", "gene1")]
    assert fwd.bank_fingerprint == rev.bank_fingerprint
    if "corr_obs" in fwd.roi_correlations_df.columns:
        common = ["sample_id", "roi_id"]
        merged = fwd.roi_correlations_df[common + ["corr_obs"]].merge(
            rev.roi_correlations_df[common + ["corr_obs"]], on=common,
            suffixes=("_fwd", "_rev"),
        )
        finite = merged.dropna(subset=["corr_obs_fwd", "corr_obs_rev"])
        if not finite.empty:
            assert (finite["corr_obs_fwd"] - finite["corr_obs_rev"]).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# One reordered/superset run — proves A6 (ordering) + A7 (add-pair no-perturb)
# ---------------------------------------------------------------------------


def test_reordered_superset_preserves_shared_pair_frames(
    baseline_result, supported_df, base_config,
):
    superset_pairs = [
        ("G3", "G4"), ("gene1", "G3"),
        ("gene2", "gene1"), ("gene1", "gene2"),
    ]
    superset_result = run_sample(
        expression_df=supported_df, config=base_config,
        pair_features=superset_pairs, dataset_id="mock",
        pair_identities=PAIR_IDENTITIES,
    )
    assert (
        superset_result.spatial_bank.support_bank.fingerprint
        == baseline_result.spatial_bank.support_bank.fingerprint
    )
    for pair in baseline_result.pair_results:
        base = baseline_result.pair_results[pair]
        alt = superset_result.pair_results[pair]
        for field in ROI_GRAIN_FIELDS:
            pd.testing.assert_frame_equal(
                getattr(base, field).reset_index(drop=True),
                getattr(alt, field).reset_index(drop=True),
            )


# ---------------------------------------------------------------------------
# Pair canonicalisation — parametrised, no pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad,pattern", [
    ([], "pair_features must be non-empty"),
    ([("gene1",)], "must be a 2-tuple"),
    ([("gene1", "gene2", "G3")], "must be a 2-tuple"),
    ([("", "gene2")], "must be non-empty"),
    ([("gene1", "")], "must be non-empty"),
    ([(1, "gene2")], "must be str"),
    ([("gene1", "gene2"), ("gene1", "gene2")], "duplicate pair"),
])
def test_pair_canonicalisation_rejects_malformed(
    supported_df, base_config, bad, pattern,
):
    with pytest.raises(ValueError, match=pattern):
        run_sample(
            expression_df=supported_df, config=base_config,
            pair_features=bad, dataset_id="mock",
        )


# ---------------------------------------------------------------------------
# Fingerprint identity from direct construction — no pipeline
# ---------------------------------------------------------------------------


def test_support_affecting_vs_pair_only_config_changes(supported_df, base_config):
    """Bank fingerprint depends on tessellation/knn/support inputs;
    permutations is a pair-scope knob and must NOT alter it."""

    source = prepare_source_frame(supported_df)
    sst = build_source_spot_table_from_frame(source)
    base_kwargs = dict(
        sample_id="mock", source_spot_table=sst,
        n_replicates=base_config.run_config.tessellation.n, base_seed=base_config.run_config.seed,
    )
    fp_base = build_sample_support_fingerprint(config=base_config, **base_kwargs)

    # Pair-only knob (permutations) must not shift the fingerprint.
    fp_pair_only = build_sample_support_fingerprint(
        config=base_config.model_copy(update={"permutations": 5}),
        **base_kwargs,
    )
    assert fp_pair_only == fp_base

    # Support-affecting knob (tessellation.n) must shift it. n_replicates
    # threads into the fingerprint, so this shift is unambiguous.
    variant = base_config.model_copy(update={
        "tessellation": base_config.run_config.tessellation.model_copy(update={"n": 5}),
    })
    fp_supp = build_sample_support_fingerprint(
        config=variant, **{**base_kwargs, "n_replicates": variant.tessellation.n},
    )
    assert fp_supp != fp_base


# ---------------------------------------------------------------------------
# A13 — sole legacy-adapter integration
# ---------------------------------------------------------------------------


def test_legacy_run_analysis_matches_streaming_sink_capture(
    supported_df, base_config,
):
    """Legacy adapter dict == public sink capture on the same pair —
    proves the sink and legacy shapes are aligned by construction."""
    legacy = run_analysis(
        supported_df.copy(), base_config, dataset_id=1,
        feature_a_id="Neuron", feature_b_id="AC",
    )

    captured: dict = {}

    def _sink(_pk, pair_dict):
        captured.update(pair_dict)

    run_sample(
        expression_df=supported_df.copy(), config=base_config,
        pair_features=[(GENE1, GENE2)], dataset_id="1",
        pair_sink=_sink,
        pair_identities={(GENE1, GENE2): ("Neuron", "AC")},
    )
    df_keys = {
        "roi_enrichment", "roi_correlations", "roi_support_status",
        "spot_ledger_df", "roi_tessellation_correlations",
        "component_geometry_df", "component_domain_regions_df",
        "component_region_replicate_status_df",
        "roi_replicate_correlations_df", "roi_replicate_enrichment_df",
        "roi_tessellation_enrichment_df", "roi_consensus_correlations_df",
        "roi_consensus_enrichment_df",
        "roi_source_moran_df", "roi_tessellation_moran_df",
        "roi_consensus_moran_df",
        "consensus_hexes_df", "tessellation_tiles_df",
    }
    for key in df_keys:
        pd.testing.assert_frame_equal(
            legacy[key].reset_index(drop=True),
            captured[key].reset_index(drop=True),
        )


def test_tile_support_and_pooling_support_run_once_per_sample_including_sink(
    supported_df, base_config,
):
    """Compute-once contract holds under the streaming sink path:
    Tile geometry/WKT + ROI/context support fire once per sample; only
    pair values, density binding, and joint consensus projection fire per
    pair. Consensus rows/WKT fire once per sample and marginal consensus
    values once per unique feature."""
    pairs = [("gene1", "gene2"), ("gene1", "G3")]
    unique_features = {feature for pair in pairs for feature in pair}
    with patch(
        "tilescope.analyzers.sample_authority.build_tessellation_tile_support",
        wraps=build_tessellation_tile_support,
    ) as tile_support_spy, patch(
        "tilescope.analyzers.sample_authority.build_tessellation_tiles_df",
        wraps=_real_build_tiles,
    ) as tiles_spy, patch(
        "tilescope.analyzers.sample_authority.run_pair_consensus_from_bank",
        wraps=run_pair_consensus_from_bank,
    ) as consensus_spy, patch(
        "tilescope.analyzers.sample_authority.build_tessellation_support_index",
        wraps=build_tessellation_support_index,
    ) as support_index_spy, patch(
        "tilescope.analyzers.sample_authority.bind_tessellation_pooling_index",
        wraps=bind_tessellation_pooling_index,
    ) as bind_spy, patch(
        "tilescope.analyzers.sample_authority.build_consensus_spatial_frame",
        wraps=build_consensus_spatial_frame,
    ) as consensus_frame_spy, patch(
        "tilescope.analyzers.sample_authority.build_feature_consensus_frame",
        wraps=build_feature_consensus_frame,
    ) as consensus_feature_spy:
        with patch(
            "tilescope.analyzers.sample_authority.build_parent_roi_support_index",
            wraps=build_parent_roi_support_index,
        ) as parent_support_spy, patch(
            "tilescope.analyzers.sample_authority."
            "build_feature_parent_roi_projection",
            wraps=build_feature_parent_roi_projection,
        ) as parent_feature_spy:
            with patch(
                "tilescope.analyzers.sample_authority."
                "build_sample_feature_interface_response_df",
                wraps=build_sample_feature_interface_response_df,
            ) as interface_spy, patch(
                "tilescope.analyzers.sample_authority."
                "prepare_component_radial_shell_geometry",
                wraps=prepare_component_radial_shell_geometry,
            ) as radial_geometry_spy:
                run_sample(
                    expression_df=supported_df.copy(), config=base_config,
                    pair_features=pairs, dataset_id="1",
                    pair_sink=lambda pk, d: None,
                    pair_identities=PAIR_IDENTITIES,
                )
    assert tiles_spy.call_count == len(pairs), (
        f"build_tessellation_tiles_df: expected {len(pairs)}x, "
        f"got {tiles_spy.call_count}"
    )
    assert consensus_spy.call_count == len(pairs), (
        f"run_pair_consensus_from_bank: expected {len(pairs)}x, "
        f"got {consensus_spy.call_count}"
    )
    assert tile_support_spy.call_count == 1
    assert support_index_spy.call_count == 1
    assert consensus_frame_spy.call_count == 1
    assert consensus_feature_spy.call_count == len(unique_features)
    assert parent_support_spy.call_count == 1
    assert parent_feature_spy.call_count == len(unique_features)
    assert interface_spy.call_count == 1
    assert radial_geometry_spy.call_count == 1
    assert bind_spy.call_count == len(pairs), (
        f"bind_tessellation_pooling_index: expected {len(pairs)}x, "
        f"got {bind_spy.call_count}"
    )
