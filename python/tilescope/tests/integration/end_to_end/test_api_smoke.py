"""One public-API pipeline execution asserts every downstream contract.

`return_minimal=True` and `export=True` are orthogonal — one CSV call
covers CSV ingestion, exact minimal-return shape, expected CSVs, stale
artifact purge, PDF materialisation, and retired-key contracts.
"""
import pandas as pd
import pytest

from tilescope.api import run_tilescope_from_csv
from tilescope.cols import (
    COMPONENT_ID, G1_ID, G2_ID, PARENT_ENRICHMENT_STATUS, REPLICATE_INDEX,
    ROI_ID, SAMPLE_ID,
)
from tilescope.tests.helpers.paths import FIXTURES_DIR


STALE_ARTIFACT_NAMES = (
    "spot_enrichment.csv",
    "spot_correlation.csv",
    "summary_cross_scale.csv",
)

MINIMAL_RETURN_KEYS = {
    "roi_enrichment", "roi_correlations", "roi_support_status",
    "roi_replicate_correlations_df", "roi_tessellation_correlations",
    "roi_replicate_enrichment_df",
    "roi_tessellation_enrichment_df", "roi_consensus_correlations_df",
    "roi_consensus_enrichment_df",
    "roi_source_moran_df", "roi_tessellation_moran_df",
    "roi_consensus_moran_df", "roi_cross_grain_sensitivity_df",
    "roi_pair_interface_response_df",
    "component_radial_shell_correlogram_df",
    "consensus_hexes_df", "tessellation_tiles_df",
    "run_metadata", "config", "export_paths",
}


@pytest.fixture(scope="module")
def api_smoke_run(tmp_path_factory):
    """One CSV pipeline execution — export + return_minimal + html_report=False.
    Stale artifacts pre-planted so the purge invariant is exercised."""
    out = tmp_path_factory.mktemp("api-smoke")
    for stale in STALE_ARTIFACT_NAMES:
        (out / stale).write_text("stale,contents\n1,2\n")
    csv = FIXTURES_DIR / "grid_square_hole_and_circle.csv"
    results = run_tilescope_from_csv(
        input_csv=csv, profile="debugging", output_dir=out,
        export=True, html_report=False, return_minimal=True,
        feature_a_id="Neuron", feature_b_id="AC",
    )
    return out, results


def test_return_minimal_exact_public_key_set(api_smoke_run):
    _, results = api_smoke_run
    assert set(results.keys()) == MINIMAL_RETURN_KEYS
    for retired in (
        "tessellation_correlations", "spot_metrics", "spot_correlations",
    ):
        assert retired not in results


def test_stale_artifacts_purged_and_fresh_exports_present(api_smoke_run):
    out, results = api_smoke_run
    assert results["export_paths"]
    for stale in STALE_ARTIFACT_NAMES:
        assert not (out / stale).exists(), (
            f"retired artifact {stale!r} survived startup purge"
        )
    for produced in (
        "effective_config.json", "roi_correlations.csv",
        "roi_enrichment.csv", "roi_cross_grain_sensitivity.csv",
        "expression_summary.pdf",
    ):
        path = out / produced
        assert path.exists(), f"expected {produced} to be emitted"
        assert path.stat().st_size > 0


def test_cross_grain_sensitivity_frame_shape(api_smoke_run):
    """The materialised roi_cross_grain_sensitivity frame carries the
    canonical row-key columns and the closed 14-rows-per-ROI row set.
    Catches misspelled exporter dict keys and adapter/producer regressions
    that no unit test can see in isolation."""
    out, results = api_smoke_run
    csv_path = out / "roi_cross_grain_sensitivity.csv"
    df = pd.read_csv(csv_path)
    assert not df.empty, "roi_cross_grain_sensitivity.csv is empty"
    for col in (
        SAMPLE_ID, ROI_ID, G1_ID, G2_ID,
        "statistic", "channel", "moran_direction",
        "reference_surface", "target_surface",
        "comparison_status", "delta_effect",
    ):
        assert col in df.columns, f"missing canonical column {col!r}"
    # Closed row set: 14 rows per ROI (7 triples x 2 target surfaces)
    per_roi = df.groupby([SAMPLE_ID, ROI_ID]).size()
    assert (per_roi == 14).all(), (
        f"expected 14 rows per (sample, roi); saw counts {per_roi.unique().tolist()!r}"
    )


def test_roi_grain_contracts_on_exported_csvs(api_smoke_run):
    out, _ = api_smoke_run
    corr = pd.read_csv(out / "roi_correlations.csv")
    assert not corr.empty
    assert ROI_ID in corr.columns
    assert COMPONENT_ID not in corr.columns
    assert G1_ID in corr.columns and G2_ID in corr.columns

    enrich = pd.read_csv(out / "roi_enrichment.csv")
    assert not enrich.empty
    assert ROI_ID in enrich.columns
    assert COMPONENT_ID not in enrich.columns
    assert PARENT_ENRICHMENT_STATUS in enrich.columns


def test_roi_replicate_correlations_composite_key(api_smoke_run):
    _, results = api_smoke_run
    rrc = results["roi_replicate_correlations_df"]
    assert rrc is not None
    for col in (SAMPLE_ID, ROI_ID, REPLICATE_INDEX):
        assert col in rrc.columns
    assert not rrc.duplicated([SAMPLE_ID, ROI_ID, REPLICATE_INDEX]).any()


def test_component_radial_shell_correlogram_export_shape(api_smoke_run):
    """The component-grain radial descriptor lands on disk with the
    long-form per-component-per-shell shape, keys both roi_id and
    component_id, and covers every retained source component with
    exactly n_shells rows (component × shell composite key is unique)."""
    out, results = api_smoke_run
    path = out / "component_radial_shell_correlogram.csv"
    assert path.exists() and path.stat().st_size > 0
    df = pd.read_csv(path)
    assert not df.empty
    for col in (
        SAMPLE_ID, ROI_ID, COMPONENT_ID, "shell_bin_index",
        "shell_edge_lower_normalized", "shell_edge_upper_normalized",
        "shell_midpoint_normalized", "shell_symmetric",
        "component_radial_shell_status",
        "n_anchors_per_shell", "fraction_anchors_per_shell",
        "n_unordered_pairs_per_shell",
        "median_nn_spacing", "component_rho_0",
    ):
        assert col in df.columns, f"missing column {col!r}"
    assert not df.duplicated([SAMPLE_ID, COMPONENT_ID, "shell_bin_index"]).any()
    # Same frame is exposed on the minimal return path.
    mem = results["component_radial_shell_correlogram_df"]
    assert mem is not None and len(mem) == len(df)
