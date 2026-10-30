"""Support-policy provenance + realized-tile summary in run metadata.

One shared pipeline run — both the policy pins and the derived
`realized_tile_spots` summary come from the same `run_metadata` object,
so a fresh execution per assertion is redundant.
"""
import pytest

from tilescope.analyzers import run_analysis
from tilescope.defaults import SEED_COUNT_FORMULA, SUPPORT_POLICY_VERSION
from tilescope.settings import load_run_config
from tilescope.tests.helpers.paths import FIXTURES_DIR
from tilescope.tilescope_io.readers import read_expression_csv


@pytest.fixture(scope="module")
def tessellation_metadata(tmp_path_factory):
    out = tmp_path_factory.mktemp("tess-metadata")
    df = read_expression_csv(FIXTURES_DIR / "test_int_coords.csv")
    config = load_run_config(
        profile="standard",
        set_overrides={
            "output_dir": str(out),
            "tessellation": {"n": 2},
            "consensus": {"min_tess_coverage_fraction": 1.0, "min_spots_per_hex": 3},
        },
    )
    return run_analysis(
        df, config, feature_a_id="Neuron", feature_b_id="AC",
    )["run_metadata"]["tessellation"]


def test_tessellation_metadata_pins_policy_and_realized_tile_summary(
    tessellation_metadata,
):
    tess = tessellation_metadata
    assert (
        tess["support_policy_version"]
        == SUPPORT_POLICY_VERSION
        == "technology-support-v1"
    )
    assert tess["technology"] == "visium"
    assert tess["source_adjacency"]["method"] == "fixed_radius"
    assert tess["target_observations_per_tile"] == 5
    assert tess["resolved_voronoi_fraction"] == pytest.approx(1.0 / 5.0)
    assert tess["resolved_min_spots_per_tile"] == 3
    assert tess["min_component_observations"] == 15
    assert tess["tile_value_aggregation"] == "mean_per_source_observation"
    assert tess["seed_count_formula"] == SEED_COUNT_FORMULA

    summary = tess["realized_tile_spots"]
    assert set(summary) == {"n_tiles", "median", "q25", "q75"}
    assert summary["n_tiles"] > 0
    assert summary["q25"] <= summary["median"] <= summary["q75"]
    assert summary["q25"] >= 3
