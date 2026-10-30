"""Unit tests for the TileScope programmatic API surface.

Covers:
  - config helpers (make_run_config, validate_run_config)
  - DataFrame validation (validate_tilescope_dataframe)
  - CSV / DF route error paths that raise BEFORE run_analysis is invoked
  - Override resolution paths that don't need to touch the pipeline

The one meaningful full-pipeline test lives in
tests/integration/end_to_end/test_api_smoke.py — the CSV route is
covered end-to-end there. The DF route shares the downstream
run_analysis boundary, so its coverage here is mock-based: verify the
route validates before spending pipeline time.
"""
import unittest.mock as mock

import pandas as pd
import pytest

from tilescope.api import (
    make_run_config,
    run_tilescope_from_csv,
    run_tilescope_from_df,
    validate_run_config,
    validate_tilescope_dataframe,
)
from tilescope.settings import ResolvedRunConfig
from tilescope.tests.helpers.paths import FIXTURES_DIR


class TestConfigValidation:
    """validate_run_config accepts dicts + models; make_run_config layers overrides."""

    def test_validate_run_config_with_dict(self):
        config_dict = {"tessellation": {"n": 10}}
        validated = validate_run_config(config_dict)
        assert isinstance(validated, ResolvedRunConfig)
        assert validated.run_config.tessellation.n == 10

    def test_validate_run_config_with_model(self):
        resolved = make_run_config(profile="standard")
        validated = validate_run_config(resolved.run_config)
        assert isinstance(validated, ResolvedRunConfig)
        assert validated.run_config == resolved.run_config

    def test_validate_run_config_rejects_stale_derived_policy(self):
        resolved = make_run_config(profile="standard")
        stale_tessellation = resolved.run_config.tessellation.model_copy(
            update={"n": 1},
        )
        stale_run_config = resolved.run_config.model_copy(
            update={"tessellation": stale_tessellation},
        )
        stale = resolved.model_copy(update={"run_config": stale_run_config})
        with pytest.raises(ValueError, match="min_tess_per_hex"):
            validate_run_config(stale)

    def test_make_run_config_layers_profile_overrides_and_kwargs(self):
        config = make_run_config(
            profile="standard",
            overrides={"tessellation": {"n": 7}},
            seed=999,
        )
        assert isinstance(config, ResolvedRunConfig)
        assert config.run_config.tessellation.n == 7
        assert config.run_config.seed == 999


class TestDataframeValidation:
    """validate_tilescope_dataframe checks the required-columns contract."""

    def test_validate_tilescope_dataframe_valid(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0], "y": [3.0, 4.0],
            "gene1": [0.5, 0.6], "gene2": [0.7, 0.8],
            "roi_id": ["a", "b"],
        })
        validate_tilescope_dataframe(df)  # should not raise

    def test_validate_tilescope_dataframe_missing_column(self):
        df = pd.DataFrame({"x": [1.0], "y": [3.0], "gene1": [0.5], "gene2": [0.7]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_tilescope_dataframe(df)

    @pytest.mark.parametrize(
        "bad_value", ["not-a-number", float("nan"), float("inf"), float("-inf")],
    )
    def test_validate_tilescope_dataframe_rejects_invalid_numeric(self, bad_value):
        df = pd.DataFrame({
            "x": [1.0], "y": [3.0], "gene1": [bad_value],
            "gene2": [0.7], "roi_id": ["a"],
        })
        with pytest.raises((ValueError, TypeError)):
            validate_tilescope_dataframe(df)

    def test_validate_tilescope_dataframe_empty(self):
        df = pd.DataFrame(columns=["x", "y", "gene1", "gene2", "roi_id"])
        with pytest.raises(ValueError):
            validate_tilescope_dataframe(df)


class TestOverrideResolution:
    """Overrides land where they're supposed to — tested at the config layer,
    without paying the cost of run_analysis. (The end-to-end CSV smoke test in
    integration/end_to_end/test_api_smoke.py covers full-pipeline plumbing.)"""

    def test_csv_route_accepts_make_run_config_result(self):
        config = make_run_config(profile="debugging")
        with mock.patch(
            "tilescope.api.core.run_analysis",
            return_value={"run_metadata": {}},
        ) as run_analysis_mock:
            results = run_tilescope_from_csv(
                FIXTURES_DIR / "grid_square_hole_and_circle.csv",
                run_config=config,
                export=False,
                feature_a_id="Neuron",
                feature_b_id="AC",
            )

        assert results["config"] == config
        assert run_analysis_mock.call_args.args[1] == config

    def test_df_route_uses_shared_export_finalizer(self, tmp_path):
        config = make_run_config(profile="debugging")
        df = pd.read_csv(FIXTURES_DIR / "grid_square_hole_and_circle.csv")
        with (
            mock.patch(
                "tilescope.api.core.run_analysis",
                return_value={"run_metadata": {}},
            ),
            mock.patch(
                "tilescope.api.core.export_results", return_value={"csv": "x"},
            ) as export_mock,
            mock.patch("tilescope.api.core.dump_effective_config") as dump_mock,
        ):
            result = run_tilescope_from_df(
                df=df, config=config, output_dir=tmp_path, export=True,
                feature_a_id="Neuron", feature_b_id="AC",
            )

        export_mock.assert_called_once()
        dump_mock.assert_called_once()
        assert result["config"].run_config.output_dir == str(tmp_path)
        assert result["export_paths"] == {"csv": "x"}


class TestErrorPathsSkipPipeline:
    """Load-bearing invariant: errors raise BEFORE run_analysis is invoked so
    users don't wait minutes to learn their DataFrame is invalid or their CSV
    doesn't exist."""

    def test_missing_csv_raises_without_running_pipeline(self, tmp_path):
        with mock.patch("tilescope.api.core.run_analysis") as run_analysis_mock:
            with pytest.raises(FileNotFoundError):
                run_tilescope_from_csv(
                    input_csv="does-not-exist.csv",
                    output_dir=tmp_path,
                    export=False,
                    feature_a_id="Neuron",
                    feature_b_id="AC",
                )
        run_analysis_mock.assert_not_called()

    def test_invalid_dataframe_raises_without_running_pipeline(self, tmp_path):
        invalid_df = pd.DataFrame({"x": [1.0], "y": [2.0]})  # missing gene columns + roi_id
        config = make_run_config(profile="debugging")
        with mock.patch("tilescope.api.core.run_analysis") as run_analysis_mock:
            with pytest.raises(ValueError, match="Missing required columns"):
                run_tilescope_from_df(
                    df=invalid_df,
                    config=config,
                    output_dir=tmp_path,
                    export=False,
                    feature_a_id="Neuron",
                    feature_b_id="AC",
                )
        run_analysis_mock.assert_not_called()

    def test_html_without_export_raises_before_pipeline(self):
        df = pd.read_csv(FIXTURES_DIR / "grid_square_hole_and_circle.csv")
        config = make_run_config(profile="debugging")
        with mock.patch("tilescope.api.core.run_analysis") as run_analysis_mock:
            with pytest.raises(ValueError, match="html_report=True requires export=True"):
                run_tilescope_from_df(
                    df=df, config=config, export=False, html_report=True,
                    feature_a_id="Neuron", feature_b_id="AC",
                )
        run_analysis_mock.assert_not_called()
