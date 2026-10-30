"""Semantic coverage for the historical cross-grain backfill."""
from __future__ import annotations

from tilescope.api import run_tilescope_from_csv
from tilescope.tests.helpers.paths import FIXTURES_DIR


import json
from pathlib import Path

import pandas as pd
import pytest

from tilescope.analyzers.roi_cross_grain_sensitivity import (
    PRODUCER_CONTRACT_VERSION,
)
from tilescope.cols import ROI_ID
from tilescope.tools.backfill_cross_grain import (
    OUTPUT_FILENAME,
    RUN_METADATA_FILENAME,
    TILESCOPE_VERSION,
    BackfillStatus,
    _INGREDIENTS,
    _cli,
    backfill_corpus,
    backfill_run,
    write_backfill_manifest,
)


def _seed_run_dir(
    root: Path,
    source_run: Path,
    *,
    copy_output: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for filename, _attr in _INGREDIENTS:
        (root / filename).write_bytes((source_run / filename).read_bytes())
    (root / RUN_METADATA_FILENAME).write_bytes(
        (source_run / RUN_METADATA_FILENAME).read_bytes()
    )
    if copy_output:
        (root / OUTPUT_FILENAME).write_bytes(
            (source_run / OUTPUT_FILENAME).read_bytes()
        )
    return root


@pytest.fixture(scope="module")
def live_run_dir(tmp_path_factory) -> Path:

    output_dir = tmp_path_factory.mktemp("backfill-live-source")
    run_tilescope_from_csv(
        input_csv=FIXTURES_DIR / "grid_square_hole_and_circle.csv",
        profile="debugging",
        output_dir=output_dir,
        export=True,
        html_report=False,
        return_minimal=True,
        feature_a_id="Neuron",
        feature_b_id="AC",
    )
    return output_dir


def test_backfill_matches_live_output_and_preserves_run_metadata(
    live_run_dir,
    tmp_path,
):
    target = _seed_run_dir(tmp_path / "target", live_run_dir)
    metadata_before = (target / RUN_METADATA_FILENAME).read_bytes()

    result = backfill_run(target)

    assert result.status == BackfillStatus.WRITTEN, result.reason
    key = [
        "sample_id", "roi_id", "g1_id", "g2_id", "statistic",
        "channel", "moran_direction", "target_surface",
    ]
    live = pd.read_csv(live_run_dir / OUTPUT_FILENAME).sort_values(key)
    backfilled = pd.read_csv(target / OUTPUT_FILENAME).sort_values(key)
    pd.testing.assert_frame_equal(
        backfilled.reset_index(drop=True),
        live.reset_index(drop=True),
        check_dtype=False,
    )
    assert result.n_rows == 14 * result.n_rois
    assert (result.g1_id, result.g2_id) == ("Neuron", "AC")
    assert result.original_run_core_version is not None
    assert result.original_analysis_policy_version is not None
    assert result.output_sha256 is not None
    assert (target / RUN_METADATA_FILENAME).read_bytes() == metadata_before
    assert not list(target.glob(".backfill_*"))


def test_existing_output_skips_unless_force_is_requested(live_run_dir, tmp_path):
    target = _seed_run_dir(
        tmp_path / "target", live_run_dir, copy_output=True,
    )

    skipped = backfill_run(target)
    rewritten = backfill_run(target, force=True)

    assert skipped.status == BackfillStatus.SKIPPED_EXISTS
    assert skipped.n_rows == 14 * skipped.n_rois
    assert skipped.output_sha256 is not None
    assert rewritten.status == BackfillStatus.WRITTEN


def test_schema_valid_but_truncated_output_is_replaced(live_run_dir, tmp_path):
    target = _seed_run_dir(
        tmp_path / "target", live_run_dir, copy_output=True,
    )
    output = pd.read_csv(target / OUTPUT_FILENAME)
    output.iloc[[0]].to_csv(target / OUTPUT_FILENAME, index=False)

    result = backfill_run(target)

    rewritten = pd.read_csv(target / OUTPUT_FILENAME)
    assert result.status == BackfillStatus.WRITTEN
    assert len(rewritten) == 14 * int(rewritten[ROI_ID].nunique())


def test_incompatible_inputs_are_reported_not_raised(live_run_dir, tmp_path):
    targets: list[tuple[Path, str]] = []

    missing_metadata = _seed_run_dir(tmp_path / "missing-meta", live_run_dir)
    (missing_metadata / RUN_METADATA_FILENAME).unlink()
    targets.append((missing_metadata, RUN_METADATA_FILENAME))

    malformed_metadata = _seed_run_dir(tmp_path / "bad-meta", live_run_dir)
    (malformed_metadata / RUN_METADATA_FILENAME).write_text("{not json")
    targets.append((malformed_metadata, "malformed"))

    missing_ingredient = _seed_run_dir(tmp_path / "missing-input", live_run_dir)
    (missing_ingredient / "roi_source_moran.csv").unlink()
    targets.append((missing_ingredient, "roi_source_moran.csv"))

    missing_identity = _seed_run_dir(tmp_path / "missing-id", live_run_dir)
    frame = pd.read_csv(missing_identity / "roi_correlations.csv")
    frame.drop(columns=["sample_id"]).to_csv(
        missing_identity / "roi_correlations.csv", index=False,
    )
    targets.append((missing_identity, "sample_id"))

    wrong_identity = _seed_run_dir(tmp_path / "wrong-id", live_run_dir)
    frame = pd.read_csv(wrong_identity / "roi_correlations.csv")
    frame["g1_id"] = "wrong_feature"
    frame.to_csv(wrong_identity / "roi_correlations.csv", index=False)
    targets.append((wrong_identity, "disagrees with run_metadata"))

    for target, reason_fragment in targets:
        result = backfill_run(target)
        assert result.status == BackfillStatus.INCOMPATIBLE_SCHEMA
        assert reason_fragment in (result.reason or "")


def test_corpus_manifest_records_results_and_both_provenances(
    live_run_dir,
    tmp_path,
):
    root = tmp_path / "corpus"
    _seed_run_dir(root / "good", live_run_dir)
    bad = _seed_run_dir(root / "bad", live_run_dir)
    (bad / "roi_source_moran.csv").unlink()

    results = backfill_corpus(root)
    manifest_path = write_backfill_manifest(root, results)
    payload = json.loads(manifest_path.read_text())

    assert payload["n_runs_walked"] == 2
    assert payload["status_counts"] == {
        "written": 1, "incompatible_schema": 1,
    }
    assert payload["backfill_producer_core_version"] == TILESCOPE_VERSION
    assert payload["producer_contract_version"] == PRODUCER_CONTRACT_VERSION
    assert "backfill_producer_git_sha" in payload
    written = next(row for row in payload["runs"] if row["status"] == "written")
    assert written["n_rows"] == 14 * written["n_rois"]
    assert written["original_run_core_version"] is not None
    assert written["original_analysis_policy_version"] is not None
    assert "original_run_config_hash" not in written
    assert not list(root.glob(".backfill_*"))


def test_cli_exit_code_tracks_corpus_completeness(live_run_dir, tmp_path):
    clean = tmp_path / "clean"
    _seed_run_dir(clean / "sample", live_run_dir)
    assert _cli([str(clean)]) == 0
    assert len(list(clean.glob("cross_grain_backfill_manifest_*.json"))) == 1

    incomplete = tmp_path / "incomplete"
    bad = _seed_run_dir(incomplete / "sample", live_run_dir)
    (bad / "roi_source_moran.csv").unlink()
    assert _cli([str(incomplete)]) == 1
    manifest = next(incomplete.glob("cross_grain_backfill_manifest_*.json"))
    assert json.loads(manifest.read_text())["status_counts"] == {
        "incompatible_schema": 1,
    }
