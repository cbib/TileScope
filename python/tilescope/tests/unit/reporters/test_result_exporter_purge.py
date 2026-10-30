"""The exporter clears stale owned artifacts without touching user files."""
from pathlib import Path

from tilescope.reporters.result_exporter import _purge_managed_and_retired


def test_purge_removes_each_owned_artifact_kind_and_preserves_unrelated(
    tmp_path: Path,
):
    removed = (
        "roi_correlations.csv",
        "summary_cross_scale.csv",
        "tessellation_7_correlation.csv",
    )
    preserved = "notes.csv"
    for name in (*removed, preserved):
        (tmp_path / name).write_text("content")

    _purge_managed_and_retired(tmp_path)

    assert all(not (tmp_path / name).exists() for name in removed)
    assert (tmp_path / preserved).read_text() == "content"
