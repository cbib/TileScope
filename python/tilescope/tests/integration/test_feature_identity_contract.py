"""End-to-end pair-identity contract.

Semantic identity supplied via `feature_a_id`/`feature_b_id` on the
public API must land unchanged on every downstream surface. If any of
these disagree, downstream PDFs and CSVs would carry contradicting
labels — the exact failure mode the identity contract was written to
prevent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tilescope.api import run_tilescope_from_csv


_A_ID = "Neuron"
_B_ID = "AC"


def _make_input_csv(path: Path) -> Path:
    # Minimal viable substrate: two spatially-supported ROIs, enough
    # spots to clear the default eligibility floor.
    rng = np.random.default_rng(0)
    rows = []
    for roi in ("A", "B"):
        xs = rng.uniform(0, 20, 80) + (0 if roi == "A" else 40)
        ys = rng.uniform(0, 20, 80)
        g1 = rng.normal(0, 1, 80)
        g2 = rng.normal(0, 1, 80) + (0.3 if roi == "A" else -0.3)
        for x, y, a, b in zip(xs, ys, g1, g2):
            rows.append({"x": x, "y": y, "gene1": a, "gene2": b, "roi_id": roi})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def input_csv(tmp_path: Path) -> Path:
    return _make_input_csv(tmp_path / "input.csv")


def test_identity_propagates_across_every_pair_owned_surface(input_csv, tmp_path):
    """Same semantic ids must appear in the dict, run_metadata.json,
    every pair-owned CSV, and the visualization runner's inputs."""
    out = tmp_path / "out"
    result = run_tilescope_from_csv(
        input_csv=input_csv,
        output_dir=str(out),
        export=True,
        feature_a_id=_A_ID,
        feature_b_id=_B_ID,
    )
    # Dict authority.
    assert result["feature_g1_id"] == _A_ID
    assert result["feature_g2_id"] == _B_ID
    # Metadata authority.
    metadata = json.loads((out / "run_metadata.json").read_text())
    assert metadata["feature_identity"] == {
        "feature_g1_id": _A_ID, "feature_g2_id": _B_ID,
    }
    # Every pair-owned CSV must carry matching g1_id/g2_id.
    pair_csvs = (
        "roi_correlations.csv",
        "roi_enrichment.csv",
        "roi_replicate_correlations.csv",
        "roi_replicate_enrichment.csv",
        "roi_tessellation_correlations.csv",
        "roi_tessellation_enrichment.csv",
        "roi_consensus_correlations.csv",
        "roi_consensus_enrichment.csv",
        "roi_source_moran.csv",
        "roi_tessellation_moran.csv",
        "roi_consensus_moran.csv",
    )
    for name in pair_csvs:
        csv_path = out / name
        assert csv_path.exists(), f"missing pair CSV {name}"
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        assert "g1_id" in df.columns and "g2_id" in df.columns, (
            f"{name} lacks g1_id/g2_id"
        )
        distinct = df[["g1_id", "g2_id"]].drop_duplicates()
        assert len(distinct) == 1, f"{name}: multiple id pairs {distinct!r}"
        assert distinct.iloc[0]["g1_id"] == _A_ID, f"{name} g1_id"
        assert distinct.iloc[0]["g2_id"] == _B_ID, f"{name} g2_id"


@pytest.mark.parametrize(
    "a,b",
    [
        (None, None),
        (_A_ID, None),
        (None, _B_ID),
        ("gene1", "gene2"),
    ],
)
def test_missing_or_computational_identity_raises(input_csv, tmp_path, a, b):
    """Canonical column names must never become biological identities."""
    with pytest.raises(ValueError, match="biological feature identit|reserved"):
        run_tilescope_from_csv(
            input_csv=input_csv, output_dir=str(tmp_path / "wont_reach"),
            export=False, feature_a_id=a, feature_b_id=b,
        )
