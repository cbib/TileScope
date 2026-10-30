"""Scientific pair-pattern checks on the typed production path.

These replace scenario assertions that depended on the retired self-context,
mixed-grain enrichment path.  Each target ROI has an explicit independent
context ROI; correlations are evaluated on the ensemble consensus surface.
"""
from tilescope.cols import CORRELATION_VALUE, VALUE_SURFACE


import numpy as np
import pandas as pd

from tilescope.analyzers import run_analysis
from tilescope.cols import ROI_ID
from tilescope.settings import load_run_config


def _fixture(pattern: str) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    x, y = np.meshgrid(np.arange(12, dtype=float), np.arange(12, dtype=float))
    x = x.ravel()
    y = y.ravel()
    noise1 = rng.normal(0.0, 0.15, len(x))
    noise2 = rng.normal(0.0, 0.15, len(x))
    if pattern == "concordant":
        signal = 10.0 * np.exp(-((x - 5.5) ** 2 + (y - 5.5) ** 2) / 12.0)
        g1 = 1.0 + signal + noise1
        g2 = 1.0 + signal + noise2
    elif pattern == "discordant":
        g1 = 1.0 + 9.0 * (x < 6.0) + noise1
        g2 = 1.0 + 9.0 * (x >= 6.0) + noise2
    elif pattern == "checkerboard":
        g1 = 1.0 + 9.0 * ((x.astype(int) + y.astype(int)) % 2) + noise1
        g2 = 1.0 + 9.0 * ((x.astype(int) - y.astype(int)) % 3 == 0) + noise2
    else:
        raise ValueError(pattern)

    target = pd.DataFrame({
        "x": x,
        "y": y,
        "gene1": g1,
        "gene2": g2,
        "roi_id": "target",
    })
    context = pd.DataFrame({
        "x": x + 30.0,
        "y": y,
        "gene1": 2.0 + rng.normal(0.0, 0.5, len(x)),
        "gene2": 2.0 + rng.normal(0.0, 0.5, len(x)),
        "roi_id": "context",
    })
    return pd.concat([target, context], ignore_index=True)


def _run(pattern: str, output_dir):
    config = load_run_config(
        profile="standard",
        set_overrides={
            "output_dir": str(output_dir),
            "permutations": 0,
            "roi_filters": {
                "min_roi_spots": 20,
                "min_context_spots": 20,
            },
            "tessellation": {"n": 5},
            "consensus": {"min_tess_coverage_fraction": 0.4, "min_spots_per_hex": 3},
        },
    )
    return run_analysis(
        _fixture(pattern), config,
        feature_a_id="Neuron", feature_b_id="AC",
    )


def _target_consensus_correlation(results) -> float:
    """Target ROI's pooled consensus Pearson on the ensemble surface."""
    rows = results["roi_consensus_correlations_df"]
    ensemble = rows.loc[
        (rows[ROI_ID].astype(str) == "target")
        & (rows[VALUE_SURFACE] == "consensus_ensemble"),
        CORRELATION_VALUE,
    ]
    return float(ensemble.iloc[0])


def test_concordant_blob_remains_positive_after_ensemble_projection(tmp_path):
    assert _target_consensus_correlation(_run("concordant", tmp_path)) > 0.5


def test_discordant_halves_remain_negative_after_ensemble_projection(tmp_path):
    assert _target_consensus_correlation(_run("discordant", tmp_path)) < -0.5


def test_checkerboard_exposes_partition_dependent_replicate_dispersion(tmp_path):
    results = _run("checkerboard", tmp_path)
    rows = results["roi_replicate_correlations_df"]
    values = rows.loc[
        rows[ROI_ID] == "target", CORRELATION_VALUE
    ].dropna().to_numpy()
    assert len(values) >= 3
    assert float(np.ptp(values)) > 1e-6
