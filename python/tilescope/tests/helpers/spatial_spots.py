"""
Deterministic spatial-spot fixture for TileScope tests.

Minimal internal builder for tests that need spots on a regular
grid with explicit ROI labels and finite gene1/gene2 values. NOT a
scientific simulator: does not model correlations, hotspots, or
any biological pattern.

For scientific benchmarking, use the sister-project scenario
generator at ../Tilescope-results/scenario_benchmark/, not this
module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_grid_spots(
    nx: int = 20,
    ny: int = 20,
    n_rois: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a `(nx*ny, 5)` DataFrame with columns
    `x, y, gene1, gene2, roi_id`.

    Coordinates form a regular `nx × ny` integer grid. Spots are
    split into `n_rois` ROIs by vertical bands of equal width (so
    ROI sizes are predictable). `gene1` and `gene2` are seeded
    standard-normal noise — finite, mean ~0, std ~1, no
    correlation structure claimed. `roi_id` is a string label."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    x = xs.ravel()
    y = ys.ravel()

    # Vertical bands assign ROIs by x position — deterministic and
    # produces contiguous connected components. Requires n_rois <= nx
    # so every band contains at least one column of spots; otherwise
    # the "predictable nonempty ROIs" contract in the docstring
    # would silently break.
    if n_rois < 1:
        raise ValueError("n_rois must be >= 1")
    if n_rois > nx:
        raise ValueError(f"n_rois ({n_rois}) must be <= nx ({nx}) so every band is nonempty")
    band_edges = np.linspace(0, nx, n_rois + 1)
    roi_ids = np.digitize(x, band_edges[1:-1]) if n_rois > 1 else np.zeros(len(x), dtype=int)

    return pd.DataFrame({
        "x": x,
        "y": y,
        "gene1": rng.normal(size=len(x)),
        "gene2": rng.normal(size=len(x)),
        "roi_id": roi_ids.astype(int).astype(str),
    })


def make_supported_multifeature_spots(
    nx: int = 20, ny: int = 20, n_rois: int = 4, seed: int = 0,
) -> pd.DataFrame:
    """Deterministic supported cohort for multi-pair authority tests.

    Same grid layout as `make_grid_spots` — every ROI passes the default
    KNN splitter + `min_component_spots=20` support gate. Overwrites the
    placeholder-normal `gene1`/`gene2` from `make_grid_spots` with real
    non-degenerate patterns and adds two extra features `G3`, `G4`, so
    multi-pair Pearson / enrichment / Moran contracts have both
    legacy-named and novel-named feature axes to exercise.

    Retained fields: `x`, `y`, `gene1`, `gene2`, `G3`, `G4`, `roi_id`.
    """
    df = make_grid_spots(nx=nx, ny=ny, n_rois=n_rois, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(df)
    # gene1: gradient in x (positive slope).
    # gene2: gradient in y (negative slope).
    # G3:    log-normal noise.
    # G4:    bimodal by ROI-block.
    df["gene1"] = df["x"] * 0.1 + rng.normal(scale=0.2, size=n)
    df["gene2"] = (df["y"].max() - df["y"]) * 0.1 + rng.normal(scale=0.2, size=n)
    df["G3"] = np.exp(rng.normal(loc=0.0, scale=0.5, size=n))
    df["G4"] = df["roi_id"].astype(int).to_numpy() * 1.5 + rng.normal(scale=0.3, size=n)
    return df
