"""Single-ROI regression: context-independent estimands survive
when there is no external context to enrich against.

The retained invariants are:
- Parent-grain correlation (`roi_correlations`) is non-empty — the
  correlation gate is `min_roi_spots` only, so a single-ROI sample
  still produces its own row.
- Parent-grain enrichment (`roi_enrichment`) is empty — the
  enrichment gate additionally requires `min_context_spots`, and
  a single-ROI sample has no external context.
- ROI × replicate tessellation correlations
  (`roi_replicate_correlations_df`) are non-empty — tessellation is
  context-independent.

The retired component-spot machinery (`spot_metrics` /
`spot_correlations`) is not asserted; those keys no longer exist.
"""
from __future__ import annotations

from tilescope.cols import PARENT_CORRELATION_STATUS, PARENT_ENRICHMENT_STATUS


from pathlib import Path

import numpy as np
import pandas as pd

from tilescope.analyzers.orchestrator import run_analysis
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, ROI_ID
from tilescope.settings.loader import load_run_config


_HEX_DY = np.sqrt(3.0) / 2.0


def _hex_lattice(rows: int, cols: int, x0: float = 0.0, y0: float = 0.0,
                 spacing: float = 1.0) -> np.ndarray:
    pts = []
    for r in range(rows):
        y = y0 + r * spacing * _HEX_DY
        x_offset = (spacing / 2.0) if (r % 2 == 1) else 0.0
        for c in range(cols):
            x = x0 + x_offset + c * spacing
            pts.append((x, y))
    return np.array(pts, dtype=float)


def test_single_roi_tessellation_correlation_survive_empty_context(tmp_path: Path):
    # 60-spot single-ROI sample; deterministic gene values so
    # correlation is computable.
    coords = _hex_lattice(6, 10, x0=0.0)  # 60 spots
    df = pd.DataFrame({
        ROI_ID: ["A"] * 60,
        COORD_X: coords[:, 0],
        COORD_Y: coords[:, 1],
        GENE1: np.arange(60, dtype=float),
        GENE2: np.arange(60, dtype=float),
    })
    assert len(df) == 60

    config = load_run_config(
        path=None, profile=None,
        flag_overrides={"output_dir": str(tmp_path)},
        set_overrides=None,
    )

    # Must not abort. Pre-flight uses the tessellation-eligible set
    # and the enrichment step is guarded downstream.
    results = run_analysis(
        df, config, feature_a_id="Neuron", feature_b_id="AC",
    )

    # Parent-grain correlation: total table, one row for A with
    # FINITE status. Correlation gate is `min_roi_spots` only —
    # context-independent.
    roi_correlations = results.get("roi_correlations")
    assert roi_correlations is not None
    assert (roi_correlations[ROI_ID].astype(str) == "A").all()
    a_corr_status = roi_correlations[
        roi_correlations[ROI_ID].astype(str) == "A"
    ].iloc[0][PARENT_CORRELATION_STATUS]
    assert a_corr_status == "finite", (
        f"single-ROI correlation must be FINITE (context-independent); "
        f"got {a_corr_status!r}"
    )

    # Parent-grain enrichment: total table, one row for A with
    # INSUFFICIENT_CONTEXT_SPOTS. The row is PRESENT with a typed
    # status; consumers filter on the status, not on row absence.
    roi_enrichment = results.get("roi_enrichment")
    assert roi_enrichment is not None
    assert (roi_enrichment[ROI_ID].astype(str) == "A").all()
    a_enr_status = roi_enrichment[
        roi_enrichment[ROI_ID].astype(str) == "A"
    ].iloc[0][PARENT_ENRICHMENT_STATUS]
    assert a_enr_status == "insufficient_context_spots", (
        f"single-ROI enrichment must record INSUFFICIENT_CONTEXT_SPOTS; "
        f"got {a_enr_status!r}"
    )

    # ROI × replicate tessellation correlations: present. Same
    # context-independence rationale as the parent-grain correlation.
    # Canonical frame keyed by (sample_id, roi_id, replicate_index).
    replicate_corr_df = results.get("roi_replicate_correlations_df")
    assert replicate_corr_df is not None and not replicate_corr_df.empty, (
        "ROI × replicate correlations must exist in single-ROI samples."
    )
    assert (replicate_corr_df[ROI_ID] == "A").all()

    # Retirement contract: the two retired keys must be absent.
    assert "spot_metrics" not in results
    assert "spot_correlations" not in results
