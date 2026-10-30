"""source_row_id lineage through the pre-analysis helper pipeline.

The stable per-row identifier `SOURCE_ROW_ID` must survive each
pipeline stage as a subset of
the **immediately preceding stage's** ID set (never fabricated,
never reintroduced after being dropped upstream), with no duplication
or type drift. Exact ID sets are asserted at each stage.

This test exercises the production helpers (`prepare_source_frame`,
`apply_splitter_and_filter`, `apply_scattered_gate_mask`) rather than
the raw underlying stage functions — so if the orchestrator's helper
wiring or the helpers' internal validation regress, the test catches
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.pre_analysis import (
    _validate_source_row_identity,
    apply_scattered_gate_mask,
    apply_splitter_and_filter,
    prepare_source_frame,
)
from tilescope.cols import COORD_X, COORD_Y, GENE1, GENE2, ROI_ID, SOURCE_ROW_ID


def _make_fixture() -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Three ROIs with distinguishable fates through the pipeline.

    ROI A — 10 spots on a tight 1-D lattice: survives every stage.
    ROI B — 2 spots (whole-ROI below min_component_spots). The
        splitter treats it as one small component (cc_idx=0,
        `roi_splitter.py:45-49`) so it survives split; then
        `filter_components(min_component_spots=3)` drops it.
    ROI C — 5 tightly-packed spots at a distant location: survives
        split and filter, then dropped by the scattered-gate helper
        called with `scattered_rois=frozenset({"C"})`.

    Returns the raw fixture (no SOURCE_ROW_ID column — that is
    `prepare_source_frame`'s job) and expected surviving IDs per
    stage. IDs are the row positions the snapshot will assign
    (0..16), which the test asserts explicitly.
    """
    roi_a_xy = [(float(i), 0.0) for i in range(10)]                # rows 0..9
    roi_b_xy = [(100.0, 0.0), (101.0, 0.0)]                        # rows 10, 11
    roi_c_xy = [(500.0 + 0.1 * i, 500.0) for i in range(5)]         # rows 12..16

    rows: list[dict] = []
    for x, y in roi_a_xy:
        rows.append({ROI_ID: "A", COORD_X: x, COORD_Y: y, GENE1: 1.0, GENE2: 1.0})
    for x, y in roi_b_xy:
        rows.append({ROI_ID: "B", COORD_X: x, COORD_Y: y, GENE1: 1.0, GENE2: 1.0})
    for x, y in roi_c_xy:
        rows.append({ROI_ID: "C", COORD_X: x, COORD_Y: y, GENE1: 1.0, GENE2: 1.0})

    df = pd.DataFrame(rows)

    expected = {
        "source": list(range(17)),
        # Splitter keeps ROI B as one small component (cc_idx=0) —
        # nothing dropped here.
        "split": list(range(17)),
        # filter_components(min_component_spots=3) drops ROI B's
        # 2-spot component; ROI A (10) and ROI C (5) survive.
        "filtered": [*range(10), *range(12, 17)],
        # apply_scattered_gate_mask removes ROI C (rows 12..16).
        "gated": list(range(10)),
    }
    return df, expected


def test_source_row_id_survives_pipeline_stages():
    df_raw, expected = _make_fixture()

    source_df = prepare_source_frame(df_raw)
    assert set(source_df[SOURCE_ROW_ID].tolist()) == set(expected["source"])

    sf = apply_splitter_and_filter(
        source_df,
        knn_k=3,
        knn_gamma=None,
        knn_threshold="local",
        knn_symmetrize="mutual",
        min_component_spots=3,
        canvas=None,
    )
    assert set(sf.split_df[SOURCE_ROW_ID].tolist()) == set(expected["split"])
    assert (
        set(sf.component_expression_df[SOURCE_ROW_ID].tolist())
        == set(expected["filtered"])
    )

    gated = apply_scattered_gate_mask(
        sf.component_expression_df,
        frozenset({"C"}),
        sample_median_nn=1.0,
    )
    assert set(gated[SOURCE_ROW_ID].tolist()) == set(expected["gated"])


def test_prepare_source_frame_rejects_preexisting_column():
    df, _ = _make_fixture()
    df[SOURCE_ROW_ID] = np.int64(42)
    with pytest.raises(ValueError, match=r"reserved internal column"):
        prepare_source_frame(df)


def test_validator_rejects_missing_column():
    df = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(ValueError, match=r"column .* missing"):
        _validate_source_row_identity(df, allowed_ids=frozenset(), stage="unit")


def test_validator_rejects_null_ids():
    df = pd.DataFrame({SOURCE_ROW_ID: pd.Series([0, 1, pd.NA], dtype="Int64")})
    with pytest.raises(ValueError, match=r"null value"):
        _validate_source_row_identity(df, allowed_ids=frozenset({0, 1}), stage="unit")


def test_validator_rejects_wrong_dtype():
    df = pd.DataFrame({SOURCE_ROW_ID: np.array([0, 1, 2], dtype=np.int32)})
    with pytest.raises(ValueError, match=r"dtype"):
        _validate_source_row_identity(
            df, allowed_ids=frozenset({0, 1, 2}), stage="unit",
        )


def test_validator_rejects_duplicates():
    df = pd.DataFrame({SOURCE_ROW_ID: np.array([0, 1, 1], dtype=np.int64)})
    with pytest.raises(ValueError, match=r"duplicate ID"):
        _validate_source_row_identity(
            df, allowed_ids=frozenset({0, 1}), stage="unit",
        )


def test_validator_rejects_fabricated_ids():
    df = pd.DataFrame({SOURCE_ROW_ID: np.array([0, 1, 99], dtype=np.int64)})
    with pytest.raises(ValueError, match=r"not present in the preceding-stage"):
        _validate_source_row_identity(
            df, allowed_ids=frozenset({0, 1, 2}), stage="unit",
        )
