"""Contract tests for :func:`build_roi_source_moran_df`.

Bivariate Moran on spot/cell values + canonical spot/cell adjacency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tilescope.analyzers.parent_roi_outcomes import ParentMoranStatus
from tilescope.analyzers.parent_roi_source_moran import (
    build_roi_source_moran_df as _build_roi_source_moran_df,
)
from tilescope.cols import (
    ADJACENCY_SUPPORTED_FRACTION, MORAN_DIRECTION, MORAN_VALUE,
    N_ADJACENCY_SUPPORTED_UNITS, PARENT_MORAN_STATUS, ROI_ID,
)
from tilescope.spatial.source_spot_adjacency import (
    SourceSpotAdjacency, build_source_spot_adjacency,
)
from tilescope.spatial.source_spot_table import (
    build_source_spot_table_from_frame,
)


def build_roi_source_moran_df(**kwargs):
    return _build_roi_source_moran_df(
        g1_id="Neuron", g2_id="AC", **kwargs,
    )


def _adjacency_from_grid(coords, roi_ids):
    df = pd.DataFrame({
        "x": [c[0] for c in coords], "y": [c[1] for c in coords],
        "roi_id": roi_ids,
        "source_row_id": list(range(len(coords))),
    })
    table = build_source_spot_table_from_frame(df)
    return build_source_spot_adjacency(table, radius_factor=1.5)


def _support_status(roi_ids):
    unique = sorted(set(roi_ids))
    return pd.DataFrame({
        ROI_ID: unique,
        "source_spots": [roi_ids.count(r) for r in unique],
        "spatial_support_class": ["supported"] * len(unique),
    })


def _line_coords(n: int, y: float = 0.0):
    return [(float(i), y) for i in range(n)]


def test_positive_spatial_autocorrelation_yields_positive_moran():
    coords = _line_coords(6)
    roi_ids = ["A"] * 6
    adj = _adjacency_from_grid(coords, roi_ids)
    ramp = np.linspace(0.0, 5.0, 6)
    df = build_roi_source_moran_df(
        source_spot_adjacency=adj,
        source_values_g1=ramp,
        source_values_g2=ramp + 0.5,
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=True,
    )
    finite = df[df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert (finite[MORAN_VALUE] > 0.3).all()


def test_anti_correlated_gradients_yield_negative_moran():
    coords = _line_coords(8)
    roi_ids = ["A"] * 8
    adj = _adjacency_from_grid(coords, roi_ids)
    ramp = np.linspace(0.0, 7.0, 8)
    df = build_roi_source_moran_df(
        source_spot_adjacency=adj,
        source_values_g1=ramp,
        source_values_g2=-ramp,
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=True,
    )
    finite = df[df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert (finite[MORAN_VALUE] < -0.3).all()


def test_zero_variance_on_a_reports_typed_status_in_both_directions():
    """A constant A must yield ZERO_VARIANCE_A in BOTH directions.

    Direction 1 (g1_lag_g2) passes X=A (constant) → ZERO_VARIANCE_X.
    Direction 2 (g2_lag_g1) passes Y=A (constant) → ZERO_VARIANCE_Y.
    Both must map back to biological A being constant.
    """
    coords = _line_coords(5)
    roi_ids = ["A"] * 5
    adj = _adjacency_from_grid(coords, roi_ids)
    constant = np.ones(5)
    varying = np.arange(5.0)
    df = build_roi_source_moran_df(
        source_spot_adjacency=adj,
        source_values_g1=constant,
        source_values_g2=varying,
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=True,
    )
    statuses = dict(zip(df[MORAN_DIRECTION], df[PARENT_MORAN_STATUS]))
    assert statuses["g1_lag_g2"] == ParentMoranStatus.ZERO_VARIANCE_A.value
    assert statuses["g2_lag_g1"] == ParentMoranStatus.ZERO_VARIANCE_A.value
    assert df[MORAN_VALUE].isna().all()


def test_zero_variance_on_b_reports_zero_variance_b_in_both_directions():
    """Symmetric test: constant B yields ZERO_VARIANCE_B either direction."""
    coords = _line_coords(5)
    roi_ids = ["A"] * 5
    adj = _adjacency_from_grid(coords, roi_ids)
    varying = np.arange(5.0)
    constant = np.ones(5)
    df = build_roi_source_moran_df(
        source_spot_adjacency=adj,
        source_values_g1=varying,
        source_values_g2=constant,
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=True,
    )
    statuses = dict(zip(df[MORAN_DIRECTION], df[PARENT_MORAN_STATUS]))
    assert statuses["g1_lag_g2"] == ParentMoranStatus.ZERO_VARIANCE_B.value
    assert statuses["g2_lag_g1"] == ParentMoranStatus.ZERO_VARIANCE_B.value


def test_bidirectional_structure_asymmetric_for_shifted_signals():
    # Bivariate Moran is direction-sensitive: shifted signals produce
    # different I(A|B) vs I(B|A). Both must be finite.
    coords = _line_coords(10)
    roi_ids = ["A"] * 10
    adj = _adjacency_from_grid(coords, roi_ids)
    x = np.arange(10.0)
    a = np.sin(x)
    b = np.sin(x + 0.5)
    df = build_roi_source_moran_df(
        source_spot_adjacency=adj,
        source_values_g1=a, source_values_g2=b,
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=True,
    )
    finite = df[df[PARENT_MORAN_STATUS] == ParentMoranStatus.FINITE.value]
    assert len(finite) == 2
    assert finite[MORAN_VALUE].notna().all()


def test_sparse_roi_adjacency_reports_insufficient_support():
    coords = _line_coords(6)
    roi_ids = ["A"] * 6
    table = build_source_spot_table_from_frame(pd.DataFrame({
        "x": [x for x, _ in coords],
        "y": [y for _, y in coords],
        "roi_id": roi_ids,
        "source_row_id": list(range(6)),
    }))
    adjacency = SourceSpotAdjacency(
        source_spot_table=table,
        method="fixed_radius",
        method_parameters=(),
        sample_median_nn=1.0,
        edges=np.asarray([[0, 1]], dtype=np.int64),
        neighbors=((1,), (0,), (), (), (), ()),
        within_roi_components=(),
    )
    df = build_roi_source_moran_df(
        source_spot_adjacency=adjacency,
        source_values_g1=np.arange(6.0),
        source_values_g2=np.arange(6.0),
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=True,
    )
    assert (
        df[PARENT_MORAN_STATUS]
        == ParentMoranStatus.INSUFFICIENT_ADJACENCY_SUPPORT.value
    ).all()
    assert (df[N_ADJACENCY_SUPPORTED_UNITS] == 2).all()
    assert np.allclose(df[ADJACENCY_SUPPORTED_FRACTION], 1.0 / 3.0)
    assert df[MORAN_VALUE].isna().all()


def test_moran_disabled_still_emits_total_table():
    coords = _line_coords(4)
    roi_ids = ["A"] * 4
    adj = _adjacency_from_grid(coords, roi_ids)
    df = build_roi_source_moran_df(
        source_spot_adjacency=adj,
        source_values_g1=np.arange(4.0),
        source_values_g2=np.arange(4.0),
        support_status_df=_support_status(roi_ids),
        sample_id="S",
        moran_enabled=False,
    )
    assert len(df) == 2
    assert (
        df[PARENT_MORAN_STATUS] == ParentMoranStatus.MORAN_DISABLED.value
    ).all()
    assert df[MORAN_VALUE].isna().all()
