"""Semantic contract tests for :func:`build_component_radial_shell_correlogram_df`.

Three tests, one per contract axis:

1. Scale-normalization invariance: multiplying input coordinates by a
   positive scalar does not change `shell_symmetric` values; only the
   raw `median_nn_spacing` scales.
2. Persistent vs short-range curve at matched rho_0 = 1: a long-
   correlation Gaussian-smoothed field keeps positive correlation
   across the outer shell; a short-correlation field decays fast. Both
   cases use g1 == g2 so rho_0 is exactly 1 by construction and the
   test discriminates on curve shape rather than rediscovering rho_0.
3. Typed insufficient-anchor behaviour: a shell whose eligible-anchor
   count / fraction fails the gate carries INSUFFICIENT_ANCHOR_SUPPORT
   while adequately-supported shells remain FINITE for the same
   component.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from tilescope.analyzers.component_radial_shell_correlogram import (
    ComponentRadialShellStatus,
    build_component_radial_shell_correlogram_df,
)
from tilescope.cols import (
    COMPONENT_RADIAL_SHELL_STATUS, COMPONENT_RHO_0,
    FRACTION_ANCHORS_PER_SHELL, MEDIAN_NN_SPACING,
    N_ANCHORS_PER_SHELL, N_UNORDERED_PAIRS_PER_SHELL,
    SHELL_BIN_INDEX, SHELL_SYMMETRIC,
)
from tilescope.settings.models import RadialShellPolicyModel
from tilescope.spatial.feature_projection import (
    FeatureSpatialProjection, FeatureThresholds,
)
from tilescope.spatial.sample_support_bank import (
    ComponentSupport, SampleSupportBank, SampleSupportFingerprint,
    TessellationBank,
)
from tilescope.spatial.source_spot_adjacency import build_source_spot_adjacency
from tilescope.spatial.source_spot_table import SourceSpotTable


def _fingerprint() -> SampleSupportFingerprint:
    return SampleSupportFingerprint(
        sample_id="S", source_frame_hash="",
        support_policy_version="", source_adjacency_config_hash="",
        scattered_support_policy_hash="",
        component_geometry_contract_hash="",
        n_replicates_planned=1, base_seed=0,
        tessellation_config_hash="",
    )


def _make_bank(x: np.ndarray, y: np.ndarray) -> SampleSupportBank:
    n = int(x.size)
    fp = _fingerprint()
    sst = SourceSpotTable(
        spot_id=tuple(str(i) for i in range(n)),
        x=x, y=y, roi_id=("R0",) * n,
    )
    adj = build_source_spot_adjacency(sst)
    xmin, xmax = float(x.min()) - 1.0, float(x.max()) + 1.0
    ymin, ymax = float(y.min()) - 1.0, float(y.max()) + 1.0
    comp = ComponentSupport(
        component_id="c0", parent_roi_id="R0",
        points=np.column_stack([x, y]),
        source_row_ids=np.arange(n, dtype=np.int64),
        polygon=box(xmin, ymin, xmax, ymax),
    )
    tess = TessellationBank(
        n_replicates_planned=1, base_seed=0,
        tessellations={"c0": MappingProxyType({})},
        region_diagnostics={"c0": ()},
        region_replicate_diagnostics={"c0": ()},
    )
    return SampleSupportBank(
        sample_id="S", fingerprint=fp,
        source_spots=sst, source_spot_adjacency=adj,
        components={"c0": comp},
        component_domains={},
        tessellation_bank=tess,
    )


def _projection(name: str, values: np.ndarray,
                fp: SampleSupportFingerprint) -> FeatureSpatialProjection:
    thresholds = FeatureThresholds(
        q_high=0.9, q_low=0.1, resolved_high=1.0, resolved_low=0.0,
        n_source_values_used=int(values.size),
    )
    return FeatureSpatialProjection(
        feature_name=name, bank_fingerprint=fp,
        source_values=values, thresholds=thresholds,
        tile_marginal_sums={}, hex_ensemble_projection={},
        hex_direct_projection={},
    )


def _make_grid(nx: int, ny: int, scale: float = 1.0):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    x = xs.ravel().astype(float) * scale
    y = ys.ravel().astype(float) * scale
    return x, y


def _run(*, x, y, g1, g2, policy=None) -> pd.DataFrame:
    bank = _make_bank(x, y)
    proj_a = _projection("a", g1, bank.fingerprint)
    proj_b = _projection("b", g2, bank.fingerprint)
    return build_component_radial_shell_correlogram_df(
        support_bank=bank,
        projection_g1=proj_a, projection_g2=proj_b,
        radial_shell_policy=policy or RadialShellPolicyModel(),
        g1_id="a", g2_id="b",
    )


def test_scale_normalization_invariance():
    """Scaling the input coordinates uniformly must leave shell values
    unchanged; only median_nn_spacing tracks the raw scale."""
    rng = np.random.default_rng(2026_09_03)
    x, y = _make_grid(10, 10)
    n = x.size
    smooth = np.sin(x / 2.0) + np.cos(y / 2.0)
    g1 = smooth + 0.05 * rng.normal(size=n)
    g2 = smooth + 0.05 * rng.normal(size=n)
    df1 = _run(x=x, y=y, g1=g1, g2=g2)
    df10 = _run(x=x * 10.0, y=y * 10.0, g1=g1, g2=g2)
    for k in range(df1[SHELL_BIN_INDEX].max() + 1):
        r1 = df1[df1[SHELL_BIN_INDEX] == k].iloc[0]
        r10 = df10[df10[SHELL_BIN_INDEX] == k].iloc[0]
        assert r1[COMPONENT_RADIAL_SHELL_STATUS] == r10[COMPONENT_RADIAL_SHELL_STATUS]
        if r1[COMPONENT_RADIAL_SHELL_STATUS] == ComponentRadialShellStatus.FINITE.value:
            assert r1[SHELL_SYMMETRIC] == pytest.approx(
                r10[SHELL_SYMMETRIC], rel=1e-9, abs=1e-9,
            )
    m1 = df1[MEDIAN_NN_SPACING].iloc[0]
    m10 = df10[MEDIAN_NN_SPACING].iloc[0]
    assert m10 == pytest.approx(m1 * 10.0, rel=1e-9)


def test_persistent_vs_short_range_curve_matched_at_rho_0():
    """Two deterministic identical-copies fields (rho_0 = 1 exactly) with
    different spatial correlation lengths yield distinguishable shell
    curves: the long-correlation field keeps positive correlation at
    the outer shell; the short-correlation field decays fast."""
    x, y = _make_grid(20, 20)
    rng = np.random.default_rng(2026_09_03)
    noise_a = rng.normal(size=(20, 20))
    noise_b = rng.normal(size=(20, 20))
    smooth_long = gaussian_filter(noise_a, sigma=5.0).ravel()
    smooth_short = gaussian_filter(noise_b, sigma=0.7).ravel()
    df_long = _run(x=x, y=y, g1=smooth_long, g2=smooth_long)
    df_short = _run(x=x, y=y, g1=smooth_short, g2=smooth_short)

    assert df_long[COMPONENT_RHO_0].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert df_short[COMPONENT_RHO_0].iloc[0] == pytest.approx(1.0, abs=1e-9)

    finite_long = df_long[
        df_long[COMPONENT_RADIAL_SHELL_STATUS]
        == ComponentRadialShellStatus.FINITE.value
    ]
    finite_short = df_short[
        df_short[COMPONENT_RADIAL_SHELL_STATUS]
        == ComponentRadialShellStatus.FINITE.value
    ]
    assert not finite_long.empty and not finite_short.empty
    last_bin = int(min(
        finite_long[SHELL_BIN_INDEX].max(),
        finite_short[SHELL_BIN_INDEX].max(),
    ))
    late_long = finite_long[finite_long[SHELL_BIN_INDEX] == last_bin][
        SHELL_SYMMETRIC
    ].iloc[0]
    late_short = finite_short[finite_short[SHELL_BIN_INDEX] == last_bin][
        SHELL_SYMMETRIC
    ].iloc[0]
    assert late_long > late_short + 0.3


def test_insufficient_anchor_support_is_per_shell():
    """Two spatially-separated clusters give shell 0 adequate support
    (adjacent within-cluster pairs) but the outermost shell has almost
    no eligible anchors -- inter-cluster distances are far outside the
    outer shell, so the fraction floor fails while the count / fraction
    diagnostics remain populated."""
    x = np.concatenate([np.arange(6, dtype=float),
                        100.0 + np.arange(6, dtype=float)])
    y = np.zeros_like(x)
    g1 = np.linspace(0.0, 1.0, x.size)
    g2 = np.linspace(0.0, 1.0, x.size)
    df = _run(x=x, y=y, g1=g1, g2=g2)
    row0 = df[df[SHELL_BIN_INDEX] == 0].iloc[0]
    assert row0[COMPONENT_RADIAL_SHELL_STATUS] == ComponentRadialShellStatus.FINITE.value
    assert row0[N_ANCHORS_PER_SHELL] == 12
    row3 = df[df[SHELL_BIN_INDEX] == 3].iloc[0]
    assert row3[COMPONENT_RADIAL_SHELL_STATUS] == (
        ComponentRadialShellStatus.INSUFFICIENT_ANCHOR_SUPPORT.value
    )
    assert np.isnan(row3[SHELL_SYMMETRIC])
    assert row3[N_ANCHORS_PER_SHELL] <= 4

    # Expression variance changes availability, not physical shell support.
    constant = _run(x=x, y=y, g1=np.ones_like(g1), g2=g2)
    assert set(constant[COMPONENT_RADIAL_SHELL_STATUS]) == {
        ComponentRadialShellStatus.ZERO_VARIANCE_GENE1.value
    }
    for column in (
        N_ANCHORS_PER_SHELL,
        FRACTION_ANCHORS_PER_SHELL,
        N_UNORDERED_PAIRS_PER_SHELL,
    ):
        np.testing.assert_allclose(constant[column], df[column])
