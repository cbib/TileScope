"""Shared color-norm resolvers for the visualization layer.

Public helper reused by density panels in ``expression.py`` and
``consensus.py`` so both renderers pick colormap + norm from the
same robust rule on the same union of values.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np


def resolve_expression_norm(values: np.ndarray, default_cmap: str):
    """Return ``(cmap_name, norm)`` from a robust percentile of ``values``.

    Robust range: 1st-99th percentile of finite values (resists a
    handful of elevated spots stretching the whole colormap). If the
    robust range crosses zero: diverging ``RdBu_r`` centred at zero
    via ``TwoSlopeNorm``. Otherwise: ``default_cmap`` with
    ``Normalize`` over the robust range.

    The norm is RESOLVED FROM the input; extreme values may saturate
    at the colorbar endpoints by design.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default_cmap, mcolors.Normalize(vmin=0.0, vmax=1.0)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
    lo, hi = float(lo), float(hi)
    if lo < 0.0 < hi:
        vabs = max(abs(lo), abs(hi))
        return "RdBu_r", mcolors.TwoSlopeNorm(
            vmin=-vabs, vcenter=0.0, vmax=vabs,
        )
    return default_cmap, mcolors.Normalize(vmin=lo, vmax=hi)
