"""
Single source of truth for sufficient statistics field mapping.

This is the only place that defines mapping/order for sufficient statistics.
Do not list these columns anywhere else.

Locks field order to SufficientStats dataclass structure using dataclasses.fields().
If someone adds/removes a field but forgets the map, contract tests will fail.
"""

from dataclasses import fields
from typing import Dict, List, Any
from tilescope.metrics.pair_stats_core import SufficientStats

from tilescope.cols import (
    N_SPOTS, SUM_G1, SUM_G2,
    N_G1_HIGH, N_G2_HIGH, N_BOTH_HIGH,
    N_G1_LOW, N_G2_LOW, N_BOTH_LOW,
)


# Mapping from dataclass field -> column name (n_obs handled separately per scale)
ATTR_TO_COL = {
    "sum_g1": SUM_G1,
    "sum_g2": SUM_G2,
    "n_g1_high": N_G1_HIGH,
    "n_g2_high": N_G2_HIGH,
    "n_both_high": N_BOTH_HIGH,
    "n_g1_low": N_G1_LOW,
    "n_g2_low": N_G2_LOW,
    "n_both_low": N_BOTH_LOW,
}


def sufficient_stat_columns(n_obs_col: str = N_SPOTS) -> List[str]:
    """Return ordered list of column names for sufficient statistics.

    Order locked to SufficientStats dataclass field order.
    n_obs_col: Column name for observation count (N_SPOTS for tiles, etc.)
    """
    cols = []
    for f in fields(SufficientStats):
        if f.name == "n_obs":
            cols.append(n_obs_col)
        else:
            cols.append(ATTR_TO_COL[f.name])
    return cols


def sufficient_stats_row(stats: "SufficientStats", n_obs_col: str = N_SPOTS) -> Dict[str, Any]:
    """Convert SufficientStats to dict with proper column names.

    Maps dataclass fields to column constants from cols.py.
    n_obs_col: Column name for observation count (N_SPOTS for tiles, etc.)
    """
    cols = sufficient_stat_columns(n_obs_col)
    vals = [getattr(stats, f.name) for f in fields(SufficientStats)]
    return dict(zip(cols, vals))
