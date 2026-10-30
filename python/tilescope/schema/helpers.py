"""Small coordinate + identity-formatting helpers used across the
pipeline.
"""

import numpy as np
import pandas as pd

from tilescope.cols import COORD_X, COORD_Y


def extract_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Canonical (N, 2) float coordinate array from ``COORD_X`` /
    ``COORD_Y``. Single source of truth for coordinate extraction."""
    return df[[COORD_X, COORD_Y]].to_numpy(dtype=float)


def format_roi_display(roi_id: str) -> str:
    """Format an ROI ID for human-readable display (numeric when
    possible, string otherwise). UI/plot/log use only; core pipeline
    keeps IDs as strings."""
    try:
        return str(int(roi_id))
    except (ValueError, TypeError):
        return str(roi_id)


def natural_sort_key(value: object) -> tuple:
    """Key for natural (integer-aware) ordering of IDs presented as
    strings. Integer-parsable IDs sort before non-integer IDs."""
    s = str(value)
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)
