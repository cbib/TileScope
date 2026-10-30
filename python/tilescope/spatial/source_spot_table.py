"""Typed sample-scope spot/cell substrate: spot id + coords + ROI.

Frozen, recursively-immutable typed record. Feature attachment
belongs to the projection layer, not this one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from tilescope.cols import COORD_X, COORD_Y, ROI_ID, SOURCE_ROW_ID


class FeatureNonfiniteRejectedError(ValueError):
    """Raised when a requested feature contains NaN or infinity at ingest."""


@dataclass(frozen=True)
class SourceSpotTable:
    """Frozen sample-scope substrate: spot_id + coords + ROI, input row order canonical.

    `x` / `y` arrays are defensively copied + read-only; `spot_id` values must be unique.
    """
    spot_id: Tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    roi_id: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spot_id, tuple):
            object.__setattr__(self, "spot_id", tuple(self.spot_id))
        if not isinstance(self.roi_id, tuple):
            object.__setattr__(self, "roi_id", tuple(self.roi_id))
        for name in ("x", "y"):
            # Defensive copy: `np.array(..., copy=True)` isolates from
            # any pre-existing writable alias, then freeze the copy.
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            if arr.ndim != 1:
                raise ValueError(
                    f"SourceSpotTable.{name} must be 1-D; got shape "
                    f"{arr.shape!r}."
                )
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        n = len(self.spot_id)
        if not (n == len(self.roi_id) == self.x.shape[0] == self.y.shape[0]):
            raise ValueError(
                "SourceSpotTable: spot_id / x / y / roi_id length mismatch "
                f"(n_spot_id={len(self.spot_id)}, n_roi_id={len(self.roi_id)}, "
                f"n_x={self.x.shape[0]}, n_y={self.y.shape[0]})."
            )
        if n and len(set(self.spot_id)) != n:
            dup = {sid for sid in self.spot_id if list(self.spot_id).count(sid) > 1}
            raise ValueError(
                f"SourceSpotTable: spot_id must be unique within the sample; "
                f"{len(dup)} duplicate id(s) present, e.g. {sorted(dup)[:5]!r}."
            )
        if n and not np.isfinite(self.x).all():
            raise ValueError("SourceSpotTable.x has non-finite values.")
        if n and not np.isfinite(self.y).all():
            raise ValueError("SourceSpotTable.y has non-finite values.")

    @property
    def n_spots(self) -> int:
        return len(self.spot_id)


def build_source_spot_table_from_frame(
    df: pd.DataFrame,
    *,
    x_col: str = COORD_X,
    y_col: str = COORD_Y,
    roi_col: str = ROI_ID,
    spot_id_col: str = SOURCE_ROW_ID,
) -> SourceSpotTable:
    """Build a `SourceSpotTable` from the sample's source frame.

    Row order in `df` is preserved as canonical bank order. Rejects
    NaN in identity + coordinate + ROI columns with a typed error.
    Feature columns (if present) are NOT copied: the bank does not
    carry gene values.
    """
    for col in (x_col, y_col, roi_col):
        if col not in df.columns:
            raise ValueError(
                f"build_source_spot_table_from_frame: missing column {col!r}."
            )
    if spot_id_col not in df.columns:
        # Legacy single-pair API: synthesise stable spot_id from row position.
        spot_id = tuple(str(i) for i in range(len(df)))
    else:
        raw_ids = df[spot_id_col].to_numpy()
        if pd.isna(raw_ids).any():
            raise ValueError(
                f"build_source_spot_table_from_frame: column {spot_id_col!r} "
                "contains NaN — spot identity must be defined for every row."
            )
        spot_id = tuple(str(v) for v in raw_ids)

    for col in (x_col, y_col, roi_col):
        if df[col].isna().any():
            raise ValueError(
                f"build_source_spot_table_from_frame: column {col!r} "
                "contains NaN — support-affecting columns must be defined "
                "for every row."
            )

    return SourceSpotTable(
        spot_id=spot_id,
        x=df[x_col].to_numpy(dtype=float),
        y=df[y_col].to_numpy(dtype=float),
        roi_id=tuple(str(v) for v in df[roi_col].to_numpy()),
    )


def reject_feature_nonfinite(
    df: pd.DataFrame, feature_columns: Tuple[str, ...],
) -> None:
    """Reject requested feature values that are NaN or infinite."""
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(
                f"reject_feature_nonfinite: requested feature column {col!r} "
                "missing from the source frame."
            )
        finite = np.isfinite(df[col].to_numpy(dtype=float))
        if not finite.all():
            n_bad = int((~finite).sum())
            raise FeatureNonfiniteRejectedError(
                f"Feature column {col!r} contains {n_bad} non-finite values; "
                "requested feature columns must contain only finite numbers."
            )


def hash_source_frame_rows(table: SourceSpotTable) -> str:
    """Row-order-preserving SHA-256 of `(spot_id, x, y, roi_id)` tuples.

    This IS canonical identity: a reordered frame produces a
    different hash. Float serialisation uses `repr()` to preserve
    full precision.
    """
    h = hashlib.sha256()
    for spot_id, x, y, roi_id in zip(
        table.spot_id, table.x, table.y, table.roi_id,
    ):
        h.update(f"{spot_id}|{x!r}|{y!r}|{roi_id}\n".encode("utf-8"))
    return h.hexdigest()
