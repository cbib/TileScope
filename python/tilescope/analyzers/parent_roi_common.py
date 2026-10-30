"""Cross-support ROI-grain identity + lookup helpers used by both
tessellation-surface analyzers and the consensus + Moran analyzer.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from tilescope.cols import ROI_ID
from tilescope.schema.helpers import natural_sort_key


def group_components_by_parent(
    component_to_roi: Dict[str, str],
) -> Dict[str, List[str]]:
    """Group component IDs by their biological parent ROI, natural-sorted.

    Natural sort produces a canonical pooling order that is invariant
    across replicates and independent of caller dict insertion order.
    """
    grouped: Dict[str, List[str]] = {}
    for comp_id, raw_parent in component_to_roi.items():
        comp_id_s = str(comp_id)
        parent = str(raw_parent)
        grouped.setdefault(parent, []).append(comp_id_s)
    for parent in grouped:
        grouped[parent].sort(key=natural_sort_key)
    return grouped


def lookup_parent_support_row(
    support_status_df: pd.DataFrame,
    parent_roi_id: str,
) -> pd.Series:
    """Fetch the SINGLE support row for a parent ROI, or raise.

    Zero rows or duplicate rows are both invariant failures; the
    support classifier owns a one-to-one relationship with source
    parents.
    """
    m = support_status_df.loc[
        support_status_df[ROI_ID].astype(str) == str(parent_roi_id)
    ]
    if len(m) == 0:
        raise ValueError(
            f"Invariant violation: roi_support_status has no row for "
            f"parent ROI {parent_roi_id!r}. Every biological ROI with "
            "tessellation-eligible components must appear in the support "
            "classifier output."
        )
    if len(m) > 1:
        raise ValueError(
            f"Invariant violation: roi_support_status has {len(m)} rows "
            f"for parent ROI {parent_roi_id!r}; exactly one expected."
        )
    return m.iloc[0]


def resolve_sample_id_from_typed_record(
    typed_sample_id: str,
    caller_sample_id: str,
) -> str:
    """Return the sample_id to emit on ROI-grain rows.

    Identity authority is the typed sample-wide result record; the
    caller may pass an override, but a non-empty override that
    disagrees is a wiring failure (two identity authorities).
    """
    typed = str(typed_sample_id)
    override = str(caller_sample_id) if caller_sample_id else ""
    if override and override != typed:
        raise ValueError(
            f"sample_id identity conflict: caller supplied "
            f"{override!r} but the typed sample-wide record is "
            f"{typed!r}. Two identity authorities are not permitted; "
            "pass only one, or leave sample_id unset."
        )
    return typed
