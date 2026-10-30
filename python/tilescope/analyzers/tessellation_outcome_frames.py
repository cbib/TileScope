"""Canonical materialiser for the tessellation-stage public surface.

Built inside the sample analysis authority and consumed verbatim by
statistics and exporters. Never reconstruct these frames inside a reporter.

Frames produced:

    component_tessellation_eligibility_fields
                                         merged into component_geometry.csv
    component_domain_regions_fields      merged into component_domain_regions.csv
    component_region_replicate_status    new public audit table

Public identity prefix: `(sample_id, roi_id, component_id, ...)` on
every public frame. Column names live in `cols.py` (the DataFrame
naming authority); the tuples below are the frozen column sets +
order.

Multipart-tile telemetry is NOT surfaced here: it lives on the
internal tile table (consumed by consensus / enrichment / viz).
Consumers that need multipart counts derive them from `tile.polygon`
at their presentation boundary.

Dtypes are frozen per the DTYPE_MAP below and applied identically
to empty and populated frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from tilescope import cols
from tilescope.analyzers.tessellation_analysis import SampleTessellationResult


# ---------------------------------------------------------------------------
# Column contracts (LOCKED). Every column referenced here is defined
# in `cols.py` (the DataFrame naming authority). Tuples are immutable
# so downstream consumers cannot mutate the schema in place.
# ---------------------------------------------------------------------------


COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS: Tuple[str, ...] = (
    cols.SAMPLE_ID,
    cols.ROI_ID,
    cols.COMPONENT_ID,
    cols.N_TESSELLATION_ELIGIBLE_REGIONS,
    cols.N_SPOTS_IN_TESSELLATION_ELIGIBLE_REGIONS,
    cols.N_SPOTS_EXCLUDED_BY_TESSELLATION_ELIGIBILITY,
    cols.TESSELLATED_DOMAIN_AREA,
)

COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS: Tuple[str, ...] = (
    cols.SAMPLE_ID,
    cols.ROI_ID,
    cols.COMPONENT_ID,
    cols.DOMAIN_REGION_ID,
    cols.TESSELLATION_ELIGIBLE,
    cols.TESSELLATION_INELIGIBILITY_REASON,
)

COMPONENT_REGION_REPLICATE_STATUS_COLUMNS: Tuple[str, ...] = (
    cols.SAMPLE_ID,
    cols.ROI_ID,
    cols.COMPONENT_ID,
    cols.DOMAIN_REGION_ID,
    cols.REPLICATE_INDEX,
    cols.N_TILES,
    cols.REGION_STATUS,
)

# Composite keys per public-surface contract. Used for sorting +
# uniqueness assertion on every frame.
KEY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "component_tessellation_eligibility_fields": (
        cols.SAMPLE_ID, cols.COMPONENT_ID,
    ),
    "component_domain_regions_fields": (
        cols.SAMPLE_ID, cols.COMPONENT_ID, cols.DOMAIN_REGION_ID,
    ),
    "component_region_replicate_status": (
        cols.SAMPLE_ID, cols.COMPONENT_ID, cols.DOMAIN_REGION_ID,
        cols.REPLICATE_INDEX,
    ),
}


# Dtype map. Frozen contract. Nullable pandas types (`Int64`,
# `Float64`, `string`) admit NA where the
# domain permits it; classic `bool` is used for `tessellation_eligible`
# because eligibility is total (an NA is a contract failure, not a
# representable state). Required non-null columns are enforced by
# the key-uniqueness check below.
DTYPE_MAP: Dict[str, str] = {
    cols.SAMPLE_ID: "string",
    cols.ROI_ID: "string",
    cols.COMPONENT_ID: "string",
    cols.DOMAIN_REGION_ID: "Int64",
    cols.REPLICATE_INDEX: "Int64",
    cols.N_TESSELLATION_ELIGIBLE_REGIONS: "Int64",
    cols.N_SPOTS_IN_TESSELLATION_ELIGIBLE_REGIONS: "Int64",
    cols.N_SPOTS_EXCLUDED_BY_TESSELLATION_ELIGIBILITY: "Int64",
    cols.TESSELLATED_DOMAIN_AREA: "Float64",
    cols.TESSELLATION_ELIGIBLE: "bool",   # classic bool: no NA representable
    cols.TESSELLATION_INELIGIBILITY_REASON: "string",
    cols.N_TILES: "Int64",
    cols.REGION_STATUS: "string",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TessellationOutcomeTables:
    """Three DataFrames covering the public tessellation-stage surface.

    Frozen so consumers cannot rebind the frames. Column order,
    dtypes, and row order are guaranteed by the module-level column
    and dtype contracts above.
    Downstream exporters may drop or reorder for their own layout
    but must not add tessellation-stage columns from parallel
    derivations, and must not silently coerce dtypes away from the
    contract.
    """
    component_tessellation_eligibility_fields: pd.DataFrame
    component_domain_regions_fields: pd.DataFrame
    component_region_replicate_status: pd.DataFrame


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_tessellation_outcome_tables(
    sample_result: SampleTessellationResult,
) -> TessellationOutcomeTables:
    """Project the sample-wide primary records into the three public
    tables. Pure function of `sample_result`: no I/O, no config
    lookup, no external state.

    Ordering: each frame is sorted by its declared composite key.
    Uniqueness and null-freeness of key columns are asserted
    per frame so the public surface is deterministic even if a valid
    `SampleTessellationResult` were constructed outside
    `run_tessellation_stage`.

    """
    sample_id = sample_result.sample_id
    n_replicates_planned = sample_result.n_replicates_planned

    geom_rows: list[dict] = []
    region_rows: list[dict] = []
    status_rows: list[dict] = []

    for cid, comp_result in sample_result.components.items():
        # Public column name for the biological ROI foreign key is
        # `roi_id` regardless of the internal field name.
        roi_id = comp_result.parent_roi_id
        region_diagnostics = comp_result.region_diagnostics
        eligible_diagnostics = (
            comp_result.tessellation_eligible_region_diagnostics
        )
        n_tessellation_eligible_regions = len(eligible_diagnostics)
        n_spots_in_tessellation_eligible_regions = sum(
            diagnostic.n_component_spots
            for diagnostic in eligible_diagnostics
        )
        n_spots_excluded_by_tessellation_eligibility = sum(
            d.n_component_spots for d in region_diagnostics
            if not d.is_tessellation_eligible
        )
        tessellated_domain_area = sum(
            diagnostic.region_area for diagnostic in eligible_diagnostics
        )
        geom_rows.append({
            cols.SAMPLE_ID: sample_id,
            cols.ROI_ID: roi_id,
            cols.COMPONENT_ID: cid,
            cols.N_TESSELLATION_ELIGIBLE_REGIONS:
                n_tessellation_eligible_regions,
            cols.N_SPOTS_IN_TESSELLATION_ELIGIBLE_REGIONS:
                n_spots_in_tessellation_eligible_regions,
            cols.N_SPOTS_EXCLUDED_BY_TESSELLATION_ELIGIBILITY:
                n_spots_excluded_by_tessellation_eligibility,
            cols.TESSELLATED_DOMAIN_AREA: tessellated_domain_area,
        })

        for rd in comp_result.region_diagnostics:
            region_rows.append({
                cols.SAMPLE_ID: sample_id,
                cols.ROI_ID: roi_id,
                cols.COMPONENT_ID: cid,
                cols.DOMAIN_REGION_ID: rd.domain_region_id,
                cols.TESSELLATION_ELIGIBLE: rd.is_tessellation_eligible,
                cols.TESSELLATION_INELIGIBILITY_REASON: (
                    rd.ineligibility_reason.value
                    if rd.ineligibility_reason is not None else ""
                ),
            })

        for rr in comp_result.region_replicate_diagnostics:
            status_rows.append({
                cols.SAMPLE_ID: sample_id,
                cols.ROI_ID: roi_id,
                cols.COMPONENT_ID: cid,
                cols.DOMAIN_REGION_ID: rr.domain_region_id,
                cols.REPLICATE_INDEX: rr.replicate_index,
                cols.N_TILES: rr.n_tiles,
                cols.REGION_STATUS: rr.region_status.value,
            })

    return TessellationOutcomeTables(
        component_tessellation_eligibility_fields=_frame_finalise(
            geom_rows,
            COMPONENT_TESSELLATION_ELIGIBILITY_COLUMNS,
            key=KEY_COLUMNS["component_tessellation_eligibility_fields"],
            label="component_tessellation_eligibility_fields",
        ),
        component_domain_regions_fields=_frame_finalise(
            region_rows,
            COMPONENT_DOMAIN_REGIONS_TESSELLATION_COLUMNS,
            key=KEY_COLUMNS["component_domain_regions_fields"],
            label="component_domain_regions_fields",
        ),
        component_region_replicate_status=_frame_finalise(
            status_rows,
            COMPONENT_REGION_REPLICATE_STATUS_COLUMNS,
            key=KEY_COLUMNS["component_region_replicate_status"],
            label="component_region_replicate_status",
        ),
    )


def _frame_finalise(
    rows: list[dict],
    columns: Tuple[str, ...],
    *,
    key: Tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    """Build a DataFrame with the locked column order and dtypes,
    then sort by `key` and assert key columns are non-null and
    unique. Empty input still produces a zero-row frame with the
    right columns AND dtypes."""
    if rows:
        frame = pd.DataFrame(rows, columns=list(columns))
    else:
        frame = pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    # Apply the dtype map. `astype` on a nullable dtype accepts NA
    # values; on classic `bool` it does not (correct: eligibility
    # is total).
    frame = frame.astype({c: DTYPE_MAP[c] for c in columns})
    # Public composite-key columns are required to be non-null.
    for k in key:
        if frame[k].isna().any():
            n_bad = int(frame[k].isna().sum())
            raise ValueError(
                f"{label}: key column {k!r} has {n_bad} null value(s); "
                f"key columns must be non-null."
            )
    # Uniqueness check on the composite key.
    dupe_mask = frame.duplicated(subset=list(key), keep=False)
    if dupe_mask.any():
        dupe_keys = (
            frame.loc[dupe_mask, list(key)].drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{label}: duplicate composite key {list(key)}; "
            f"duplicated key values (first 10): {dupe_keys[:10]}"
        )
    return frame.sort_values(list(key), kind="mergesort").reset_index(drop=True)
