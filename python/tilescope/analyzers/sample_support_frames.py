"""Canonical sample-support analysis frames.

Builds the typed ``spot_ledger``, ``component_geometry``, and
``component_domain_regions`` surfaces. Reporters serialize these frames
verbatim; they do not own or recompute them.
"""
import hashlib
import json


from typing import Dict

import numpy as np
import pandas as pd

from tilescope.cols import (
    ROI_ID, SOURCE_ROW_ID,
)


def build_spot_ledger(
    original_df: pd.DataFrame,
    post_split_df: pd.DataFrame,
    post_filter_df: pd.DataFrame,
    *,
    sample_tess_result,
    support_status_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extended spot ledger: one row per input spot.

    Records the pre-tessellation retention chain (splitter +
    component-min filter) and the post-tessellation aggregation-support
    state (parent-ROI scattered class, canonical domain-region
    membership, per-region tessellation eligibility). Every row matches
    exactly one of the six patterns in `docs/CSV_EXPORT_CONTRACT.md`,
    enforced at return.

    All joins go through `SOURCE_ROW_ID` value, never positional index;
    a caller with a non-default index would silently misalign every
    domain-region join. `support_status_df` and every retained
    non-scattered component_id in `sample_tess_result` are required;
    missing entries raise ValueError.
    """
    for label, df in (
        ("original_df", original_df),
        ("post_split_df", post_split_df),
        ("post_filter_df", post_filter_df),
    ):
        if SOURCE_ROW_ID not in df.columns:
            raise ValueError(
                f"build_spot_ledger: {label} missing required column "
                f"{SOURCE_ROW_ID!r}; every ledger input must carry the "
                f"integer source-row identity."
            )
    if ROI_ID not in original_df.columns:
        raise ValueError(
            f"build_spot_ledger: original_df missing required column "
            f"{ROI_ID!r}; ledger reads the parent-ROI label directly "
            f"from the source frame."
        )
    if support_status_df is None or support_status_df.empty:
        raise ValueError(
            "build_spot_ledger: support_status_df is required; the "
            "scattered-ROI gate output disambiguates the six-row "
            "truth table."
        )
    original_parents = set(original_df[ROI_ID].astype(str).unique())
    support_roi_series = support_status_df["roi_id"].astype(str)
    support_dupes = support_roi_series[support_roi_series.duplicated()]
    if len(support_dupes) > 0:
        raise ValueError(
            f"build_spot_ledger: support_status_df has duplicate "
            f"roi_id row(s): {sorted(set(support_dupes))[:10]!r}. "
            f"parent_scattered lookup requires unique parent rows."
        )
    scattered_by_roi = {
        str(r["roi_id"]): (str(r["spatial_support_class"]) == "scattered")
        for _, r in support_status_df.iterrows()
    }
    missing_support = original_parents - set(scattered_by_roi)
    if missing_support:
        raise ValueError(
            f"build_spot_ledger: support_status_df missing entries for "
            f"parent ROI(s) {sorted(missing_support)[:10]!r} present in "
            f"original_df; support classifier / analyzer contract "
            f"violation."
        )

    if "comp_id" in post_split_df.columns:
        raw_comp_id = post_split_df["comp_id"]
        if pd.api.types.is_numeric_dtype(raw_comp_id):
            # Numeric comp_ids: cast int -> str so float64 columns
            # (e.g. 10.0) don't astype to "10.0" and diverge from
            # downstream string comparisons.
            comp_id_series = raw_comp_id.astype("int64").astype(str)
        else:
            comp_id_series = raw_comp_id.astype(str)
        # Keyed by SOURCE_ROW_ID (integer), never positional index.
        split_src_ids = post_split_df[SOURCE_ROW_ID].astype(np.int64)
        comp_id_by_src: dict = dict(zip(split_src_ids, comp_id_series))
        comp_sizes: dict = comp_id_series.groupby(comp_id_series).size().to_dict()
    else:
        comp_id_by_src = {}
        comp_sizes = {}

    post_split_srcs = set(post_split_df[SOURCE_ROW_ID].astype(np.int64))
    post_filter_srcs = set(post_filter_df[SOURCE_ROW_ID].astype(np.int64))

    src_to_region_by_component: dict = {}
    eligibility_by_component: dict = {}
    tess_components: set = set()
    if sample_tess_result is not None:
        for cid, comp_result in sample_tess_result.components.items():
            cid_str = str(cid)
            tess_components.add(cid_str)
            src_to_region_by_component[cid_str] = dict(
                comp_result.domain_partition.canonical_ids_by_source_row
            )
            eligibility_by_component[cid_str] = {
                d.domain_region_id: (
                    d.is_tessellation_eligible,
                    (d.ineligibility_reason.value
                     if d.ineligibility_reason is not None else None),
                )
                for d in comp_result.region_diagnostics
            }

    # Every retained non-scattered component must appear in
    # sample_tess_result; silently emitting null domain_region_id would
    # collide with the "retained spot outside every region" pattern.
    filter_src_to_comp: dict = {}
    if "comp_id" in post_filter_df.columns:
        filter_src_series = post_filter_df[SOURCE_ROW_ID].astype(np.int64)
        raw_filter_comp = post_filter_df["comp_id"]
        if pd.api.types.is_numeric_dtype(raw_filter_comp):
            filter_comp_series = raw_filter_comp.astype("int64").astype(str)
        else:
            filter_comp_series = raw_filter_comp.astype(str)
        filter_src_to_comp = dict(zip(filter_src_series, filter_comp_series))
    retained_components: set = set()
    filter_src_to_roi: dict = {}
    if ROI_ID in post_filter_df.columns:
        _f_srcs = post_filter_df[SOURCE_ROW_ID].astype(np.int64)
        _f_rois = post_filter_df[ROI_ID].astype(str)
        filter_src_to_roi = dict(zip(_f_srcs, _f_rois))
    for src, cid in filter_src_to_comp.items():
        parent_roi_for_src = filter_src_to_roi.get(src, "")
        if scattered_by_roi.get(parent_roi_for_src, False):
            continue  # scattered parents have no spatial pipeline
        retained_components.add(cid)
    missing_from_tess = retained_components - tess_components
    if sample_tess_result is not None and missing_from_tess:
        raise ValueError(
            f"build_spot_ledger: sample_tess_result missing entries "
            f"for retained non-scattered component(s) "
            f"{sorted(missing_from_tess)[:10]!r}; tessellation stage / "
            f"analyzer contract violation."
        )

    src_ids_iter = original_df[SOURCE_ROW_ID].astype(np.int64)
    roi_ids_iter = original_df[ROI_ID].astype(str)
    rows: list = []
    for src_id, roi_id in zip(src_ids_iter, roi_ids_iter):
        src_id = int(src_id)
        parent_scattered = scattered_by_roi[roi_id]  # guaranteed present

        if src_id not in post_split_srcs:
            rows.append({
                "spot_index": src_id,
                "roi_id": roi_id,
                "component_id": None,
                "component_size": None,
                "component_status": "small_connected_component",
                "exclusion_reason": "below_split_min_component_spots",
                "parent_scattered": parent_scattered,
                "domain_region_id": None,
                "tessellation_eligible": None,
                "tessellation_ineligibility_reason": None,
            })
            continue

        comp_id = comp_id_by_src.get(src_id)
        comp_size = comp_sizes.get(comp_id)

        if src_id not in post_filter_srcs:
            rows.append({
                "spot_index": src_id,
                "roi_id": roi_id,
                "component_id": comp_id,
                "component_size": comp_size,
                "component_status": "small_connected_component",
                "exclusion_reason": "below_filter_min_component_spots",
                "parent_scattered": parent_scattered,
                "domain_region_id": None,
                "tessellation_eligible": None,
                "tessellation_ineligibility_reason": None,
            })
            continue

        # Retained-component branch: only supported (non-scattered)
        # parents have a sample_tess_result entry.
        cid_str = str(comp_id)
        domain_region_id = None
        tessellation_eligible = None
        tessellation_ineligibility_reason = None
        if not parent_scattered:
            src_map = src_to_region_by_component.get(cid_str)
            # Absence here is impossible after the completeness check
            # above; keep the defensive read.
            if src_map is not None:
                rid = src_map.get(src_id)
                if rid is not None:
                    domain_region_id = int(rid)
                    elig = eligibility_by_component.get(cid_str, {}).get(rid)
                    if elig is not None:
                        tessellation_eligible = bool(elig[0])
                        tessellation_ineligibility_reason = elig[1]

        rows.append({
            "spot_index": src_id,
            "roi_id": roi_id,
            "component_id": comp_id,
            "component_size": comp_size,
            "component_status": "retained_component",
            "exclusion_reason": None,
            "parent_scattered": parent_scattered,
            "domain_region_id": domain_region_id,
            "tessellation_eligible": tessellation_eligible,
            "tessellation_ineligibility_reason":
                tessellation_ineligibility_reason,
        })

    frame = pd.DataFrame(rows).sort_values("spot_index").reset_index(drop=True)
    # Nullable in-memory dtypes so pd.NA survives programmatic use.
    if not frame.empty:
        frame = frame.astype({
            "domain_region_id": "Int64",
            "component_size": "Int64",
            "parent_scattered": "boolean",
            "tessellation_eligible": "boolean",
        })
        _assert_truth_table_rows(frame)
    return frame


def _assert_truth_table_rows(frame: pd.DataFrame) -> None:
    """Assert every row matches exactly one of the six declared row
    patterns in `docs/CSV_EXPORT_CONTRACT.md`; raises ValueError on
    any mismatch."""
    def _pattern(row) -> str:
        cs = row["component_status"]
        er = row["exclusion_reason"]
        cid_null = pd.isna(row["component_id"])
        ps = bool(row["parent_scattered"])
        rid_null = pd.isna(row["domain_region_id"])
        te = row["tessellation_eligible"]
        te_null = pd.isna(te)
        reason_null = pd.isna(row["tessellation_ineligibility_reason"])

        if cs == "small_connected_component":
            if er == "below_split_min_component_spots":
                return "A_split_dropped" if (
                    cid_null and rid_null and te_null and reason_null
                ) else "invalid"
            if er == "below_filter_min_component_spots":
                return "B_filter_dropped" if (
                    not cid_null and rid_null and te_null and reason_null
                ) else "invalid"
            return "invalid"
        if cs != "retained_component":
            return "invalid"
        # retained_component branch (four sub-patterns).
        if ps and rid_null and te_null and reason_null:
            return "C_scattered_parent"
        if (not ps) and rid_null and te_null and reason_null:
            return "D_retained_outside_domain"
        if (not ps) and (not rid_null) and te is True and reason_null:
            return "E_eligible_region"
        if (not ps) and (not rid_null) and te is False and (not reason_null):
            return "F_ineligible_region"
        return "invalid"

    patterns = frame.apply(_pattern, axis=1)
    invalid_mask = patterns == "invalid"
    if invalid_mask.any():
        first_bad = frame.loc[invalid_mask].head(5).to_dict(orient="records")
        raise ValueError(
            f"build_spot_ledger: {int(invalid_mask.sum())} row(s) do not "
            f"match any of the six declared truth-table patterns in "
            f"docs/CSV_EXPORT_CONTRACT.md. First bad rows: {first_bad!r}"
        )


GEOMETRY_CONTRACT_NAME = "bounded_spot_support_owned_acquisition"
GEOMETRY_CONTRACT_VERSION = 1


def compute_geometry_contract_parameter_fingerprint(
    *,
    component_support_radius_factor: float,
    acquisition_support_radius_factor: float,
    ineligible_spot_exclusion_radius_factor: float,
) -> str:
    """Stable 8-byte blake2b fingerprint of the radius-factor triple.

    Version bumps encode changes to the mathematical composition of the
    component domain; radius-factor changes yield a new fingerprint at
    the same version. Consumers key provenance on the (name, version,
    fingerprint) triple.
    """
    payload = json.dumps(sorted([
        ("component_support_radius_factor",
         float(component_support_radius_factor)),
        ("acquisition_support_radius_factor",
         float(acquisition_support_radius_factor)),
        ("ineligible_spot_exclusion_radius_factor",
         float(ineligible_spot_exclusion_radius_factor)),
    ])).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


_GEOMETRY_AREA_ABS_TOL = 1e-9
_GEOMETRY_AREA_REL_TOL = 1e-9

_COMPONENT_GEOMETRY_COLUMNS: tuple = (
    "sample_id", "roi_id", "component_id",
    "geometry_wkt", "component_domain_area", "geometry_status",
    "geometry_contract_name", "geometry_contract_version",
    "geometry_contract_parameter_fingerprint",
    "sample_median_nn",
    "component_support_radius_factor",
    "acquisition_support_radius_factor",
    "ineligible_spot_exclusion_radius_factor",
    "n_all_domain_regions", "n_regions_with_spots",
    "n_component_spots", "n_spots_in_domain",
    "n_supported_spots_outside_domain",
    "ownership_excluded_area", "acquisition_excluded_area",
    "n_tessellation_eligible_regions",
    "n_spots_in_tessellation_eligible_regions",
    "n_spots_excluded_by_tessellation_eligibility",
    "tessellated_domain_area",
)

_COMPONENT_GEOMETRY_DTYPES: dict = {
    "sample_id": "string",
    "roi_id": "string",
    "component_id": "string",
    "geometry_wkt": "string",
    "component_domain_area": "Float64",
    "geometry_status": "string",
    "geometry_contract_name": "string",
    "geometry_contract_version": "Int64",
    "geometry_contract_parameter_fingerprint": "string",
    "sample_median_nn": "Float64",
    "component_support_radius_factor": "Float64",
    "acquisition_support_radius_factor": "Float64",
    "ineligible_spot_exclusion_radius_factor": "Float64",
    "n_all_domain_regions": "Int64",
    "n_regions_with_spots": "Int64",
    "n_component_spots": "Int64",
    "n_spots_in_domain": "Int64",
    "n_supported_spots_outside_domain": "Int64",
    "ownership_excluded_area": "Float64",
    "acquisition_excluded_area": "Float64",
    "n_tessellation_eligible_regions": "Int64",
    "n_spots_in_tessellation_eligible_regions": "Int64",
    "n_spots_excluded_by_tessellation_eligibility": "Int64",
    "tessellated_domain_area": "Float64",
}


def build_component_geometry(
    *,
    sample_id,
    component_polygons: Dict[str, object],
    component_to_roi: Dict[str, str],
    component_domain_diagnostics,
    sample_median_nn: float,
    component_support_radius_factor: float,
    acquisition_support_radius_factor: float,
    ineligible_spot_exclusion_radius_factor: float,
    tessellation_eligibility_frame,
) -> pd.DataFrame:
    """Canonical component_geometry.csv per
    docs/CSV_EXPORT_CONTRACT.md.

    One row per analysis component. Composite key
    (sample_id, component_id); component_id restarts per sample.
    Column set + dtypes are locked. Coherence invariants (raise
    ValueError): parent_roi_id agrees with component_to_roi;
    n_spots_in_domain <= n_component_spots; on ok geometries,
    polygon.area matches diag.final_domain_area within tolerance.
    Eligibility join is strict one-to-one; components without
    tessellation output leave the five eligibility fields null.
    """
    fingerprint = compute_geometry_contract_parameter_fingerprint(
        component_support_radius_factor=component_support_radius_factor,
        acquisition_support_radius_factor=acquisition_support_radius_factor,
        ineligible_spot_exclusion_radius_factor=(
            ineligible_spot_exclusion_radius_factor
        ),
    )
    sample_id_str = str(sample_id)
    diag_by_component = {
        str(d.component_id): d for d in component_domain_diagnostics
    }
    # Strict one-to-one eligibility lookup by (sample_id, component_id).
    # Duplicate rows raise; missing rows for a component in
    # component_polygons yield a null block; extra rows raise.
    elig_lookup: dict = {}
    if not tessellation_eligibility_frame.empty:
        emitted_components = {str(c) for c in component_polygons.keys()}
        for _, row in tessellation_eligibility_frame.iterrows():
            row_sample = str(row["sample_id"])
            row_cid = str(row["component_id"])
            if row_sample != sample_id_str:
                # Different sample: this builder emits rows for one
                # sample only.
                continue
            key = (row_sample, row_cid)
            if row_cid not in emitted_components:
                raise ValueError(
                    f"build_component_geometry: eligibility frame carries "
                    f"row for component_id={row_cid!r} that is absent from "
                    f"component_polygons; coherence failure."
                )
            if key in elig_lookup:
                raise ValueError(
                    f"build_component_geometry: eligibility frame has "
                    f"duplicate row for {key!r}; eligibility join is "
                    f"strict one-to-one."
                )
            elig_lookup[key] = row

    rows: list = []
    for comp_id, polygon in component_polygons.items():
        comp_id_str = str(comp_id)
        if comp_id_str not in component_to_roi:
            raise KeyError(
                f"build_component_geometry: component_to_roi missing "
                f"entry for component_id={comp_id_str!r}; every "
                f"geometry component must map to a parent ROI."
            )
        if comp_id_str not in diag_by_component:
            raise KeyError(
                f"build_component_geometry: component_domain_diagnostics "
                f"missing entry for component_id={comp_id_str!r}; every "
                f"geometry component must carry a ComponentDomainDiagnostic."
            )
        diag = diag_by_component[comp_id_str]
        roi_id = component_to_roi[comp_id_str]

        if str(diag.parent_roi_id) != str(roi_id):
            raise ValueError(
                f"build_component_geometry: identity drift for "
                f"component_id={comp_id_str!r}; component_to_roi says "
                f"parent={roi_id!r} but "
                f"ComponentDomainDiagnostic.parent_roi_id="
                f"{diag.parent_roi_id!r}."
            )
        if diag.n_spots_in_domain > diag.n_component_spots:
            raise ValueError(
                f"build_component_geometry: impossible accounting for "
                f"component_id={comp_id_str!r}: "
                f"n_spots_in_domain={diag.n_spots_in_domain} > "
                f"n_component_spots={diag.n_component_spots}."
            )
        n_outside_domain = int(
            diag.n_component_spots - diag.n_spots_in_domain
        )

        if polygon is None or polygon.is_empty:
            geom_wkt = ""
        else:
            geom_wkt = polygon.wkt
            # On ok geometries polygon.area must match diag's
            # authoritative final_domain_area.
            if diag.geometry_status == "ok":
                poly_area = float(polygon.area)
                diag_area = float(diag.final_domain_area)
                tol = _GEOMETRY_AREA_ABS_TOL + _GEOMETRY_AREA_REL_TOL * max(
                    abs(poly_area), abs(diag_area),
                )
                if abs(poly_area - diag_area) > tol:
                    raise ValueError(
                        f"build_component_geometry: area drift for "
                        f"component_id={comp_id_str!r}: polygon.area="
                        f"{poly_area!r} vs diagnostic.final_domain_area="
                        f"{diag_area!r} (tol={tol!r})."
                    )

        elig_row = elig_lookup.get((sample_id_str, comp_id_str))
        rows.append({
            "sample_id": sample_id_str,
            "roi_id": str(roi_id),
            "component_id": comp_id_str,
            "geometry_wkt": geom_wkt,
            "component_domain_area": (
                float(diag.final_domain_area)
                if diag.final_domain_area is not None else float("nan")
            ),
            "geometry_status": diag.geometry_status,
            "geometry_contract_name": GEOMETRY_CONTRACT_NAME,
            "geometry_contract_version": int(GEOMETRY_CONTRACT_VERSION),
            "geometry_contract_parameter_fingerprint": fingerprint,
            "sample_median_nn": float(sample_median_nn),
            "component_support_radius_factor":
                float(component_support_radius_factor),
            "acquisition_support_radius_factor":
                float(acquisition_support_radius_factor),
            "ineligible_spot_exclusion_radius_factor":
                float(ineligible_spot_exclusion_radius_factor),
            "n_all_domain_regions": int(diag.n_domain_regions),
            "n_regions_with_spots": int(diag.n_regions_with_spots),
            "n_component_spots": int(diag.n_component_spots),
            "n_spots_in_domain": int(diag.n_spots_in_domain),
            "n_supported_spots_outside_domain": n_outside_domain,
            "ownership_excluded_area": float(diag.ownership_excluded_area),
            "acquisition_excluded_area": float(diag.acquisition_excluded_area),
            "n_tessellation_eligible_regions": (
                int(elig_row["n_tessellation_eligible_regions"])
                if elig_row is not None else pd.NA
            ),
            "n_spots_in_tessellation_eligible_regions": (
                int(elig_row["n_spots_in_tessellation_eligible_regions"])
                if elig_row is not None else pd.NA
            ),
            "n_spots_excluded_by_tessellation_eligibility": (
                int(elig_row["n_spots_excluded_by_tessellation_eligibility"])
                if elig_row is not None else pd.NA
            ),
            "tessellated_domain_area": (
                float(elig_row["tessellated_domain_area"])
                if elig_row is not None else pd.NA
            ),
        })

    return _finalise_frame(
        rows, _COMPONENT_GEOMETRY_COLUMNS, _COMPONENT_GEOMETRY_DTYPES,
        key=("sample_id", "component_id"),
        label="component_geometry",
    )


def _finalise_frame(
    rows: list, columns: tuple, dtypes: dict,
    *, key: tuple, label: str,
) -> pd.DataFrame:
    """Build a DataFrame with locked column order + dtypes; sort by
    key; assert key columns are non-null and unique. Empty inputs
    produce zero-row frames with the right columns and dtypes."""
    if rows:
        frame = pd.DataFrame(rows, columns=list(columns))
    else:
        frame = pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    frame = frame.astype({c: dtypes[c] for c in columns})
    for k in key:
        if frame[k].isna().any():
            n_bad = int(frame[k].isna().sum())
            raise ValueError(
                f"{label}: key column {k!r} has {n_bad} null value(s); "
                f"key columns must be non-null."
            )
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


_COMPONENT_DOMAIN_REGIONS_COLUMNS: tuple = (
    "sample_id", "roi_id", "component_id", "domain_region_id",
    "area", "n_component_spots", "n_holes",
    "geometry_status", "source_row_fingerprint",
    "tessellation_eligible", "tessellation_ineligibility_reason",
)

_COMPONENT_DOMAIN_REGIONS_DTYPES: dict = {
    "sample_id": "string",
    "roi_id": "string",
    "component_id": "string",
    "domain_region_id": "Int64",
    "area": "Float64",
    "n_component_spots": "Int64",
    "n_holes": "Int64",
    "geometry_status": "string",
    "source_row_fingerprint": "string",
    "tessellation_eligible": "boolean",
    "tessellation_ineligibility_reason": "string",
}


def build_component_domain_regions(
    *,
    sample_id,
    region_diagnostics,
    component_to_roi: Dict[str, str],
    tessellation_domain_regions_frame,
) -> pd.DataFrame:
    """Canonical component_domain_regions.csv.

    One row per (component, canonical domain region). Composite key
    (sample_id, component_id, domain_region_id). Region geometry from
    `DomainRegionDiagnostic` merged with typed tessellation-eligibility
    fields. Every component_id in region_diagnostics must appear in
    component_to_roi (else KeyError); eligibility join is strict
    one-to-one; regions absent from the eligibility frame keep those
    two fields null. `source_row_fingerprint` is a stable 16-byte
    digest of the sorted SOURCE_ROW_IDs for cross-run diffability.
    """
    sample_id_str = str(sample_id)
    diag_keys = {
        (str(d.component_id), int(d.domain_region_id))
        for d in region_diagnostics
    }
    elig_lookup: dict = {}
    if not tessellation_domain_regions_frame.empty:
        for _, row in tessellation_domain_regions_frame.iterrows():
            row_sample = str(row["sample_id"])
            if row_sample != sample_id_str:
                continue
            row_cid = str(row["component_id"])
            row_rid = int(row["domain_region_id"])
            if (row_cid, row_rid) not in diag_keys:
                raise ValueError(
                    f"build_component_domain_regions: eligibility frame "
                    f"carries row for (component_id={row_cid!r}, "
                    f"domain_region_id={row_rid}) that is absent from "
                    f"region_diagnostics; coherence failure."
                )
            key = (row_sample, row_cid, row_rid)
            if key in elig_lookup:
                raise ValueError(
                    f"build_component_domain_regions: eligibility "
                    f"frame has duplicate row for {key!r}; join "
                    f"is strict one-to-one."
                )
            elig_lookup[key] = row

    rows: list = []
    for d in region_diagnostics:
        cid_str = str(d.component_id)
        if cid_str not in component_to_roi:
            raise KeyError(
                f"build_component_domain_regions: component_to_roi "
                f"missing entry for component_id={cid_str!r}; every "
                f"region diagnostic must map to a parent ROI."
            )
        rid = int(d.domain_region_id)
        elig_row = elig_lookup.get((sample_id_str, cid_str, rid))
        rows.append({
            "sample_id": sample_id_str,
            "roi_id": str(component_to_roi[cid_str]),
            "component_id": cid_str,
            "domain_region_id": rid,
            "area": float(d.area),
            "n_component_spots": int(d.n_component_spots),
            "n_holes": int(d.n_holes),
            "geometry_status": str(d.geometry_status),
            "source_row_fingerprint": str(d.source_row_fingerprint),
            "tessellation_eligible": (
                bool(elig_row["tessellation_eligible"])
                if elig_row is not None else pd.NA
            ),
            "tessellation_ineligibility_reason": (
                str(elig_row["tessellation_ineligibility_reason"])
                if elig_row is not None
                and elig_row["tessellation_ineligibility_reason"] != ""
                else pd.NA
            ),
        })

    return _finalise_frame(
        rows, _COMPONENT_DOMAIN_REGIONS_COLUMNS,
        _COMPONENT_DOMAIN_REGIONS_DTYPES,
        key=("sample_id", "component_id", "domain_region_id"),
        label="component_domain_regions",
    )
