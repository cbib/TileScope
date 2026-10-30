"""Scientific per-hex enrichment surface (analysis, not serialization).

Builds the canonical ``consensus_hexes`` frame: one row per
(component, hex). Per-hex enrichment ratios are ROI-context-derived:
each hex's target mass is normalised by its own spot mass and divided
by the ROI-level context rate from ``roi_consensus_enrichment_df``.
The shared :func:`compute_enrichment_ratios_from_counts` primitive
runs vectorized once per ROI over all its finite-mass hexes; per-hex
per-ratio statuses come directly from that call and are consumed
downstream by the ROI Moran analyzer.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from tilescope import cols
from tilescope.analyzers.consensus_frames import ConsensusAnalysisFrames
from tilescope.metrics.enrichment import (
    EnrichmentRatioStatus,
    compute_enrichment_ratios_from_counts,
)


CONSENSUS_HEXES_COLUMNS: Tuple[str, ...] = (
    cols.SAMPLE_ID,
    cols.ROI_ID,
    cols.COMPONENT_ID,
    cols.HEX_ID,
    cols.DOMAIN_REGION_ID,
    cols.LATTICE_COL,
    cols.LATTICE_ROW,
    cols.CENTER_X,
    cols.CENTER_Y,
    cols.HEX_GEOMETRY_WKT,
    cols.HEX_AREA,
    cols.N_TESSELLATIONS_COVERED,
    cols.OBS_EQUIV_SPOT_MASS,
    cols.OBS_EQUIV_G1_SUM_MASS,
    cols.OBS_EQUIV_G2_SUM_MASS,
    cols.OBS_EQUIV_G1_HIGH_MASS,
    cols.OBS_EQUIV_G2_HIGH_MASS,
    cols.OBS_EQUIV_BOTH_HIGH_MASS,
    cols.OBS_EQUIV_G1_LOW_MASS,
    cols.OBS_EQUIV_G2_LOW_MASS,
    cols.OBS_EQUIV_BOTH_LOW_MASS,
    cols.GENE1_DENSITY_ENSEMBLE,
    cols.GENE2_DENSITY_ENSEMBLE,
    cols.GENE1_DENSITY_STABILITY,
    cols.GENE2_DENSITY_STABILITY,
    cols.N_SPOTS_DIRECT,
    cols.SUM_G1_DIRECT,
    cols.SUM_G2_DIRECT,
    cols.GENE1_DENSITY_DIRECT,
    cols.GENE2_DENSITY_DIRECT,
    cols.PARENT_CONSENSUS_ENRICHMENT_STATUS,
    cols.HEX_ENRICHMENT_STATUS,
    cols.HEX_G1_ENRICHMENT,
    cols.HEX_G2_ENRICHMENT,
    cols.HEX_JOINT_HIGH_ENRICHMENT_FC,
    cols.HEX_JOINT_LOW_ENRICHMENT_FC,
    cols.HEX_G1_ENRICHMENT_STATUS,
    cols.HEX_G2_ENRICHMENT_STATUS,
    cols.HEX_JOINT_HIGH_ENRICHMENT_STATUS,
    cols.HEX_JOINT_LOW_ENRICHMENT_STATUS,
)


KEY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "consensus_hexes": (
        cols.SAMPLE_ID, cols.COMPONENT_ID, cols.HEX_ID,
    ),
}


DTYPE_MAP: Dict[str, str] = {
    cols.SAMPLE_ID: "string",
    cols.ROI_ID: "string",
    cols.COMPONENT_ID: "string",
    cols.HEX_ID: "Int64",
    cols.DOMAIN_REGION_ID: "Int64",
    cols.LATTICE_COL: "Int64",
    cols.LATTICE_ROW: "Int64",
    cols.CENTER_X: "Float64",
    cols.CENTER_Y: "Float64",
    cols.HEX_GEOMETRY_WKT: "string",
    cols.HEX_AREA: "Float64",
    cols.N_TESSELLATIONS_COVERED: "Int64",
    cols.OBS_EQUIV_SPOT_MASS: "Float64",
    cols.OBS_EQUIV_G1_SUM_MASS: "Float64",
    cols.OBS_EQUIV_G2_SUM_MASS: "Float64",
    cols.OBS_EQUIV_G1_HIGH_MASS: "Float64",
    cols.OBS_EQUIV_G2_HIGH_MASS: "Float64",
    cols.OBS_EQUIV_BOTH_HIGH_MASS: "Float64",
    cols.OBS_EQUIV_G1_LOW_MASS: "Float64",
    cols.OBS_EQUIV_G2_LOW_MASS: "Float64",
    cols.OBS_EQUIV_BOTH_LOW_MASS: "Float64",
    cols.GENE1_DENSITY_ENSEMBLE: "Float64",
    cols.GENE2_DENSITY_ENSEMBLE: "Float64",
    cols.GENE1_DENSITY_STABILITY: "Float64",
    cols.GENE2_DENSITY_STABILITY: "Float64",
    cols.N_SPOTS_DIRECT: "Int64",
    cols.SUM_G1_DIRECT: "Float64",
    cols.SUM_G2_DIRECT: "Float64",
    cols.GENE1_DENSITY_DIRECT: "Float64",
    cols.GENE2_DENSITY_DIRECT: "Float64",
    cols.PARENT_CONSENSUS_ENRICHMENT_STATUS: "string",
    cols.HEX_ENRICHMENT_STATUS: "string",
    cols.HEX_G1_ENRICHMENT: "Float64",
    cols.HEX_G2_ENRICHMENT: "Float64",
    cols.HEX_JOINT_HIGH_ENRICHMENT_FC: "Float64",
    cols.HEX_JOINT_LOW_ENRICHMENT_FC: "Float64",
    cols.HEX_G1_ENRICHMENT_STATUS: "string",
    cols.HEX_G2_ENRICHMENT_STATUS: "string",
    cols.HEX_JOINT_HIGH_ENRICHMENT_STATUS: "string",
    cols.HEX_JOINT_LOW_ENRICHMENT_STATUS: "string",
}


HEX_STATUS_FINITE = "finite"
HEX_STATUS_PARTIAL = "partial_ratio_availability"
HEX_STATUS_CONTEXT_PROPORTION_ZERO = "context_proportion_zero"
HEX_STATUS_ROI_UNAVAILABLE = "roi_enrichment_unavailable"
HEX_STATUS_NO_HEX_OBS = "no_hex_observations"

# Aggregate ROI-grain statuses that carry per-ratio numeric contract
# populated. Everything else is structurally unavailable.
_ROI_STATUS_FINITE = "finite"
_ROI_STATUS_CONTEXT_PROPORTION_ZERO = "context_proportion_zero"
_ROI_STATUSES_WITH_PER_RATIO = frozenset({
    _ROI_STATUS_FINITE, _ROI_STATUS_CONTEXT_PROPORTION_ZERO,
})

_ENSEMBLE_SURFACE = "consensus_ensemble"

# Mass provenance columns (target-side hex mass ↔ ROI-side context
# mass) for the four canonical ratios, in the same order as
# `cols.ENRICHMENT_RATIO_STATUS_PAIRS`. Derived jointly with the hex
# and ROI value/status pairings so the pairing has one authority.
_HEX_RATIO_MASS_PROVENANCE = (
    (cols.OBS_EQUIV_G1_HIGH_MASS, cols.CONTEXT_OBS_EQUIV_G1_HIGH_MASS),
    (cols.OBS_EQUIV_G2_HIGH_MASS, cols.CONTEXT_OBS_EQUIV_G2_HIGH_MASS),
    (cols.OBS_EQUIV_BOTH_HIGH_MASS, cols.CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS),
    (cols.OBS_EQUIV_BOTH_LOW_MASS, cols.CONTEXT_OBS_EQUIV_BOTH_LOW_MASS),
)
_RATIO_SPEC = tuple(
    (tgt_mass, ctx_mass, hex_val, hex_status, roi_val, roi_status)
    for (tgt_mass, ctx_mass), (hex_val, hex_status), (roi_val, roi_status)
    in zip(
        _HEX_RATIO_MASS_PROVENANCE,
        cols.HEX_ENRICHMENT_RATIO_STATUS_PAIRS,
        cols.ENRICHMENT_RATIO_STATUS_PAIRS,
    )
)

# Context baseline keys expected by the primitive. Populated from the
# ROI consensus enrichment row's mass columns.
_CONTEXT_BASELINE_MASS_COLS = (
    (cols.CONTEXT_OBS_EQUIV_G1_HIGH_MASS, "ctx_n_g1_high"),
    (cols.CONTEXT_OBS_EQUIV_G2_HIGH_MASS, "ctx_n_g2_high"),
    (cols.CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS, "ctx_n_both_high"),
    (cols.CONTEXT_OBS_EQUIV_G1_LOW_MASS, "ctx_n_g1_low"),
    (cols.CONTEXT_OBS_EQUIV_G2_LOW_MASS, "ctx_n_g2_low"),
    (cols.CONTEXT_OBS_EQUIV_BOTH_LOW_MASS, "ctx_n_both_low"),
)

_COHERENCE_RTOL = 1e-9
_COHERENCE_ATOL = 1e-12


def build_consensus_hexes_df(
    *,
    consensus_frames: ConsensusAnalysisFrames,
    roi_consensus_enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Decorate the consensus hex substrate with per-hex enrichment
    ratios + four per-ratio statuses + a derived aggregate hex status.

    The shared primitive runs once per ROI over its finite-mass hex
    rows; each numeric ratio column is populated iff its own
    ``hex_*_status == finite``. The aggregate ``hex_enrichment_status``
    is a projection over structural gates + the four per-ratio
    statuses, never a numerical authority.
    """
    sample_id = consensus_frames.sample_id
    hex_observations_df = consensus_frames.hex_observations
    roi_index = _index_roi_enrichment(roi_consensus_enrichment_df, sample_id)

    for roi_id in hex_observations_df[cols.ROI_ID].astype(str).unique():
        if roi_id not in roi_index:
            raise KeyError(
                "build_consensus_hexes_df: no "
                f"(sample_id={sample_id!r}, roi_id={roi_id!r}, "
                f"value_surface={_ENSEMBLE_SURFACE!r}) row on "
                "roi_consensus_enrichment_df; the ROI context surface must "
                "cover every component's parent ROI."
            )

    decorations = _empty_decoration_frame(len(hex_observations_df))
    obs = hex_observations_df.reset_index(drop=True)
    decorations.index = obs.index

    for roi_id, group in obs.groupby(obs[cols.ROI_ID].astype(str), sort=False):
        roi_row = roi_index[str(roi_id)]
        _fill_roi_decorations(decorations, group, roi_row)

    combined = pd.concat([obs, decorations], axis=1)
    frame = combined.reindex(columns=list(CONSENSUS_HEXES_COLUMNS))
    frame = frame.astype({c: DTYPE_MAP[c] for c in CONSENSUS_HEXES_COLUMNS})
    frame = frame.sort_values(
        list(KEY_COLUMNS["consensus_hexes"]), kind="mergesort",
    ).reset_index(drop=True)
    _enforce_per_ratio_coupling(frame)
    _enforce_aggregate_status_derivation(frame)
    _enforce_weighted_coherence(frame, roi_index)
    return frame


def _empty_decoration_frame(n_rows: int) -> pd.DataFrame:
    """Preallocate the ten decoration columns as NA:
    ``parent_consensus_enrichment_status`` (copied verbatim), the
    derived ``hex_enrichment_status``, four per-hex numeric ratios,
    and four per-hex per-ratio statuses."""
    return pd.DataFrame({
        cols.PARENT_CONSENSUS_ENRICHMENT_STATUS: [pd.NA] * n_rows,
        cols.HEX_ENRICHMENT_STATUS: [pd.NA] * n_rows,
        cols.HEX_G1_ENRICHMENT: [pd.NA] * n_rows,
        cols.HEX_G2_ENRICHMENT: [pd.NA] * n_rows,
        cols.HEX_JOINT_HIGH_ENRICHMENT_FC: [pd.NA] * n_rows,
        cols.HEX_JOINT_LOW_ENRICHMENT_FC: [pd.NA] * n_rows,
        cols.HEX_G1_ENRICHMENT_STATUS: [pd.NA] * n_rows,
        cols.HEX_G2_ENRICHMENT_STATUS: [pd.NA] * n_rows,
        cols.HEX_JOINT_HIGH_ENRICHMENT_STATUS: [pd.NA] * n_rows,
        cols.HEX_JOINT_LOW_ENRICHMENT_STATUS: [pd.NA] * n_rows,
    })


def _fill_roi_decorations(
    decorations: pd.DataFrame, group: pd.DataFrame, roi_row: dict,
) -> None:
    parent_status = str(roi_row[cols.PARENT_CONSENSUS_ENRICHMENT_STATUS])
    decorations.loc[group.index, cols.PARENT_CONSENSUS_ENRICHMENT_STATUS] = (
        parent_status
    )

    if parent_status not in _ROI_STATUSES_WITH_PER_RATIO:
        decorations.loc[group.index, cols.HEX_ENRICHMENT_STATUS] = (
            HEX_STATUS_ROI_UNAVAILABLE
        )
        return

    spot_mass = group[cols.OBS_EQUIV_SPOT_MASS].astype(float)
    no_obs_mask = spot_mass.isna() | (spot_mass == 0.0)
    decorations.loc[
        group.index[no_obs_mask], cols.HEX_ENRICHMENT_STATUS,
    ] = HEX_STATUS_NO_HEX_OBS
    active_idx = group.index[~no_obs_mask]
    if len(active_idx) == 0:
        return

    active = group.loc[active_idx]
    _validate_active_hex_masses(active)

    counts = pd.DataFrame({
        cols.N_G1_HIGH: active[cols.OBS_EQUIV_G1_HIGH_MASS].astype(float).to_numpy(),
        cols.N_G2_HIGH: active[cols.OBS_EQUIV_G2_HIGH_MASS].astype(float).to_numpy(),
        cols.N_BOTH_HIGH: active[cols.OBS_EQUIV_BOTH_HIGH_MASS].astype(float).to_numpy(),
        cols.N_G1_LOW: active[cols.OBS_EQUIV_G1_LOW_MASS].astype(float).to_numpy(),
        cols.N_G2_LOW: active[cols.OBS_EQUIV_G2_LOW_MASS].astype(float).to_numpy(),
        cols.N_BOTH_LOW: active[cols.OBS_EQUIV_BOTH_LOW_MASS].astype(float).to_numpy(),
    }, index=active_idx)
    denominators = pd.Series(
        active[cols.OBS_EQUIV_SPOT_MASS].astype(float).to_numpy(),
        index=active_idx,
    )
    context_baselines = _context_baselines_from_roi_row(roi_row)
    ratios = compute_enrichment_ratios_from_counts(
        counts=counts, denominators=denominators,
        context_baselines=context_baselines,
    )

    for (
        _tgt, _ctx,
        hex_val_col, hex_status_col,
        primitive_val_col, primitive_status_col,
    ) in _RATIO_SPEC:
        primitive_status = ratios[primitive_status_col].astype(str)
        decorations.loc[active_idx, hex_status_col] = primitive_status.values
        finite_mask = (
            primitive_status.values == EnrichmentRatioStatus.FINITE.value
        )
        finite_idx = active_idx[finite_mask]
        if len(finite_idx):
            decorations.loc[finite_idx, hex_val_col] = (
                ratios.loc[finite_idx, primitive_val_col].astype(float).values
            )

    per_ratio_status_frame = decorations.loc[
        active_idx, list(cols.HEX_ENRICHMENT_RATIO_STATUS_COLUMNS)
    ]
    finite_count = (
        per_ratio_status_frame == EnrichmentRatioStatus.FINITE.value
    ).sum(axis=1)
    aggregate = np.where(
        finite_count == 4, HEX_STATUS_FINITE,
        np.where(
            finite_count == 0, HEX_STATUS_CONTEXT_PROPORTION_ZERO,
            HEX_STATUS_PARTIAL,
        ),
    )
    decorations.loc[active_idx, cols.HEX_ENRICHMENT_STATUS] = aggregate


def _validate_active_hex_masses(active: pd.DataFrame) -> None:
    threshold_val = active[cols.OBS_EQUIV_BOTH_HIGH_MASS]
    if threshold_val.isna().any():
        offender = active.loc[threshold_val.isna(), cols.HEX_ID].iloc[0]
        raise ValueError(
            f"consensus_hexes: hex_id={int(offender)} has non-zero "
            "spot mass but missing threshold mass fields; "
            "run_consensus_stage must be invoked with thresholds."
        )


def _context_baselines_from_roi_row(roi_row: dict) -> dict:
    ctx_total = float(roi_row[cols.CONTEXT_OBS_EQUIV_SPOT_MASS])
    baselines = {"ctx_n_total": ctx_total}
    for mass_col, key in _CONTEXT_BASELINE_MASS_COLS:
        baselines[key] = float(roi_row[mass_col])
    return baselines


def _index_roi_enrichment(
    roi_enrichment_df: pd.DataFrame, sample_id: str,
) -> Dict[str, dict]:
    if roi_enrichment_df is None or roi_enrichment_df.empty:
        return {}
    if cols.VALUE_SURFACE not in roi_enrichment_df.columns:
        raise ValueError(
            f"consensus_hexes: roi_consensus_enrichment_df is missing the "
            f"{cols.VALUE_SURFACE!r} column; cannot select the ensemble "
            f"surface for per-hex enrichment derivation."
        )
    df = roi_enrichment_df[
        (roi_enrichment_df[cols.VALUE_SURFACE].astype(str) == _ENSEMBLE_SURFACE)
        & (roi_enrichment_df[cols.SAMPLE_ID].astype(str) == sample_id)
    ]
    dup_keys = df[df.duplicated(subset=[cols.ROI_ID], keep=False)][cols.ROI_ID]
    if len(dup_keys):
        raise ValueError(
            f"consensus_hexes: roi_consensus_enrichment_df has duplicate "
            f"(sample_id={sample_id!r}, roi_id, "
            f"value_surface={_ENSEMBLE_SURFACE!r}) rows; ROI enrichment "
            f"context must be single-valued per ROI. Duplicated roi_id(s): "
            f"{sorted(set(str(r) for r in dup_keys))!r}."
        )
    return {
        str(row[cols.ROI_ID]): dict(row)
        for _, row in df.iterrows()
    }


_VALID_HEX_RATIO_STATUS_VALUES = frozenset(
    s.value for s in EnrichmentRatioStatus
)
_HEX_STATUSES_WITH_PER_RATIO = frozenset({
    HEX_STATUS_FINITE, HEX_STATUS_PARTIAL, HEX_STATUS_CONTEXT_PROPORTION_ZERO,
})


def _enforce_per_ratio_coupling(frame: pd.DataFrame) -> None:
    """Each numeric ratio populated iff its own per-ratio status is
    finite. Active numerically-eligible hexes must carry a valid
    EnrichmentRatioStatus enum value on every per-ratio status column;
    NA/unknown status would slip past the aggregate derivation and
    the ratio populated-iff check simultaneously.
    """
    if frame.empty:
        return
    finite_value = EnrichmentRatioStatus.FINITE.value
    active_mask = frame[cols.HEX_ENRICHMENT_STATUS].astype(str).isin(
        _HEX_STATUSES_WITH_PER_RATIO,
    )
    for (
        _tgt, _ctx, val_col, status_col, _roi_val, _roi_status,
    ) in _RATIO_SPEC:
        finite_mask = frame[status_col].fillna("").astype(str) == finite_value
        populated_mask = frame[val_col].notna()
        if not finite_mask.equals(populated_mask):
            offender_rows = frame.loc[finite_mask ^ populated_mask].head(1)
            raise ValueError(
                f"consensus_hexes: {val_col!r} populated iff "
                f"{status_col!r} == 'finite' invariant violated. "
                f"Example row: {offender_rows.to_dict(orient='records')}"
            )
        # On active rows, every per-ratio status must be a valid enum
        # value (never NA / unknown string); on inactive rows it must
        # be NA.
        active_status_valid = frame.loc[active_mask, status_col].astype(str).isin(
            _VALID_HEX_RATIO_STATUS_VALUES,
        )
        if not bool(active_status_valid.all()):
            offender = frame.loc[
                active_mask & ~active_status_valid, [status_col, val_col],
            ].head(1).to_dict(orient="records")
            raise ValueError(
                f"consensus_hexes: {status_col!r} must be one of "
                f"{sorted(_VALID_HEX_RATIO_STATUS_VALUES)!r} on active "
                f"(numerically-eligible) rows; got {offender}"
            )
        inactive_status_populated = frame.loc[~active_mask, status_col].notna()
        if bool(inactive_status_populated.any()):
            offender = frame.loc[
                (~active_mask) & inactive_status_populated,
                [cols.HEX_ENRICHMENT_STATUS, status_col],
            ].head(1).to_dict(orient="records")
            raise ValueError(
                f"consensus_hexes: {status_col!r} must be NA on "
                f"structurally-unavailable rows; got {offender}"
            )


def _enforce_aggregate_status_derivation(frame: pd.DataFrame) -> None:
    """`hex_enrichment_status` is a derived projection over structural
    state + the four per-ratio statuses. Reconstruct + compare."""
    if frame.empty:
        return
    finite_value = EnrichmentRatioStatus.FINITE.value
    parent_status = frame[cols.PARENT_CONSENSUS_ENRICHMENT_STATUS].astype(str)
    parent_ok = parent_status.isin(_ROI_STATUSES_WITH_PER_RATIO)
    spot_mass = frame[cols.OBS_EQUIV_SPOT_MASS].astype(float)
    has_hex_obs = spot_mass.notna() & (spot_mass > 0.0)
    finite_count = pd.DataFrame({
        c: (frame[c].fillna("").astype(str) == finite_value)
        for c in cols.HEX_ENRICHMENT_RATIO_STATUS_COLUMNS
    }).sum(axis=1)

    expected = pd.Series(
        [HEX_STATUS_CONTEXT_PROPORTION_ZERO] * len(frame),
        index=frame.index, dtype="object",
    )
    expected[finite_count == 4] = HEX_STATUS_FINITE
    expected[(finite_count > 0) & (finite_count < 4)] = HEX_STATUS_PARTIAL
    expected[parent_ok & ~has_hex_obs] = HEX_STATUS_NO_HEX_OBS
    expected[~parent_ok] = HEX_STATUS_ROI_UNAVAILABLE

    observed = frame[cols.HEX_ENRICHMENT_STATUS].astype(str)
    mismatch = observed != expected
    if mismatch.any():
        offender = frame.loc[mismatch].head(1).to_dict(orient="records")
        raise ValueError(
            "consensus_hexes: hex_enrichment_status not derivable from "
            f"structural gates + per-ratio statuses. Example: {offender}"
        )


def _enforce_weighted_coherence(
    frame: pd.DataFrame, roi_index: Dict[str, dict],
) -> None:
    """For each ROI, the weighted mean of hexes whose per-ratio status
    is finite must equal that ratio's ROI-grain FC to floating-point
    tolerance. Checked per ratio (not aggregate)."""
    if frame.empty:
        return
    finite_value = EnrichmentRatioStatus.FINITE.value
    for (
        _tgt, _ctx, val_col, status_col,
        roi_val_col, roi_status_col,
    ) in _RATIO_SPEC:
        ratio_finite = frame[status_col].fillna("").astype(str) == finite_value
        for roi_id, group in frame.loc[ratio_finite].groupby(cols.ROI_ID):
            roi_row = roi_index.get(str(roi_id))
            if roi_row is None:
                continue
            roi_finite = (
                str(roi_row.get(roi_status_col, "")) == finite_value
            )
            if not roi_finite:
                raise ValueError(
                    f"consensus_hexes: roi_id={roi_id!r} has hexes with "
                    f"{status_col!r}='finite' but ROI-grain "
                    f"{roi_val_col!r} is not finite — indicates a stale "
                    "ROI enrichment row."
                )
            weights = group[cols.OBS_EQUIV_SPOT_MASS].astype(float).to_numpy()
            w_sum = float(weights.sum())
            if w_sum <= 0.0:
                continue
            roi_fc = float(roi_row[roi_val_col])
            hex_vals = group[val_col].astype(float).to_numpy()
            weighted_mean = float((weights * hex_vals).sum() / w_sum)
            if not np.isclose(
                weighted_mean, roi_fc,
                rtol=_COHERENCE_RTOL, atol=_COHERENCE_ATOL,
            ):
                raise ValueError(
                    "consensus_hexes: weighted-hex mean does not match ROI "
                    f"FC for roi_id={roi_id!r} column={val_col!r} — "
                    f"weighted_mean={weighted_mean!r} roi_fc={roi_fc!r} "
                    f"delta={weighted_mean - roi_fc!r}."
                )
