"""Parent-ROI tessellation correlation: ROI × replicate primary
builder + ROI-grain reducer.

Emits Pearson (or configured mode) pooled once per (ROI, replicate)
over canonically sorted sibling-component tile means, plus the
reduced-estimator permutation null on the ROI summary grain. Never
averages component correlations; sibling-component values are pooled
before the estimator, not after.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd

from tilescope.analyzers.parent_roi_common import (
    lookup_parent_support_row,
)
from tilescope.analyzers.parent_roi_outcomes import (
    ParentReplicateCorrelationStatus,
    ParentTessellationCorrelationStatus,
    parent_replicate_correlation_status_from_reason,
)
from tilescope.analyzers.parent_roi_tessellation_common import (
    TessellationPoolingIndex,
)
from tilescope.cols import (
    CORR_MEDIAN, CORR_Q25, CORR_Q75,
    CORRELATION_VALUE,
    CORR_P_ANALYTIC, CORR_P_ANALYTIC_CONDITIONAL, CORR_Q_ANALYTIC_CONDITIONAL,
    CORR_P_PERM_REDUCED, CORR_Q_PERM_REDUCED,
    ELIGIBLE_SPOT_COVERAGE_FRACTION,
    FDR_FAMILY_SIZE,
    G1_ID, G2_ID,
    MODE,
    N_CONTRIBUTED_SPOTS, N_CONTRIBUTING_COMPONENTS, N_CONTRIBUTING_REGIONS,
    N_ELIGIBLE_COMPONENTS, N_ELIGIBLE_REGIONS, N_ELIGIBLE_SPOTS,
    N_PERMUTATIONS_REDUCED, N_PLANNED_REPLICATES, N_VALID_PERMUTATIONS,
    N_SOURCE_SPOTS, N_TILES_CONTRIBUTED, N_TILES_CONTRIBUTED_MEDIAN,
    PARENT_REPLICATE_CORRELATION_STATUS,
    PARENT_TESSELLATION_CORRELATION_STATUS,
    PERM_SEED,
    REPLICATE_INDEX, ROI_ID, SAMPLE_ID, SAMPLE_SIZE,
    SOURCE_ROI_SPOT_COVERAGE_FRACTION,
)
from tilescope.enums import CorrelationMode
from tilescope.metrics.correlation import (
    UndefinedCorrelationError,
    compute_correlation_from_arrays,
)
from tilescope.stats.fdr import apply_bh_within_families
from tilescope.schema.helpers import natural_sort_key
from tilescope.schema.table_validation import (
    RoiReplicateCorrelationSchema,
    RoiTessellationCorrelationSchema,
    assert_columns_populated_iff,
)
from tilescope.stats.reduced_estimator_null import (
    ReducedEstimatorNullInvariantError,
    compute_synchronized_median_null_p_value,
)

if TYPE_CHECKING:
    from tilescope.analyzers.tessellation_analysis import SampleTessellationResult
    from tilescope.spatial.feature_projection import FeatureSpatialProjection

logger = logging.getLogger(__name__)


_MIN_TILES_FOR_CORRELATION = 3


_ROI_REPLICATE_CORR_COLS = list(RoiReplicateCorrelationSchema.columns)


def build_roi_replicate_correlations_df(
    *,
    pooling_index: TessellationPoolingIndex,
    support_status_df: pd.DataFrame,
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
    mode: str = CorrelationMode.PEARSON.value,
) -> pd.DataFrame:
    """ROI × replicate primary table for tessellation correlation:
    raw realization-level effect + support diagnostics only.

    Exactly one row per ``(sample_id, source_roi_id, replicate_index)``
    over the source ROIs in ``support_status_df`` × the planned
    replicates ``0..sample_tess_result.n_replicates_planned-1``.
    All-scattered samples emit rows too.

    Pearson is computed once per row over canonically pooled sibling-
    component tile means (natural-sorted component order), never as
    an average of component correlations.
    """
    typed_sample_id = pooling_index.sample_id
    caller_sample_id = str(sample_id) if sample_id else ""
    if caller_sample_id and caller_sample_id != typed_sample_id:
        raise ValueError(
            f"sample_id identity conflict: caller supplied "
            f"{caller_sample_id!r} but the typed tessellation sample_id is "
            f"{typed_sample_id!r}. Two identity authorities are not "
            "permitted; pass only one, or leave sample_id unset."
        )
    resolved_sample_id = typed_sample_id
    n_planned = pooling_index.n_replicates_planned
    all_parents = pooling_index.roi_ids

    rows: List[dict] = []
    for parent_roi in all_parents:
        support_row = lookup_parent_support_row(support_status_df, parent_roi)
        support = pooling_index.support_by_roi[parent_roi]
        if int(support_row["source_spots"]) != support.n_source_spots:
            raise ValueError(
                f"Source-support disagreement for ROI {parent_roi!r}."
            )
        n_source_spots = support.n_source_spots
        n_eligible_components_roi = support.n_eligible_components
        n_eligible_regions_roi = support.n_eligible_regions
        n_eligible_spots_roi = support.n_eligible_spots

        base_row = {
            SAMPLE_ID: resolved_sample_id,
            ROI_ID: str(parent_roi),
            G1_ID: str(g1_id),
            G2_ID: str(g2_id),
            MODE: mode,
            N_SOURCE_SPOTS: n_source_spots,
            N_ELIGIBLE_SPOTS: n_eligible_spots_roi,
            N_ELIGIBLE_COMPONENTS: n_eligible_components_roi,
            N_ELIGIBLE_REGIONS: n_eligible_regions_roi,
        }

        if n_eligible_components_roi == 0:
            source_cov = 0.0 if n_source_spots > 0 else float("nan")
            for t in range(n_planned):
                rows.append({
                    **base_row,
                    REPLICATE_INDEX: t,
                    PARENT_REPLICATE_CORRELATION_STATUS: (
                        ParentReplicateCorrelationStatus
                        .NO_TESSELLATION_ELIGIBLE_COMPONENT.value
                    ),
                    N_CONTRIBUTING_COMPONENTS: 0,
                    N_CONTRIBUTING_REGIONS: 0,
                    N_TILES_CONTRIBUTED: 0,
                    N_CONTRIBUTED_SPOTS: 0,
                    ELIGIBLE_SPOT_COVERAGE_FRACTION: float("nan"),
                    SOURCE_ROI_SPOT_COVERAGE_FRACTION: source_cov,
                    CORRELATION_VALUE: float("nan"),
                    SAMPLE_SIZE: pd.NA,
                    CORR_P_ANALYTIC_CONDITIONAL: float("nan"),
                    CORR_Q_ANALYTIC_CONDITIONAL: float("nan"),
                    FDR_FAMILY_SIZE: pd.NA,
                })
            continue

        for t in range(n_planned):
            pool = pooling_index.pool_by_roi_replicate[(parent_roi, t)]
            n_contributing_components_roi = pool.n_contributing_components
            n_contributing_regions_roi = pool.n_contributing_regions
            n_tiles_contributed_roi = pool.n_tiles
            n_contributed_spots_roi = pool.n_spots

            if n_contributed_spots_roi > n_eligible_spots_roi:
                raise ValueError(
                    f"Invariant violation for parent ROI {parent_roi!r} "
                    f"at replicate {t}: n_contributed_spots "
                    f"({n_contributed_spots_roi}) > n_eligible_spots "
                    f"({n_eligible_spots_roi}). Pooled tile assignments "
                    f"overshot the eligible source-row set — construction "
                    f"failure (leaked / double-counted assignment)."
                )

            eligible_cov = (
                n_contributed_spots_roi / n_eligible_spots_roi
                if n_eligible_spots_roi > 0 else float("nan")
            )
            source_cov = (
                n_contributed_spots_roi / n_source_spots
                if n_source_spots > 0 else float("nan")
            )

            row = {
                **base_row,
                REPLICATE_INDEX: t,
                N_CONTRIBUTING_COMPONENTS: n_contributing_components_roi,
                N_CONTRIBUTING_REGIONS: n_contributing_regions_roi,
                N_TILES_CONTRIBUTED: n_tiles_contributed_roi,
                N_CONTRIBUTED_SPOTS: n_contributed_spots_roi,
                ELIGIBLE_SPOT_COVERAGE_FRACTION: eligible_cov,
                SOURCE_ROI_SPOT_COVERAGE_FRACTION: source_cov,
                CORRELATION_VALUE: float("nan"),
                SAMPLE_SIZE: pd.NA,
                CORR_P_ANALYTIC_CONDITIONAL: float("nan"),
                CORR_Q_ANALYTIC_CONDITIONAL: float("nan"),
                FDR_FAMILY_SIZE: pd.NA,
            }

            if n_contributing_components_roi == 0:
                status = (
                    ParentReplicateCorrelationStatus.NO_CONTRIBUTING_COMPONENTS
                )
            elif (
                n_contributing_regions_roi < n_eligible_regions_roi
                or n_contributed_spots_roi < n_eligible_spots_roi
            ):
                status = (
                    ParentReplicateCorrelationStatus.INCOMPLETE_SUPPORT_COVERAGE
                )
            elif n_tiles_contributed_roi < _MIN_TILES_FOR_CORRELATION:
                status = ParentReplicateCorrelationStatus.INSUFFICIENT_TILES
            else:
                gene1_density, gene2_density = pooling_index.density_values(
                    pool,
                )
                try:
                    result = compute_correlation_from_arrays(
                        gene1_density,
                        gene2_density,
                        mode=mode,
                        n_perms=0,
                    )
                except UndefinedCorrelationError as e:
                    status = parent_replicate_correlation_status_from_reason(
                        e.reason,
                    )
                else:
                    corr = float(result["corr_obs"])
                    if math.isfinite(corr) and -1.0 <= corr <= 1.0:
                        status = ParentReplicateCorrelationStatus.FINITE
                        row[CORRELATION_VALUE] = corr
                        row[SAMPLE_SIZE] = int(result["sample_size"])
                        row[CORR_P_ANALYTIC_CONDITIONAL] = float(
                            result[CORR_P_ANALYTIC]
                        )
                    else:
                        status = (
                            ParentReplicateCorrelationStatus
                            .UNDEFINED_CORRELATION_OTHER
                        )

            row[PARENT_REPLICATE_CORRELATION_STATUS] = status.value
            rows.append(row)

    df = pd.DataFrame(rows, columns=_ROI_REPLICATE_CORR_COLS)
    if not df.empty:
        df[SAMPLE_SIZE] = df[SAMPLE_SIZE].astype("Int64")
        df = _apply_replicate_correlation_fdr(df)
    df = RoiReplicateCorrelationSchema.validate(df, lazy=True)
    _enforce_finite_null_coupling(df)
    return df


_REPLICATE_FDR_GROUP_KEYS: tuple = (
    SAMPLE_ID, REPLICATE_INDEX, G1_ID, G2_ID, MODE,
)


def _apply_replicate_correlation_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """BH q per replicate; family keyed by (sample_id, replicate_index,
    g1_id, g2_id, mode). Q values are conditional diagnostics; never
    aggregate across replicates."""
    finite_mask = df[PARENT_REPLICATE_CORRELATION_STATUS] == (
        ParentReplicateCorrelationStatus.FINITE.value
    )
    return apply_bh_within_families(
        df,
        family_keys=_REPLICATE_FDR_GROUP_KEYS,
        p_to_q_columns=((CORR_P_ANALYTIC_CONDITIONAL, CORR_Q_ANALYTIC_CONDITIONAL),),
        populated_mask=finite_mask,
    )


def _enforce_finite_null_coupling(df: pd.DataFrame) -> None:
    """Enforce FINITE ↔ populated coupling for the finite-only numeric
    fields (correlation_value, sample_size, corr_p_analytic_conditional)."""
    finite_mask = df[PARENT_REPLICATE_CORRELATION_STATUS] == (
        ParentReplicateCorrelationStatus.FINITE.value
    )
    assert_columns_populated_iff(
        df,
        (CORRELATION_VALUE, SAMPLE_SIZE, CORR_P_ANALYTIC_CONDITIONAL),
        finite_mask,
        surface_name="roi_replicate_correlations_df",
        condition_label="parent_replicate_correlation_status == 'finite'",
    )


_TESS_CORR_SUMMARY_COLS = list(RoiTessellationCorrelationSchema.columns)

_REDUCED_NULL_FDR_GROUP_KEYS: tuple = (SAMPLE_ID, G1_ID, G2_ID, "mode")


def _derive_per_roi_null_seed(
    *, config_seed: int, sample_id: str, roi_id: str,
    g1_id: str, g2_id: str, mode: str,
) -> int:
    """Deterministic 32-bit seed for one ROI's reduced-estimator null."""
    digest = hashlib.blake2b(
        f"{int(config_seed)}|{sample_id}|{roi_id}|{g1_id}|{g2_id}|{mode}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


def _build_reduced_null_inputs_for_roi(
    *,
    sample_tess_result,
    projection_g1: "FeatureSpatialProjection",
    projection_g2: "FeatureSpatialProjection",
    component_ids: List[str],
    finite_replicate_indices: List[int],
    n_replicates_planned: int,
):
    """Assemble per-ROI inputs for the synchronized-median null.

    ``component_ids`` must already be in canonical (natural-sorted)
    order. Slots for replicates absent from the finite set are
    populated with empty per-component maps so the primitive's
    index-based access works uniformly.
    """
    source_g1 = np.asarray(projection_g1.source_values, dtype=float)
    source_g2 = np.asarray(projection_g2.source_values, dtype=float)
    source_g1_by_component: Dict[str, np.ndarray] = {}
    source_g2_by_component: Dict[str, np.ndarray] = {}
    for cid in component_ids:
        support = sample_tess_result.support_bank.components[cid]
        positions = np.asarray(support.source_row_ids, dtype=np.int64)
        source_g1_by_component[cid] = source_g1[positions]
        source_g2_by_component[cid] = source_g2[positions]
    tile_memberships: List[Dict[str, List[np.ndarray]]] = [
        {} for _ in range(n_replicates_planned)
    ]
    finite_set = set(int(t) for t in finite_replicate_indices)
    for cid in component_ids:
        comp = sample_tess_result.components[cid]
        for _region_id, per_rep in comp.tessellations.items():
            for t, tess in per_rep.items():
                if int(t) not in finite_set:
                    continue
                slot = tile_memberships[int(t)].setdefault(cid, [])
                for tile in tess.tiles:
                    slot.append(np.asarray(tile.point_indices))
    return source_g1_by_component, source_g2_by_component, tile_memberships


def compute_roi_tessellation_correlations(
    *,
    roi_replicate_correlations_df: pd.DataFrame,
    support_status_df: pd.DataFrame,
    sample_tess_result: "SampleTessellationResult | None" = None,
    projection_g1: "FeatureSpatialProjection | None" = None,
    projection_g2: "FeatureSpatialProjection | None" = None,
    n_permutations: int = 0,
    seed: int = 0,
    sample_id: str = "",
    g1_id: str,
    g2_id: str,
    mode: str = CorrelationMode.PEARSON.value,
) -> pd.DataFrame:
    """Reader-facing ROI summary. Pure reduction of the per-replicate
    primary table.

    - Correlation quantiles + n_valid_replicates over FINITE rows only.
    - Tile + coverage medians over ALL planned replicates (zero
      contribution is a real outcome, not a missing value).
    - When ``n_permutations > 0`` and the sample result plus both feature
      projections are supplied,
      computes the reduced-estimator permutation null (synchronized
      Gene-2 permutation reused across every FINITE replicate) and BH
      q per ``(sample_id, g1_id, g2_id, mode)`` family.
    """
    all_parents = sorted(
        support_status_df[ROI_ID].astype(str).unique(),
        key=natural_sort_key,
    )
    frame_parents = set(
        roi_replicate_correlations_df[ROI_ID].astype(str).unique()
    )
    unknown_parents = frame_parents - set(all_parents)
    if unknown_parents:
        raise ValueError(
            f"compute_roi_tessellation_correlations: per-replicate "
            f"frame carries ROI(s) {sorted(unknown_parents)!r} absent "
            f"from support_status_df — identity drift between the "
            "primary builder and the reducer."
        )
    frame_sample_ids = set(
        roi_replicate_correlations_df[SAMPLE_ID].astype(str).unique()
    ) if not roi_replicate_correlations_df.empty else set()
    if len(frame_sample_ids) > 1:
        raise ValueError(
            f"compute_roi_tessellation_correlations: per-replicate frame "
            f"carries multiple sample_id values {sorted(frame_sample_ids)!r}; "
            "one sample per call."
        )
    caller_sample_id = str(sample_id) if sample_id else ""
    frame_sample_id = next(iter(frame_sample_ids), "")
    if caller_sample_id and frame_sample_id and caller_sample_id != frame_sample_id:
        raise ValueError(
            f"sample_id identity conflict: caller supplied "
            f"{caller_sample_id!r} but per-replicate frame carries "
            f"{frame_sample_id!r}."
        )
    resolved_sample_id = frame_sample_id or caller_sample_id
    finite_status = ParentReplicateCorrelationStatus.FINITE.value

    if bool(projection_g1 is None) != bool(projection_g2 is None):
        raise ValueError(
            "compute_roi_tessellation_correlations: projection_g1 and "
            "projection_g2 must be supplied together."
        )
    if n_permutations > 0 and sample_tess_result is not None:
        if projection_g1 is None or projection_g2 is None:
            raise ValueError(
                "compute_roi_tessellation_correlations: reduced-estimator "
                "permutations require explicit projection_g1 + projection_g2; "
                "implicit gene1/gene2 source columns are not an authority."
            )
        bank_fp = sample_tess_result.support_bank.fingerprint
        if (
            projection_g1.bank_fingerprint != bank_fp
            or projection_g2.bank_fingerprint != bank_fp
        ):
            raise ValueError(
                "compute_roi_tessellation_correlations: feature projection "
                "fingerprint disagrees with the sample support bank."
            )

    rows: List[dict] = []
    for parent_roi in all_parents:
        support_row = lookup_parent_support_row(support_status_df, parent_roi)
        support_class = str(support_row["spatial_support_class"])
        n_source_spots = int(support_row["source_spots"])

        replicate_rows = roi_replicate_correlations_df.loc[
            roi_replicate_correlations_df[ROI_ID].astype(str) == str(parent_roi)
        ]
        n_planned = int(len(replicate_rows))

        base = {
            SAMPLE_ID: resolved_sample_id,
            ROI_ID: str(parent_roi),
            "spatial_support_class": support_class,
            G1_ID: str(g1_id),
            G2_ID: str(g2_id),
            "mode": mode,
            N_PLANNED_REPLICATES: n_planned,
            N_SOURCE_SPOTS: n_source_spots,
        }

        if n_planned == 0:
            raise ValueError(
                f"Invariant violation: roi_replicate_correlations_df has "
                f"no rows for parent ROI {parent_roi!r}; the primary "
                f"builder must emit one row per source ROI × planned "
                f"replicate."
            )
        actual_indices = sorted(replicate_rows[REPLICATE_INDEX].tolist())
        expected_indices = list(range(n_planned))
        if actual_indices != expected_indices:
            raise ValueError(
                f"Invariant violation for parent ROI {parent_roi!r}: "
                f"replicate_index set {actual_indices} does not equal "
                f"expected 0..{n_planned - 1} (duplicates or gaps)."
            )

        finite_mask = (
            replicate_rows[PARENT_REPLICATE_CORRELATION_STATUS] == finite_status
        )
        n_valid = int(finite_mask.sum())

        finite_replicate_indices: List[int] = []
        if n_valid > 0:
            finite_values = replicate_rows.loc[
                finite_mask, CORRELATION_VALUE
            ].astype(float).to_numpy()
            corr_median = float(np.median(finite_values))
            corr_q25 = float(np.quantile(finite_values, 0.25))
            corr_q75 = float(np.quantile(finite_values, 0.75))
            status = ParentTessellationCorrelationStatus.FINITE.value
            finite_replicate_indices = [
                int(t) for t in
                replicate_rows.loc[finite_mask, REPLICATE_INDEX].tolist()
            ]
        else:
            corr_median = corr_q25 = corr_q75 = float("nan")
            no_eligible_status = (
                ParentReplicateCorrelationStatus
                .NO_TESSELLATION_ELIGIBLE_COMPONENT.value
            )
            if (
                replicate_rows[PARENT_REPLICATE_CORRELATION_STATUS]
                == no_eligible_status
            ).all():
                status = (
                    ParentTessellationCorrelationStatus
                    .NO_TESSELLATION_ELIGIBLE_COMPONENT.value
                )
            else:
                status = (
                    ParentTessellationCorrelationStatus.NO_VALID_REPLICATES.value
                )

        tiles_median = float(np.median(
            replicate_rows[N_TILES_CONTRIBUTED].astype(float).to_numpy()
        ))
        eligible_cov_values = (
            replicate_rows[ELIGIBLE_SPOT_COVERAGE_FRACTION]
            .astype(float)
            .to_numpy()
        )
        source_cov_values = (
            replicate_rows[SOURCE_ROI_SPOT_COVERAGE_FRACTION]
            .astype(float)
            .to_numpy()
        )
        eligible_cov_median = (
            float(np.nanmedian(eligible_cov_values))
            if not np.all(np.isnan(eligible_cov_values)) else float("nan")
        )
        source_cov_median = (
            float(np.nanmedian(source_cov_values))
            if not np.all(np.isnan(source_cov_values)) else float("nan")
        )

        p_perm_reduced: float = float("nan")
        n_valid_perm: object = pd.NA
        per_roi_seed: object = pd.NA
        if (
            status == ParentTessellationCorrelationStatus.FINITE.value
            and n_permutations > 0
            and sample_tess_result is not None
        ):
            component_to_roi = dict(sample_tess_result.component_to_roi)
            comp_ids_for_roi = sorted(
                (cid for cid, r in component_to_roi.items() if str(r) == str(parent_roi)),
                key=natural_sort_key,
            )
            (
                source_g1_by_component,
                source_g2_by_component,
                tile_memberships,
            ) = _build_reduced_null_inputs_for_roi(
                sample_tess_result=sample_tess_result,
                projection_g1=projection_g1,
                projection_g2=projection_g2,
                component_ids=comp_ids_for_roi,
                finite_replicate_indices=finite_replicate_indices,
                n_replicates_planned=int(sample_tess_result.n_replicates_planned),
            )
            per_roi_seed = _derive_per_roi_null_seed(
                config_seed=int(seed),
                sample_id=str(resolved_sample_id),
                roi_id=str(parent_roi),
                g1_id=str(g1_id), g2_id=str(g2_id), mode=str(mode),
            )
            try:
                null_result = compute_synchronized_median_null_p_value(
                    source_g1_by_component=source_g1_by_component,
                    source_g2_by_component=source_g2_by_component,
                    tile_memberships=tile_memberships,
                    valid_replicate_indices=finite_replicate_indices,
                    t_obs=float(corr_median),
                    n_permutations=int(n_permutations),
                    seed=int(per_roi_seed),
                    mode=str(mode),
                )
            except ReducedEstimatorNullInvariantError as exc:
                # Eligibility invariance failed: keep the observed
                # statistic but emit NaN p/q/seed/n_valid; the row
                # drops from the BH family via the "iff p populated"
                # family-count rule.
                logger.warning(
                    "reduced-estimator null skipped for ROI %s "
                    "(eligibility invariance failed): %s",
                    parent_roi, exc,
                )
                p_perm_reduced = float("nan")
                n_valid_perm = pd.NA
                per_roi_seed = pd.NA
            else:
                p_perm_reduced = float(null_result.p_value)
                n_valid_perm = int(null_result.n_valid_permutations)

        rows.append({
            **base,
            PARENT_TESSELLATION_CORRELATION_STATUS: status,
            "n_valid_replicates": n_valid,
            CORR_MEDIAN: corr_median,
            CORR_Q25: corr_q25,
            CORR_Q75: corr_q75,
            CORR_P_PERM_REDUCED: p_perm_reduced,
            CORR_Q_PERM_REDUCED: float("nan"),
            N_PERMUTATIONS_REDUCED: int(n_permutations),
            N_VALID_PERMUTATIONS: n_valid_perm,
            PERM_SEED: per_roi_seed,
            N_TILES_CONTRIBUTED_MEDIAN: tiles_median,
            "eligible_spot_coverage_fraction_median": eligible_cov_median,
            "source_roi_spot_coverage_fraction_median": source_cov_median,
        })

    df = pd.DataFrame(rows, columns=_TESS_CORR_SUMMARY_COLS)
    if df.empty:
        return df
    df = _apply_reduced_null_fdr(df)
    df[N_PERMUTATIONS_REDUCED] = df[N_PERMUTATIONS_REDUCED].astype("Int64")
    df[N_VALID_PERMUTATIONS] = df[N_VALID_PERMUTATIONS].astype("Int64")
    df[PERM_SEED] = df[PERM_SEED].astype("Int64")
    df[FDR_FAMILY_SIZE] = df[FDR_FAMILY_SIZE].astype("Int64")
    return RoiTessellationCorrelationSchema.validate(df, lazy=True)


def _apply_reduced_null_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """BH q per (sample_id, g1_id, g2_id, mode) family over rows with a
    finite ``corr_p_perm_reduced``; ``fdr_family_size`` is populated on
    every row (finite or not)."""
    populated_mask = df[CORR_P_PERM_REDUCED].notna()
    return apply_bh_within_families(
        df,
        family_keys=_REDUCED_NULL_FDR_GROUP_KEYS,
        p_to_q_columns=((CORR_P_PERM_REDUCED, CORR_Q_PERM_REDUCED),),
        populated_mask=populated_mask,
    )
