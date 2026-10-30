"""Synchronized permutation null for the reduced tessellation-summary correlation ``T_obs = median_t(r_t)``.

One Gene-2 permutation per null draw, restricted to the ROI's eligible-spot/cell union across the frozen valid replicate set and reused across every replicate via retained tile point indices. If any per-component per-replicate union differs from the ROI-level union, :class:`ReducedEstimatorNullInvariantError` is raised. Draws yielding an invalid ``r_{t,b}`` on any valid replicate are rejected and redrawn, capped at ``attempt_cap_multiplier * n_permutations``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from tilescope.enums import CorrelationMode
from tilescope.component_data import require_integer_point_indices
from tilescope.metrics.correlation import (
    UndefinedCorrelationError,
    compute_correlation_from_arrays,
)


@dataclass(frozen=True)
class ReducedEstimatorNullResult:
    """Outcome of a synchronized-median-null p-value draw batch."""
    p_value: float
    n_valid_permutations: int


class ReducedEstimatorNullExhaustedError(ValueError):
    """Raised when the primitive cannot obtain the requested number of
    valid null draws within the configured attempt cap."""


class ReducedEstimatorNullInvariantError(ValueError):
    """Raised when the per-component eligible-spot-index set differs across replicates in the frozen valid set; the synchronized-Gene-2 null requires exactly one eligible pool per ROI x component."""


def compute_synchronized_median_null_p_value(
    *,
    source_g1_by_component: Dict[str, np.ndarray],
    source_g2_by_component: Dict[str, np.ndarray],
    tile_memberships: Sequence[Dict[str, List[np.ndarray]]],
    valid_replicate_indices: Sequence[int],
    t_obs: float,
    n_permutations: int,
    seed: int,
    mode: str = CorrelationMode.PEARSON.value,
    attempt_cap_multiplier: int = 10,
) -> ReducedEstimatorNullResult:
    """Two-tailed bounded p-value for ``T_obs = median_t(r_t)`` under a synchronized Gene-2 permutation null.

    Same ``seed`` + same inputs yields byte-identical ``p_value`` and ``n_valid_permutations``.
    """
    if n_permutations < 1:
        raise ValueError(
            f"compute_synchronized_median_null_p_value: n_permutations "
            f"must be >= 1, got {n_permutations}."
        )
    if attempt_cap_multiplier < 1:
        raise ValueError(
            f"compute_synchronized_median_null_p_value: "
            f"attempt_cap_multiplier must be >= 1, got "
            f"{attempt_cap_multiplier}."
        )
    valid_indices = list(int(t) for t in valid_replicate_indices)
    if not valid_indices:
        raise ValueError(
            "compute_synchronized_median_null_p_value: "
            "valid_replicate_indices is empty; T_obs must have at "
            "least one FINITE replicate."
        )

    # Canonical component order frozen from source_g1's keys; every
    # tile-membership frame is queried against this list.
    component_ids = list(source_g1_by_component.keys())
    if list(source_g2_by_component.keys()) != component_ids:
        raise ValueError(
            "compute_synchronized_median_null_p_value: "
            "source_g1_by_component and source_g2_by_component must "
            "expose the same component keys in the same order."
        )
    for cid in component_ids:
        g1 = np.asarray(source_g1_by_component[cid], dtype=float)
        g2 = np.asarray(source_g2_by_component[cid], dtype=float)
        if g1.shape != g2.shape or g1.ndim != 1:
            raise ValueError(
                f"compute_synchronized_median_null_p_value: component "
                f"{cid!r} source vectors mismatch: g1.shape={g1.shape}, "
                f"g2.shape={g2.shape}; both must be 1-D and equal length."
            )

    # Per-component eligible-spot set = union of tile point_indices across
    # every frozen valid replicate. INVARIANT: each per-replicate union must
    # equal this ROI-level union, else a spot assigned in one replicate but
    # not another would land in a tile after permutation that it never
    # populated in the observed data.
    eligible_indices_by_comp: Dict[str, np.ndarray] = {}
    per_replicate_union_by_comp: Dict[int, Dict[str, np.ndarray]] = {}
    for t in valid_indices:
        if t >= len(tile_memberships):
            raise ValueError(
                f"compute_synchronized_median_null_p_value: "
                f"tile_memberships has no entry for replicate index "
                f"{t} (len={len(tile_memberships)})."
            )
        replicate_map = tile_memberships[t]
        per_replicate_union_by_comp[t] = {}
        for cid in component_ids:
            if cid not in replicate_map:
                continue
            source_len = int(np.asarray(source_g1_by_component[cid]).shape[0])
            union_positions: List[np.ndarray] = []
            for point_indices in replicate_map[cid]:
                idx = require_integer_point_indices(point_indices, upper_bound=source_len)
                if idx.size == 0:
                    continue
                union_positions.append(idx)
            replicate_union = (
                np.unique(np.concatenate(union_positions))
                if union_positions else np.zeros(0, dtype=np.int64)
            )
            per_replicate_union_by_comp[t][cid] = replicate_union
    for cid in component_ids:
        all_unions = [
            per_replicate_union_by_comp[t][cid]
            for t in valid_indices
            if cid in per_replicate_union_by_comp[t]
        ]
        if all_unions:
            roi_union = np.unique(np.concatenate(all_unions))
        else:
            roi_union = np.zeros(0, dtype=np.int64)
        eligible_indices_by_comp[cid] = roi_union
    # Enforce the invariance so an ineligible-spot value cannot reach a
    # tile after permutation.
    for cid in component_ids:
        roi_union = eligible_indices_by_comp[cid]
        for t in valid_indices:
            replicate_union = per_replicate_union_by_comp[t].get(cid)
            if replicate_union is None:
                # Component missing on a valid replicate: eligible-set mismatch
                # (empty vs non-empty union).
                if roi_union.size:
                    raise ReducedEstimatorNullInvariantError(
                        f"compute_synchronized_median_null_p_value: "
                        f"component {cid!r} missing from replicate {t} "
                        f"but present in other frozen valid replicates "
                        f"(ROI-level eligible size {roi_union.size}); "
                        f"synchronized-Gene-2 null undefined."
                    )
                continue
            if replicate_union.shape != roi_union.shape or not np.array_equal(
                replicate_union, roi_union,
            ):
                only_in_replicate = np.setdiff1d(replicate_union, roi_union)
                only_in_union = np.setdiff1d(roi_union, replicate_union)
                raise ReducedEstimatorNullInvariantError(
                    f"compute_synchronized_median_null_p_value: "
                    f"component {cid!r} eligible-spot set differs at "
                    f"replicate {t} vs the ROI-level union "
                    f"(|replicate|={replicate_union.size}, "
                    f"|union|={roi_union.size}, "
                    f"|only_in_replicate|={only_in_replicate.size}, "
                    f"|only_in_union|={only_in_union.size}); "
                    f"synchronized-Gene-2 null undefined for this ROI."
                )

    # Build eligible-only per-component source vectors. Gene 2 is permuted;
    # Gene 1 is fixed but uses the same local remap for consistency.
    eligible_g2_by_comp: Dict[str, np.ndarray] = {
        cid: np.asarray(source_g2_by_component[cid], dtype=float)[
            eligible_indices_by_comp[cid]
        ]
        for cid in component_ids
    }
    # LUT mapping original source index to position in the eligible-only
    # slice, used to remap each tile's point_indices onto the permuted slice.
    local_position_by_comp: Dict[str, np.ndarray] = {}
    for cid in component_ids:
        source_len = int(np.asarray(source_g1_by_component[cid]).shape[0])
        # Sentinel -1 for ineligible spots (unreachable given the invariance).
        lut = -np.ones(source_len, dtype=np.int64)
        eligible = eligible_indices_by_comp[cid]
        lut[eligible] = np.arange(eligible.size, dtype=np.int64)
        local_position_by_comp[cid] = lut

    per_comp_sizes = [
        int(eligible_indices_by_comp[cid].size) for cid in component_ids
    ]
    offsets = np.cumsum([0] + per_comp_sizes)
    pooled_g2 = np.concatenate(
        [eligible_g2_by_comp[cid] for cid in component_ids]
    ) if component_ids else np.zeros(0, dtype=float)

    # Precompute per-replicate Gene-1 density vectors (canonical component /
    # tile order) and paired remapped tile point-index arrays into the
    # eligible-only permuted slice.
    per_replicate_g1_densities: Dict[int, np.ndarray] = {}
    per_replicate_tile_specs: Dict[int, List] = {}
    for t in valid_indices:
        replicate_map = tile_memberships[t]
        g1_pool: List[float] = []
        specs: List = []
        for cid in component_ids:
            if cid not in replicate_map:
                continue
            comp_g1 = np.asarray(
                source_g1_by_component[cid], dtype=float,
            )
            source_len = comp_g1.shape[0]
            lut = local_position_by_comp[cid]
            for point_indices in replicate_map[cid]:
                idx = require_integer_point_indices(point_indices, upper_bound=source_len)
                n_spots = int(idx.shape[0])
                if n_spots == 0:
                    continue
                g1_sum = float(comp_g1[idx].sum())
                remapped = lut[idx]
                # Sentinel -1 is unreachable given the invariance above; asserted
                # to catch a future contract break.
                if np.any(remapped < 0):
                    raise ValueError(
                        f"compute_synchronized_median_null_p_value: "
                        f"tile at replicate {t} component {cid!r} "
                        f"references ineligible spot/cell indices "
                        f"after eligibility restriction; invariance "
                        f"check should have caught this."
                    )
                g1_pool.append(g1_sum / n_spots)
                specs.append((cid, remapped, n_spots))
        per_replicate_g1_densities[t] = np.asarray(g1_pool, dtype=float)
        per_replicate_tile_specs[t] = specs

    abs_t_obs = float(abs(t_obs))
    rng = np.random.default_rng(int(seed))
    attempt_cap = int(attempt_cap_multiplier) * int(n_permutations)
    n_valid = 0
    n_invalid = 0
    n_more_extreme = 0
    n_attempts = 0
    while n_valid < n_permutations:
        if n_attempts >= attempt_cap:
            raise ReducedEstimatorNullExhaustedError(
                f"compute_synchronized_median_null_p_value: could not "
                f"obtain {n_permutations} valid null draws within "
                f"{attempt_cap} attempts "
                f"(n_permutations={n_permutations}, "
                f"n_attempts={n_attempts}, n_invalid_draws={n_invalid}, "
                f"n_replicates_in_frozen_set={len(valid_indices)})."
            )
        n_attempts += 1
        perm_g2 = rng.permutation(pooled_g2)
        perm_g2_by_comp: Dict[str, np.ndarray] = {
            cid: perm_g2[offsets[i]:offsets[i + 1]]
            for i, cid in enumerate(component_ids)
        }
        replicate_r_values: List[float] = []
        draw_valid = True
        for t in valid_indices:
            specs = per_replicate_tile_specs[t]
            if not specs:
                # A "frozen valid" replicate with zero tile entries is a
                # caller contract violation, not a null-draw failure.
                raise ValueError(
                    f"compute_synchronized_median_null_p_value: "
                    f"replicate {t} in valid_replicate_indices has "
                    f"zero tile-membership entries; the caller must "
                    f"only pass replicates that produced tile "
                    f"contributions."
                )
            g2_pool = np.empty(len(specs), dtype=float)
            for j, (cid, remapped, n_spots) in enumerate(specs):
                # `remapped` indexes into the eligible-only permuted slice.
                g2_pool[j] = float(perm_g2_by_comp[cid][remapped].sum()) / n_spots
            g1_pool = per_replicate_g1_densities[t]
            try:
                result = compute_correlation_from_arrays(
                    g1_pool, g2_pool, mode=mode, n_perms=0,
                )
            except UndefinedCorrelationError:
                draw_valid = False
                break
            except ValueError:
                # Too-few-observations or length mismatch on a frozen valid
                # replicate is a caller contract violation.
                raise
            r = float(result["corr_obs"])
            if not np.isfinite(r):
                draw_valid = False
                break
            replicate_r_values.append(r)
        if not draw_valid:
            n_invalid += 1
            continue
        t_b = float(np.median(replicate_r_values))
        if abs(t_b) >= abs_t_obs:
            n_more_extreme += 1
        n_valid += 1

    p_value = (1 + n_more_extreme) / (n_valid + 1)
    return ReducedEstimatorNullResult(
        p_value=float(p_value),
        n_valid_permutations=int(n_valid),
    )


__all__ = [
    "ReducedEstimatorNullResult",
    "ReducedEstimatorNullExhaustedError",
    "ReducedEstimatorNullInvariantError",
    "compute_synchronized_median_null_p_value",
]
