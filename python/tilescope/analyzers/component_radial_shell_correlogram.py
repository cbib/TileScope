"""Component-grain spot/cell radial shell cross-correlogram.

One long-form row per component × shell for every TileScope-retained
source component (i.e. every entry of ``SampleSupportBank.components``,
not every component of every ROI -- components excluded upstream by
the sample support authority never appear here). **Component-grain,
not ROI-grain**: shell edges are normalised by each component's median
nearest-neighbour spacing (dimensionless), and spatial persistence is
a component-local property. Pooling curves across a ROI's disjoint
components would change the estimand, so this descriptor is
deliberately component-grain and not rolled up to ROI. Every row
retains both ``roi_id`` and ``component_id`` for cross-join.

The estimator is spot/cell: values come from
``FeatureSpatialProjection.source_values`` indexed by each
``ComponentSupport.source_row_ids``; coordinates come from
``ComponentSupport.points``. The component polygon is not consulted --
membership is graph-defined by the bank -- so the descriptor does not
inherit the polygon-construction estimand.

The symmetric shell curve is the primary descriptor: it averages the
``A -> B`` and ``B -> A`` per-anchor products so the exported value is
symmetric in the two feature vectors. Directional curves are computed
internally to form the symmetric average but are not exported --
their asymmetry can reflect anchor eligibility, sampling density,
degree variation, or boundary effects rather than biological
directionality, and requires independent validation before use.

Per-shell availability: ``INSUFFICIENT_ANCHOR_SUPPORT`` when either
the anchor count or the anchor fraction floor fails. Component-level
statuses (``INSUFFICIENT_SPOTS`` / ``MEDIAN_NN_UNDEFINED`` /
``ZERO_VARIANCE_*``) are replicated across every shell row for the
component when the estimator cannot produce any shell value.

Ported from the sister validated implementation at
``analyses/descriptor_benchmark_v1/implementations/radial.py`` in the
GREENWALD-AC-OPC-01 cycle; the sister diagnostics ``quad_segs`` +
polygon-annulus coverage matrix + polygon-outside classification are
retired here because they have no exported QC role and this estimator
is not polygon-aware.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tilescope.analyzers.parent_roi_common import (
    resolve_sample_id_from_typed_record,
)
from tilescope.cols import (
    COMPONENT_ID, COMPONENT_RADIAL_SHELL_STATUS, COMPONENT_RHO_0,
    FRACTION_ANCHORS_PER_SHELL, G1_ID, G2_ID, MEDIAN_NN_SPACING,
    N_ANCHORS_PER_SHELL, N_UNORDERED_PAIRS_PER_SHELL,
    ROI_ID, SAMPLE_ID, SHELL_BIN_INDEX,
    SHELL_EDGE_LOWER_NORMALIZED, SHELL_EDGE_UPPER_NORMALIZED,
    SHELL_MIDPOINT_NORMALIZED, SHELL_SYMMETRIC,
)
from tilescope.settings.models import RadialShellPolicyModel
from tilescope.spatial.errors import BankFingerprintMismatchError
from tilescope.spatial.feature_projection import FeatureSpatialProjection
from tilescope.spatial.sample_support_bank import SampleSupportBank


class ComponentRadialShellStatus(str, Enum):
    """Per-shell availability for the component-grain radial cross-correlogram.

    Component-level statuses (``INSUFFICIENT_SPOTS`` /
    ``MEDIAN_NN_UNDEFINED`` / ``ZERO_VARIANCE_*``) are replicated
    across every shell row for the component when the estimator
    cannot produce any shell value. ``INSUFFICIENT_ANCHOR_SUPPORT``
    is the only per-shell fail: some shells may be ``FINITE`` while
    others are not.
    """
    FINITE = "finite"
    INSUFFICIENT_ANCHOR_SUPPORT = "insufficient_anchor_support"
    INSUFFICIENT_SPOTS = "insufficient_spots"
    MEDIAN_NN_UNDEFINED = "median_nn_undefined"
    ZERO_VARIANCE_GENE1 = "zero_variance_gene1"
    ZERO_VARIANCE_GENE2 = "zero_variance_gene2"
    ZERO_VARIANCE_GENE1_AND_GENE2 = "zero_variance_gene1_and_gene2"


_VARIATION_FLOOR = 1e-12
_PAIR_DISTANCE_BLOCK_SIZE = 131072


@dataclass(frozen=True)
class PackedShellNeighbours:
    """Sorted neighbours in flat int64 storage, with one slice per location."""
    indices: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        self.indices.flags.writeable = False
        self.offsets.flags.writeable = False

    def __getitem__(self, position: int) -> np.ndarray:
        return self.indices[self.offsets[position]:self.offsets[position + 1]]


def _pack_shell_neighbours(pairs: np.ndarray, n_points: int) -> PackedShellNeighbours:
    """Pack symmetric edges in the exact order of the previous sorted pair loop.

    That loop visits smaller-index neighbours first, then larger-index
    neighbours, both in ascending order. Sorting by anchor and neighbour
    reproduces this order without Python objects for individual neighbours.
    """
    anchors = np.concatenate((pairs[:, 0], pairs[:, 1]))
    neighbours = np.concatenate((pairs[:, 1], pairs[:, 0]))
    offsets = np.empty(n_points + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.bincount(anchors, minlength=n_points), out=offsets[1:])
    order = np.lexsort((neighbours, anchors))
    return PackedShellNeighbours(indices=neighbours[order], offsets=offsets)


def _pair_distances(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Keep coordinate temporaries bounded without changing per-pair arithmetic."""
    distances = np.empty(len(pairs), dtype=float)
    for start in range(0, len(pairs), _PAIR_DISTANCE_BLOCK_SIZE):
        stop = min(start + _PAIR_DISTANCE_BLOCK_SIZE, len(pairs))
        block = pairs[start:stop]
        distances[start:stop] = np.linalg.norm(
            coords[block[:, 0]] - coords[block[:, 1]], axis=1,
        )
    return distances


@dataclass(frozen=True)
class PreparedRadialShellGeometry:
    """Feature-independent shell neighbours for one source component."""
    n_points: int
    shell_edges_normalized: Tuple[float, ...]
    median_nn_spacing: float
    neighbours_by_shell: Tuple[PackedShellNeighbours, ...]
    n_anchors: Tuple[int, ...]
    anchor_fraction: Tuple[float, ...]
    n_pairs: Tuple[int, ...]


def prepare_component_radial_shell_geometry(
    *,
    support_bank: SampleSupportBank,
    radial_shell_policy: RadialShellPolicyModel,
) -> Mapping[str, PreparedRadialShellGeometry]:
    """Prepare radial shell geometry once for every retained component."""
    shell_edges = tuple(
        float(edge) for edge in radial_shell_policy.shell_edges_normalized
    )
    n_shells = len(shell_edges) - 1
    prepared = {
        str(component_id): _prepare_shell_geometry(
            coords=np.asarray(support.points, dtype=float),
            shell_edges=shell_edges,
            n_shells=n_shells,
        )
        for component_id, support in support_bank.components.items()
    }
    return MappingProxyType(prepared)


def build_component_radial_shell_correlogram_df(
    *,
    support_bank: SampleSupportBank,
    projection_g1: FeatureSpatialProjection,
    projection_g2: FeatureSpatialProjection,
    radial_shell_policy: RadialShellPolicyModel,
    g1_id: str,
    g2_id: str,
    prepared_geometry: Optional[
        Mapping[str, PreparedRadialShellGeometry]
    ] = None,
) -> pd.DataFrame:
    """One row per component × shell with the symmetric shell value +
    per-shell anchor QC + per-component rho_0 and median_nn_spacing.
    Both projections are validated against ``support_bank.fingerprint``
    so mismatched substrates cannot silently associate."""
    if projection_g1.bank_fingerprint != support_bank.fingerprint:
        raise BankFingerprintMismatchError(
            f"projection_g1 ({projection_g1.feature_name!r}) was built "
            "against a different SampleSupportBank."
        )
    if projection_g2.bank_fingerprint != support_bank.fingerprint:
        raise BankFingerprintMismatchError(
            f"projection_g2 ({projection_g2.feature_name!r}) was built "
            "against a different SampleSupportBank."
        )
    resolved_sample_id = resolve_sample_id_from_typed_record(
        str(support_bank.sample_id), "",
    )
    values_g1 = np.asarray(projection_g1.source_values, dtype=float)
    values_g2 = np.asarray(projection_g2.source_values, dtype=float)

    shell_edges = tuple(
        float(e) for e in radial_shell_policy.shell_edges_normalized
    )
    n_shells = len(shell_edges) - 1
    if n_shells < 1:
        raise ValueError(
            f"radial_shell.shell_edges_normalized must have at least 2 "
            f"entries; got {shell_edges!r}."
        )
    midpoints = np.array([
        0.5 * (shell_edges[k] + shell_edges[k + 1])
        for k in range(n_shells)
    ])
    min_anchors = int(radial_shell_policy.min_anchors_per_shell)
    min_fraction = float(radial_shell_policy.min_anchor_fraction)

    rows: List[dict] = []
    for component_id in sorted(support_bank.components.keys()):
        support = support_bank.components[component_id]
        geometry = None
        if prepared_geometry is not None:
            geometry = prepared_geometry.get(str(component_id))
            if geometry is None:
                raise ValueError(
                    f"Prepared radial geometry is missing component "
                    f"{component_id!r}."
                )
            if geometry.shell_edges_normalized != shell_edges:
                raise ValueError(
                    f"Prepared radial geometry for {component_id!r} uses "
                    "different normalized shell edges."
                )
        rows.extend(_component_rows(
            support=support,
            values_g1=values_g1, values_g2=values_g2,
            shell_edges=shell_edges, midpoints=midpoints,
            n_shells=n_shells,
            min_anchors=min_anchors, min_fraction=min_fraction,
            sample_id=resolved_sample_id,
            g1_id=str(g1_id), g2_id=str(g2_id),
            prepared_geometry=geometry,
        ))
    columns = [
        SAMPLE_ID, ROI_ID, COMPONENT_ID, G1_ID, G2_ID,
        SHELL_BIN_INDEX,
        SHELL_EDGE_LOWER_NORMALIZED, SHELL_EDGE_UPPER_NORMALIZED,
        SHELL_MIDPOINT_NORMALIZED, SHELL_SYMMETRIC,
        COMPONENT_RADIAL_SHELL_STATUS,
        N_ANCHORS_PER_SHELL, FRACTION_ANCHORS_PER_SHELL,
        N_UNORDERED_PAIRS_PER_SHELL,
        MEDIAN_NN_SPACING, COMPONENT_RHO_0,
    ]
    return pd.DataFrame(rows, columns=columns)


def _component_rows(
    *, support, values_g1: np.ndarray, values_g2: np.ndarray,
    shell_edges: Tuple[float, ...], midpoints: np.ndarray, n_shells: int,
    min_anchors: int, min_fraction: float,
    sample_id: str, g1_id: str, g2_id: str,
    prepared_geometry: Optional[PreparedRadialShellGeometry],
) -> List[dict]:
    component_id = str(support.component_id)
    parent_roi_id = str(support.parent_roi_id)
    src_ids = np.asarray(support.source_row_ids, dtype=np.int64)
    g1_comp = values_g1[src_ids]
    g2_comp = values_g2[src_ids]
    finite = np.isfinite(g1_comp) & np.isfinite(g2_comp)
    if prepared_geometry is not None and not finite.all():
        raise ValueError(
            f"Prepared radial geometry for component {component_id!r} "
            "requires finite feature values on every component spot."
        )
    coords = np.asarray(support.points, dtype=float)[finite]
    g1 = g1_comp[finite]
    g2 = g2_comp[finite]
    n = int(coords.shape[0])
    if n < 2:
        return _uniform_rows(
            sample_id=sample_id, roi_id=parent_roi_id,
            component_id=component_id, g1_id=g1_id, g2_id=g2_id,
            shell_edges=shell_edges, midpoints=midpoints, n_shells=n_shells,
            status=ComponentRadialShellStatus.INSUFFICIENT_SPOTS,
            median_nn=None, rho_0=None,
        )
    shell_geometry = prepared_geometry or _prepare_shell_geometry(
        coords=coords, shell_edges=shell_edges, n_shells=n_shells,
    )
    if shell_geometry.n_points != n:
        raise ValueError(
            f"Prepared radial geometry for component {component_id!r} has "
            f"{shell_geometry.n_points} points; expected {n}."
        )
    median_nn = float(shell_geometry.median_nn_spacing)
    if not np.isfinite(median_nn) or median_nn <= 0:
        return _uniform_rows(
            sample_id=sample_id, roi_id=parent_roi_id,
            component_id=component_id, g1_id=g1_id, g2_id=g2_id,
            shell_edges=shell_edges, midpoints=midpoints, n_shells=n_shells,
            status=ComponentRadialShellStatus.MEDIAN_NN_UNDEFINED,
            median_nn=None, rho_0=None,
        )
    g1_var = bool(np.std(g1) > _VARIATION_FLOOR)
    g2_var = bool(np.std(g2) > _VARIATION_FLOOR)
    if not g1_var and not g2_var:
        variance_status = (
            ComponentRadialShellStatus.ZERO_VARIANCE_GENE1_AND_GENE2
        )
    elif not g1_var:
        variance_status = ComponentRadialShellStatus.ZERO_VARIANCE_GENE1
    elif not g2_var:
        variance_status = ComponentRadialShellStatus.ZERO_VARIANCE_GENE2
    else:
        variance_status = None
    if variance_status is not None:
        return _uniform_rows(
            sample_id=sample_id, roi_id=parent_roi_id,
            component_id=component_id, g1_id=g1_id, g2_id=g2_id,
            shell_edges=shell_edges, midpoints=midpoints, n_shells=n_shells,
            status=variance_status,
            median_nn=float(median_nn), rho_0=None,
            shell_geometry=shell_geometry,
        )
    z_a = (g1 - np.mean(g1)) / np.std(g1)
    z_b = (g2 - np.mean(g2)) / np.std(g2)
    rho_0 = float(np.mean(z_a * z_b))

    per_shell = _evaluate_shells(
        z_a=z_a, z_b=z_b, shell_geometry=shell_geometry,
        n_shells=n_shells,
        min_anchors=min_anchors, min_fraction=min_fraction,
    )
    rows: List[dict] = []
    for k in range(n_shells):
        rows.append({
            SAMPLE_ID: sample_id, ROI_ID: parent_roi_id,
            COMPONENT_ID: component_id, G1_ID: g1_id, G2_ID: g2_id,
            SHELL_BIN_INDEX: k,
            SHELL_EDGE_LOWER_NORMALIZED: float(shell_edges[k]),
            SHELL_EDGE_UPPER_NORMALIZED: float(shell_edges[k + 1]),
            SHELL_MIDPOINT_NORMALIZED: float(midpoints[k]),
            SHELL_SYMMETRIC: per_shell["shell_symmetric"][k],
            COMPONENT_RADIAL_SHELL_STATUS: per_shell["status"][k].value,
            N_ANCHORS_PER_SHELL: int(per_shell["n_anchors"][k]),
            FRACTION_ANCHORS_PER_SHELL: float(per_shell["anchor_fraction"][k]),
            N_UNORDERED_PAIRS_PER_SHELL: int(per_shell["n_pairs"][k]),
            MEDIAN_NN_SPACING: float(median_nn),
            COMPONENT_RHO_0: rho_0,
        })
    return rows


def _prepare_shell_geometry(
    *, coords: np.ndarray, shell_edges: Tuple[float, ...], n_shells: int,
) -> PreparedRadialShellGeometry:
    """Prepare feature-independent shell neighbours once per component."""
    n = int(coords.shape[0])
    tree = cKDTree(coords)
    dd, _ = tree.query(coords, k=2)
    median_nn = float(np.median(dd[:, 1]))
    if not np.isfinite(median_nn) or median_nn <= 0:
        return PreparedRadialShellGeometry(
            n_points=n,
            shell_edges_normalized=shell_edges,
            median_nn_spacing=median_nn,
            neighbours_by_shell=(),
            n_anchors=(0,) * n_shells,
            anchor_fraction=(0.0,) * n_shells,
            n_pairs=(0,) * n_shells,
        )

    shell_edges_raw = np.asarray(shell_edges, dtype=float) * median_nn
    pairs_arr = tree.query_pairs(r=float(shell_edges_raw[-1]), output_type="ndarray")
    distances = _pair_distances(coords, pairs_arr)

    per_anchor_by_shell = []
    n_anchors = np.zeros(n_shells, dtype=int)
    anchor_fraction = np.zeros(n_shells, dtype=float)
    n_pairs = np.zeros(n_shells, dtype=int)
    for k in range(n_shells):
        mask = (
            (distances >= float(shell_edges_raw[k]))
            & (distances < float(shell_edges_raw[k + 1]))
        )
        pairs_k = pairs_arr[mask]
        n_pairs[k] = int(pairs_k.shape[0])
        per_anchor = _pack_shell_neighbours(pairs_k, n)
        per_anchor_by_shell.append(per_anchor)
        n_anchors[k] = np.count_nonzero(np.diff(per_anchor.offsets))
        anchor_fraction[k] = n_anchors[k] / n if n else 0.0
    return PreparedRadialShellGeometry(
        n_points=n,
        shell_edges_normalized=shell_edges,
        median_nn_spacing=median_nn,
        neighbours_by_shell=tuple(per_anchor_by_shell),
        n_anchors=tuple(int(value) for value in n_anchors),
        anchor_fraction=tuple(float(value) for value in anchor_fraction),
        n_pairs=tuple(int(value) for value in n_pairs),
    )


def _evaluate_shells(
    *, z_a: np.ndarray, z_b: np.ndarray,
    shell_geometry: PreparedRadialShellGeometry, n_shells: int,
    min_anchors: int, min_fraction: float,
) -> dict:
    """Per-shell symmetric estimator on the membership-observed
    (clipped-observed) profile. `has_neighbour[a, k]` is the
    per-shell eligibility mask; both count AND fraction floors must
    hold to emit `shell_symmetric[k]`."""
    shell_symmetric = np.full(n_shells, np.nan)
    n_anchors = shell_geometry.n_anchors
    anchor_fraction = shell_geometry.anchor_fraction
    status = [
        ComponentRadialShellStatus.INSUFFICIENT_ANCHOR_SUPPORT
    ] * n_shells

    for k in range(n_shells):
        per_anchor = shell_geometry.neighbours_by_shell[k]
        eligible = np.diff(per_anchor.offsets) > 0
        n_elig = int(n_anchors[k])
        if n_elig < min_anchors or anchor_fraction[k] < min_fraction:
            continue
        products_ab = np.empty(n_elig, dtype=float)
        products_ba = np.empty(n_elig, dtype=float)
        eligible_idx = np.where(eligible)[0]
        for local_i, a_idx in enumerate(eligible_idx):
            nb = per_anchor[a_idx]
            products_ab[local_i] = float(z_a[a_idx]) * float(np.mean(z_b[nb]))
            products_ba[local_i] = float(z_b[a_idx]) * float(np.mean(z_a[nb]))
        shell_symmetric[k] = 0.5 * (
            float(np.mean(products_ab)) + float(np.mean(products_ba))
        )
        status[k] = ComponentRadialShellStatus.FINITE
    return {
        "shell_symmetric": shell_symmetric,
        "n_anchors": n_anchors,
        "anchor_fraction": anchor_fraction,
        "n_pairs": shell_geometry.n_pairs,
        "status": status,
    }


def _uniform_rows(
    *, sample_id: str, roi_id: str, component_id: str,
    g1_id: str, g2_id: str,
    shell_edges: Tuple[float, ...], midpoints: np.ndarray, n_shells: int,
    status: ComponentRadialShellStatus,
    median_nn, rho_0,
    shell_geometry: Optional[PreparedRadialShellGeometry] = None,
) -> List[dict]:
    """Emit n_shells rows with NaN estimates carrying `status` uniformly.
    Used for component-level fails (INSUFFICIENT_SPOTS /
    MEDIAN_NN_UNDEFINED / ZERO_VARIANCE_*)."""
    rows: List[dict] = []
    for k in range(n_shells):
        rows.append({
            SAMPLE_ID: sample_id, ROI_ID: roi_id,
            COMPONENT_ID: component_id, G1_ID: g1_id, G2_ID: g2_id,
            SHELL_BIN_INDEX: k,
            SHELL_EDGE_LOWER_NORMALIZED: float(shell_edges[k]),
            SHELL_EDGE_UPPER_NORMALIZED: float(shell_edges[k + 1]),
            SHELL_MIDPOINT_NORMALIZED: float(midpoints[k]),
            SHELL_SYMMETRIC: np.nan,
            COMPONENT_RADIAL_SHELL_STATUS: status.value,
            N_ANCHORS_PER_SHELL: (
                int(shell_geometry.n_anchors[k])
                if shell_geometry is not None else 0
            ),
            FRACTION_ANCHORS_PER_SHELL: (
                float(shell_geometry.anchor_fraction[k])
                if shell_geometry is not None else 0.0
            ),
            N_UNORDERED_PAIRS_PER_SHELL: (
                int(shell_geometry.n_pairs[k])
                if shell_geometry is not None else 0
            ),
            MEDIAN_NN_SPACING: (
                float(median_nn) if median_nn is not None else np.nan
            ),
            COMPONENT_RHO_0: (
                float(rho_0) if rho_0 is not None else np.nan
            ),
        })
    return rows


__all__ = [
    "ComponentRadialShellStatus",
    "PreparedRadialShellGeometry",
    "build_component_radial_shell_correlogram_df",
    "prepare_component_radial_shell_geometry",
]
