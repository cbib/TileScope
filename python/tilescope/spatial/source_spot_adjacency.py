"""Canonical source-observation acquisition adjacency.

The user-facing technology selects one internally versioned graph rule:
Visium uses the established fixed-radius lattice reconstruction and Xenium
uses locally pruned Delaunay connectivity. Feature-independent and reused by
scattered-ROI
eligibility classification and by directional source-grain
estimators that need cross-ROI edge selection, interface
endpoints, or within-ROI hop distance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from tilescope.spatial.source_spot_table import SourceSpotTable
from tilescope.spatial.spacing import median_nn_distance


DEFAULT_SAMPLE_SUPPORT_FACTOR = 1.5
XENIUM_DELAUNAY_LOCAL_FACTOR = 2.5
XENIUM_DELAUNAY_GLOBAL_CAP_FACTOR = 4.0


UNREACHABLE_HOP = np.int64(-1)


@dataclass(frozen=True)
class SourceRoiComponent:
    """One connected component of a parent ROI's induced source graph."""
    roi_id: str
    positions: np.ndarray
    x_centroid: float

    def __post_init__(self) -> None:
        positions = np.array(self.positions, dtype=np.int64, copy=True)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("SourceRoiComponent.positions must be non-empty 1D.")
        positions.sort()
        positions.flags.writeable = False
        object.__setattr__(self, "positions", positions)

    @property
    def n_spots(self) -> int:
        return int(self.positions.size)


@dataclass(frozen=True)
class SourceSpotAdjacency:
    """Frozen sample-wide source-observation connectivity.

    `edges` are canonically ordered (`i < j`, lexicographic); `neighbors`
    are per-spot sorted tuples for deterministic traversal.
    """
    source_spot_table: SourceSpotTable
    method: Literal["fixed_radius", "locally_pruned_delaunay"]
    method_parameters: Tuple[Tuple[str, float], ...]
    sample_median_nn: float
    edges: np.ndarray
    neighbors: Tuple[Tuple[int, ...], ...]
    within_roi_components: Tuple[SourceRoiComponent, ...]

    def __post_init__(self) -> None:
        arr = np.array(self.edges, dtype=np.int64, copy=True)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"SourceSpotAdjacency.edges must be (E, 2); got {arr.shape!r}."
            )
        if arr.shape[0] and (arr[:, 0] >= arr[:, 1]).any():
            raise ValueError(
                "SourceSpotAdjacency.edges must be canonically ordered "
                "with i < j on every row."
            )
        arr.flags.writeable = False
        object.__setattr__(self, "edges", arr)
        if not isinstance(self.neighbors, tuple) or any(
            not isinstance(nb, tuple) for nb in self.neighbors
        ):
            object.__setattr__(
                self, "neighbors", tuple(tuple(nb) for nb in self.neighbors),
            )
        if len(self.neighbors) != self.source_spot_table.n_spots:
            raise ValueError(
                "SourceSpotAdjacency.neighbors length "
                f"({len(self.neighbors)}) must match n_spots "
                f"({self.source_spot_table.n_spots})."
            )
        if tuple(sorted(self.method_parameters)) != self.method_parameters:
            raise ValueError(
                "SourceSpotAdjacency.method_parameters must be sorted by key."
            )
        if not isinstance(self.within_roi_components, tuple):
            object.__setattr__(
                self, "within_roi_components",
                tuple(self.within_roi_components),
            )

    @property
    def n_spots(self) -> int:
        return self.source_spot_table.n_spots

    def cross_roi_edges(
        self, anchor_roi_id: str, responder_roi_id: str,
    ) -> np.ndarray:
        """Edges spanning (anchor, responder); col 0 anchor idx, col 1 responder idx.

        Rows are sorted lexicographically by (anchor_idx, responder_idx).
        """
        if anchor_roi_id == responder_roi_id:
            raise ValueError(
                "cross_roi_edges requires distinct anchor and responder "
                f"ROIs; got {anchor_roi_id!r}."
            )
        if self.edges.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int64)
        roi = np.asarray(self.source_spot_table.roi_id)
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        ri = roi[i]
        rj = roi[j]
        forward = (ri == anchor_roi_id) & (rj == responder_roi_id)
        reverse = (ri == responder_roi_id) & (rj == anchor_roi_id)
        n_fwd = int(forward.sum())
        n_rev = int(reverse.sum())
        out = np.empty((n_fwd + n_rev, 2), dtype=np.int64)
        out[:n_fwd, 0] = i[forward]
        out[:n_fwd, 1] = j[forward]
        out[n_fwd:, 0] = j[reverse]
        out[n_fwd:, 1] = i[reverse]
        order = np.lexsort((out[:, 1], out[:, 0]))
        return out[order]

    def interface_endpoints(
        self, anchor_roi_id: str, responder_roi_id: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unique anchor-side and responder-side indices of the anchor↔responder interface."""
        cross = self.cross_roi_edges(anchor_roi_id, responder_roi_id)
        return np.unique(cross[:, 0]), np.unique(cross[:, 1])

    def hop_distance_within_roi(
        self, roi_id: str, seed_indices: Iterable[int],
    ) -> np.ndarray:
        """BFS hop distance on the ROI-induced subgraph.

        Returned length-`n_spots` int64 array carries `UNREACHABLE_HOP`
        (-1) for spots outside `roi_id` AND for in-ROI spots with no
        within-ROI path to any seed.
        """
        roi = np.asarray(self.source_spot_table.roi_id)
        in_roi = roi == roi_id
        seeds = np.asarray(list(seed_indices), dtype=np.int64)
        if seeds.size and ((seeds < 0) | (seeds >= self.n_spots)).any():
            raise IndexError(
                "hop_distance_within_roi: seed indices out of bounds for "
                f"n_spots={self.n_spots}."
            )
        if seeds.size and not in_roi[seeds].all():
            raise ValueError(
                "hop_distance_within_roi: every seed index must belong to "
                f"roi_id {roi_id!r}."
            )
        hop = np.full(self.n_spots, UNREACHABLE_HOP, dtype=np.int64)
        if seeds.size == 0:
            return hop
        hop[seeds] = 0
        queue: deque[int] = deque(int(s) for s in np.unique(seeds))
        while queue:
            u = queue.popleft()
            next_h = int(hop[u]) + 1
            for v in self.neighbors[u]:
                if not in_roi[v]:
                    continue
                if hop[v] != UNREACHABLE_HOP:
                    continue
                hop[v] = next_h
                queue.append(v)
        return hop


def _build_within_roi_components(
    source_spot_table: SourceSpotTable,
    neighbors: Tuple[Tuple[int, ...], ...],
) -> Tuple[SourceRoiComponent, ...]:
    """Traverse each ROI-induced graph once and retain all components."""
    roi_labels = np.asarray(source_spot_table.roi_id)
    components: list[SourceRoiComponent] = []
    for roi_id in sorted(np.unique(roi_labels)):
        positions = np.flatnonzero(roi_labels == roi_id)
        in_roi = set(int(position) for position in positions)
        visited: set[int] = set()
        roi_components: list[SourceRoiComponent] = []
        for start in positions:
            start_i = int(start)
            if start_i in visited:
                continue
            stack = [start_i]
            members: list[int] = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                members.append(node)
                stack.extend(
                    neighbor for neighbor in neighbors[node]
                    if neighbor in in_roi and neighbor not in visited
                )
            member_positions = np.asarray(sorted(members), dtype=np.int64)
            roi_components.append(SourceRoiComponent(
                roi_id=str(roi_id),
                positions=member_positions,
                x_centroid=float(source_spot_table.x[member_positions].mean()),
            ))
        roi_components.sort(key=lambda item: (-item.n_spots, item.x_centroid))
        components.extend(roi_components)
    return tuple(components)


def build_locally_pruned_delaunay_edges(
    coordinates: np.ndarray,
    *,
    local_factor: float = XENIUM_DELAUNAY_LOCAL_FACTOR,
    global_cap_factor: float = XENIUM_DELAUNAY_GLOBAL_CAP_FACTOR,
    reference_median_nn: float | None = None,
) -> np.ndarray:
    """Return canonical locally pruned Delaunay edges.

    Retain an edge only when its
    length is at most ``local_factor × max(nn_i, nn_j)`` and at most
    ``global_cap_factor × reference_median_nn``.  The local condition adapts
    to heterogeneous cell density; the global cap blocks macroscopic bridges.
    Collinear inputs use the one-dimensional Delaunay analogue (consecutive
    points along their principal axis).
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (n, 2); "
            f"got {coords.shape!r}."
        )
    if not np.isfinite(coords).all():
        raise ValueError("coordinates must be finite.")
    if not np.isfinite(local_factor) or local_factor <= 0:
        raise ValueError("local_factor must be positive and finite.")
    if not np.isfinite(global_cap_factor) or global_cap_factor <= 0:
        raise ValueError("global_cap_factor must be positive and finite.")
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    if len(np.unique(coords, axis=0)) != n:
        raise ValueError(
            "The Xenium Delaunay graph requires unique coordinates."
        )

    if reference_median_nn is None:
        reference_median_nn = median_nn_distance(coords)
    if not np.isfinite(reference_median_nn) or reference_median_nn <= 0:
        raise ValueError(
            "reference_median_nn must be positive and finite; "
            f"got {reference_median_nn!r}."
        )

    if n == 2:
        candidates = np.array([[0, 1]], dtype=np.int64)
    else:
        try:
            simplices = Delaunay(coords).simplices
            candidates = np.vstack((
                simplices[:, [0, 1]],
                simplices[:, [1, 2]],
                simplices[:, [0, 2]],
            ))
            candidates = np.unique(np.sort(candidates, axis=1), axis=0)
        except QhullError:
            centered = coords - coords.mean(axis=0)
            if np.linalg.matrix_rank(centered) > 1:
                raise
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ axes[0]
            order = np.argsort(projection, kind="mergesort")
            candidates = np.column_stack([order[:-1], order[1:]])
            candidates = np.unique(np.sort(candidates, axis=1), axis=0)

    local_nn = cKDTree(coords).query(coords, k=2)[0][:, 1]
    lengths = np.linalg.norm(
        coords[candidates[:, 0]] - coords[candidates[:, 1]], axis=1,
    )
    local_limit = local_factor * np.maximum(
        local_nn[candidates[:, 0]], local_nn[candidates[:, 1]],
    )
    global_limit = global_cap_factor * float(reference_median_nn)
    kept = candidates[(lengths <= local_limit) & (lengths <= global_limit)]
    if not len(kept):
        return np.empty((0, 2), dtype=np.int64)
    order = np.lexsort((kept[:, 1], kept[:, 0]))
    return np.asarray(kept[order], dtype=np.int64)


def build_source_spot_adjacency(
    source_spot_table: SourceSpotTable,
    *,
    technology: Literal["visium", "xenium"] = "visium",
    radius_factor: float = DEFAULT_SAMPLE_SUPPORT_FACTOR,
    local_factor: float = XENIUM_DELAUNAY_LOCAL_FACTOR,
    global_cap_factor: float = XENIUM_DELAUNAY_GLOBAL_CAP_FACTOR,
) -> SourceSpotAdjacency:
    """Materialise the technology-selected source graph exactly once."""
    if technology not in ("visium", "xenium"):
        raise ValueError(f"Unsupported technology: {technology!r}")
    if not np.isfinite(radius_factor) or radius_factor <= 0:
        raise ValueError(
            f"radius_factor must be positive and finite; got {radius_factor!r}"
        )
    if not np.isfinite(local_factor) or local_factor <= 0:
        raise ValueError(
            f"local_factor must be positive and finite; got {local_factor!r}"
        )
    if not np.isfinite(global_cap_factor) or global_cap_factor <= 0:
        raise ValueError(
            "global_cap_factor must be positive and finite; got "
            f"{global_cap_factor!r}"
        )
    if technology == "visium":
        method = "fixed_radius"
        method_parameters = (("radius_factor", float(radius_factor)),)
    else:
        method = "locally_pruned_delaunay"
        method_parameters = tuple(sorted((
            ("global_cap_factor", float(global_cap_factor)),
            ("local_factor", float(local_factor)),
        )))

    n = source_spot_table.n_spots
    if n == 0:
        return SourceSpotAdjacency(
            source_spot_table=source_spot_table,
            method=method, method_parameters=method_parameters,
            sample_median_nn=float("nan"),
            edges=np.empty((0, 2), dtype=np.int64),
            neighbors=(),
            within_roi_components=(),
        )
    if n == 1:
        return SourceSpotAdjacency(
            source_spot_table=source_spot_table,
            method=method, method_parameters=method_parameters,
            sample_median_nn=float("nan"),
            edges=np.empty((0, 2), dtype=np.int64),
            neighbors=((),),
            within_roi_components=_build_within_roi_components(
                source_spot_table, ((),),
            ),
        )
    coords = np.column_stack([source_spot_table.x, source_spot_table.y])
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < 2:
        # Degenerate: no meaningful spacing, return an empty graph.
        neighbors = tuple(() for _ in range(n))
        return SourceSpotAdjacency(
            source_spot_table=source_spot_table,
            method=method, method_parameters=method_parameters,
            sample_median_nn=float("nan"),
            edges=np.empty((0, 2), dtype=np.int64),
            neighbors=neighbors,
            within_roi_components=_build_within_roi_components(
                source_spot_table, neighbors,
            ),
        )
    sample_median_nn = median_nn_distance(unique_coords)
    if technology == "visium":
        threshold = float(radius_factor) * float(sample_median_nn)
        pairs = cKDTree(coords).query_pairs(threshold)
        edges_arr = (
            np.array(sorted(pairs), dtype=np.int64)
            if pairs else np.empty((0, 2), dtype=np.int64)
        )
    else:
        edges_arr = build_locally_pruned_delaunay_edges(
            coords,
            local_factor=float(local_factor),
            global_cap_factor=float(global_cap_factor),
            reference_median_nn=float(sample_median_nn),
        )

    neighbors_list: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges_arr:
        neighbors_list[int(i)].append(int(j))
        neighbors_list[int(j)].append(int(i))
    neighbors = tuple(tuple(sorted(nb)) for nb in neighbors_list)

    return SourceSpotAdjacency(
        source_spot_table=source_spot_table,
        method=method, method_parameters=method_parameters,
        sample_median_nn=float(sample_median_nn),
        edges=edges_arr,
        neighbors=neighbors,
        within_roi_components=_build_within_roi_components(
            source_spot_table, neighbors,
        ),
    )
