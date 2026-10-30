"""Storage and row-order regression tests for compact tile membership."""
import numpy as np
import pytest

from tilescope.tessellation.builder import _group_point_indices
from tilescope.analyzers.region_tessellation import _remap_point_indices


@pytest.mark.parametrize("n_rows,n_tiles", [(0, 0), (0, 4), (20, 1), (500, 37)])
def test_grouping_matches_boolean_selection(n_rows, n_tiles):
    rng = np.random.default_rng(173)
    assignment = rng.integers(-1, max(1, n_tiles), size=n_rows)
    groups = _group_point_indices(assignment, n_tiles)
    assert len(groups) == n_tiles
    values = rng.normal(size=(n_rows, 2))
    for tile_id, idx in enumerate(groups):
        mask = assignment == tile_id
        np.testing.assert_array_equal(idx, np.flatnonzero(mask))
        assert values[idx].sum(axis=0).tobytes() == values[mask].sum(axis=0).tobytes()


def test_membership_storage_scales_with_cells_not_cells_times_tiles():
    n_rows, n_tiles = 300_000, 20_000
    groups = _group_point_indices(np.arange(n_rows) % n_tiles, n_tiles)
    assert sum(g.nbytes for g in groups) == n_rows * np.dtype(np.int64).itemsize
    assert sum(g.size for g in groups) == n_rows
    mapped = _remap_point_indices(groups[0], np.arange(n_rows), component_row_count=n_rows)
    assert mapped.nbytes == 15 * np.dtype(np.int64).itemsize


def test_remapping_sorts_and_deduplicates_like_boolean_selection():
    mapping = np.array([30, 10, 20, 10])
    out = _remap_point_indices(np.array([0, 3, 1]), mapping, component_row_count=40)
    np.testing.assert_array_equal(out, [10, 30])


@pytest.mark.parametrize("indices", [np.array([], dtype=np.int64), np.array([], dtype=np.int32)])
def test_empty_membership_does_not_allocate_component_mask(indices):
    out = _remap_point_indices(indices, np.array([3, 2, 1]), component_row_count=10**9)
    assert out.dtype == np.int64 and out.nbytes == 0


@pytest.mark.parametrize("indices,mapping", [
    (np.array([0.0]), np.array([0])),
    (np.array([[True]]), np.array([0])),
    (np.array([0]), np.array([-1])),
    (np.array([0]), np.array([10])),
])
def test_invalid_membership_rejected(indices, mapping):
    with pytest.raises(ValueError):
        _remap_point_indices(indices, mapping, component_row_count=10)
