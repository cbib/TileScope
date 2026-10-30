"""Reject obsolete masks at input boundaries; preserve integer order and bytes."""
import numpy as np
import pytest
from shapely.geometry import box

from tilescope.component_data import Tile, require_integer_point_indices
from tilescope.stats.reduced_estimator_null import compute_synchronized_median_null_p_value


@pytest.mark.parametrize('indices', [
    np.array([True, False]), np.array([], dtype=bool), np.array([0.0]),
    np.array([[0]]), np.array([-1]), np.array([2**64-1], dtype=np.uint64),
])
def test_tile_rejects_invalid_membership(indices):
    with pytest.raises(ValueError, match='point_indices'):
        Tile(tile_id=0, polygon=box(0, 0, 1, 1), point_indices=indices,
             n_spots=1, area=1.)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.uint32])
def test_integer_order_is_preserved_without_boolean_conversion(dtype):
    indices = np.array([4, 1, 3], dtype=dtype)
    tile = Tile(tile_id=0, polygon=box(0, 0, 1, 1), point_indices=indices,
                n_spots=3, area=1.)
    np.testing.assert_array_equal(tile.point_indices, [4, 1, 3])
    assert tile.point_indices.dtype == np.int64
    assert not tile.point_indices.flags.writeable
    values = np.array([1e16, -1e16, 1., 2., 3.])
    assert values[indices].sum().tobytes() == values[tile.point_indices].sum().tobytes()


def test_bounds_checks_and_empty_membership():
    with pytest.raises(ValueError, match='out of range'):
        require_integer_point_indices(np.array([3]), upper_bound=3)
    indices = np.array([], dtype=np.int64)
    assert require_integer_point_indices(indices, upper_bound=0) is indices


@pytest.mark.parametrize('indices', [np.array([True]*6), np.array([0.0]),
                                     np.array([-1]), np.array([6])])
def test_permutation_api_rejects_invalid_raw_memberships(indices):
    with pytest.raises(ValueError, match='point_indices'):
        compute_synchronized_median_null_p_value(
            source_g1_by_component={'C': np.arange(6.)},
            source_g2_by_component={'C': np.arange(6.)},
            tile_memberships=[{'C': [indices]}], valid_replicate_indices=[0],
            t_obs=.5, n_permutations=1, seed=42,
        )
