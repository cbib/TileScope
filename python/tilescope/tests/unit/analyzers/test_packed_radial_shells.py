"""Exact regression against the retired list-based radial neighbourhoods."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from tilescope.analyzers import component_radial_shell_correlogram as radial
from tilescope.analyzers.tessellation_tile_frames import _iter_records


def legacy_neighbours(coords, edges):
    tree = cKDTree(coords)
    median = float(np.median(tree.query(coords, k=2)[0][:, 1]))
    pairs = np.asarray(sorted(tree.query_pairs(float(edges[-1] * median))), dtype=int).reshape(-1, 2)
    distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    shells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = pairs[(distances >= lo * median) & (distances < hi * median)]
        neighbours = [[] for _ in coords]
        for a, b in selected:
            neighbours[a].append(int(b))
            neighbours[b].append(int(a))
        shells.append(neighbours)
    return shells


@pytest.mark.parametrize('kind', ['grid', 'irregular', 'collinear', 'disconnected'])
@pytest.mark.parametrize('edges', [(0.5, 1.5, 2.5, 4.5, 8.5), (1., 2., 3., 4.)])
def test_neighbours_and_reductions_are_byte_identical(kind, edges):
    rng = np.random.default_rng(92)
    coords = np.stack(np.meshgrid(np.arange(10), np.arange(8)), axis=-1).reshape(-1, 2).astype(float)
    if kind == 'irregular': coords += rng.normal(0, .2, coords.shape)
    if kind == 'collinear': coords = np.column_stack([np.arange(80), np.zeros(80)])
    if kind == 'disconnected': coords[40:] += 100
    old = legacy_neighbours(coords, edges)
    packed = radial._prepare_shell_geometry(coords=coords, shell_edges=edges, n_shells=len(edges)-1)
    a, b = rng.normal(size=(2, len(coords)))
    expected = []
    for shell, compact in zip(old, packed.neighbours_by_shell):
        eligible = np.array([bool(nb) for nb in shell])
        for position, nb in enumerate(shell):
            np.testing.assert_array_equal(compact[position], nb)
        assert not compact.indices.flags.writeable
        assert not compact.offsets.flags.writeable
        ab, ba = [], []
        for pos in np.where(eligible)[0]:
            nb = np.array(shell[pos], dtype=int)
            ab.append(float(a[pos]) * float(np.mean(b[nb])))
            ba.append(float(b[pos]) * float(np.mean(a[nb])))
        expected.append(.5 * (float(np.mean(ab)) + float(np.mean(ba))) if len(ab) else np.nan)
    new = radial._evaluate_shells(z_a=a, z_b=b, shell_geometry=packed, n_shells=len(edges)-1,
                                  min_anchors=1, min_fraction=0)
    assert new['shell_symmetric'].tobytes() == np.asarray(expected).tobytes()
    assert packed.n_pairs == tuple(sum(map(len, shell)) // 2 for shell in old)


@pytest.mark.parametrize('block_size', [1, 7, 100000])
def test_batched_distances_preserve_bits(monkeypatch, block_size):
    rng = np.random.default_rng(173)
    coords = rng.normal(size=(125, 2))
    pairs = cKDTree(coords).query_pairs(1.7, output_type='ndarray')
    expected = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    monkeypatch.setattr(radial, '_PAIR_DISTANCE_BLOCK_SIZE', block_size)
    assert radial._pair_distances(coords, pairs).tobytes() == expected.tobytes()


def test_empty_shell_and_linear_storage():
    empty = radial._pack_shell_neighbours(np.empty((0, 2), dtype=int), 100)
    assert empty.indices.size == 0 and not empty.offsets.any()
    pairs = np.column_stack([np.zeros(2000, dtype=int), np.arange(1, 2001)])
    packed = radial._pack_shell_neighbours(pairs[::-1], 2001)
    np.testing.assert_array_equal(packed[0], np.arange(1, 2001))
    assert packed.indices.nbytes + packed.offsets.nbytes == 8 * (4000 + 2002)


def test_neighbour_query_never_constructs_python_pair_sets(monkeypatch):
    original = radial.cKDTree
    class NumericTree:
        def __init__(self, coords): self.tree = original(coords)
        def query(self, *args, **kwargs): return self.tree.query(*args, **kwargs)
        def query_pairs(self, *args, **kwargs):
            assert kwargs.get('output_type') == 'ndarray'
            return self.tree.query_pairs(*args, **kwargs)
    monkeypatch.setattr(radial, 'cKDTree', NumericTree)
    radial._prepare_shell_geometry(coords=np.column_stack([np.arange(12), np.zeros(12)]),
                                   shell_edges=(0.5, 1.5, 3.5), n_shells=2)


def test_row_iterator_preserves_values_without_materializing_all_records(monkeypatch):
    frame = pd.DataFrame({'s': ['a','b'], 'n': pd.Series([1, 2], dtype='Int64'), 'x': [1.2, 3.4]})
    expected = frame.to_dict('records')
    monkeypatch.setattr(pd.DataFrame, 'to_dict', lambda *args, **kwargs:
                        pytest.fail('Streamed rows must not call to_dict'))
    assert list(_iter_records(frame)) == expected
