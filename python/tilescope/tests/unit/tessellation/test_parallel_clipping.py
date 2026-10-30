from contextlib import contextmanager
import threading
import numpy as np
import pytest
from shapely.geometry import box
from tilescope.tessellation.clipping import ordered_clips
from tilescope.tessellation import engine_geos

def test_order_workers_and_bounded_submission():
    seen = []
    lock = threading.Lock()
    def work(item):
        with lock:
            seen.append((item, threading.get_ident()))
        return item * 2
    with ordered_clips(work, range(101)) as results:
        assert next(results) == 0
        assert len(seen) <= 32
        assert list(results) == list(range(2, 202, 2))
    assert len(set(t for _, t in seen)) <= 2
    assert threading.get_ident() not in {t for _, t in seen}

def test_small_case_stays_serial():
    with ordered_clips(lambda _: threading.get_ident(), range(3)) as result:
        assert list(result) == [threading.get_ident()] * 3

def test_exception_order_and_cleanup():
    workers = []
    def work(item):
        workers.append(threading.current_thread())
        if item == 7:
            raise ValueError('first in input order')
        if item == 12:
            raise RuntimeError('later')
        return item
    with pytest.raises(ValueError, match='first in input order'):
        with ordered_clips(work, range(99)) as result:
            assert [next(result) for _ in range(7)] == list(range(7))
            next(result)
    assert all(not t.is_alive() for t in workers)

def test_early_exit_joins_workers():
    workers = []
    def work(item):
        workers.append(threading.current_thread())
        return item
    with ordered_clips(work, range(100)) as result:
        assert next(result) == 0
    assert all(not t.is_alive() for t in workers)

@pytest.mark.parametrize('hole', [False, True])
def test_exact_engine_wkb(monkeypatch, hole):
    region = box(0, 0, 20, 20)
    if hole:
        region = region.difference(box(8, 8, 12, 12))
    seeds = np.asarray([(x, y) for x in range(1, 20, 2) for y in range(1, 20, 2)
                        if not (hole and 8 < x < 12 and 8 < y < 12)], dtype=float)
    engine = engine_geos.GeosVoronoiEngine(engine_geos.GeosVoronoiConfig())
    threaded = engine.build(region, seeds, d1=2)
    @contextmanager
    def serial(function, items):
        yield map(function, items)
    monkeypatch.setattr(engine_geos, 'ordered_clips', serial)
    reference = engine.build(region, seeds, d1=2)
    assert threaded.seed_index == reference.seed_index
    assert [p.wkb for p in threaded.tiles] == [p.wkb for p in reference.tiles]
    assert [p.area for p in threaded.tiles] == [p.area for p in reference.tiles]
