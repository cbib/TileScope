"""Bounded two-thread clipping with unchanged operations and result order."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice


@contextmanager
def ordered_clips(function, items):
    """Run independent clips, joining workers even on errors or early exit.

    At most 32 tasks are submitted at once. Results and exceptions are yielded
    in input order; no geometry reductions run concurrently.
    """
    # Avoid thread startup for small regions. No geometry operation changes.
    if len(items) < 32:
        yield map(function, items)
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        def results():
            iterator = iter(items)
            while chunk := list(islice(iterator, 32)):
                yield from pool.map(function, chunk)
        yield results()
