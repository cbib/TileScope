"""Deterministic seed derivation from a configured base seed and
opaque identity parts.

Used wherever a stage needs a per-call RNG seed that must be:

- reproducible across processes (independent of ``PYTHONHASHSEED``),
- controllable by the user's ``config.seed`` (folded into the digest),
- call-order-independent (a permutation seed depends only on the
  named identity of the operation, not on how many other seeds were
  drawn beforehand),
- distinct across operations that share the same identity but differ
  by a caller-supplied tag (e.g. the two directions of bivariate
  Moran within one component).

Callers pass ``base_seed`` (typically ``config.seed``) plus any
combination of identity fields (sample id, component id, replicate
index, operation tag) as positional arguments. All parts are
``str``-coerced and pipe-joined before hashing.
"""
from __future__ import annotations

import hashlib


def derive_stable_seed(base_seed: int, *identity_parts: object) -> int:
    """Return a stable 32-bit seed derived from ``base_seed`` and
    ``identity_parts``.

    Algorithm: BLAKE2b (8-byte digest) over
    ``"|".join([str(base_seed), *map(str, identity_parts)])``,
    then reduced modulo ``2**32 - 1`` to a 32-bit range that fits
    NumPy's :class:`numpy.random.Generator` seed contract.

    Never uses Python's built-in ``hash(...)`` (which is
    ``PYTHONHASHSEED``-dependent for strings).
    """
    key = "|".join([str(base_seed)] + [str(part) for part in identity_parts])
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1)
