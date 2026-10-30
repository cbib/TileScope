"""Contract pins for ``derive_stable_seed``.

Three semantic properties: determinism (same inputs → same output),
base-seed sensitivity (changing ``config.seed`` changes the output),
and independence from ``PYTHONHASHSEED`` (BLAKE2b digest of a
pipe-joined key, not Python's built-in ``hash(...)``).
"""
import os
import subprocess
import sys

import pytest

from tilescope.stable_seed import derive_stable_seed


def test_derive_stable_seed_is_deterministic_across_calls():
    a = derive_stable_seed(42, "sampleA", "compX", 3, "replicate_correlation")
    b = derive_stable_seed(42, "sampleA", "compX", 3, "replicate_correlation")
    assert a == b


@pytest.mark.parametrize(
    "left, right",
    [
        # base_seed folds into the digest
        ((42, "s", "c", 0, "op"), (43, "s", "c", 0, "op")),
        # each identity part influences the output
        ((42, "s1", "c", 0, "op"), (42, "s2", "c", 0, "op")),
        ((42, "s", "c1", 0, "op"), (42, "s", "c2", 0, "op")),
        ((42, "s", "c", 0, "op"), (42, "s", "c", 1, "op")),
        ((42, "s", "c", 0, "opA"), (42, "s", "c", 0, "opB")),
    ],
    ids=["base_seed", "sample_id", "component_id",
         "replicate_index", "operation_tag"],
)
def test_derive_stable_seed_is_sensitive_to_each_identity_part(left, right):
    assert derive_stable_seed(*left) != derive_stable_seed(*right)


def test_derive_stable_seed_is_pythonhashseed_independent():
    """Two separate Python interpreters with different
    ``PYTHONHASHSEED`` values must produce the same seed for the
    same inputs. Guards against accidental use of ``hash(...)``.
    """
    script = (
        "from tilescope.stable_seed import derive_stable_seed; "
        "print(derive_stable_seed(42, 'sampleA', 'compX', 3, 'replicate_correlation'))"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "0"}
    env_b = {**os.environ, "PYTHONHASHSEED": "17"}
    out_a = subprocess.check_output([sys.executable, "-c", script], env=env_a).strip()
    out_b = subprocess.check_output([sys.executable, "-c", script], env=env_b).strip()
    assert out_a == out_b
