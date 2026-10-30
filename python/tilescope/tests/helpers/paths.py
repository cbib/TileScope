"""Absolute paths to test fixtures and the shared output directory.

Test callers reference `FIXTURES_DIR / "..."` instead of
CWD-relative strings, so `pytest` runs cleanly from anywhere (repo
root, python/tilescope, CI working directory).

Follow-up: several tests still write to TESTS_OUTPUT_DIR (a
persistent directory) where they should be using pytest's
`tmp_path`. Migrating them is a separate cleanup.
"""
from pathlib import Path

# tests/helpers/paths.py -> tests/helpers -> tests -> fixtures
FIXTURES_DIR: Path = Path(__file__).resolve().parent.parent / "fixtures"

# Persistent output collector next to the fixtures; kept for tests that
# haven't been migrated to tmp_path yet.
TESTS_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "tests_output"
