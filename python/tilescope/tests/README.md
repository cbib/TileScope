# Test suite

Run commands from the repository root with the project virtual environment.

## Full suite

```bash
venv/bin/pytest python/tilescope/tests -q
```

Use collection when an exact current test count is needed:

```bash
venv/bin/pytest python/tilescope/tests --collect-only -q
```

## Layout

- `unit/`: focused module and schema behavior.
- `integration/`: interactions between analyzers, configuration, API, and
  pipeline stages.
- `scenarios/`: spatial-pattern and MAUP behavior.
- `e2e/`: complete execution and identity contracts.
- `helpers/`: shared fixtures and builders.
- `fixtures/`: small input CSV and schema fixtures.

## Focused runs

```bash
# Configuration
venv/bin/pytest python/tilescope/tests/unit/config -q

# Spatial support and tessellation
venv/bin/pytest \
  python/tilescope/tests/unit/spatial \
  python/tilescope/tests/unit/tessellation -q

# Analyzers
venv/bin/pytest python/tilescope/tests/unit/analyzers -q

# Visualization
venv/bin/pytest python/tilescope/tests/unit/viz -q

# API
venv/bin/pytest python/tilescope/tests/unit/test_api.py -q

# End-to-end contracts
venv/bin/pytest python/tilescope/tests/e2e -q
```

The test dependencies do not include `pytest-xdist`; do not document or rely
on `-n` parallel execution. No project-wide `slow` marker is registered.

## Before committing

Run the affected focused tests, then the full suite. Visualization changes also
need a representative real-output smoke because rendering correctness is not
fully captured by unit tests.
