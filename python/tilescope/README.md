# TileScope Python package

TileScope analyzes pairs of spatial expression fields on a fixed biological ROI
partition. It exposes an installed CLI, a Python API, typed CSV outputs, and PDF
diagnostics.

## Install

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e "python[test]"
```

Python 3.10 or newer is required.

## CLI

```bash
tilescope data.csv --profile standard --feature-a-id featureA --feature-b-id featureB --output results/
```

The input must contain `x`, `y`, `gene1`, `gene2`, and `roi_id`.
`featureA` and `featureB` correspond to the actual genes or signatures
whose expression/scores are contained on columns `gene1` and `gene2`.
Use `--config-dataset` when source column names differ.

Useful options:

```bash
tilescope --help
tilescope data.csv --profile exploratory --feature-a-id featureA --feature-b-id featureB
tilescope data.csv --config-run run.config.json --feature-a-id featureA --feature-b-id featureB
tilescope data.csv --set tessellation.n=20 --feature-a-id featureA --feature-b-id featureB
```

Configuration precedence is:
`--set` > explicit CLI flags > user JSON > profile > packaged defaults.
See [../../docs/CONFIGURATION.md](../../docs/CONFIGURATION.md).

## Python API

```python
from tilescope.api import run_tilescope_from_csv

results = run_tilescope_from_csv(
    "data.csv",
    profile="standard",
    output_dir="results/",
    export=True,
)
```

For in-memory DataFrames and resolved configuration objects, see
[api/README.md](api/README.md).

Low-level tile membership uses one-dimensional integer position arrays.
`Tile.point_indices` and raw memberships passed to the tessellation permutation
routine reject Boolean masks and floating-point indices. Convert an existing
mask explicitly with `numpy.flatnonzero(mask)`. Tile repair requires a finite,
nonnegative `edge_tol`; its unused `verbose` argument has been removed.

## Outputs

The main scientific results are biological-ROI tables for source,
tessellation, and consensus representations. Tile and hex tables are
observation surfaces; support tables record construction and availability.
The provisional interface-response table is directional and indexed by ordered
ROI pair.

See [../../docs/CSV_EXPORT_CONTRACT.md](../../docs/CSV_EXPORT_CONTRACT.md) for
the complete current file list and grain definitions.

Generated reports are:

- `expression_summary.pdf`
- `consensus_tessellation_summary.pdf`
- `tessellation_densities_*.pdf`
- `support_provenance.pdf`
- optional `analysis_report.html`

## Tests

From the repository root:

```bash
venv/bin/pytest python/tilescope/tests -q
```

Focused commands are listed in [tests/README.md](tests/README.md).

## Companion screening tool

TileScope includes one multi-gene ROI enrichment screening utility:

```bash
python -m tilescope.tools.tilescope_enrichment_screening \
  --input data.csv \
  --output screening/
```

See [tools/README.md](tools/README.md).

## License

MIT. See [LICENSE](LICENSE).
