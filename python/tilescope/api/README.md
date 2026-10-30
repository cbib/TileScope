# TileScope Python API

Import the public API from `tilescope.api`.

## CSV input

```python
from tilescope.api import run_tilescope_from_csv

results = run_tilescope_from_csv(
    "data.csv",
    profile="standard",
    output_dir="results/",
    export=True,
    html_report=False,
    feature_a_id="Neuron",
    feature_b_id="AC",
)
```

The CSV route accepts:

- a JSON path or nested dictionary through `run_config`,
- an optional dataset-column mapping through `dataset_schema`,
- profile, flag, and `set_overrides` configuration layers,
- required biological feature identifiers,
- optional CSV/PDF export and HTML generation.

`html_report` defaults to `False`.

## DataFrame input

```python
import pandas as pd
from tilescope.api import make_run_config, run_tilescope_from_df

df = pd.read_csv("data.csv")
config = make_run_config(
    profile="standard",
    overrides={"tessellation": {"n": 20}},
)

results = run_tilescope_from_df(
    df,
    config,
    export=False,
    feature_a_id="Neuron",
    feature_b_id="AC",
)
```

The DataFrame must already contain canonical `x`, `y`, `gene1`,
`gene2`, and `roi_id` columns. `make_run_config()` and
`validate_run_config()` return a frozen `ResolvedRunConfig`.

## Configuration helpers

```python
from tilescope.api import make_run_config, validate_run_config

resolved = make_run_config(profile="standard", seed=42)
resolved = validate_run_config({
    "tessellation": {"n": 10},
})
```

Resolved user values live under `resolved.run_config`; frozen method values
live under `resolved.resolved_policy`. See
[../../../docs/CONFIGURATION.md](../../../docs/CONFIGURATION.md).

## Result surface

The full return dictionary contains canonical source/support objects plus these
scientific and publication frames:

- `roi_enrichment`
- `roi_correlations`
- `roi_support_status`
- `roi_replicate_correlations_df`
- `roi_tessellation_correlations`
- `roi_replicate_enrichment_df`
- `roi_tessellation_enrichment_df`
- `roi_consensus_correlations_df`
- `roi_consensus_enrichment_df`
- `roi_source_moran_df`
- `roi_tessellation_moran_df`
- `roi_consensus_moran_df`
- `roi_pair_interface_response_df`
- `tessellation_tiles_df`
- `consensus_hexes_df`

It also carries `run_metadata`, the resolved `config`, and
`export_paths` when export is enabled.

`return_minimal=True` currently retains:

- source ROI enrichment and correlation,
- replicate enrichment/correlation,
- tessellation enrichment,
- consensus enrichment/correlation,
- all three Moran frames,
- tile and consensus-hex frames,
- metadata, configuration, and export paths.

It excludes heavy source/geometry objects, the tessellation-correlation summary,
and the provisional ROI-pair interface frame.

The current CSV grain and typed-status contract is documented in
[../../../docs/CSV_EXPORT_CONTRACT.md](../../../docs/CSV_EXPORT_CONTRACT.md).

## R through reticulate

```r
library(reticulate)
use_python("/path/to/TileScope/venv/bin/python", required = TRUE)

api <- import("tilescope.api")
results <- api$run_tilescope_from_csv(
  input_csv = "data.csv",
  profile = "standard",
  export = FALSE,
  return_minimal = TRUE,
  feature_a_id = "Neuron",
  feature_b_id = "AC"
)
```

`roi_id` must be a biological ROI/cluster label, not a sample identifier.
See [R_INTEGRATION_GUIDE.md](R_INTEGRATION_GUIDE.md).

## Version

```python
from tilescope.api import __version__
print(__version__)
```

## Tests

From the repository root:

```bash
venv/bin/pytest python/tilescope/tests/unit/test_api.py -q
```
