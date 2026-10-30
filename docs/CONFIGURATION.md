# Configuration

TileScope separates user choices from the frozen scientific method.

## Resolution order

Configuration layers are merged from lowest to highest priority:

1. packaged defaults,
2. named profile,
3. user JSON,
4. explicit CLI flags,
5. `--set` overrides.

`profile=None` and `profile=""` apply no profile. Use
`profile="standard"` explicitly when the standard runtime profile is wanted.

## Resolved configuration

`load_run_config()` and `make_run_config()` return a frozen
`ResolvedRunConfig`:

```python
resolved.run_config       # user-configurable values
resolved.resolved_policy  # frozen method values, version, and hash
```

Do not mutate either model. Derive a new value with `model_copy`:

```python
new_run = resolved.run_config.model_copy(
    update={"output_dir": "results/"}
)
resolved = resolved.model_copy(update={"run_config": new_run})
```

`model_copy(update=...)` bypasses Pydantic's validators — safe for
purely informational fields like `output_dir`, but unsafe for fields
that feed the resolver. Any change to `tessellation.n`,
`consensus.min_tess_coverage_fraction`, `technology`, or
`technology_options.xenium_local_factor` must go
through `load_run_config()` or `make_run_config()` so the
`resolved_policy` is recomputed. `ResolvedRunConfig` reasserts the
resolver identities on construction, so a stale envelope raises
`ValidationError` instead of silently applying inconsistent values.

## User-configurable fields

- `technology` (`visium` or `xenium`)
- `technology_options.xenium_local_factor` (optional; Xenium only)
- `tessellation.n`
- `tessellation.target_observations_per_tile`
- `tessellation.min_observations_per_tile`
- `tessellation.min_component_observations`
- `consensus.min_spots_per_hex`
- `consensus.min_tess_coverage_fraction`
- `consensus.bivariate_morans.visualize`
- `roi_filters.min_roi_spots`
- `roi_filters.min_context_spots`
- `visualization.show_stability`
- `permutations`
- `seed`
- `output_dir`
- `verbose`

`consensus.min_tess_coverage_fraction` is a float in `(0, 1]` (default
`2/3`); it resolves to `resolved_policy.consensus.min_tess_per_hex =
math.ceil(fraction * tessellation.n)`, which is always in
`[1, tessellation.n]`. `min_observations_per_tile` must not exceed
`target_observations_per_tile`. The component floor must yield at least three
nominal seeds under
`round(min_component_observations / target_observations_per_tile)`.

`technology` selects the graph method. Visium resolves to fixed-radius
adjacency; Xenium resolves to locally pruned Delaunay adjacency. Xenium's
local pruning factor defaults to `2.5` and may be overridden with
`technology_options.xenium_local_factor` after a geometry audit; the global
long-edge cap remains frozen. The selected graph is also the ROI-component
connectivity authority. The resolved method and parameters are recorded under
`resolved_policy.source_adjacency`.

Method-definition values such as expression quantiles, component-domain
radii, graph-construction method/global cap, consensus sizing, and
Moran computation live under `resolved_policy`. Supplying one of those paths
in user configuration raises `ConfigMigrationError`, even when the requested
value equals the frozen value.

## CLI examples

```bash
tilescope data.csv --profile standard
tilescope data.csv --config-run run.config.json
tilescope data.csv --profile standard --n-tessellations 20
tilescope data.csv --set consensus.min_tess_coverage_fraction=0.7
tilescope data.csv --set technology=xenium \
  --set technology_options.xenium_local_factor=2.0
```

An explicitly supplied CLI flag wins over the corresponding JSON/profile
value. Flags left unspecified do not overwrite lower layers.

## Python examples

Build a resolved configuration:

```python
from tilescope.api import make_run_config

resolved = make_run_config(
    profile="standard",
    overrides={"tessellation": {"n": 20}},
    seed=42,
)
```

For the CSV API, a nested dictionary or JSON path may be supplied through
`run_config`; `profile`, `flag_overrides`, and `set_overrides` expose the
same layering rules.

## Dataset schema

`--config-dataset` describes source-column mappings. It is independent of run
configuration.

```json
{
  "columns": {
    "array_row": "x",
    "array_col": "y",
    "feature_a": "gene1",
    "feature_b": "gene2",
    "cluster": "roi_id"
  }
}
```

The DataFrame API expects canonical columns directly:
`x`, `y`, `gene1`, `gene2`, and `roi_id`.

Tile and consensus "density" values retain their established meaning: the
mean feature value per contributing spot or cell observation. For Xenium this is a
mean per cell; for Visium it is a mean per spot. It is not expression divided
by geometric area. The effective policy records
`tile_value_aggregation=mean_per_source_observation`.

## Validation

Validation is owned by the stage with enough information to enforce it:

- Pydantic validates and freezes configuration during loading.
- The sample-analysis entry point validates source columns and feature
  identities.
- Support, tessellation, and consensus stages emit typed availability for
  data-dependent eligibility.
- Publication DataFrames are validated by
  `tilescope.schema.table_validation`.

## Adding a field

1. Add it to the appropriate model in `tilescope.settings.models`.
2. Add its packaged default and any profile override.
3. Add a CLI flag only if ordinary users need it.
4. Decide whether it is a runtime choice or a versioned method-definition
   value.
5. Read it from `config.run_config` or `config.resolved_policy`, as
   appropriate.
6. Add focused configuration tests.
