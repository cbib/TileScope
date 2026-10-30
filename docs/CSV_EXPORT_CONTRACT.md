# CSV export contract

`tilescope.reporters.result_exporter` is the sole CSV writer. Before each
export it removes every managed output and known retired artifact from the
target directory. It then writes each non-empty canonical DataFrame once.

Absence of a CSV therefore means its canonical frame was absent or empty; it
must not be interpreted as a finite zero result.

## Identity rules

- `sample_id` identifies the source sample.
- `roi_id` always identifies the biological source-parent ROI.
- `component_id` identifies computational support within a parent ROI.
- `domain_region_id` identifies a connected component-domain region.
- Replicate, tile, hex, direction, value-surface, feature, and ordered-pair
  columns extend those identities where applicable.

Component and constructed-unit rows retain `roi_id` as parent provenance.
No export replaces `roi_id` with `component_id`.

## Biological ROI tables

| File | Row grain | Meaning |
|---|---|---|
| `roi_enrichment.csv` | ROI | Spot/cell marginal/joint enrichment and distributional effects |
| `roi_correlations.csv` | ROI | Spot/cell Pearson correlation and inference |
| `roi_support_status.csv` | ROI | Spot/cell support class and component accounting |
| `roi_replicate_correlations.csv` | ROI × tessellation replicate | Pearson on pooled sibling-component tile means |
| `roi_tessellation_correlations.csv` | ROI | Reduction across the replicate correlation surface |
| `roi_replicate_enrichment.csv` | ROI × tessellation replicate | Same-replicate ROI-versus-other-ROI effects |
| `roi_tessellation_enrichment.csv` | ROI | Reduction across independently available replicate effects |
| `roi_consensus_correlations.csv` | ROI × value surface | Pearson on pooled consensus hex densities |
| `roi_consensus_enrichment.csv` | ROI × value surface | Consensus ROI-versus-other-ROI effects |
| `roi_source_moran.csv` | ROI × direction | Bivariate Moran on spot/cell expression and spot/cell adjacency |
| `roi_tessellation_moran.csv` | ROI × direction | Bivariate Moran on tile densities, reduced across valid replicates |
| `roi_consensus_moran.csv` | ROI × value surface × direction | Bivariate Moran on consensus densities |
| `roi_cross_grain_sensitivity.csv` | ROI × estimand triple × target surface | Source-vs-construction empirical sensitivity table (see below) |

Each estimator has a typed status column. Numeric values are populated only
when the corresponding status permits them. Discrete support/accounting fields
remain populated on unavailable rows where their values are defined.

The source, tessellation, and consensus Moran estimators consume continuous
expression or density values; they are not computed from enrichment
classifications. At every technology and grain, a finite Moran value requires
at least three units—and at least half of all finite units—to participate in
the estimator's adjacency. The Moran exports record both
`n_adjacency_supported_units` and `adjacency_supported_fraction`; failures of
this common eligibility rule carry `insufficient_adjacency_support` rather
than a numeric value. Constructed-support Moran also requires distance-based
neighbour discrimination. If every region block is a complete graph, moving
the units cannot change the weights; the row carries
`no_spatial_discrimination` instead of a numeric Moran value.

## Cross-grain sensitivity table

`roi_cross_grain_sensitivity.csv` reports how each ROI's headline
estimator changes when computed on a construction surface instead of the
spot/cell surface. Row grain is:

```text
sample × ROI × (statistic, channel, moran_direction) × target_surface
```

Every declared ROI contributes exactly **14 rows**: 7 estimand triples
(1 correlation + 4 enrichment channels + 2 Moran directions) × 2 target
surfaces. Row absence is a producer bug; unavailability is a typed row
with an unavailable `raw_status` and a populated `undefined_reason`.

**Two comparisons per estimand.** The reference is always
`source_native`. The two targets are:

- `tessellation_median` — random-zoning sensitivity captured by the
  tessellation replicate summary and effect-scale IQR.
- `consensus_ensemble` — headline source-to-constructed-support
  sensitivity on the ensemble hex surface.

`consensus_direct` sensitivity is not a row in this table; downstream
code obtains it by joining the existing `roi_consensus_*.csv` surfaces.

**Effect scale.** Sign, delta, and sign-preservation are computed on the
null-centred effect scale, never on raw values. For correlation and
Moran the transform is identity. For enrichment it is
`E(r) = r / (1 + r) - 0.5`, with the typed saturation
`positive_over_zero_context` mapped to `+0.5`. Sharing a signed delta
across families without this transform would silently invert sign
whenever an enrichment ratio crosses 1.

**Comparability.** `comparison_status` takes one of
`both_comparable`, `reference_unavailable`, `target_unavailable`,
`both_unavailable`. It is derived from effect availability, not from
raw-status strings. `delta_effect`, `abs_delta_effect`, and
`sign_preserved` are populated only when `both_comparable`.

**Tessellation-only fields.** `tessellation_effect_iqr`,
`n_comparable_replicates`, `n_planned_replicates` are populated on
`tessellation_median` rows only and null on `consensus_ensemble` rows.

**Semantic identity.** The `g1_id` / `g2_id` columns carry the biological
feature identities passed as `feature_a_id` / `feature_b_id` to the API.
Both are required. The reserved computational input columns are never used
as biological labels.

Provenance for the table lives on the pair-run's `run_metadata.json` and
`effective_config.json`; it is not repeated per row.

## Ordered ROI-pair diagnostic

`roi_pair_interface_response.csv` is provisional. Its grain is:

```text
sample × ordered(anchor ROI, responder ROI) × responder feature × surface
```

It emits spot/cell, tessellation, consensus-direct, and
consensus-ensemble rows. Near/buffer/far support is defined once by spot/cell
hop distance from the A–B interface, then projected to constructed surfaces.
Unavailable pairs and surfaces carry `availability_status`.

## Component-grain descriptor

`component_radial_shell_correlogram.csv` is the single deliberately
component-grain descriptor. Its grain is:

```text
sample × component × shell
```

Every row retains both `roi_id` and `component_id`. Shell edges are
normalised by the component's median nearest-neighbour spacing
(dimensionless) and spatial persistence is a component-local
property; pooling curves across a ROI's disjoint components would
change the estimand, so this descriptor is not rolled up to ROI.
Rows cover every TileScope-retained source component — components
excluded upstream by the sample support authority never appear here.
Availability is per-shell (`insufficient_anchor_support` when the
count / fraction gate fails) with a small set of component-level
fails replicated across all shell rows (`insufficient_spots` /
`median_nn_undefined` / `zero_variance_*`).

## Support and construction tables

| File | Row grain | Meaning |
|---|---|---|
| `spot_ledger.csv` | spot or cell | Spot/cell identity, component assignment, and support eligibility |
| `component_geometry.csv` | component | Parent identity, geometry WKT, areas, and construction diagnostics |
| `component_domain_regions.csv` | component × domain region | Region geometry and tessellation eligibility |
| `component_region_replicate_status.csv` | component × region × replicate | Tessellation construction outcome |

Component-domain geometry includes ordinary sampling faces supported by the
canonical source-neighbour threshold while retaining genuine acquisition gaps.

## Observation surfaces

| File | Row grain | Meaning |
|---|---|---|
| `tessellation_tiles.csv` | component × region × replicate × tile | Tile geometry, membership-derived masses, and pair densities |
| `consensus_hexes.csv` | component × region × hex | Direct and ensemble densities, masses, stability, and per-ratio enrichment status |

Tiles and hexagons are observation units, not biological reporting ROIs.
Direct and ensemble consensus values remain separate surfaces.

## Validation authorities

The authoritative column names and dtypes live in:

- `tilescope.cols`
- `tilescope.schema.table_validation`
- the focused `build_*_df` producer for each table

Status vocabularies live beside their estimators in
`tilescope.analyzers.parent_roi_outcomes`,
`tilescope.analyzers.partition_analysis`, and the focused analyzer modules.
Consumers must gate a numeric field using its associated status rather than
inferring availability from NaN alone.

