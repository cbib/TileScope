# TileScope architecture

TileScope is a Python analysis engine for pairs of continuous spatial features.
Biological ROI labels are supplied by the caller and remain stable throughout
the pipeline.

## Public entry points

- CLI: `tilescope`
- CSV/DataFrame API: `tilescope.api`
- Multi-pair engine: `tilescope.run_sample`

## Identity and grain

TileScope keeps these identities distinct:

- `sample_id`: acquisition/sample identity.
- `roi_id`: biological parent ROI supplied with the source data.
- `component_id`: connected support component created inside a parent ROI.
- `domain_region_id`: connected region of a component domain.
- tile and hex identifiers: constructed observation units.

`roi_id` is never replaced by `component_id`. ROI statistics pool eligible
sibling components back to their biological parent.

## Analysis layers

```text
sample
├── SampleSupportBank                 feature-independent
│   ├── SourceSpotTable
│   ├── SourceSpotAdjacency
│   ├── support components and domains
│   ├── tessellation ensemble
│   ├── radial shell geometry
│   └── reusable sparse Moran weights
├── ConsensusBank                     feature-independent
│   ├── consensus hex geometry
│   ├── direct source membership
│   └── tile-to-hex overlap
├── FeatureSpatialProjection          one per requested feature
│   ├── spot/cell values and thresholds
│   ├── tile marginals
│   ├── consensus marginals
│   └── ordered-interface responses
└── PairStatisticResult               one per requested feature pair
    ├── source-ROI statistics
    ├── tessellation and consensus statistics
    ├── bivariate Moran surfaces
    ├── radial cross-products
    ├── selected feature-interface lanes
    └── cross-grain sensitivity
```

Feature-independent geometry is built once. Feature projections are reused
across pairs. Pair modules consume those authorities and do not rebuild
component domains, tessellations, membership, radial shells, interface bands,
or spatial weights for an identical retained support.

## Support and component domains

Spots and cells are split into connected components by the adaptive k-nearest-
neighbour connectivity operator. This split describes computational support;
it does not redefine biological ROIs or biological contact.

`SourceSpotAdjacency` is the canonical source-grain acquisition-neighbour
graph. It is built once using the technology-selected adjacency policy and is
stored in `SampleSupportBank`. It supports source Moran, cross-ROI edge selection, and
within-ROI hop distance. Its within-ROI component memberships are also built
once; support classification and the ROI splitter project from those same
memberships rather than traversing the graph independently.

Component domains combine:

1. component support,
2. parent-ROI ownership, and
3. acquisition support.

Component and acquisition support include source-observation footprints and
ordinary sampling faces whose edges are present in the canonical source graph.
This fills normal lattice cells without bridging genuine acquisition gaps.
Parent ownership prevents overlap between biological ROIs, and ineligible
spots or cells remain explicit exclusions.

## Constructed representations

The tessellation bank contains randomized Voronoi partitions for every eligible
component-domain region. The number of replicates is user-configurable; seed
selection scale, minimum tile occupancy, and component observation floor are
dataset-configurable.

The consensus bank is built once from the tessellation bank. It owns:

- the sample-anchored hex lattice,
- direct source-to-hex membership,
- tile-to-hex overlap weights, and
- ensemble coverage.

Direct and ensemble consensus densities are distinct value surfaces.

Graphs at different grains are intentionally separate:

- source: technology-selected adjacency, reused by support construction,
  ROI components, interfaces, and source Moran,
- tessellation Moran: region-blocked kNN graph on tile centroids,
- consensus Moran: region-blocked kNN graph on hex centres.

They answer different grain-specific questions and are not a universal graph
abstraction.

## Statistics

Source, tessellation, and consensus estimators emit biological-ROI rows with
typed availability. Ineligible ROIs or constructed surfaces are represented by
status rather than silently changing the estimand.

Bivariate Moran uses continuous expression/density values:

- spots or cells on spot/cell adjacency,
- tile densities per replicate, summarized across valid replicates,
- direct and ensemble consensus densities on hex adjacency.

The provisional interface-response estimator is directional. It defines
near/buffer/far bands once on the spot/cell adjacency and projects their mass to
tessellation and consensus surfaces; constructed graphs do not redefine source
hop distance.

## Execution and export

`tilescope.analyzers.sample_authority` owns computation. Pair results may be
streamed through `pair_sink`; the one-pair API captures the same result shape.

`tilescope.reporters.result_exporter` is the sole CSV writer. It purges managed
and retired artifacts before writing non-empty canonical DataFrames.
Canonical sample-support and tessellation frames are materialized under
`tilescope.analyzers`; reporter modules do not construct scientific tables.
Visualization consumes the same exported/canonical frames and does not
recompute statistics.

The current CSV surface is documented in
[CSV_EXPORT_CONTRACT.md](CSV_EXPORT_CONTRACT.md).

## Configuration and validation

`ResolvedRunConfig` contains:

- `run_config`: user-configurable runtime and reporting choices,
- `resolved_policy`: frozen method-definition values plus version and hash.

Pydantic validates configuration. Producers validate publication DataFrames
against schemas in `tilescope.schema.table_validation`.

See [CONFIGURATION.md](CONFIGURATION.md).

## R boundary

The tracked R package provides Seurat extraction, candidate-generation,
storage, and visualization helpers. It does not own analytical algorithms and
does not currently wrap Python execution. R callers may export canonical CSV
input or call `tilescope.api` directly through `reticulate`.
