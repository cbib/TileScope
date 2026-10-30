<p align="center">
  <img src="assets/tilescope_telescope_logo.svg" alt="TileScope" width="100%">
</p>

# TileScope

**TileScope quantifies how pairs of continuous spatial features — genes or
transcriptional-program scores — are organised within predefined tissue
regions and near their boundaries, while making the sensitivity of each
estimate to spatial aggregation explicit.**

The engine runs on Visium spots and single-cell-resolution spatial data
(Xenium). It preserves biological ROI labels throughout, distinguishes
their spatially disconnected components, builds a repeated tessellation
ensemble and a consensus hexagonal grid, and reports every statistic
alongside its availability and support on each of these representations.

## Why

Spatial co-expression analyses commonly combine a within-ROI correlation
with a regional enrichment statistic. These two numbers cannot
distinguish, for example, broad co-elevation from opposing gradients,
and neither reports how its value would change if the observations were
grouped differently. TileScope addresses both: it measures the same ROI
with complementary statistics on three spatial supports (source
observations, tessellation tiles, consensus hexagons) and surfaces the
differences between these supports as an explicit output. The paper
below reports a benchmark showing which combinations of these
measurements distinguish which spatial arrangements, and applies the
framework to published glioblastoma datasets, a mouse glioblastoma
dataset generated for the study, and Xenium single-cell data.

## Citation

If you use TileScope, please cite:

> *TileScope: multi-support spatial statistics for pairs of continuous
> features in spatial transcriptomics data.* Manuscript submitted.
> [GitHub](https://github.com/cbib/TileScope) · companion analysis
> repository: [cbib/TileScope-results](https://github.com/cbib/TileScope-results) 
>  · companion data bundles: [Zenodo DOI](https://zenodo.org/records/23535235) 


## Install

TileScope requires Python 3.10 or newer. The engine lives under
`python/tilescope`; install from the repository root in editable mode:

```bash
git clone https://github.com/cbib/TileScope.git
cd TileScope
python -m venv venv
source venv/bin/activate
pip install -e "python[test]"
```

The HTML report embeds its figures through `pdf2image`, which needs the
poppler utilities on the system `PATH`:

```bash
sudo apt-get install poppler-utils   # Debian/Ubuntu
brew install poppler                 # macOS
conda install -c conda-forge poppler # conda environments
```

Without poppler the run still completes and writes all CSVs and PDFs,
but `analysis_report.html` is generated without its figures.

## Minimum run

Prepare one CSV per sample with columns `x`, `y`, `gene1`, `gene2`, and
`roi_id`. `roi_id` is the biological ROI label (e.g., a BayesSpace
cluster assignment); TileScope does not re-partition the data. Columns
`gene1` and `gene2` are the computational slots — the real feature
names propagate separately as `feature_a_id` / `feature_b_id`.

**The engine is pairwise by design.** Each invocation analyses exactly
one pair of features. If you work from a per-sample score bank with
many signatures (one row per observation, one column per signature),
select the two columns of interest, rename them to `gene1` and
`gene2`, and pass the real signature names as `feature_a_id` /
`feature_b_id`. The companion
[cbib/TileScope-results](https://github.com/cbib/TileScope-results)
repository contains a small reusable helper,
`tilescope_results.pair_inputs`, which materialises a transient
per-pair CSV from a score bank, and two cohort runners that wrap it
to loop over all pairs of a study:
`analyses/greenwald_full/run_all_pairs.py` (human GBM) and
`analyses/mouse_gbm_spatial_programs/phase6_harmonized_run_tilescope.py`
(mouse). Both accept `--input-dir` (containing `scores/<sample>.csv.gz`,
`pairs.csv` and the shared run config) and `--output-dir` (where the
per-pair run trees are written). The score bank itself is never
modified.

Command line:

```bash
tilescope data.csv --profile standard --feature-a-id featureA --feature-b-id featureB --output results/
```

Programmatic:

```python
from tilescope.api import run_tilescope_from_csv

results = run_tilescope_from_csv(
    "data.csv",
    profile="standard",
    output_dir="results/",
    export=True,
    html_report=True,
    feature_a_id="Neuron",
    feature_b_id="AC",
)
```

Both routes write a report directory containing the CSVs listed below
plus `analysis_report.html` summarising the run visually.

## What the pipeline does

For each sample TileScope builds the spatial substrate once:

1. **Spot/cell observations and their graph** — spots or cells with their
   neighbourhood edges, respecting ROI labels.
2. **Connected components** inside each ROI — handled as construction
   units; the ROI identity is preserved.
3. **Tessellation ensemble** — a bank of Voronoi-style partitions
   within each component, used to assess scale sensitivity.
4. **Consensus hexagonal grid** — a single aggregated surface derived
   from the ensemble.

Each ordered pair of features is then projected onto each surface, and
the same scientific statistics are computed on each. The engine reports
their numerical values alongside availability flags and support counts,
so sign changes or dropouts caused by aggregation are visible rather
than hidden.

## Outputs

Outputs are grouped by the scientific question they answer. Full
column-level definitions are in
[docs/CSV_EXPORT_CONTRACT.md](docs/CSV_EXPORT_CONTRACT.md); the HTML
report (`analysis_report.html`) is explained in
[docs/HOW_TO_READ_THE_REPORT.md](docs/HOW_TO_READ_THE_REPORT.md).

**Co-expression and spatial co-variation**
- `roi_correlations.csv`, `roi_consensus_correlations.csv`,
  `roi_replicate_correlations.csv`, `roi_tessellation_correlations.csv`
  — Pearson at the four supports.
- `roi_source_moran.csv`, `roi_tessellation_moran.csv`,
  `roi_consensus_moran.csv` — bivariate Moran's I in both directions.

**Regional enrichment**
- `roi_enrichment.csv`, `roi_consensus_enrichment.csv`,
  `roi_replicate_enrichment.csv`, `roi_tessellation_enrichment.csv`
  — overrepresentation of each feature in the ROI relative to the
  other ROIs in the sample.

**Expression near boundaries**
- `roi_pair_interface_response.csv` — directional ordered-ROI-pair
  contrast between spots near the shared boundary and spots farther
  within the responder ROI.

**Scale sensitivity**
- `roi_cross_grain_sensitivity.csv` — how each statistic changes when
  observations are aggregated into tiles or hexagons; flags sign
  changes and large magnitude differences.

**Support and provenance**
- `roi_support_status.csv`, `spot_ledger.csv`,
  `component_geometry.csv`, `component_domain_regions.csv`,
  `component_region_replicate_status.csv`, `tessellation_tiles.csv`,
  `consensus_hexes.csv` — exactly which observations contributed to
  each statistic.

**Reports**
- `analysis_report.html`, `expression_summary.pdf`,
  `consensus_tessellation_summary.pdf`, `support_provenance.pdf`,
  `tessellation_densities_*.pdf`, `run_metadata.json`,
  `effective_config.json`.

## Reproducing the paper

The scripts that assemble the paper's figures and the supplementary
data bundles live in the companion repository
[cbib/TileScope-results](https://github.com/cbib/TileScope-results) 
and on the [Zenodo DOI 10.5281/zenodo.23535234](https://zenodo.org/records/23535235).
The four Supplementary Data ZIPs attached to the paper package the
numerical evidence for Figures 2, 3, 4 and the Xenium feasibility
section — reviewers can rebuild each panel's spot/cell values from those
bundles without the raw upstream datasets. End-to-end regeneration
from raw expression values additionally requires the original Visium /
Xenium data, obtained through the accessions listed in the Data
Availability section.

## R interface

The `r/` package is a thin bridge for Seurat users: a single function,
`extract_data_from_seurat_to_csv()`, writes TileScope-ready CSVs from
a Seurat spatial object. The R package deliberately does not wrap the
Python engine — after generating the CSVs, run the `tilescope` CLI,
or call `tilescope.api` from R through `reticulate`. See
[python/tilescope/api/R_INTEGRATION_GUIDE.md](python/tilescope/api/R_INTEGRATION_GUIDE.md).

## Documentation

- [Python package guide](python/tilescope/README.md)
- [Python API](python/tilescope/api/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [CSV export contract](docs/CSV_EXPORT_CONTRACT.md)
- [How to read the HTML report](docs/HOW_TO_READ_THE_REPORT.md)
- [R integration guide](python/tilescope/api/R_INTEGRATION_GUIDE.md)
- [Tests](python/tilescope/tests/README.md)

## Tests

```bash
venv/bin/pytest python/tilescope/tests -q
```

## License

MIT. See [python/tilescope/LICENSE](python/tilescope/LICENSE).
