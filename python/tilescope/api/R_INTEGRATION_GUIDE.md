# R integration

TileScope has one Python analysis engine. The tracked R package prepares
Seurat-derived inputs; it does not wrap execution of the Python pipeline.

## ROI identity

Every TileScope input row requires a biological `roi_id`: the fixed spatial
cluster/compartment label used for analysis. Do not substitute
`Seurat::orig.ident`, which usually identifies the sample.

## Export from Seurat, then run the CLI

The R package exports pairwise CSVs through
`extract_data_from_seurat_to_csv()`:

```r
library(TileScope)

extract_data_from_seurat_to_csv(
  seurat_obj = object,
  sample_id = "sample_1",
  assay = "Spatial",
  layer = "data",
  metadata_col = "spatial_cluster",
  input_data = c("GFAP", "PDGFRA"),
  out_path = "tilescope_inputs/"
)
```

Then run:

```bash
tilescope tilescope_inputs/sample_1_expression_data_GFAP_PDGFRA.csv \
  --profile standard \
  --output results/  \
  --feature-a-id GFAP \
  --feature-b-id PDGFRA
```

## Direct reticulate call

```r
library(reticulate)

use_python("/path/to/TileScope/venv/bin/python", required = TRUE)
stopifnot(py_module_available("tilescope.api"))

api <- import("tilescope.api")
results <- api$run_tilescope_from_csv(
  input_csv = "data.csv",
  profile = "standard",
  export = FALSE,
  return_minimal = TRUE,
  feature_a_id = "GFAP",
  feature_b_id = "PDGFRA"
)

roi_correlation <- as.data.frame(results$roi_correlations)
roi_consensus <- as.data.frame(results$roi_consensus_correlations_df)
```

Use integer literals such as `10L` for Python integer configuration values.
Convert only the result frames needed in R; tile and hex frames may be large.

## Joining results to Seurat

Join ROI-grain results through the same metadata column supplied as
`metadata_col` during export:

```r
roi_result <- as.data.frame(results$roi_correlations)
roi_key <- as.character(object@meta.data$spatial_cluster)

object$tilescope_correlation <- roi_result$correlation_value[
  match(roi_key, as.character(roi_result$roi_id))
]
```

When a table contains `value_surface` or `moran_direction`, filter those
dimensions before joining so each ROI has one selected row.

The complete Python result surface is listed in [README.md](README.md).
