# Package history

TileScope began as an R-only prototype, preserved under the Git tag
`r-prototype-final`. The prototype's WSD workflow, storage container,
categorization / colocalization helpers, and visualization functions
are archived under that tag; they are not part of the current package.

The current analytical engine is the Python `tilescope` package. The
tracked R package is now a thin bridge that exports pairwise input CSVs
from a Seurat object via `extract_data_from_seurat_to_csv()`; see the
package vignette and `python/tilescope/api/R_INTEGRATION_GUIDE.md` for
the end-to-end path.
