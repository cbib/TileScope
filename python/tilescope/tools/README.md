# Companion tool

TileScope currently ships one standalone screening utility:
`tilescope.tools.tilescope_enrichment_screening`.

## Purpose

The tool screens any number of expression columns for ROI-versus-other-ROI
enrichment. It uses the main TileScope robust log2-fold-change primitive but
does not run tessellation, consensus, Moran, or pairwise spatial analysis.

Use it for exploratory gene selection, not as a substitute for the full
spatial pipeline.

## Input

The CSV contains:

- `x`, `y`,
- `roi_id`,
- one or more numeric gene/feature columns.

## Run

```bash
python -m tilescope.tools.tilescope_enrichment_screening \
  --input data.csv \
  --output screening/
```

Options:

```bash
python -m tilescope.tools.tilescope_enrichment_screening --help

python -m tilescope.tools.tilescope_enrichment_screening \
  --input data.csv \
  --output screening/ \
  --min-expression 0.5 \
  --format detailed,wide
```

## Outputs

- `gene_enrichment_detailed.csv`: one row per ROI × feature.
- `gene_enrichment_lists.csv`: semicolon-separated feature lists by verdict.
- `gene_enrichment_counts.csv`: verdict counts per ROI.

Files are written only for the requested formats.

## Verdicts

Verdicts use the fold-change thresholds from TileScope:

| Verdict | Fold change |
|---|---:|
| `strong_enriched` | ≥ 1.50 |
| `weak_enriched` | ≥ 1.20 and < 1.50 |
| `neutral` | > 0.80 and < 1.20 |
| `weak_depleted` | > 0.67 and ≤ 0.80 |
| `strong_depleted` | ≤ 0.67 |

The context for a target ROI is the union of all other ROIs.

## Test fixture

`python/tilescope/tests/fixtures/tilescope_mock_5rois_100genes.csv`
provides a small multi-gene screening fixture.
