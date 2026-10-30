# How to read a TileScope report

Every TileScope run writes an `analysis_report.html` next to its CSVs
and PDFs. The report is a compact reader-facing summary of the same
canonical outputs; every number in it comes from a CSV that ships in
the run directory. This guide explains the vocabulary once so the
report itself does not need to repeat a tutorial.

## Section order

1. **Analysis coverage** — sample-level input / spatial-support /
   tessellation counts, presented alongside an **Observed spatial
   support** orientation panel (embedded from
   `support_provenance.pdf`) that shows where the numbered
   biological ROIs sit and which observed regions contribute to
   spatial analysis. The two render side by side (orientation panel
   on the left, coverage table on the right) so the ROI numbers
   referenced below are readable up front.
2. **Expression summary** (collapsible) — per-ROI spot/cell expression,
   joint highs, and support geometry, embedded from
   `expression_summary.pdf`.
3. **Consensus & tessellation visual summary** — spatial patterns
   across spatially eligible components, embedded from
   `consensus_tessellation_summary.pdf`.
4. **Sensitivity to spatial support** — a short per-ROI reading of
   whether the main relationships persist when observations are grouped
   in different ways.
5. **Detailed support and provenance tables**:
   - **Spatial support by ROI** — connectivity classification of
     every biological ROI.
   - **Construction provenance** (collapsed) — one row per
     spatially supported component.
   - **Analysis settings** (collapsed) — enrichment thresholds and
     eligibility thresholds.
6. **Downloads** — headline canonical CSVs and PDFs; the full run
   directory is one click away.

## Sensitivity to spatial support

Every ROI receives a compact four-row table:

* within-ROI correlation;
* overlap between their high values;
* each program compared with nearby values of the other program.

The columns compare the observed spots, the middle half of results
across alternative tessellations, and the consensus map. Enrichment is
shown as a fold ratio, where `1×` means no enrichment. Other values are
correlation coefficients or Moran's I. An enrichment value of `∞`
means that the program was present while its comparison population was
zero.

The report uses three biological readings:

* **Stable** — the result stays consistent across the observed,
  tessellation and consensus supports.
* **Sensitive to zoning** — results vary meaningfully across alternative
  tessellations; this includes cases where their middle half spans both
  directions.
* **Direction changes** — the observed spots and consensus map have
  clear opposite directions. Tessellations do not trigger this label.

**No clear pattern** means the estimates are weak or do not support one
of those readings. **Not assessable** means a required result was not
available. Neither fallback is interpreted as stability.

Small operational floors prevent movements close to zero from being
called clear patterns: `0.15` for correlation, `0.10` for Moran's I and
`0.08` for enrichment on its bounded, null-centred effect scale. These
are display guides, not significance thresholds. The canonical CSVs
retain the unclassified continuous values.

## Worked example

**262_T, MES-Hyp × AC, ROI 1** (from the Greenwald 36-sample corpus):

* Observed-spot correlation is `−0.32`.
* The middle half of the alternative-tessellation correlations runs
  from `+0.36` to `+0.67`.
* Consensus-map correlation is `+0.83`.

The row reads **Direction changes** because the observed spots and
consensus map have clear opposite directions. The tessellation interval
is supporting context, not the source of that label.

## When the report is not enough

The report deliberately shows a compact subset. For anything below
this bar, open the run directory:

* per-replicate correlations / enrichments →
  `roi_replicate_correlations.csv`, `roi_replicate_enrichment.csv`;
* per-tile / per-hex geometry →
  `tessellation_tiles.csv`, `consensus_hexes.csv`;
* per-spot/cell ledger → `spot_ledger.csv`;
* the raw canonical cross-grain table → `roi_cross_grain_sensitivity.csv`;
* analysis provenance (core version, git SHA, effective config,
  timestamps) → `run_metadata.json` + `effective_config.json`.
