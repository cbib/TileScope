"""
DataFrame column name constants (single source of truth).

Every column appearing in any internal or exported DataFrame is named here.
No user-facing labels, no defaults, no methods.
"""

# --- IDENTIFICATION COLUMNS ---

ROI_ID = "roi_id"
G1_ID = "g1_id"
G2_ID = "g2_id"

# ROI_ID is the biological source-ROI (never overwritten); COMPONENT_ID is
# the geometric analysis unit from connected-component splitting.
COMPONENT_ID = "component_id"

# Assigned once at run start (int64, unique, 0..N-1); lets downstream stages
# identify supported vs ineligible rows by set difference. Run-local only.
SOURCE_ROW_ID = "source_row_id"

TESSELLATION_ID = "tessellation_id"
TILE_ID = "tile_id"

# SAMPLE_ID is a foreign key on tessellation frames; DOMAIN_REGION_ID keys
# region-level tables; REPLICATE_INDEX keys the replicate-level tables.
SAMPLE_ID = "sample_id"
DOMAIN_REGION_ID = "domain_region_id"
REPLICATE_INDEX = "replicate_index"

# --- INPUT DATA COLUMNS ---

COORD_X = "x"
COORD_Y = "y"

# Users provide pre-normalized data in these columns (default workflow).
GENE1 = "gene1"
GENE2 = "gene2"

# --- COMPUTED METRICS COLUMNS ---

# Distributional companions (robust effect sizes)
LOG2FC_G1 = "log2fc_g1"
LOG2FC_G2 = "log2fc_g2"
CLES_G1 = "CLES_g1"
CLES_G2 = "CLES_g2"

# Correlation statistics
CORR_OBS = "corr_obs"
CORR_P_ANALYTIC = "corr_p_analytic"
CORR_PERM_MEAN = "corr_perm_mean"
CORR_PERM_SD = "corr_perm_sd"
CORR_P_PERM = "corr_p_perm"

# BH q-values + conditional analytic p on ROI-grain correlation surfaces.
# `_conditional` variants belong on per-replicate rows (inference conditional on the drawn tessellation).
CORR_Q_ANALYTIC = "corr_q_analytic"
CORR_Q_PERM = "corr_q_perm"
CORR_P_ANALYTIC_CONDITIONAL = "corr_p_analytic_conditional"
CORR_Q_ANALYTIC_CONDITIONAL = "corr_q_analytic_conditional"
FDR_FAMILY_SIZE = "fdr_family_size"

# Reduced-estimator permutation null on roi_tessellation_correlations:
# two-tailed bounded p over synchronized-Gene-2 permutations.
CORR_P_PERM_REDUCED = "corr_p_perm_reduced"
CORR_Q_PERM_REDUCED = "corr_q_perm_reduced"
N_PERMUTATIONS_REDUCED = "n_permutations_reduced"
N_VALID_PERMUTATIONS = "n_valid_permutations"

# --- COMPONENT-DOMAIN TESSELLATION PUBLIC SURFACE ---

# Single naming authority for the frames produced by `tessellation_outcome_frames`,
# so consumers and exporters never hard-code the strings.

# Component-level (merged into component_geometry.csv):
N_TESSELLATION_ELIGIBLE_REGIONS = "n_tessellation_eligible_regions"
N_SPOTS_IN_TESSELLATION_ELIGIBLE_REGIONS = "n_spots_in_tessellation_eligible_regions"
N_SPOTS_EXCLUDED_BY_TESSELLATION_ELIGIBILITY = (
    "n_spots_excluded_by_tessellation_eligibility"
)
TESSELLATED_DOMAIN_AREA = "tessellated_domain_area"

# Region-level (merged into component_domain_regions.csv):
TESSELLATION_ELIGIBLE = "tessellation_eligible"
TESSELLATION_INELIGIBILITY_REASON = "tessellation_ineligibility_reason"

# Region-replicate execution:
REGION_STATUS = "region_status"

# Component-replicate correlation. Composite key = (SAMPLE_ID, COMPONENT_ID,
# REPLICATE_INDEX); ROI_ID is a foreign key, not part of the primary key.
N_ELIGIBLE_REGIONS = "n_eligible_regions"
N_CONTRIBUTING_REGIONS = "n_contributing_regions"
CORRELATION_VALUE = "correlation_value"
MODE = "mode"  # currently the constant string "pearson"

# --- METADATA / PROVENANCE COLUMNS ---

# Sample sizes. N_SPOTS is the general canonical count in enrichment tables;
# N_ROI is the spec-lock name on the correlation surface (see below).
N_ROI = "n_roi"
N_CTX = "n_ctx"
# Correlation metadata: number of (gene1, gene2) pairs used.
SAMPLE_SIZE = "sample_size"
N_TILES = "n_tiles"

N_PERMUTATIONS = "n_permutations"
PERM_SEED = "perm_seed"

# Parent-ROI eligibility gates recorded on parent tables so a reader can
# reproduce the gate decision without reaching back into the run config.
MIN_ROI_SPOTS = "min_roi_spots"
MIN_CONTEXT_SPOTS = "min_context_spots"

# Typed status columns (enums in analyzers/parent_roi_outcomes.py). Every parent-ROI
# table is total over source parents; the status disambiguates FINITE from every branch.
PARENT_CORRELATION_STATUS = "parent_correlation_status"
PARENT_ENRICHMENT_STATUS = "parent_enrichment_status"
PARENT_EFFECT_SIZE_STATUS = "parent_effect_size_status"
PARENT_TESSELLATION_CORRELATION_STATUS = (
    "parent_tessellation_correlation_status"
)
PARENT_REPLICATE_CORRELATION_STATUS = "parent_replicate_correlation_status"
PARENT_REPLICATE_ENRICHMENT_STATUS = "parent_replicate_enrichment_status"
PARENT_REPLICATE_EFFECT_SIZE_STATUS = "parent_replicate_effect_size_status"
PARENT_TESSELLATION_ENRICHMENT_STATUS = (
    "parent_tessellation_enrichment_status"
)
PARENT_TESSELLATION_EFFECT_SIZE_STATUS = (
    "parent_tessellation_effect_size_status"
)
PARENT_CONSENSUS_CORRELATION_STATUS = "parent_consensus_correlation_status"
PARENT_CONSENSUS_ENRICHMENT_STATUS = "parent_consensus_enrichment_status"
PARENT_CONSENSUS_EFFECT_SIZE_STATUS = "parent_consensus_effect_size_status"
PARENT_MORAN_STATUS = "parent_moran_status"

# --- ROI CONSENSUS + ROI MORAN SURFACES ---

# `value_surface` = PartitionValueSurface enum (voronoi_tile_mean / consensus_ensemble / consensus_direct);
# `moran_direction` = `g1_lag_g2` or `g2_lag_g1`.
VALUE_SURFACE = "value_surface"
MORAN_DIRECTION = "moran_direction"
MORAN_VALUE = "moran_value"
MORAN_REASON = "moran_reason"
N_HEXES_POOLED = "n_hexes_pooled"
N_HEXES_WITH_FINITE_DENSITY_POOLED = "n_hexes_with_finite_density_pooled"
N_CONTRIBUTING_COMPONENTS_CONSENSUS = "n_contributing_components_consensus"
N_ELIGIBLE_CONSENSUS_COMPONENTS = "n_eligible_consensus_components"
N_CONTEXT_HEXES_POOLED = "n_context_hexes_pooled"
N_CONTEXT_HEXES_WITH_FINITE_DENSITY_POOLED = (
    "n_context_hexes_with_finite_density_pooled"
)
N_CONTEXT_ELIGIBLE_CONSENSUS_COMPONENTS = (
    "n_context_eligible_consensus_components"
)
N_CONTEXT_CONTRIBUTING_COMPONENTS_CONSENSUS = (
    "n_context_contributing_components_consensus"
)

# n_units = value carriers per Moran computation (spots/hexes/tiles; tessellation reports the replicate median).
# n_adjacency_entries = nonzero directed entries in the row-standardized weight matrix (undirected edge = two entries).
N_UNITS = "n_units"
N_ADJACENCY_ENTRIES = "n_adjacency_entries"
N_ADJACENCY_SUPPORTED_UNITS = "n_adjacency_supported_units"
ADJACENCY_SUPPORTED_FRACTION = "adjacency_supported_fraction"

# Source-grain Moran extras.
N_FINITE_SPOTS = "n_finite_spots"

# Tessellation-grain Moran replicate-reduction fields. moran_value is the
# median across replicates; P25 / P75 report spread.
N_REPLICATES = "n_replicates"
N_FINITE_REPLICATES = "n_finite_replicates"
MORAN_VALUE_P25_ACROSS_REPLICATES = "moran_value_p25_across_replicates"
MORAN_VALUE_P75_ACROSS_REPLICATES = "moran_value_p75_across_replicates"

# --- TESSELLATION-GRAIN CORRELATION SUMMARY (roi_tessellation_correlations.csv) ---

# corr_median = median across finite replicates; q25/q75 = spread on the identity effect scale.
# n_tiles_contributed_median = replicate-median tile count; n_planned_replicates = requested budget.
CORR_MEDIAN = "corr_median"
CORR_Q25 = "corr_q25"
CORR_Q75 = "corr_q75"
N_PLANNED_REPLICATES = "n_planned_replicates"
N_TILES_CONTRIBUTED_MEDIAN = "n_tiles_contributed_median"

# --- ROI x REPLICATE CORRELATION TABLE (roi_replicate_correlations.csv) ---

# Discrete spot + region + component coverage counts; no polygonal area
# quantities on the reporting surface.
N_SOURCE_SPOTS = "n_source_spots"
N_ELIGIBLE_SPOTS = "n_eligible_spots"
N_CONTRIBUTED_SPOTS = "n_contributed_spots"
N_ELIGIBLE_COMPONENTS = "n_eligible_components"
N_CONTRIBUTING_COMPONENTS = "n_contributing_components"
N_TILES_CONTRIBUTED = "n_tiles_contributed"
ELIGIBLE_SPOT_COVERAGE_FRACTION = "eligible_spot_coverage_fraction"
SOURCE_ROI_SPOT_COVERAGE_FRACTION = "source_roi_spot_coverage_fraction"

# --- ROI x REPLICATE ENRICHMENT TABLE (roi_replicate_enrichment.csv) ---

# For each (roi, replicate) row the context is the union of tiles from every
# OTHER biological ROI at the SAME replicate. Every row emits both sides.
N_CONTEXT_SOURCE_SPOTS = "n_context_source_spots"
N_CONTEXT_ELIGIBLE_SPOTS = "n_context_eligible_spots"
N_CONTEXT_CONTRIBUTED_SPOTS = "n_context_contributed_spots"
N_CONTEXT_ELIGIBLE_COMPONENTS = "n_context_eligible_components"
N_CONTEXT_CONTRIBUTING_COMPONENTS = "n_context_contributing_components"
N_CONTEXT_ELIGIBLE_REGIONS = "n_context_eligible_regions"
N_CONTEXT_CONTRIBUTING_REGIONS = "n_context_contributing_regions"
N_CONTEXT_TILES_CONTRIBUTED = "n_context_tiles_contributed"
CONTEXT_ELIGIBLE_SPOT_COVERAGE_FRACTION = (
    "context_eligible_spot_coverage_fraction"
)
CONTEXT_SOURCE_ROI_SPOT_COVERAGE_FRACTION = (
    "context_source_roi_spot_coverage_fraction"
)

# --- CANONICAL CONSENSUS SURFACE COLUMNS (consensus_hexes.csv) ---
# See docs/CSV_EXPORT_CONTRACT.md.

HEX_ID = "hex_id"
HEX_AREA = "hex_area"
HEX_GEOMETRY_WKT = "hex_geometry_wkt"
LATTICE_COL = "lattice_col"
LATTICE_ROW = "lattice_row"
CENTER_X = "center_x"
CENTER_Y = "center_y"

# `obs_equiv_*_mass` = raw ensemble accumulator / `n_tessellations_covered` (publishing both would duplicate authorities).
# `n_*` stays reserved for literal integer counts; direct-side fields follow the same convention. See docs/CSV_EXPORT_CONTRACT.md.
OBS_EQUIV_SPOT_MASS = "obs_equiv_spot_mass"
OBS_EQUIV_G1_SUM_MASS = "obs_equiv_g1_sum_mass"
OBS_EQUIV_G2_SUM_MASS = "obs_equiv_g2_sum_mass"
OBS_EQUIV_G1_HIGH_MASS = "obs_equiv_g1_high_mass"
OBS_EQUIV_G2_HIGH_MASS = "obs_equiv_g2_high_mass"
OBS_EQUIV_BOTH_HIGH_MASS = "obs_equiv_both_high_mass"
OBS_EQUIV_G1_LOW_MASS = "obs_equiv_g1_low_mass"
OBS_EQUIV_G2_LOW_MASS = "obs_equiv_g2_low_mass"
OBS_EQUIV_BOTH_LOW_MASS = "obs_equiv_both_low_mass"

# Context-side masses on roi_consensus_enrichment.csv (pooled from every OTHER ROI's eligible sibling components).
# Ratio reconstruction: target masses / `obs_equiv_spot_mass`, context masses / `context_obs_equiv_spot_mass`.
CONTEXT_OBS_EQUIV_SPOT_MASS = "context_obs_equiv_spot_mass"
CONTEXT_OBS_EQUIV_G1_SUM_MASS = "context_obs_equiv_g1_sum_mass"
CONTEXT_OBS_EQUIV_G2_SUM_MASS = "context_obs_equiv_g2_sum_mass"
CONTEXT_OBS_EQUIV_G1_HIGH_MASS = "context_obs_equiv_g1_high_mass"
CONTEXT_OBS_EQUIV_G2_HIGH_MASS = "context_obs_equiv_g2_high_mass"
CONTEXT_OBS_EQUIV_BOTH_HIGH_MASS = "context_obs_equiv_both_high_mass"
CONTEXT_OBS_EQUIV_G1_LOW_MASS = "context_obs_equiv_g1_low_mass"
CONTEXT_OBS_EQUIV_G2_LOW_MASS = "context_obs_equiv_g2_low_mass"
CONTEXT_OBS_EQUIV_BOTH_LOW_MASS = "context_obs_equiv_both_low_mass"
GENE1_DENSITY_ENSEMBLE = "gene1_density_ensemble"
GENE2_DENSITY_ENSEMBLE = "gene2_density_ensemble"
N_SPOTS_DIRECT = "n_spots_direct"
SUM_G1_DIRECT = "sum_g1_direct"
SUM_G2_DIRECT = "sum_g2_direct"
GENE1_DENSITY_DIRECT = "gene1_density_direct"
GENE2_DENSITY_DIRECT = "gene2_density_direct"

# --- PER-HEX ENRICHMENT SURFACE (consensus_hexes.csv) ---

# Effect-only, ROI-context-derived: target = per-hex ensemble mass / spot mass;
# context = same ratio pooled from roi_consensus_enrichment.csv for the parent ROI.

# `hex_enrichment_status` is a derived diagnostic projection (structural state + the four
# per-ratio statuses); numeric hex ratios are gated by their own per-ratio status column.
HEX_ENRICHMENT_STATUS = "hex_enrichment_status"
HEX_G1_ENRICHMENT = "hex_g1_enrichment"
HEX_G2_ENRICHMENT = "hex_g2_enrichment"
HEX_JOINT_HIGH_ENRICHMENT_FC = "hex_joint_high_enrichment_fc"
HEX_JOINT_LOW_ENRICHMENT_FC = "hex_joint_low_enrichment_fc"
# Per-ratio primitive statuses at the hex grain: the authoritative gate
# for the four numeric hex ratios.
HEX_G1_ENRICHMENT_STATUS = "hex_g1_enrichment_status"
HEX_G2_ENRICHMENT_STATUS = "hex_g2_enrichment_status"
HEX_JOINT_HIGH_ENRICHMENT_STATUS = "hex_joint_high_enrichment_status"
HEX_JOINT_LOW_ENRICHMENT_STATUS = "hex_joint_low_enrichment_status"
HEX_ENRICHMENT_RATIO_STATUS_PAIRS = (
    (HEX_G1_ENRICHMENT, HEX_G1_ENRICHMENT_STATUS),
    (HEX_G2_ENRICHMENT, HEX_G2_ENRICHMENT_STATUS),
    (HEX_JOINT_HIGH_ENRICHMENT_FC, HEX_JOINT_HIGH_ENRICHMENT_STATUS),
    (HEX_JOINT_LOW_ENRICHMENT_FC, HEX_JOINT_LOW_ENRICHMENT_STATUS),
)
HEX_ENRICHMENT_RATIO_STATUS_COLUMNS = tuple(
    status_col for _, status_col in HEX_ENRICHMENT_RATIO_STATUS_PAIRS
)

# --- TESSELLATION METADATA ---
TILE_AREA = "tile_area"
TILE_GEOMETRY_WKT = "tile_geometry_wkt"

# Density columns (generic; used for tiles and hexagons)
GENE1_DENSITY = "g1_density"
GENE2_DENSITY = "g2_density"
GENE1_DENSITY_STABILITY = "g1_density_stability"
GENE2_DENSITY_STABILITY = "g2_density_stability"

# Geometry columns
CENTER_X = "center_x"
CENTER_Y = "center_y"

# Consensus metadata
N_TESSELLATIONS_COVERED = "n_tessellations_covered"

# --- SUFFICIENT STATISTICS FOR PERCENTILE-BASED ENRICHMENT (tile level) ---

# N_SPOTS is the general canonical count; N_ROI is the spec-lock name on the correlation surface.
N_SPOTS = "n_spots"
SUM_G1 = "sum_g1"
SUM_G2 = "sum_g2"
N_G1_HIGH = "n_g1_high"
N_G2_HIGH = "n_g2_high"
N_BOTH_HIGH = "n_both_high"
N_G1_LOW = "n_g1_low"
N_G2_LOW = "n_g2_low"
N_BOTH_LOW = "n_both_low"

# Enrichment ratios (percentile-based fold-changes)
JOINT_HIGH_ENRICHMENT_FC = "joint_high_enrichment_fc"
JOINT_LOW_ENRICHMENT_FC = "joint_low_enrichment_fc"
GENE1_ENRICHMENT = "g1_enrichment"
GENE2_ENRICHMENT = "g2_enrichment"

# Per-ratio primitive status columns (values: EnrichmentRatioStatus).
GENE1_ENRICHMENT_STATUS = "g1_enrichment_status"
GENE2_ENRICHMENT_STATUS = "g2_enrichment_status"
JOINT_HIGH_ENRICHMENT_STATUS = "joint_high_enrichment_status"
JOINT_LOW_ENRICHMENT_STATUS = "joint_low_enrichment_status"

# Shared (value, status) pairing consumed by every caller of the enrichment
# primitive. Single source of truth; do not duplicate this tuple.
ENRICHMENT_RATIO_STATUS_PAIRS = (
    (GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS),
    (GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS),
    (JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS),
    (JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS),
)
ENRICHMENT_RATIO_STATUS_COLUMNS = tuple(
    status_col for _, status_col in ENRICHMENT_RATIO_STATUS_PAIRS
)

# Enrichment classification (primary label column)
ENRICHMENT_DIRECTION = "enrichment_direction"

# Sample-wide percentile thresholds
Q_G1_THRESHOLD = "q_g1_threshold"
Q_G2_THRESHOLD = "q_g2_threshold"
Q_G1_LOW_THRESHOLD = "q_g1_low_threshold"
Q_G2_LOW_THRESHOLD = "q_g2_low_threshold"

# --- COMPONENT-GRAIN RADIAL SHELL CROSS-CORRELOGRAM COLUMNS ---
SHELL_BIN_INDEX = "shell_bin_index"
SHELL_EDGE_LOWER_NORMALIZED = "shell_edge_lower_normalized"
SHELL_EDGE_UPPER_NORMALIZED = "shell_edge_upper_normalized"
SHELL_MIDPOINT_NORMALIZED = "shell_midpoint_normalized"
SHELL_SYMMETRIC = "shell_symmetric"
COMPONENT_RADIAL_SHELL_STATUS = "component_radial_shell_status"
N_ANCHORS_PER_SHELL = "n_anchors_per_shell"
FRACTION_ANCHORS_PER_SHELL = "fraction_anchors_per_shell"
N_UNORDERED_PAIRS_PER_SHELL = "n_unordered_pairs_per_shell"
MEDIAN_NN_SPACING = "median_nn_spacing"
COMPONENT_RHO_0 = "component_rho_0"

__all__ = [name for name in globals() if name.isupper()]
