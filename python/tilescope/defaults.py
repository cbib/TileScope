"""Runtime defaults for TileScope configuration.

The versioned analysis policy owns method-definition values; runtime, reporting, and bounded eligibility knobs remain configurable. Column names and labels live in cols.py and user_configs/dataset.schema.json.
"""

import json
import datetime
from pathlib import Path
from typing import Final, Dict, Any, Optional


# Canonical location for the packaged run.config.json path; other modules
# must import _PACKAGED_CONFIG instead of recomputing the path.
def load_packaged_defaults() -> Dict[str, Any]:
    """Load packaged default configuration from JSON."""
    defaults_path = Path(__file__).parent / "tilescope_defaults" / "run.config.json"
    with open(defaults_path, 'r') as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith('_')}


_PACKAGED_CONFIG: Final[Dict[str, Any]] = load_packaged_defaults()


def load_profile(profile_name: str) -> Dict[str, Any]:
    """Load a packaged profile configuration (exploratory, standard, debugging)."""
    profile_path = Path(__file__).parent / "tilescope_defaults" / f"{profile_name}_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile '{profile_name}' not found at {profile_path}")

    with open(profile_path, 'r') as f:
        config = json.load(f)

    return {k: v for k, v in config.items() if not k.startswith('_')}


def dump_effective_config(config: Dict[str, Any], output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Write fully-resolved effective configuration to file for reproducibility."""
    effective_config = {
        "_metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "description": "Effective TileScope configuration (fully resolved)",
            **(metadata or {})
        },
        **config
    }

    with open(output_path, 'w') as f:
        json.dump(effective_config, f, indent=2)

    print(f"[CONFIG] Effective configuration written to: {output_path}")


# Non-configurable algorithm constants (not user tunables).

ANALYSIS_POLICY_VERSION: Final[str] = "standard-v3"
STANDARD_V3_FROZEN_VALUES: Final[Dict[str, Any]] = {
    "analysis_policy_version": ANALYSIS_POLICY_VERSION,
    "thresholds.q_high": 0.75,
    "thresholds.q_low": 0.25,
    "consensus.occupancy_safety": 1.2,
    "consensus.area_sizing_method": "winsor_mean",
    "consensus.winsor_limits": [0.0, 0.98],
    "consensus.base_area_quantile": 0.5,
    "consensus.max_hex_area_fraction_of_roi": 0.25,
    "consensus.bivariate_morans.compute": True,
    "consensus.bivariate_morans.k_neighbors": 6,
    "consensus.min_hex_roi_overlap_fraction": None,
    "consensus.adaptive_overlap.small_roi_max_spots": 150,
    "consensus.adaptive_overlap.medium_roi_max_spots": 400,
    "consensus.adaptive_overlap.small_roi_overlap": 0.04,
    "consensus.adaptive_overlap.medium_roi_overlap": 0.05,
    "consensus.adaptive_overlap.large_roi_overlap": 0.07,
    "component_domain_radii.component_support_radius_factor": 0.6,
    "component_domain_radii.acquisition_support_radius_factor": 0.6,
    "component_domain_radii.ineligible_spot_exclusion_radius_factor": 0.6,
    # Radial shell edges in units of the component's median nearest-neighbour
    # spacing; a shell contributes only when both anchor floors hold.
    "radial_shell.shell_edges_normalized": [0.5, 1.5, 2.5, 4.5, 8.5],
    "radial_shell.min_anchors_per_shell": 10,
    "radial_shell.min_anchor_fraction": 0.5,
}

EPSILON: Final[float] = 1e-6  # Zero-division / float-comparison tolerance.

CONSENSUS_HEX_SCALING_FACTOR: Final[float] = 1.0  # Hex radius multiplier (1.0 = no scaling).

# Canvas visualization defaults (derived from data, not in JSON).
DEFAULT_TARGET_PX_PER_NN: Final[float] = 30.0
DEFAULT_TARGET_PX_PER_STEP: Final[int] = 1
DEFAULT_CANVAS_MARGIN_PX: Final[int] = 20

# Normalization parameter dict keys (internal, not configurable).
NORM_GENE1_MIN: Final[str] = "gene1_min"
NORM_GENE2_MIN: Final[str] = "gene2_min"

# Log2 fold-change computation thresholds.
DEFAULT_TRIM_PROPORTION: Final[float] = 0.08  # 8% per tail.
FC_SMALL_SAMPLE_THRESHOLD: Final[int] = 10  # Use trimmed mean if n < 10.
FC_WINSORIZE_THRESHOLD: Final[int] = 20  # Use winsorized mean if 10 <= n < 20.

# Sliver filter thresholds (combined d1-based + median-area).
SLIVER_TAU_D1: Final[float] = 0.03  # Primary: area < tau_d1 * d1^2 (scale-invariant).
SLIVER_TAU_MED: Final[float] = 0.01  # Safety net: area < tau_med * median_area.

# Tessellation validation thresholds (Canvas-based).
VALIDATION_BOUNDARY_THRESH_MULT: Final[float] = 1e-3  # Boundary suspect: gap < mult * d1.
VALIDATION_MAX_BOUNDARY_FRAC: Final[float] = 0.02  # Max fraction of boundary-suspect points.

# ROI overlap detection thresholds (filter out negligible geometric artifacts).
OVERLAP_MIN_AREA: Final[float] = 1.0
OVERLAP_MIN_FRAC: Final[float] = 0.001

# Enrichment classification thresholds: 5-level (strong/weak enriched, neutral, weak/strong depleted).
DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD: Final[float] = 1.5
DEFAULT_FC_WEAK_ENRICHMENT_THRESHOLD: Final[float] = 1.2
DEFAULT_FC_WEAK_DEPLETION_THRESHOLD: Final[float] = 0.8
DEFAULT_FC_STRONG_DEPLETION_THRESHOLD: Final[float] = 0.67

DEFAULT_CLES_STRONG_ENRICHMENT_THRESHOLD: Final[float] = 0.6
DEFAULT_CLES_WEAK_ENRICHMENT_THRESHOLD: Final[float] = 0.55
DEFAULT_CLES_WEAK_DEPLETION_THRESHOLD: Final[float] = 0.45
DEFAULT_CLES_STRONG_DEPLETION_THRESHOLD: Final[float] = 0.4

DEFAULT_LOG2FC_STRONG_ENRICHMENT_THRESHOLD: Final[float] = 1.5
DEFAULT_LOG2FC_WEAK_ENRICHMENT_THRESHOLD: Final[float] = 1.0
DEFAULT_LOG2FC_WEAK_DEPLETION_THRESHOLD: Final[float] = -1.0
DEFAULT_LOG2FC_STRONG_DEPLETION_THRESHOLD: Final[float] = -1.5

# Configurable defaults, computed from packaged JSON. Used as function default
# arguments and by tests that call functions directly.
DEFAULT_MIN_ROI_SPOTS: Final[int] = _PACKAGED_CONFIG["roi_filters"]["min_roi_spots"]
DEFAULT_MIN_CONTEXT_SPOTS: Final[int] = _PACKAGED_CONFIG["roi_filters"]["min_context_spots"]
DEFAULT_N_TESSELLATIONS: Final[int] = _PACKAGED_CONFIG["tessellation"]["n"]
DEFAULT_PERMUTATIONS: Final[int] = _PACKAGED_CONFIG["permutations"]

# Frozen policy defaults sourced from STANDARD_V3_FROZEN_VALUES so any change
# to frozen values happens in one place.
DEFAULT_Q_PERCENTILE: Final[float] = STANDARD_V3_FROZEN_VALUES["thresholds.q_high"]
DEFAULT_Q_LOW_PERCENTILE: Final[float] = STANDARD_V3_FROZEN_VALUES["thresholds.q_low"]
DEFAULT_MIN_COMPONENT_SPOTS: Final[int] = 15

# Frozen tessellation support policy. Production analysis reads these values
# from RunConfigModel.tessellation; direct low-level callers use these defaults.
SUPPORT_POLICY_VERSION: Final[str] = "technology-support-v1"
DEFAULT_VORONOI_FRACTION: Final[float] = 0.20
DEFAULT_MIN_TILE_SPOTS: Final[int] = 3
SEED_COUNT_FORMULA: Final[str] = (
    "max(1, round(n_region_observations / target_observations_per_tile))"
)

__all__ = [
    "load_profile",
    "dump_effective_config",
    "EPSILON",
    "CONSENSUS_HEX_SCALING_FACTOR",
    "DEFAULT_TARGET_PX_PER_NN",
    "DEFAULT_TARGET_PX_PER_STEP",
    "DEFAULT_CANVAS_MARGIN_PX",
    "NORM_GENE1_MIN",
    "NORM_GENE2_MIN",
    "DEFAULT_TRIM_PROPORTION",
    "FC_SMALL_SAMPLE_THRESHOLD",
    "FC_WINSORIZE_THRESHOLD",
    "SLIVER_TAU_D1",
    "SLIVER_TAU_MED",
    "VALIDATION_BOUNDARY_THRESH_MULT",
    "VALIDATION_MAX_BOUNDARY_FRAC",
    "OVERLAP_MIN_AREA",
    "OVERLAP_MIN_FRAC",
    "DEFAULT_FC_STRONG_ENRICHMENT_THRESHOLD",
    "DEFAULT_FC_WEAK_ENRICHMENT_THRESHOLD",
    "DEFAULT_FC_WEAK_DEPLETION_THRESHOLD",
    "DEFAULT_FC_STRONG_DEPLETION_THRESHOLD",
    "DEFAULT_CLES_STRONG_ENRICHMENT_THRESHOLD",
    "DEFAULT_CLES_WEAK_ENRICHMENT_THRESHOLD",
    "DEFAULT_CLES_WEAK_DEPLETION_THRESHOLD",
    "DEFAULT_CLES_STRONG_DEPLETION_THRESHOLD",
    "DEFAULT_LOG2FC_STRONG_ENRICHMENT_THRESHOLD",
    "DEFAULT_LOG2FC_WEAK_ENRICHMENT_THRESHOLD",
    "DEFAULT_LOG2FC_WEAK_DEPLETION_THRESHOLD",
    "DEFAULT_LOG2FC_STRONG_DEPLETION_THRESHOLD",
    "ANALYSIS_POLICY_VERSION",
    "STANDARD_V3_FROZEN_VALUES",
    "DEFAULT_MIN_TILE_SPOTS",
    "DEFAULT_Q_PERCENTILE",
    "DEFAULT_Q_LOW_PERCENTILE",
    "DEFAULT_MIN_ROI_SPOTS",
    "DEFAULT_MIN_CONTEXT_SPOTS",
    "DEFAULT_MIN_COMPONENT_SPOTS",
    "DEFAULT_N_TESSELLATIONS",
    "DEFAULT_VORONOI_FRACTION",
    "DEFAULT_PERMUTATIONS",
    "SUPPORT_POLICY_VERSION",
    "SEED_COUNT_FORMULA",
]
