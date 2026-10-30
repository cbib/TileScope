"""Determinism and translation invariance of the typed ensemble path.

Consumes the canonical `consensus_hexes_df` surface (CORE-OUT-01)
via composite key (sample_id, component_id, hex_id).
"""

import numpy as np

from tilescope.analyzers import run_analysis
from tilescope.cols import (
    COMPONENT_ID, GENE1_DENSITY_ENSEMBLE, GENE2_DENSITY_ENSEMBLE,
    HEX_AREA, HEX_ID,
)
from tilescope.settings import load_run_config
from tilescope.tests.helpers import make_grid_spots


def _config(output_dir):
    return load_run_config(
        profile="standard",
        set_overrides={
            "output_dir": str(output_dir),
            "permutations": 0,
            "tessellation": {"n": 3},
            "consensus": {"min_tess_coverage_fraction": 2.0 / 3.0, "min_spots_per_hex": 3},
        },
    )


def _consensus_fingerprint(results):
    hexes = results["consensus_hexes_df"].sort_values(
        [COMPONENT_ID, HEX_ID], kind="mergesort",
    )
    rows = []
    for _, row in hexes.iterrows():
        rows.append((
            str(row[COMPONENT_ID]),
            int(row[HEX_ID]),
            float(row[HEX_AREA]),
            # Ensemble density (nullable) — coerce NA to NaN for
            # comparability.
            float(row[GENE1_DENSITY_ENSEMBLE])
            if row[GENE1_DENSITY_ENSEMBLE] is not None
            and not (isinstance(row[GENE1_DENSITY_ENSEMBLE], float)
                     and row[GENE1_DENSITY_ENSEMBLE] != row[GENE1_DENSITY_ENSEMBLE])
            else float("nan"),
            float(row[GENE2_DENSITY_ENSEMBLE])
            if row[GENE2_DENSITY_ENSEMBLE] is not None
            and not (isinstance(row[GENE2_DENSITY_ENSEMBLE], float)
                     and row[GENE2_DENSITY_ENSEMBLE] != row[GENE2_DENSITY_ENSEMBLE])
            else float("nan"),
        ))
    return rows


def test_same_seed_is_bit_deterministic(tmp_path):
    spots = make_grid_spots(nx=12, ny=12, n_rois=2, seed=42)
    first = run_analysis(
        spots, _config(tmp_path / "first"),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    second = run_analysis(
        spots, _config(tmp_path / "second"),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    assert _consensus_fingerprint(first) == _consensus_fingerprint(second)


def test_coordinate_translation_preserves_ensemble_values(tmp_path):
    spots = make_grid_spots(nx=12, ny=12, n_rois=2, seed=42)
    translated = spots.copy()
    translated["x"] += 1000.0
    translated["y"] -= 700.0
    original_result = run_analysis(
        spots, _config(tmp_path / "original"),
        feature_a_id="Neuron", feature_b_id="AC",
    )
    translated_result = run_analysis(
        translated, _config(tmp_path / "translated"),
        feature_a_id="Neuron", feature_b_id="AC",
    )

    original = _consensus_fingerprint(original_result)
    shifted = _consensus_fingerprint(translated_result)
    assert [(r[0], r[1]) for r in original] == [(r[0], r[1]) for r in shifted]
    np.testing.assert_allclose(
        [r[2:] for r in original], [r[2:] for r in shifted], rtol=1e-9, atol=1e-9
    )
