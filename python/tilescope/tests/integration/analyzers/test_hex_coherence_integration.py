"""Typed consensus hex identity, geometry, and primitive coherence."""

import numpy as np

from tilescope.analyzers.consensus_stage import run_consensus_stage
from tilescope.spatial import make_canvas
from tilescope.settings import load_run_config
from tilescope.tests.integration.analyzers.test_tessellation_public_surface_chain import (
    _build_frames,
    _run_chain,
)


def test_typed_consensus_hexes_are_coherent():
    sample_tessellation_result, _ = _run_chain()
    source_df, _, _ = _build_frames()
    config = load_run_config(
        profile="standard",
        set_overrides={
        'output_dir': 'tests/tests_output',
        'consensus': {
            'min_tess_coverage_fraction': 0.1,
            'min_spots_per_hex': 1,
        },
    },
    )
    result = run_consensus_stage(
        sample_tessellation_result=sample_tessellation_result,
        canvas=make_canvas(source_df),
        sample_anchor=(
            float(source_df["x"].min()),
            float(source_df["y"].min()),
        ),
        consensus_config=config,
    )

    assert result.components
    for component_id, component in result.components.items():
        hex_ids = [hex_record.hex_id for hex_record in component.hexes]
        assert len(hex_ids) == len(set(hex_ids))
        domain = sample_tessellation_result.components[
            component_id
        ].component_data.polygon
        for hex_record in component.hexes:
            assert hex_record.polygon.is_valid
            assert not hex_record.polygon.is_empty
            assert domain.covers(hex_record.polygon.representative_point())
            assert hex_record.n_spots_direct >= 0
            assert hex_record.n_spots_ensemble_weighted >= 0
