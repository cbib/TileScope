"""Grain identity contract on ``ParentCorrelationSchema`` — the
correlation surface is ROI-grain only and requires ROI_ID on every
row. The retired component-grain metrics-schema cases went with
their schemas in CORE-HYG-04 slice D.
"""
from __future__ import annotations

from tilescope.cols import SAMPLE_ID, MODE, N_PERMUTATIONS, MIN_ROI_SPOTS, PARENT_CORRELATION_STATUS


import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from tilescope.cols import (
    G1_ID, G2_ID, ROI_ID,
    CORR_OBS, CORR_P_ANALYTIC, CORR_PERM_MEAN, CORR_PERM_SD, CORR_P_PERM,
    CORR_Q_ANALYTIC, CORR_Q_PERM, FDR_FAMILY_SIZE,
    SAMPLE_SIZE, N_ROI,
)
from tilescope.schema.table_validation import ParentCorrelationSchema


def _parent_correlation_row() -> dict:
    return {
        SAMPLE_ID: "S",
        ROI_ID: "A",
        G1_ID: "gene1",
        G2_ID: "gene2",
        MODE: "pearson",
        N_ROI: 10,
        N_PERMUTATIONS: 0,
        MIN_ROI_SPOTS: 5,
        PARENT_CORRELATION_STATUS: "finite",
        CORR_OBS: 0.5,
        CORR_P_ANALYTIC: 0.01,
        CORR_PERM_MEAN: None,
        CORR_PERM_SD: None,
        CORR_P_PERM: None,
        CORR_Q_ANALYTIC: 0.01,
        CORR_Q_PERM: None,
        FDR_FAMILY_SIZE: 1,
        SAMPLE_SIZE: 10,
    }


def test_parent_correlation_accepts_canonical_and_requires_roi_id():
    ParentCorrelationSchema.validate(pd.DataFrame([_parent_correlation_row()]), lazy=True)

    row = _parent_correlation_row()
    del row[ROI_ID]
    with pytest.raises((SchemaError, SchemaErrors)):
        ParentCorrelationSchema.validate(pd.DataFrame([row]), lazy=True)
