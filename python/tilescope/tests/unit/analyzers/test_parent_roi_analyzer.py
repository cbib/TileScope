"""Parent-ROI analyzer contract (total-tables edition).

Contracts:

1. Canonical fixture: total tables emit ONE ROW per source parent
   ROI with a typed status column; pooled per-ROI sample sizes are
   correct, context cardinality matches "all other-ROI spots", output
   frames never carry COMPONENT_ID, and descriptive enrichment
   direction is present on FINITE rows.
2. Gate split + shape independence: correlation is context-
   independent — its status is FINITE whenever `min_roi_spots` is
   satisfied, regardless of whether enrichment qualifies. Enrichment
   status is INSUFFICIENT_CONTEXT_SPOTS when only the context gate
   fails.
3. Per-ROI failure independence: a constant-gene ROI records
   ZERO_GENE1_VARIANCE on its own correlation row but does not
   suppress the other ROI's outputs; per-side context asymmetry in
   enrichment surfaces as CONTEXT_PROPORTION_ZERO while distributional
   effect sizes remain available.
4. Determinism: same input + same seed → identical permutation
   values on FINITE rows.
5. Empty input → empty frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.analyzers.parent_roi_analyzer import analyze_parent_rois
from tilescope.analyzers.parent_roi_outcomes import (
    ParentCorrelationStatus,
    ParentEffectSizeStatus,
    ParentEnrichmentStatus,
)
from tilescope.analyzers.parent_roi_projection import (
    build_feature_parent_roi_projection, build_parent_roi_support_index,
)
from tilescope.cols import (
    COMPONENT_ID, ROI_ID, SAMPLE_ID, GENE1, GENE2,
    CORR_OBS, ENRICHMENT_DIRECTION,
    GENE1_ENRICHMENT, GENE1_ENRICHMENT_STATUS,
    GENE2_ENRICHMENT, GENE2_ENRICHMENT_STATUS,
    JOINT_HIGH_ENRICHMENT_FC, JOINT_HIGH_ENRICHMENT_STATUS,
    JOINT_LOW_ENRICHMENT_FC, JOINT_LOW_ENRICHMENT_STATUS,
    LOG2FC_G1, CLES_G1,
    N_ROI, N_CTX,
    MIN_ROI_SPOTS, MIN_CONTEXT_SPOTS,
    PARENT_CORRELATION_STATUS, PARENT_ENRICHMENT_STATUS,
    PARENT_EFFECT_SIZE_STATUS,
    Q_G1_THRESHOLD, Q_G2_THRESHOLD,
    Q_G1_LOW_THRESHOLD, Q_G2_LOW_THRESHOLD,
)
from tilescope.metrics.enrichment import EnrichmentRatioStatus
from tilescope.defaults import NORM_GENE1_MIN, NORM_GENE2_MIN


THRESHOLDS = {
    Q_G1_THRESHOLD: 2.5,
    Q_G2_THRESHOLD: 2.5,
    Q_G1_LOW_THRESHOLD: 1.5,
    Q_G2_LOW_THRESHOLD: 1.5,
}
NORM_PARAMS = {NORM_GENE1_MIN: 0.1, NORM_GENE2_MIN: 0.1}


def _make_fixture():
    """Parent A: 80 pooled spots or cells (gene1 == gene2 → r=+1).
    Parent B: 60 pooled random spots or cells."""
    rng = np.random.default_rng(2026)
    a_g1 = np.concatenate([np.arange(40, dtype=float), np.arange(40, dtype=float) + 100])
    a_g2 = a_g1
    b_g1 = rng.uniform(0.5, 5.0, 60)
    b_g2 = rng.uniform(0.5, 5.0, 60)
    return pd.DataFrame({
        ROI_ID: ["A"] * 80 + ["B"] * 60,
        GENE1: np.concatenate([a_g1, b_g1]),
        GENE2: np.concatenate([a_g2, b_g2]),
    })


def _run(df, **kwargs):
    defaults = dict(
        source_expression_df=df,
        thresholds=THRESHOLDS,
        normalization_params=NORM_PARAMS,
        min_roi_spots=40,
        min_context_spots=40,
        n_permutations=0,
        sample_id="S",
    )
    defaults.update(kwargs)
    return analyze_parent_rois(**defaults)


def _finite_rois(df, status_col):
    return set(df[df[status_col] == "finite"][ROI_ID])


def test_canonical_fixture_pooling_and_output_grain():
    """Total tables: one row per source parent with typed status.
    Pooled sample sizes correct on FINITE rows; no COMPONENT_ID leak;
    provenance columns present."""
    df = _make_fixture()
    enrichment_df, correlation_df = _run(df)

    # Total tables: every source parent has a row on BOTH frames.
    assert len(correlation_df) == 2
    assert len(enrichment_df) == 2
    assert set(correlation_df[ROI_ID]) == {"A", "B"}
    assert set(enrichment_df[ROI_ID]) == {"A", "B"}

    # Both parents FINITE under the fixture gate values.
    assert _finite_rois(correlation_df, PARENT_CORRELATION_STATUS) == {"A", "B"}
    assert _finite_rois(enrichment_df, PARENT_ENRICHMENT_STATUS) == {"A", "B"}

    # sample_id stamped on every row.
    assert (correlation_df[SAMPLE_ID] == "S").all()
    assert (enrichment_df[SAMPLE_ID] == "S").all()

    # Provenance columns present.
    assert (correlation_df[MIN_ROI_SPOTS] == 40).all()
    assert (enrichment_df[MIN_ROI_SPOTS] == 40).all()
    assert (enrichment_df[MIN_CONTEXT_SPOTS] == 40).all()

    a_corr = correlation_df[correlation_df[ROI_ID] == "A"].iloc[0]
    b_corr = correlation_df[correlation_df[ROI_ID] == "B"].iloc[0]
    assert a_corr["sample_size"] == 80
    assert b_corr["sample_size"] == 60
    assert a_corr[CORR_OBS] == pytest.approx(1.0, abs=1e-6)

    a_enr = enrichment_df[enrichment_df[ROI_ID] == "A"].iloc[0]
    b_enr = enrichment_df[enrichment_df[ROI_ID] == "B"].iloc[0]
    assert a_enr[N_ROI] == 80 and a_enr[N_CTX] == 60
    assert b_enr[N_ROI] == 60 and b_enr[N_CTX] == 80

    # Grain-leak floor.
    assert COMPONENT_ID not in enrichment_df.columns
    assert COMPONENT_ID not in correlation_df.columns

    # Enrichment direction is descriptive; fake confidence/FDR fields
    # are not part of the native ROI surface.
    finite_enr = enrichment_df[
        enrichment_df[PARENT_ENRICHMENT_STATUS] == "finite"
    ]
    assert ENRICHMENT_DIRECTION in finite_enr.columns
    assert "joint_confidence" not in finite_enr.columns
    assert "joint_confidence_adj" not in finite_enr.columns
    assert (finite_enr[ENRICHMENT_DIRECTION] != "").all()


@pytest.mark.parametrize("scenario", [
    pytest.param("context_too_small", id="enrichment_blocked_by_context_gate"),
    pytest.param("other_roi_ineligible", id="enrichment_batch_has_only_one_roi"),
])
def test_correlation_emits_independently_of_enrichment_shape(scenario):
    """Correlation is context-independent: its FINITE set is unaffected
    by enrichment eligibility. Enrichment status disambiguates
    INSUFFICIENT_CONTEXT_SPOTS from INSUFFICIENT_ROI_SPOTS.
    """
    if scenario == "context_too_small":
        df = _make_fixture()
        enrichment_df, correlation_df = _run(df, min_context_spots=1000)
        # Correlation FINITE for both parents.
        assert _finite_rois(correlation_df, PARENT_CORRELATION_STATUS) == {"A", "B"}
        # Enrichment: both parents present as rows, both non-FINITE
        # with INSUFFICIENT_CONTEXT_SPOTS.
        assert len(enrichment_df) == 2
        assert set(enrichment_df[PARENT_ENRICHMENT_STATUS]) == {
            ParentEnrichmentStatus.INSUFFICIENT_CONTEXT_SPOTS.value,
        }
    else:  # other_roi_ineligible
        rng = np.random.default_rng(2028)
        b_span = np.linspace(0.5, 5.0, 20)
        df = pd.DataFrame({
            ROI_ID: ["A"] * 60 + ["B"] * 20,
            GENE1: np.concatenate([rng.uniform(0.5, 5.0, 60), b_span]),
            GENE2: np.concatenate([rng.uniform(0.5, 5.0, 60), b_span]),
        })
        enrichment_df, correlation_df = _run(df, min_context_spots=5)
        # A is FINITE on both; B is INSUFFICIENT_ROI_SPOTS on both.
        assert _finite_rois(correlation_df, PARENT_CORRELATION_STATUS) == {"A"}
        assert _finite_rois(enrichment_df, PARENT_ENRICHMENT_STATUS) == {"A"}
        b_corr_status = correlation_df[
            correlation_df[ROI_ID] == "B"
        ].iloc[0][PARENT_CORRELATION_STATUS]
        b_enr_status = enrichment_df[
            enrichment_df[ROI_ID] == "B"
        ].iloc[0][PARENT_ENRICHMENT_STATUS]
        assert b_corr_status == ParentCorrelationStatus.INSUFFICIENT_ROI_SPOTS.value
        assert b_enr_status == ParentEnrichmentStatus.INSUFFICIENT_ROI_SPOTS.value


def test_per_roi_failure_independence():
    """Constant-gene ROI records ZERO_GENE1_VARIANCE on its own
    correlation row but does not suppress the other ROI's outputs.
    Enrichment ratios surface CONTEXT_PROPORTION_ZERO on the per-side
    context asymmetry without discarding log2FC/CLES.
    """
    rng = np.random.default_rng(2027)
    b_g1 = np.linspace(0.5, 5.0, 60)
    b_g2 = np.linspace(0.5, 5.0, 60)
    df = pd.DataFrame({
        ROI_ID: ["A"] * 60 + ["B"] * 60,
        GENE1: np.concatenate([np.full(60, 3.14), b_g1]),
        GENE2: np.concatenate([rng.uniform(0.5, 5.0, 60), b_g2]),
    })
    enrichment_df, correlation_df = _run(df)

    # Correlation: A has zero variance on gene1 → ZERO_GENE1_VARIANCE.
    # B has non-zero variance on both → FINITE.
    a_corr_status = correlation_df[
        correlation_df[ROI_ID] == "A"
    ].iloc[0][PARENT_CORRELATION_STATUS]
    b_corr_status = correlation_df[
        correlation_df[ROI_ID] == "B"
    ].iloc[0][PARENT_CORRELATION_STATUS]
    assert a_corr_status == ParentCorrelationStatus.ZERO_GENE1_VARIANCE.value
    assert b_corr_status == ParentCorrelationStatus.FINITE.value

    # Enrichment: A FINITE (B's context has non-zero HIGH/LOW).
    # B CONTEXT_PROPORTION_ZERO (A's constant g1 gives zero LOW
    # context proportion), but the distributional effect remains valid.
    a_enr_status = enrichment_df[
        enrichment_df[ROI_ID] == "A"
    ].iloc[0][PARENT_ENRICHMENT_STATUS]
    b_enr_status = enrichment_df[
        enrichment_df[ROI_ID] == "B"
    ].iloc[0][PARENT_ENRICHMENT_STATUS]
    assert a_enr_status == ParentEnrichmentStatus.FINITE.value
    assert b_enr_status == ParentEnrichmentStatus.CONTEXT_PROPORTION_ZERO.value
    b_enr = enrichment_df[enrichment_df[ROI_ID] == "B"].iloc[0]
    assert b_enr[PARENT_EFFECT_SIZE_STATUS] == ParentEffectSizeStatus.FINITE.value
    assert pd.notna(b_enr[LOG2FC_G1])
    assert pd.notna(b_enr[CLES_G1])
    # Per-ratio breakdown: A's constant g1=3.14 makes context g1_low
    # proportion zero, so joint_low collapses. Marginal ratios and
    # joint_high remain finite because the individual g1/g2/both_high
    # context proportions are all positive. Aggregate
    # CONTEXT_PROPORTION_ZERO must not gate the finite marginals.
    finite = EnrichmentRatioStatus.FINITE.value
    assert b_enr[GENE1_ENRICHMENT_STATUS] == finite
    assert b_enr[GENE2_ENRICHMENT_STATUS] == finite
    assert b_enr[JOINT_HIGH_ENRICHMENT_STATUS] == finite
    assert b_enr[JOINT_LOW_ENRICHMENT_STATUS] != finite
    assert pd.notna(b_enr[GENE1_ENRICHMENT])
    assert pd.notna(b_enr[GENE2_ENRICHMENT])
    assert pd.notna(b_enr[JOINT_HIGH_ENRICHMENT_FC])
    assert pd.isna(b_enr[JOINT_LOW_ENRICHMENT_FC])
    # ENRICHMENT_DIRECTION available because both marginals are finite.
    assert pd.notna(b_enr[ENRICHMENT_DIRECTION])


def test_deterministic_reproducibility_under_same_seed():
    df = _make_fixture()
    _, corr1 = _run(df, seed=99, n_permutations=50)
    _, corr2 = _run(df, seed=99, n_permutations=50)
    for col in ("corr_p_perm", "corr_perm_mean", "corr_perm_sd"):
        for roi in ("A", "B"):
            v1 = corr1[corr1[ROI_ID] == roi].iloc[0][col]
            v2 = corr2[corr2[ROI_ID] == roi].iloc[0][col]
            assert v1 == v2 or (np.isnan(v1) and np.isnan(v2)), (
                f"{roi}.{col}: {v1} != {v2}"
            )


def test_empty_input_returns_empty_frames():
    empty = pd.DataFrame(columns=[ROI_ID, GENE1, GENE2])
    enrichment_df, correlation_df = _run(empty)
    assert enrichment_df.empty
    assert correlation_df.empty


def test_parent_projection_preserves_legacy_sort_index_order_and_marginals():
    frame = pd.DataFrame({
        ROI_ID: ["A", "A", "B", "B"],
        GENE1: [4.0, 1.0, 3.0, 2.0],
    }, index=[30, 10, 20, 0])
    support = build_parent_roi_support_index(frame)
    assert support.target_positions_by_roi["A"] == (1, 0)
    assert support.context_positions_by_roi["A"] == (3, 2)
    projection = build_feature_parent_roi_projection(
        feature_name=GENE1,
        source_values=frame[GENE1].to_numpy(dtype=float),
        high_threshold=2.5,
        low_threshold=1.5,
        support_index=support,
        gene_min_value=1.0,
        min_roi_spots=2,
        min_context_spots=2,
    )
    assert projection.target_values("A").tolist() == [1.0, 4.0]
    marginal = projection.marginals_by_roi["A"]
    assert marginal.target_sum == 5.0
    assert marginal.target_high == 1
    assert marginal.target_low == 1


def test_native_bh_exact_analytic_and_perm_and_family_boundary():
    """Native FDR: exact BH q's on the two-ROI fixture for both analytic
    and permutation p-values; non-finite rows carry null p/q but
    non-null fdr_family_size.
    """
    # Two ROIs: A r=+1 (analytic p ~ 0, perm p ~ 1/(B+1)); B random-uniform.
    df = _make_fixture()
    _, corr = _run(df, seed=99, n_permutations=99)

    # Family for BH is (sample_id, g1_id, g2_id, mode); size=2 with
    # both ROIs finite.
    a = corr[corr[ROI_ID] == "A"].iloc[0]
    b = corr[corr[ROI_ID] == "B"].iloc[0]

    # Both finite: q non-null on both.
    assert not np.isnan(a["corr_q_analytic"])
    assert not np.isnan(b["corr_q_analytic"])
    assert not np.isnan(a["corr_q_perm"])
    assert not np.isnan(b["corr_q_perm"])

    # BH monotonicity: min q equals min p * n / rank_at_min = p_min * 2 / 1.
    p_analytic = sorted([a["corr_p_analytic"], b["corr_p_analytic"]])
    q_analytic_min = p_analytic[0] * 2 / 1
    q_analytic_max = min(p_analytic[1] * 2 / 2, 1.0)
    q_analytic_at_rank_1 = min(q_analytic_min, q_analytic_max)  # step-up min
    assert min(a["corr_q_analytic"], b["corr_q_analytic"]) == pytest.approx(
        q_analytic_at_rank_1, abs=1e-12,
    )
    assert max(a["corr_q_analytic"], b["corr_q_analytic"]) == pytest.approx(
        q_analytic_max, abs=1e-12,
    )

    # Perm BH: same shape, computed from (a.corr_p_perm, b.corr_p_perm).
    p_perm = sorted([a["corr_p_perm"], b["corr_p_perm"]])
    q_perm_max = min(p_perm[1] * 2 / 2, 1.0)
    q_perm_at_1 = min(p_perm[0] * 2 / 1, q_perm_max)
    assert min(a["corr_q_perm"], b["corr_q_perm"]) == pytest.approx(
        q_perm_at_1, abs=1e-12,
    )

    # fdr_family_size populated on every row and equals finite count.
    assert (corr["fdr_family_size"] == 2).all()


def test_native_non_finite_row_retains_null_p_q_but_non_null_family_size():
    """Non-finite row: analytic + perm p and q all null; fdr_family_size
    still populated (equals count of finite p rows in the row's family).
    """
    # A: sufficient spots + variance ⇒ FINITE. B: zero variance ⇒ non-finite.
    rng = np.random.default_rng(2029)
    df = pd.DataFrame({
        ROI_ID: ["A"] * 60 + ["B"] * 60,
        GENE1: np.concatenate([
            rng.uniform(0.5, 5.0, 60), np.full(60, 3.14),
        ]),
        GENE2: np.concatenate([
            rng.uniform(0.5, 5.0, 60), rng.uniform(0.5, 5.0, 60),
        ]),
    })
    _, corr = _run(df, seed=7, n_permutations=99)

    a = corr[corr[ROI_ID] == "A"].iloc[0]
    b = corr[corr[ROI_ID] == "B"].iloc[0]

    # A finite: p + q populated.
    assert a[PARENT_CORRELATION_STATUS] == "finite"
    assert not np.isnan(a["corr_p_analytic"])
    assert not np.isnan(a["corr_q_analytic"])
    assert not np.isnan(a["corr_p_perm"])
    assert not np.isnan(a["corr_q_perm"])

    # B non-finite: p + q both NaN; family_size still populated.
    assert b[PARENT_CORRELATION_STATUS] != "finite"
    assert np.isnan(b["corr_p_analytic"])
    assert np.isnan(b["corr_q_analytic"])
    assert np.isnan(b["corr_p_perm"])
    assert np.isnan(b["corr_q_perm"])
    assert not pd.isna(b["fdr_family_size"])
    # Family size counts finite-p rows in the family; A is the only finite.
    assert int(b["fdr_family_size"]) == 1
    assert int(a["fdr_family_size"]) == 1
