"""Ledger-driven support classification: fail-loud, six-pattern coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tilescope.viz.support_overlay import (
    CAUSE_FILTER_DROPPED, CAUSE_INELIGIBLE_REGION, CAUSE_OUTSIDE_DOMAIN,
    CAUSE_SCATTERED_PARENT, CONSTRUCTIBLE, LedgerClassificationError,
    classify,
)


def _ledger_row(spot_index, cs, er=None, cid=None, scattered=False,
                region=None, eligible=None):
    return {
        "spot_index": spot_index,
        "component_status": cs,
        "exclusion_reason": er,
        "component_id": cid,
        "parent_scattered": scattered,
        "domain_region_id": region,
        "tessellation_eligible": eligible,
    }


def _source_row(source_row_id, x=0.0, y=0.0, g1=1.0, g2=1.0):
    return {"source_row_id": source_row_id, "x": x, "y": y,
            "gene1": g1, "gene2": g2}


def _classify_from(ledger_rows, source_rows):
    return classify(pd.DataFrame(ledger_rows), pd.DataFrame(source_rows))


# ------------------------------------------------------------------ #
# Six ledger patterns (A, B, C, D, E, F).
# ------------------------------------------------------------------ #


def test_pattern_a_split_dropped_maps_to_filter_dropped():
    overlay = _classify_from(
        [_ledger_row(0, cs="small_connected_component",
                     er="below_split_min_component_spots",
                     cid=None, region=None, eligible=None)],
        [_source_row(0)],
    )
    assert overlay.counts[CAUSE_FILTER_DROPPED] == 1
    assert overlay.counts[CONSTRUCTIBLE] == 0


def test_pattern_b_filter_dropped_maps_to_filter_dropped():
    overlay = _classify_from(
        [_ledger_row(0, cs="small_connected_component",
                     er="below_filter_min_component_spots",
                     cid="c1", region=None, eligible=None)],
        [_source_row(0)],
    )
    assert overlay.counts[CAUSE_FILTER_DROPPED] == 1


def test_pattern_c_scattered_parent():
    overlay = _classify_from(
        [_ledger_row(0, cs="retained_component", cid="c1",
                     scattered=True, region=None, eligible=None)],
        [_source_row(0)],
    )
    assert overlay.counts[CAUSE_SCATTERED_PARENT] == 1


def test_pattern_d_retained_outside_domain():
    overlay = _classify_from(
        [_ledger_row(0, cs="retained_component", cid="c1",
                     scattered=False, region=None, eligible=None)],
        [_source_row(0)],
    )
    assert overlay.counts[CAUSE_OUTSIDE_DOMAIN] == 1


def test_pattern_e_eligible_region_is_constructible():
    overlay = _classify_from(
        [_ledger_row(0, cs="retained_component", cid="c1",
                     scattered=False, region=0, eligible=True)],
        [_source_row(0)],
    )
    assert overlay.counts[CONSTRUCTIBLE] == 1
    assert overlay.unconstructible.empty


def test_pattern_f_ineligible_region():
    overlay = _classify_from(
        [_ledger_row(0, cs="retained_component", cid="c1",
                     scattered=False, region=0, eligible=False)],
        [_source_row(0)],
    )
    assert overlay.counts[CAUSE_INELIGIBLE_REGION] == 1


# ------------------------------------------------------------------ #
# Cause counts + rowset preservation across a mixed ledger.
# ------------------------------------------------------------------ #


def test_mixed_ledger_counts_and_constructible_only_carries_gene_cols():
    ledger = [
        _ledger_row(0, "retained_component", cid="c1", region=0, eligible=True),
        _ledger_row(1, "retained_component", cid="c1", region=0, eligible=True),
        _ledger_row(2, "small_connected_component",
                    er="below_filter_min_component_spots", cid="c2"),
        _ledger_row(3, "retained_component", cid="c3", scattered=True),
        _ledger_row(4, "retained_component", cid="c4", region=0, eligible=False),
    ]
    source = [
        _source_row(0, 0, 0, 1.0, 2.0),
        _source_row(1, 1, 0, 1.5, 2.5),
        _source_row(2, 0, 1, 3.0, 4.0),
        _source_row(3, 1, 1, 5.0, 6.0),
        _source_row(4, 2, 2, 7.0, 8.0),
    ]
    overlay = _classify_from(ledger, source)
    assert overlay.counts[CONSTRUCTIBLE] == 2
    assert overlay.counts[CAUSE_FILTER_DROPPED] == 1
    assert overlay.counts[CAUSE_SCATTERED_PARENT] == 1
    assert overlay.counts[CAUSE_INELIGIBLE_REGION] == 1
    assert list(overlay.constructible.columns) == ["x", "y", "gene1", "gene2"]
    assert set(overlay.unconstructible["cause"]) == {
        CAUSE_FILTER_DROPPED, CAUSE_SCATTERED_PARENT, CAUSE_INELIGIBLE_REGION,
    }


# ------------------------------------------------------------------ #
# Fail-loud id-set reconciliation.
# ------------------------------------------------------------------ #


def test_missing_spot_in_ledger_raises():
    ledger = [_ledger_row(0, "retained_component", cid="c1", region=0, eligible=True)]
    source = [_source_row(0), _source_row(1)]
    with pytest.raises(LedgerClassificationError, match="disagree"):
        _classify_from(ledger, source)


def test_missing_spot_in_source_raises():
    ledger = [
        _ledger_row(0, "retained_component", cid="c1", region=0, eligible=True),
        _ledger_row(1, "retained_component", cid="c1", region=0, eligible=True),
    ]
    source = [_source_row(0)]
    with pytest.raises(LedgerClassificationError, match="disagree"):
        _classify_from(ledger, source)


def test_duplicate_spot_index_raises():
    ledger = [
        _ledger_row(0, "retained_component", cid="c1", region=0, eligible=True),
        _ledger_row(0, "retained_component", cid="c1", region=0, eligible=True),
    ]
    source = [_source_row(0)]
    with pytest.raises(LedgerClassificationError, match="duplicate"):
        _classify_from(ledger, source)


def test_duplicate_source_row_id_raises():
    ledger = [_ledger_row(0, "retained_component", cid="c1", region=0, eligible=True)]
    source = [_source_row(0), _source_row(0)]
    with pytest.raises(LedgerClassificationError, match="duplicate"):
        _classify_from(ledger, source)


def test_none_frames_raise():
    with pytest.raises(LedgerClassificationError):
        classify(None, pd.DataFrame([_source_row(0)]))
    with pytest.raises(LedgerClassificationError):
        classify(pd.DataFrame([_ledger_row(0, "retained_component", cid="c1",
                                          region=0, eligible=True)]), None)


# ------------------------------------------------------------------ #
# Fail-loud on truth-table pattern violations.
# ------------------------------------------------------------------ #


def test_unknown_component_status_raises():
    ledger = [_ledger_row(0, cs="mystery_status")]
    with pytest.raises(LedgerClassificationError, match="unknown component_status"):
        _classify_from(ledger, [_source_row(0)])


def test_small_component_with_wrong_shape_raises():
    # component_status = small_* but exclusion_reason absent -> not A or B.
    ledger = [_ledger_row(0, cs="small_connected_component", er=None)]
    with pytest.raises(LedgerClassificationError, match="A or B pattern"):
        _classify_from(ledger, [_source_row(0)])


def test_retained_with_impossible_combo_raises():
    # scattered=True + region=0 + eligible=True should not exist.
    ledger = [_ledger_row(0, cs="retained_component", cid="c1",
                          scattered=True, region=0, eligible=True)]
    with pytest.raises(LedgerClassificationError, match="C/D/E/F pattern"):
        _classify_from(ledger, [_source_row(0)])
