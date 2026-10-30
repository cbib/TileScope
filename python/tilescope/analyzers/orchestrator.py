"""Legacy `run_analysis` adapter over the multi-pair authority.

`run_analysis(expression_df, config, dataset_id)` is a one-pair adapter
built on top of `_run_sample_authority`: it invokes the internal
authority with `[(GENE1, GENE2)]` and a capture sink that receives
that pair's legacy-shape export dict. No pipeline logic lives here.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from tilescope.analyzers.sample_authority import _run_sample_authority
from tilescope.cols import GENE1, GENE2, ROI_ID
from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.settings import RunConfigModel, ResolvedRunConfig

logger = logging.getLogger(__name__)


def run_analysis(
    expression_df: pd.DataFrame,
    config: ResolvedRunConfig,
    dataset_id: int = 1,
    dataset_schema: Optional[object] = None,
    *,
    feature_a_id: str,
    feature_b_id: str,
) -> Dict[str, object]:
    """One-pair adapter over the internal multi-pair authority.

    Returns the flat 27-key dict. The authority engine emits that dict
    once via a capture sink; nothing is recomputed here.

    `feature_a_id`/`feature_b_id` propagate the SEMANTIC pair identity
    down to `PairStatisticResult` (and every downstream CSV / metadata
    / renderer surface). Both are required. The canonical computational
    column names are not biological identities.
    """
    del dataset_schema  # Input mapping resolved before this boundary.
    resolved_a_id, resolved_b_id = require_semantic_feature_ids(
        feature_a_id, feature_b_id, owner="run_analysis",
    )
    sample_id = str(dataset_id)
    logger.info(
        "Running TileScope on %d spots across %d biological ROIs",
        len(expression_df), expression_df[ROI_ID].nunique(),
    )
    captured: Dict[str, object] = {}

    def _capture(_pair_key, pair_dict: Dict[str, object]) -> None:
        captured.update(pair_dict)

    pair_identities = {
        (GENE1, GENE2): (resolved_a_id, resolved_b_id),
    }

    _run_sample_authority(
        expression_df=expression_df, config=config,
        pair_features=[(GENE1, GENE2)],
        dataset_id=sample_id,
        pair_sink=_capture,
        pair_identities=pair_identities,
    )
    if not captured:
        raise RuntimeError(
            "run_analysis: capture sink saw no pair result — internal "
            "engine did not invoke it. Pipeline invariant violated."
        )
    return captured


__all__ = ["run_analysis"]
