"""Public multi-pair entry point.

`run_sample(*, expression_df, config, pair_features)` returns the typed
`SampleAnalysisResult`. Optional `pair_sink` streams a legacy-shape
per-pair export dict as each pair completes: the sink sees the pair
and drops it before the next iteration, so no pair-scope state
accumulates. The legacy `run_analysis` in `tilescope.analyzers.orchestrator`
is a one-pair adapter over the same internal engine.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from tilescope.analyzers.sample_analysis_result import (
    PairKey, SampleAnalysisResult,
)
from tilescope.analyzers.sample_authority import (
    PairSink, _run_sample_authority,
)
from tilescope.settings import RunConfigModel, ResolvedRunConfig


def run_sample(
    *,
    expression_df: pd.DataFrame,
    config: ResolvedRunConfig,
    pair_features: Sequence[PairKey],
    dataset_id: str = "1",
    pair_sink: Optional[PairSink] = None,
    pair_identities: Optional[Dict[PairKey, Tuple[str, str]]] = None,
) -> SampleAnalysisResult:
    """Multi-pair sample analysis with optional streaming per-pair export.

    Substrate stages (support / tessellation / consensus geometry) run
    exactly once per invocation; each unique feature is projected once;
    each ordered pair is computed once. Returns the typed
    `SampleAnalysisResult`.

    If `pair_sink` is provided, it is invoked once per pair with
    `(pair_key, legacy_export_dict)`. The dict shape matches the legacy
    `run_analysis` one-pair return. The sink runs inside the pair loop
    and its argument becomes garbage-collectable as soon as the sink
    returns; no per-pair export state is retained.
    """
    return _run_sample_authority(
        expression_df=expression_df, config=config,
        pair_features=pair_features, dataset_id=str(dataset_id),
        pair_sink=pair_sink, pair_identities=pair_identities,
    )
