"""Backfill roi_cross_grain_sensitivity.csv for existing pair-run outputs.

Two entry points:

- ``backfill_run(run_dir)``: single run.
- ``backfill_corpus(root)`` + ``write_backfill_manifest(...)``: walks a
  directory tree, applies backfill to every discovered run, emits one
  per-invocation manifest carrying original-run and backfill-reducer
  provenance for every entry.

The module also exposes a ``python -m tilescope.tools.backfill_cross_grain
<root> [--force]`` CLI.

Contract
--------
- Reads only the canonical ingredient CSVs the live adapter consumes.
- Consumes ``run_metadata.json`` verbatim; never modifies it.
- Every ingredient frame is required to carry ``sample_id`` /
  ``g1_id`` / ``g2_id`` columns; loader raises
  ``INCOMPATIBLE_SCHEMA`` if any frame disagrees with the metadata or
  omits an identity column.
- Reuses the existing adapter + producer verbatim; no second
  disk-specific computation lives here. Missing columns surface as
  ``INCOMPATIBLE_SCHEMA``, not ``ERROR``.
- Atomic write (tempfile in the target directory + ``os.replace``).
- Skips runs whose target CSV already exists and passes the full
  producer frame validator, unless ``force=True`` is supplied. A
  ``SKIPPED_EXISTS`` result still carries the full provenance read
  from the existing target.
- Older / incompatible runs return ``INCOMPATIBLE_SCHEMA`` and are
  not accommodated through fallback logic.
- The manifest is the source-of-truth for corpus-level provenance.
  Downstream aggregators must read it and retain BOTH the original-run
  identifiers (core version, config hash) AND the backfill-reducer
  identifiers so a copied backfilled CSV cannot be mistaken for an
  output produced by the historical core itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence

import pandas as pd

from tilescope.__version__ import __version__ as TILESCOPE_VERSION
from tilescope.feature_identity import require_semantic_feature_ids
from tilescope.analyzers.cross_grain_adapter import (
    CrossGrainPairFrames,
    build_cross_grain_inputs,
)
from tilescope.analyzers.roi_cross_grain_sensitivity import (
    CANONICAL_COLUMNS,
    PRODUCER_CONTRACT_VERSION,
    build_roi_cross_grain_sensitivity_df,
    validate_cross_grain_sensitivity_frame,
)
from tilescope.cols import G1_ID, G2_ID, ROI_ID, SAMPLE_ID

OUTPUT_FILENAME = "roi_cross_grain_sensitivity.csv"
RUN_METADATA_FILENAME = "run_metadata.json"
MANIFEST_FILENAME_PREFIX = "cross_grain_backfill_manifest"
ROWS_PER_ROI = 14  # 7 estimand triples x 2 target surfaces

# Canonical ingredient CSVs the adapter reads. Ordered for determinism
# in error messages. Each maps to the CrossGrainPairFrames attribute.
_INGREDIENTS: tuple = (
    ("roi_correlations.csv", "roi_correlations_df"),
    ("roi_enrichment.csv", "roi_enrichment_df"),
    ("roi_source_moran.csv", "roi_source_moran_df"),
    ("roi_replicate_correlations.csv", "roi_replicate_correlations_df"),
    ("roi_tessellation_correlations.csv", "roi_tessellation_correlations_df"),
    ("roi_replicate_enrichment.csv", "roi_replicate_enrichment_df"),
    ("roi_tessellation_moran.csv", "roi_tessellation_moran_df"),
    ("roi_consensus_correlations.csv", "roi_consensus_correlations_df"),
    ("roi_consensus_enrichment.csv", "roi_consensus_enrichment_df"),
    ("roi_consensus_moran.csv", "roi_consensus_moran_df"),
)


class BackfillStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED_EXISTS = "skipped_exists"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    ERROR = "error"


@dataclass(frozen=True)
class BackfillResult:
    run_dir: Path
    status: BackfillStatus
    sample_id: Optional[str] = None
    g1_id: Optional[str] = None
    g2_id: Optional[str] = None
    n_rows: Optional[int] = None
    n_rois: Optional[int] = None
    output_sha256: Optional[str] = None
    # Read verbatim from the run's own run_metadata.json. The immutable
    # run_metadata.json + effective_config.json remain on disk beside
    # each run and are the authoritative record of the original run's
    # scientific configuration; the backfill does not synthesise a
    # hash of the config (raw config hashes include operational
    # metadata like timestamps and output paths and would classify
    # scientifically-identical runs as different).
    original_run_core_version: Optional[str] = None
    original_analysis_policy_version: Optional[str] = None
    reason: Optional[str] = None


class IncompatibleSchemaError(RuntimeError):
    """Signals that a run's ingredient CSVs do not satisfy the current
    canonical export contract."""


def _read_run_metadata(run_dir: Path) -> dict:
    path = run_dir / RUN_METADATA_FILENAME
    if not path.exists():
        raise IncompatibleSchemaError(f"missing {RUN_METADATA_FILENAME}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise IncompatibleSchemaError(
            f"malformed {RUN_METADATA_FILENAME}: {e}"
        )


def _resolve_feature_identity(metadata: dict) -> tuple:
    identity = metadata.get("feature_identity") or {}
    try:
        return require_semantic_feature_ids(
            identity.get("feature_g1_id"),
            identity.get("feature_g2_id"),
            owner="run_metadata.json feature_identity",
        )
    except ValueError as exc:
        raise IncompatibleSchemaError(
            str(exc)
        ) from exc


def _read_ingredient_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        raise IncompatibleSchemaError(f"missing ingredient CSV: {filename}")
    return pd.read_csv(path)


def _load_pair_frames(run_dir: Path, g1_id: str, g2_id: str) -> CrossGrainPairFrames:
    kwargs = {"feature_g1_id": g1_id, "feature_g2_id": g2_id}
    for filename, attr in _INGREDIENTS:
        kwargs[attr] = _read_ingredient_csv(run_dir, filename)
    return CrossGrainPairFrames(**kwargs)


def _resolve_and_check_identities(
    pair_frames: CrossGrainPairFrames,
    g1_id_metadata: str,
    g2_id_metadata: str,
) -> str:
    """Assert every non-empty ingredient frame carries the required
    identity columns, that all agree, and that pair identities match
    ``run_metadata.json``. Returns the resolved sample_id."""
    sample_ids: set = set()
    for filename, attr in _INGREDIENTS:
        df = getattr(pair_frames, attr)
        if df.empty:
            continue
        for col in (SAMPLE_ID, G1_ID, G2_ID):
            if col not in df.columns:
                raise IncompatibleSchemaError(
                    f"{filename} missing required identity column {col!r}"
                )
        sample_ids.update(df[SAMPLE_ID].astype(str).unique().tolist())
        observed_g1 = set(df[G1_ID].astype(str).unique().tolist())
        if observed_g1 != {g1_id_metadata}:
            raise IncompatibleSchemaError(
                f"{filename}.{G1_ID} disagrees with run_metadata: "
                f"metadata={g1_id_metadata!r} observed={sorted(observed_g1)!r}"
            )
        observed_g2 = set(df[G2_ID].astype(str).unique().tolist())
        if observed_g2 != {g2_id_metadata}:
            raise IncompatibleSchemaError(
                f"{filename}.{G2_ID} disagrees with run_metadata: "
                f"metadata={g2_id_metadata!r} observed={sorted(observed_g2)!r}"
            )
    if not sample_ids:
        raise IncompatibleSchemaError(
            "no populated ingredient frames; cannot resolve sample_id"
        )
    if len(sample_ids) > 1:
        raise IncompatibleSchemaError(
            f"ingredient frames disagree on sample_id: {sorted(sample_ids)!r}"
        )
    return next(iter(sample_ids))


def _existing_output_is_schema_valid(path: Path) -> Optional[pd.DataFrame]:
    """Return the loaded DataFrame if the existing output passes the
    producer's full frame validator, else None. Row-count expectation
    is derived from the target's own unique roi_id set."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or ROI_ID not in df.columns:
        return None
    expected_rows = int(df[ROI_ID].astype(str).nunique()) * ROWS_PER_ROI
    try:
        validate_cross_grain_sensitivity_frame(df, expected_rows)
    except AssertionError:
        return None
    return df


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(payload: bytes, target: Path, *, prefix: str) -> None:
    """Write bytes atomically via tempfile + os.replace on the same
    filesystem so a partially-written file can never be observed."""
    fd, tmp_path = tempfile.mkstemp(
        prefix=prefix, suffix=".tmp", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _atomic_write_csv(df: pd.DataFrame, target: Path) -> str:
    """Write CSV atomically. Returns the sha256 of the written bytes."""
    payload = df.to_csv(index=False).encode("utf-8")
    _atomic_write_bytes(payload, target, prefix=".backfill_csv_")
    return _sha256_of_bytes(payload)


def _resolve_backfill_git_sha() -> Optional[str]:
    """Best-effort git rev-parse HEAD for the checkout that provides
    this module. Returns None when git is unavailable or the module
    is not inside a working tree."""
    module_dir = Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(module_dir),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _skipped_result(
    run_dir: Path, target: Path, df: pd.DataFrame,
    original_version: Optional[str], original_policy_version: Optional[str],
) -> BackfillResult:
    """Populate a SKIPPED_EXISTS result with full provenance read from
    the existing target file so the downstream manifest is complete."""
    payload = target.read_bytes()
    return BackfillResult(
        run_dir=run_dir, status=BackfillStatus.SKIPPED_EXISTS,
        sample_id=str(df[SAMPLE_ID].iloc[0]),
        g1_id=str(df[G1_ID].iloc[0]),
        g2_id=str(df[G2_ID].iloc[0]),
        n_rows=int(len(df)),
        n_rois=int(df[ROI_ID].astype(str).nunique()),
        output_sha256=_sha256_of_bytes(payload),
        original_run_core_version=original_version,
        original_analysis_policy_version=original_policy_version,
    )


def backfill_run(
    run_dir: Path,
    *,
    force: bool = False,
    csv_writer: Callable[[pd.DataFrame, Path], str] = _atomic_write_csv,
) -> BackfillResult:
    """Backfill a single run directory.

    Returns a ``BackfillResult`` describing the outcome. Never modifies
    ``run_metadata.json``. Skips existing schema-valid outputs unless
    ``force`` is set. ``csv_writer`` is an injection point for tests.
    """
    run_dir = Path(run_dir)
    target = run_dir / OUTPUT_FILENAME

    try:
        metadata = _read_run_metadata(run_dir)
    except IncompatibleSchemaError as e:
        return BackfillResult(
            run_dir=run_dir, status=BackfillStatus.INCOMPATIBLE_SCHEMA,
            reason=str(e),
        )
    original_version = str(metadata.get("tilescope_version") or "") or None
    original_policy = str(metadata.get("analysis_policy_version") or "") or None

    if target.exists() and not force:
        existing_df = _existing_output_is_schema_valid(target)
        if existing_df is not None:
            return _skipped_result(
                run_dir, target, existing_df,
                original_version, original_policy,
            )

    try:
        g1_id, g2_id = _resolve_feature_identity(metadata)
        pair_frames = _load_pair_frames(run_dir, g1_id, g2_id)
        sample_id = _resolve_and_check_identities(pair_frames, g1_id, g2_id)
    except IncompatibleSchemaError as e:
        return BackfillResult(
            run_dir=run_dir, status=BackfillStatus.INCOMPATIBLE_SCHEMA,
            reason=str(e),
            original_run_core_version=original_version,
            original_analysis_policy_version=original_policy,
        )
    except Exception as e:
        return BackfillResult(
            run_dir=run_dir, status=BackfillStatus.ERROR,
            reason=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            original_run_core_version=original_version,
            original_analysis_policy_version=original_policy,
        )

    try:
        inputs = build_cross_grain_inputs(sample_id=sample_id, pair_frames=pair_frames)
    except KeyError as e:
        # Adapter raises KeyError when an ingredient CSV lacks a
        # required column. That is a schema mismatch, not an internal
        # failure.
        return BackfillResult(
            run_dir=run_dir, status=BackfillStatus.INCOMPATIBLE_SCHEMA,
            sample_id=sample_id, g1_id=g1_id, g2_id=g2_id,
            reason=f"adapter column missing: {e}",
            original_run_core_version=original_version,
            original_analysis_policy_version=original_policy,
        )
    except Exception as e:
        return BackfillResult(
            run_dir=run_dir, status=BackfillStatus.ERROR,
            sample_id=sample_id, g1_id=g1_id, g2_id=g2_id,
            reason=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            original_run_core_version=original_version,
            original_analysis_policy_version=original_policy,
        )

    try:
        df = build_roi_cross_grain_sensitivity_df(inputs)
        sha = csv_writer(df, target)
    except Exception as e:
        return BackfillResult(
            run_dir=run_dir, status=BackfillStatus.ERROR,
            sample_id=sample_id, g1_id=g1_id, g2_id=g2_id,
            reason=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            original_run_core_version=original_version,
            original_analysis_policy_version=original_policy,
        )

    return BackfillResult(
        run_dir=run_dir, status=BackfillStatus.WRITTEN,
        sample_id=sample_id, g1_id=g1_id, g2_id=g2_id,
        n_rows=int(len(df)),
        n_rois=int(df[ROI_ID].nunique()) if not df.empty else 0,
        output_sha256=sha,
        original_run_core_version=original_version,
        original_analysis_policy_version=original_policy,
    )


# ---------------------------------------------------------------------
# Corpus driver + manifest
# ---------------------------------------------------------------------

def _default_is_run_dir(path: Path) -> bool:
    """Heuristic: a run directory has run_metadata.json AND at least
    the source-correlation ingredient CSV."""
    return (
        (path / RUN_METADATA_FILENAME).is_file()
        and (path / "roi_correlations.csv").is_file()
    )


def _walk_run_dirs(
    root: Path, is_run_dir: Callable[[Path], bool] = _default_is_run_dir,
) -> Iterator[Path]:
    """Yield every subdirectory (root inclusive) that matches
    ``is_run_dir``, sorted for determinism. Once a directory is
    identified as a run dir, its subtree is not descended further so
    nested per-pair outputs cannot double-count."""
    root = Path(root).resolve()
    if is_run_dir(root):
        yield root
        return
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames.sort()  # deterministic traversal
        here = Path(dirpath)
        matches = [d for d in dirnames if is_run_dir(here / d)]
        for d in matches:
            yield here / d
        # Don't descend into matched run dirs.
        dirnames[:] = [d for d in dirnames if d not in matches]


def backfill_corpus(
    root: Path,
    *,
    force: bool = False,
    is_run_dir: Callable[[Path], bool] = _default_is_run_dir,
) -> List[BackfillResult]:
    """Walk ``root`` and backfill every discovered run. Returns the
    ordered list of per-run outcomes; caller can hand it to
    :func:`write_backfill_manifest` for the on-disk manifest."""
    root = Path(root)
    return [backfill_run(rd, force=force) for rd in _walk_run_dirs(root, is_run_dir)]


def _serialise_result(result: BackfillResult) -> dict:
    payload = asdict(result)
    payload["run_dir"] = str(result.run_dir)
    payload["status"] = result.status.value
    return payload


def write_backfill_manifest(
    root: Path,
    results: Sequence[BackfillResult],
    *,
    output_path: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Write one manifest describing an entire backfill invocation.

    Provenance for downstream aggregators:

    - ``backfill_producer_core_version`` and ``backfill_producer_git_sha``
      identify the reducer that generated this manifest's outputs.
    - ``producer_contract_version`` names the roi_cross_grain_sensitivity
      row + column contract.
    - Per-run ``original_run_core_version`` and
      ``original_analysis_policy_version`` name the historical core that
      wrote the ingredient CSVs. The immutable ``run_metadata.json`` and
      ``effective_config.json`` remain on disk beside each run as the
      authoritative record of the original scientific configuration.

    Atomic write: the manifest is the corpus-level provenance authority,
    so a partially-written manifest must never be observable.
    """
    root = Path(root).resolve()
    ts = timestamp or datetime.now(timezone.utc)
    ts_slug = ts.strftime("%Y%m%dT%H%M%SZ")
    if output_path is None:
        output_path = root / f"{MANIFEST_FILENAME_PREFIX}_{ts_slug}.json"
    else:
        output_path = Path(output_path)
    status_counts: dict = {}
    for r in results:
        status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1
    payload = {
        "backfill_producer_core_version": TILESCOPE_VERSION,
        "backfill_producer_git_sha": _resolve_backfill_git_sha(),
        "producer_contract_version": PRODUCER_CONTRACT_VERSION,
        "backfill_timestamp_utc": ts.isoformat(),
        "root": str(root),
        "n_runs_walked": len(results),
        "status_counts": status_counts,
        "runs": [_serialise_result(r) for r in results],
    }
    _atomic_write_bytes(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        output_path, prefix=".backfill_manifest_",
    )
    return output_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tilescope.tools.backfill_cross_grain",
        description=(
            "Backfill roi_cross_grain_sensitivity.csv across every run "
            "directory found under <root>. Never modifies "
            "run_metadata.json. Skips existing schema-valid outputs "
            "unless --force is supplied."
        ),
    )
    parser.add_argument("root", type=Path, help="Corpus root to walk.")
    parser.add_argument(
        "--force", action="store_true",
        help="Rewrite existing outputs even when they pass schema.",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=None,
        help=f"Manifest output path (default: <root>/{MANIFEST_FILENAME_PREFIX}_<ts>.json).",
    )
    args = parser.parse_args(argv)
    if not args.root.exists():
        print(f"error: root does not exist: {args.root}", file=sys.stderr)
        return 2
    results = backfill_corpus(args.root, force=args.force)
    manifest_path = write_backfill_manifest(
        args.root, results, output_path=args.manifest_path,
    )
    status_counts: dict = {}
    for r in results:
        status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1
    print(f"[BACKFILL] runs={len(results)} status_counts={status_counts}")
    print(f"[BACKFILL] manifest: {manifest_path}")
    # Non-zero exit on ANY bad outcome (ERROR or INCOMPATIBLE_SCHEMA).
    # For a corpus that is known to satisfy the current export contract,
    # an incompatible-schema outcome is as much a failure as a hard
    # ERROR: incomplete backfill must not appear successful.
    bad = status_counts.get(BackfillStatus.ERROR.value, 0) + status_counts.get(
        BackfillStatus.INCOMPATIBLE_SCHEMA.value, 0,
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(_cli())
