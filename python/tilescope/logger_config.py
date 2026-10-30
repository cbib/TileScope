r"""Centralized logging configuration for TileScope.

Idempotent configurator - safe to call multiple times.
Configure once in CLI/tests, never in library code.

Logging Pattern (Debug-Aware Formatting):
    DEBUG mode (verbose=True or level='DEBUG'):
        Module name prefix + inline tags:
            logger = logging.getLogger(__name__)
            logger.info("[EXPORT] Writing consensus enrichment...")
            # Output: [analyzers.orchestrator] [EXPORT] Writing consensus enrichment...

    Non-DEBUG mode (default):
        Clean output with inline tags only:
            logger.info("[EXPORT] Writing consensus enrichment...")
            # Output: [EXPORT] Writing consensus enrichment...

    Common operation tags (inline string literals, not imported):
        [VIZ]         - Visualization operations
        [ROI_SPLIT]   - Component splitting
        [EXPORT]      - File exports
        [VALIDATION]  - Schema/data validation
        [REPAIR]      - Geometry repair
        [CONFIG]      - Config loading/merging
        [OVERLAP]     - Polygon overlap detection
        [DIST-COMP]   - Distributional companions (CLES, LOG2FC)
        [PARENT]      - Parent-ROI analyzer

    Grep patterns:
        rg "\[EXPORT\]"      # All export operations
        rg "\[ROI_SPLIT\]"   # All component splitting operations
        rg "\[VIZ\]"         # All visualization operations
"""

import logging
import sys

_CONFIGURED = False


def configure_tilescope_logging(verbose: bool = False, level: str = None) -> None:
    """Configure TileScope logging (idempotent).

    Args:
        verbose: Legacy flag for DEBUG level (backward compatible)
        level: Explicit level string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
               Overrides verbose flag if provided
    """
    global _CONFIGURED

    # Determine logging level
    if level is not None:
        log_level = getattr(logging, level.upper())
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    root_logger = logging.getLogger()

    # Always update level (allows changing verbosity)
    root_logger.setLevel(log_level)

    # Configure handler exactly once (idempotency guard)
    if not _CONFIGURED:
        # Clear any existing handlers (prevents default logging format leak)
        # This handles the case where Python's default logging ran before configure_tilescope_logging()
        root_logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)  # Let logger do the filtering

        # Conditional formatter: show module names only in DEBUG mode
        if log_level == logging.DEBUG:
            # Debug: [module_name] [INLINE_FLAG] message
            fmt = '[%(name)s] %(message)s'
        else:
            # Non-debug: [INLINE_FLAG] message (cleaner output)
            fmt = '%(message)s'

        handler.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(handler)

        # Silence noisy third-party loggers (once, regardless of verbose setting)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

        _CONFIGURED = True
