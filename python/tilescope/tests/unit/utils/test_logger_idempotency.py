"""Test logger configuration idempotency."""

import logging
from tilescope import logger_config
from tilescope.logger_config import configure_tilescope_logging


def test_idempotent():
    """[LOG] Verify configure_tilescope_logging() is idempotent - safe to call multiple times."""
    root = logging.getLogger()
    # Save state
    original_handlers = root.handlers.copy()
    original_configured = logger_config._CONFIGURED

    # Reset for deterministic test
    root.handlers.clear()
    logger_config._CONFIGURED = False

    # Call twice with different verbose settings
    configure_tilescope_logging(verbose=True)
    h1 = len(root.handlers)

    configure_tilescope_logging(verbose=False)
    h2 = len(root.handlers)

    # Verify handlers not duplicated
    assert h1 == h2 == 1, f"Expected 1 handler, got {h1} then {h2}"

    # Verify level toggles work (last call wins)
    assert root.level == logging.INFO, (
        f"Expected INFO level after verbose=False, got {logging.getLevelName(root.level)}"
    )

    print(f"✓ Logger idempotency verified: {h1} handler(s), no duplication")
    print(f"✓ Level updates work: level={logging.getLevelName(root.level)}")

    # Restore original state
    root.handlers.clear()
    for h in original_handlers:
        root.addHandler(h)
    logger_config._CONFIGURED = original_configured


def test_no_root_pollution():
    """[LOG] Verify logger configuration doesn't pollute root logger beyond expectations."""
    # conftest.py already configured logging, so root should have exactly 1 handler
    root = logging.getLogger()

    # Count handlers before (should be 1 from conftest)
    handlers_before = len(root.handlers)

    # Configure again (idempotent - should not add more handlers)
    configure_tilescope_logging(verbose=True)
    handlers_after = len(root.handlers)

    # Should be the same (idempotency)
    assert handlers_before == handlers_after, (
        f"Handler count changed: {handlers_before} -> {handlers_after}"
    )

    print(f"✓ No root pollution: {handlers_after} handler(s) (stable)")
