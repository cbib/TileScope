"""Pytest configuration for TileScope tests.

Automatically configures verbose logging for all tests.
"""
from tilescope.logger_config import configure_tilescope_logging


import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure verbose logging for all tests (session-wide, auto-applied)."""
    configure_tilescope_logging(verbose=True)
