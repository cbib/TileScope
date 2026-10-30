"""
Canvas geometry and transformation unit tests.

Tests low-level Canvas coordinate transformations and data type handling:
- Int→float conversion at data loading
- Canvas coordinate transform invariance
- Geometric equivalence across coordinate regimes
"""

import pytest
import numpy as np
from pathlib import Path

from tilescope.tilescope_io import read_expression_csv
from tilescope.spatial import Canvas, make_canvas
from tilescope.tests.helpers.paths import FIXTURES_DIR, TESTS_OUTPUT_DIR


def test_data_loading_int_to_float_conversion():
    """Unit test: verify integer coordinates are converted to floats at data loading.

    This is the foundation of the Canvas-based pipeline:
    all coordinates (int and float) are normalized to floats at the data boundary.
    """
    # Test integer grid fixture
    int_fixture = FIXTURES_DIR / "test_int_coords.csv"
    df_int = read_expression_csv(int_fixture)

    # Verify coordinates are floats (not int)
    assert df_int['x'].dtype == np.float64, f"x should be float64, got {df_int['x'].dtype}"
    assert df_int['y'].dtype == np.float64, f"y should be float64, got {df_int['y'].dtype}"
    print(f"✓ Integer coords converted to float64: x={df_int['x'].dtype}, y={df_int['y'].dtype}")

    # Verify float fixture remains float
    float_fixture = FIXTURES_DIR / "test_float_coords.csv"
    df_float = read_expression_csv(float_fixture)

    assert df_float['x'].dtype == np.float64, f"x should be float64, got {df_float['x'].dtype}"
    assert df_float['y'].dtype == np.float64, f"y should be float64, got {df_float['y'].dtype}"
    print(f"✓ Float coords remain float64: x={df_float['x'].dtype}, y={df_float['y'].dtype}")

    # Verify Canvas.from_df() always returns Canvas
    canvas_int = make_canvas(df_int)
    canvas_float = make_canvas(df_float)

    assert isinstance(canvas_int, Canvas), f"Expected Canvas, got {type(canvas_int)}"
    assert isinstance(canvas_float, Canvas), f"Expected Canvas, got {type(canvas_float)}"
    print(f"✓ Canvas.from_df() always returns Canvas")

    print(f"✅ Data loading int→float conversion: PASSED")


def test_canvas_to_px_clipping():
    """Verify Canvas.to_px() clips coordinates to canvas bounds."""
    canvas = Canvas(x0=0.0, y0=0.0, sx=1.0, sy=1.0, W=100, H=100)

    # Test in-bounds (should work normally)
    i, j = canvas.to_px(50.0, 50.0)
    assert 0 <= i < 100 and 0 <= j < 100, f"In-bounds coord should stay in-bounds: ({i}, {j})"

    # Test out-of-bounds (should clip)
    i_neg, j_neg = canvas.to_px(-10.0, -10.0)
    assert i_neg == 0 and j_neg == 0, f"Negative coords should clip to (0,0): ({i_neg}, {j_neg})"

    i_over, j_over = canvas.to_px(200.0, 200.0)
    assert i_over == 99 and j_over == 99, f"Over-bounds coords should clip to max: ({i_over}, {j_over})"

    print(f"✅ Canvas.to_px() clipping: PASSED")


def test_canvas_to_world_precision():
    """Verify Canvas.to_world() maintains coordinate precision."""
    canvas = Canvas(x0=1000.5, y0=2000.5, sx=0.25, sy=0.25, W=400, H=400)

    # Round-trip: world → pixel → world
    x_orig, y_orig = 1050.75, 2050.75
    i, j = canvas.to_px(x_orig, y_orig)
    x_recovered, y_recovered = canvas.to_world(i, j)

    # Should be close (within pixel resolution)
    x_diff = abs(x_recovered - x_orig)
    y_diff = abs(y_recovered - y_orig)

    # Tolerance = 1 pixel in world units
    pixel_size = 1.0 / canvas.sx
    assert x_diff < pixel_size, f"X round-trip error: {x_diff:.6f} >= {pixel_size}"
    assert y_diff < pixel_size, f"Y round-trip error: {y_diff:.6f} >= {pixel_size}"

    print(f"✅ Canvas.to_world() precision: PASSED")
