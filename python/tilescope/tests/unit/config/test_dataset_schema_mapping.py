"""
Test dataset schema mapping.

Verifies IC-5: A CSV with non-canonical headers maps correctly via dataset schema.
"""

from pathlib import Path

from tilescope.tilescope_io import read_expression_csv
from tilescope.settings import load_dataset_schema
from tilescope.spatial import Canvas, make_canvas
from tilescope.cols import GENE1, GENE2, ROI_ID
from tilescope.tests.helpers.paths import FIXTURES_DIR


def test_shipped_dataset_schema_is_valid_and_canonical():
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "user_configs"
        / "dataset.schema.json"
    )
    schema = load_dataset_schema(str(schema_path))

    assert schema is not None
    assert set(schema.columns.values()) == {"x", "y", "roi_id", "gene1", "gene2"}


def test_schema_mapping_noncanonical_headers():
    """Non-canonical CSV headers should map correctly via dataset schema."""
    # Load dataset schema
    schema_path = str(FIXTURES_DIR / "test_noncanonical_schema.json")
    dataset_schema = load_dataset_schema(schema_path)

    # Load CSV with non-canonical headers
    csv_path = str(FIXTURES_DIR / "test_noncanonical_headers.csv")
    df = read_expression_csv(csv_path, dataset_schema=dataset_schema)

    # Verify columns were mapped correctly
    assert ROI_ID in df.columns, "roi_id should be mapped"
    assert 'x' in df.columns, "x should be mapped"
    assert 'y' in df.columns, "y should be mapped"
    assert GENE1 in df.columns, "GENE1 should be mapped"
    assert GENE2 in df.columns, "GENE2 should be mapped"

    # Verify original column names are gone
    assert 'region_id' not in df.columns
    assert 'coord_x' not in df.columns
    assert 'coord_y' not in df.columns
    assert 'expression_a' not in df.columns
    assert 'expression_b' not in df.columns

    # Verify data integrity (spot check first row)
    assert df.iloc[0][ROI_ID] == '1'
    assert df.iloc[0]['x'] == 7.0
    assert df.iloc[0]['y'] == 10.0
    assert df.iloc[0][GENE1] == 23.0
    assert df.iloc[0][GENE2] == 10.0

    print(f"✓ Schema mapping successful: {len(df)} rows")


def test_worldspace_creation_with_schema():
    """Canvas creation should work with dataset schema."""
    schema_path = str(FIXTURES_DIR / "test_noncanonical_schema.json")
    dataset_schema = load_dataset_schema(schema_path)

    csv_path = str(FIXTURES_DIR / "test_noncanonical_headers.csv")
    df = read_expression_csv(csv_path, dataset_schema=dataset_schema)

    # Create Canvas
    canvas = make_canvas(df)

    # Should always be Canvas (world-only architecture)
    assert isinstance(canvas, Canvas), \
        f"Expected Canvas, got {type(canvas).__name__}"

    print(f"✓ Coordinate space created: {type(canvas).__name__}")


def test_worldspace_auto_detect_without_schema():
    """Without schema, Canvas should auto-detect from data."""
    # Load CSV without schema
    csv_path = str(FIXTURES_DIR / "test_int_coords.csv")
    df = read_expression_csv(csv_path, dataset_schema=None)

    # Auto-detect coordinate space
    canvas = make_canvas(df)

    assert isinstance(canvas, Canvas), \
        f"Expected Canvas, got {type(canvas).__name__}"

    print(f"✓ Coordinate space auto-detected: {type(canvas).__name__}")
