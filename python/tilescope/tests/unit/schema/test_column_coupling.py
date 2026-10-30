import pandas as pd
import pytest

from tilescope.schema.table_validation import assert_columns_populated_iff


def test_column_coupling_rejects_missing_value_inside_mask():
    frame = pd.DataFrame({"value": [1.0, pd.NA]})
    with pytest.raises(ValueError, match="must be populated"):
        assert_columns_populated_iff(
            frame, ("value",), pd.Series([True, True]),
            surface_name="surface", condition_label="finite",
        )


def test_column_coupling_rejects_value_outside_mask():
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must be NA"):
        assert_columns_populated_iff(
            frame, ("value",), pd.Series([True, False]),
            surface_name="surface", condition_label="finite",
        )
