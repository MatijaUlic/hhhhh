from src.transform import ClassSuportTransformer
import pandas as pd


def test_map_binary_column_to_int():
    transformer = ClassSuportTransformer()
    df = pd.DataFrame({"Gender": ["Female", "Female", "Female", "Male"]})

    expected_df = pd.DataFrame({"Gender": [1, 1, 1, 0]})

    transformed_df = transformer._map_binary_variables(df)

    # Test the result against the expected DataFrame
    pd.testing.assert_frame_equal(transformed_df, expected_df)
