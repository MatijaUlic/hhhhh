import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from metadata import COLUMNS_TO_DROP, BINARY_FEATURES, ONE_HOT_ENCODE_COLUMNS


class ClassSuportTransformer:
    def __init__(self):
        self.DROP_COLUMNS = COLUMNS_TO_DROP
        self.BINARY_FEATURES = BINARY_FEATURES
        self.ONE_HOT_ENCODE_COLUMNS = ONE_HOT_ENCODE_COLUMNS

    def transform_class_support(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._drop_columns(df)
        df = self._map_binary_variables(df)
        df = self._one_hot_encoding(df)

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(self.DROP_COLUMNS, axis=1)

    def _map_binary_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.BINARY_FEATURES:
            df[col] = df[col].map({"Female": 1, "Male": 0})
        return df

    def _one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder_path = "encoder.pkl"
        encoder_exists = os.path.exists(encoder_path)

        if encoder_exists:
            encoder = self._load_encoder()
        else:
            encoder = OneHotEncoder(drop="first", sparse_output=False).set_output(
                transform="pandas"
            )
            encoder.fit(df[self.ONE_HOT_ENCODE_COLUMNS])
            self._save_encoder(encoder)

        encoded_df = encoder.transform(df[self.ONE_HOT_ENCODE_COLUMNS])
        df = df.drop(columns=self.ONE_HOT_ENCODE_COLUMNS)
        df = pd.concat([df, encoded_df], axis=1)

        return df

    def _save_encoder(self, encoder) -> None:
        joblib.dump(encoder, "encoder.pkl")

    def _load_encoder(self):
        return joblib.load("encoder.pkl")

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Separate the classes
        df_y0 = df[df["Exited"] == 0].copy()
        df_y1 = df[df["Exited"] == 1].copy()

        # Find the smaller class size
        min_size = len(df_y1)

        # Randomly sample from each class
        df_y0_balanced = df_y0.sample(n=min_size, random_state=42)

        # Concatenate back together
        df_balanced = pd.concat([df_y0_balanced, df_y1])

        # Shuffle the dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        return df_balanced
