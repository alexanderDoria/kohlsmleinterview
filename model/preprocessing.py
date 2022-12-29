from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
import pandas as pd


class PreProcessor:

    def __init__(self,
                 numeric_features: List[str] = [],
                 categorical_features: List[str] = []
                 ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def x_y_split(self, df: pd.DataFrame, label: str):
        df_X = df.drop(label, axis=1)
        df_y = df[label]
        return df_X, df_y

    def create_preprocessor(self, numerical_imputer='median'):
        transformers = []
        if self.numeric_features:
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy=numerical_imputer)),
                       ("scaler", StandardScaler())]
            )
            transformers.append(
                ("num", numeric_transformer, self.numeric_features))
        if self.categorical_features:
            categorical_transformer = OneHotEncoder(
                handle_unknown="infrequent_if_exist")
            transformers.append(
                ("cat", categorical_transformer, self.categorical_features))

        preprocessor = ColumnTransformer(
            transformers=[transformers]
        )
        return preprocessor()
