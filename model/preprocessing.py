import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import List


class PreProcessor:
    """
    A class used to implement preprocessing steps for data 
    to be trained in a model. 

    ...

    Attributes
    ----------
    numeric_features : List[int]
        feature indices representing numeric values in dataset
    categorical_features : List[int]
        feature indices representing categorical values in dataset

    """

    def __init__(self,
                 numeric_features: List[str] = [],
                 categorical_features: List[str] = []
                 ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def x_y_split(self, df: pd.DataFrame, label: str) -> tuple((pd.DataFrame, pd.DataFrame)):
        """
        Splits data into features (X) and target (y)

        Parameters
        ----------
        df : DataFrame
            DataFrame of input data including X and y

        label : str
            column name of target, label, etc

        """
        df_X = df.drop(label, axis=1)
        df_y = df[label]
        return df_X, df_y

    def train_test_split(self, X: pd.DataFrame, y, split_ratio: float = 0.8):
        """
        Returns x_train, y_train, x_test, y_test given a split ratio.

        Parameters
        ----------
        X : DataFrame
            DataFrame of input data to be used for predictions

        y: array-like
            list of labels

        """
        assert 0 < split_ratio < 1.0, "split_ratio must be a value between 0 and 1"
        return train_test_split(X, y, train_size=split_ratio)

    def create_transformer(self, numerical_imputer: str = 'median') -> ColumnTransformer:
        """
        Returns column transformer containing pipeline processing steps.
        Handles numeric imputing and scaling and categoric one hot encoding.

        Parameters
        ----------
        numerical_imputer : str
            strategy to be used for simple imputing

        """
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

        transformer = ColumnTransformer(
            transformers=transformers
        )

        return transformer
