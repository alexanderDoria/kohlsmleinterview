from pandas import DataFrame
import importlib
import logging
import joblib
from sklearn.model_selection import train_test_split
import sklearn
from typing import List
from metrics import Metrics
import mlflow


class Model:
    def __init__(
        self,
        model_name: str,
        import_module: str,
        model_params: dict = {},
        model_save_name: str = ""
    ):
        self.features = []
        self.model = self.get_model(model_name, import_module, model_params)
        self.model_save_name = self.set_default_model_save_name(
            model_name, model_save_name)
        self.metrics = Metrics.infer_metrics(self.model)

    def train_test_split(self, input_data: DataFrame, split_ratio: float = 0.8):
        assert 0 < split_ratio < 1.0, "split_ratio must be a value between 0 and 1"
        return train_test_split(input_data, train_size=split_ratio)

    def get_model(
        model_name: str, import_module: str, model_params: dict
    ) -> sklearn.base.BaseEstimator:
        model_class = getattr(
            importlib.import_module(import_module), model_name)
        model = model_class(**model_params)
        return model

    def set_default_model_save_name(self, model_name: str):
        return model_name if model_name else type(self.model).__name__

    def train(self, features: List[str], target):
        self.model.fit(features, target)
        return self.model

    def predict(self, X: DataFrame):
        return self.model.predict(X)

    def evaluate(self, true, pred) -> List[dict]:
        values = []
        for metric in self.metrics:
            values.append({
                'metric_name': metric,
                'metric_value': self.metrics[metric](true, pred)
            })
        logging.info(values)
        return values

    def log_metrics(self, metrics):
        for metric in metrics:
            self.log(metric)

    def log(self, metric_name, metric_value):
        mlflow.log_metric(metric_name, metric_value)
        return

    def save_model(self, model_save_name=""):
        model_save_name = model_save_name if model_save_name else self.model_save_name
        joblib.dump(self.model, model_save_name)
