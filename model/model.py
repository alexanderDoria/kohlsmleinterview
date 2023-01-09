from pandas import DataFrame
import importlib
import logging
import sklearn
from typing import List
import mlflow
from .metrics import Metrics


class Model:
    """
    A class used to represent an ML model and relevant use cases.
    Uses sklearn and MLflow stack for training and logging.

    ...

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        sklearn model to be used for predictions
    metrics : List[dict]
        list of metrics to be used during evaluation

    """

    def __init__(
        self,
        model_name: str,
        import_module: str,
        model_params: dict = {}
    ):
        self.model = self.get_model(model_name, import_module, model_params)
        self.metrics = Metrics.infer_metrics(self.model)

    def get_model(self,
                  model_name: str, import_module: str, model_params: dict
                  ) -> sklearn.base.BaseEstimator:
        """
        Returns instantiated sklearn model defined by parameters.

        Parameters
        ----------
        model_name : str
            Name of sklearn model e.g. "RandomForestClassifier"

        import_module : str
            Name of import module of sklearn base estimator e.g. "sklearn.ensemble"

        model_params : dict
           Dictionary defining sklearn model parameters 

        """
        model_params = model_params if model_params else {}
        model_class = getattr(
            importlib.import_module(import_module), model_name)
        model = model_class(**model_params)
        return model

    def predict(self, X: DataFrame):
        """
        Returns array-like of predicted values given features.

        Parameters
        ----------
        X : DataFrame
            DataFrame of input data to be used for predictions

        """
        return self.model.predict(X)

    def evaluate(self, true, pred) -> List[dict]:
        """
        Creates evaluation metrics comparing predicted and actual values.

        Parameters
        ----------
        true : array-like
            actual values to compare predictions against

        pred : array-like
            predictions from model

        """
        values = []
        for metric in self.metrics:
            values.append({
                'metric_name': metric,
                'metric_value': round(self.metrics[metric](true, pred), 2)
            })
        logging.info(values)
        return values

    def log_metrics(self, metrics: List[dict]):
        """
        Uses the log() function to iterate through metrics returned by evaluate()

        Parameters
        ----------
        metrics : List[dict]
            metrics with metric name and metric value e.g. {'RMSE': 0.9}

        """
        for metric in metrics:
            self.log(metric['metric_name'], metric['metric_value'])

    def log(self, metric_name: str, metric_value: float):
        """
        Uses MLflow to log metric.


        Parameters
        ----------
        metric_name : str
            name of metric

        metric_value : number
            metric value

        """
        mlflow.log_metric(metric_name, metric_value)

    def mlflow_log_model(self):
        """
        Logs model within ML run folder in /artifacts created by MLflow

        """
        mlflow.sklearn.log_model(self.model, 'model')
