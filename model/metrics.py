import logging
from sklearn import metrics
from sklearn.base import is_classifier, is_regressor


class Metrics:
    @staticmethod
    def infer_metrics(model):
        if is_classifier(model):
            return {
                'F1': metrics.f1_score,
                'Accuracy': metrics.accuracy_score
            }
        elif is_regressor(model):
            return {
                'MSE': metrics.mean_squared_error,
                'MAE': metrics.mean_absolute_error,
            }
        else:
            logging.warning("Model is neither a regressor or classifier")
