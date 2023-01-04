import argparse
import joblib
from model.model import Model
from model.preprocessing import PreProcessor
from sklearn.pipeline import Pipeline
import pandas as pd
import json


parser = argparse.ArgumentParser(description='train a model')
parser.add_argument('--model_save_path',
                    help='file path to save model', required=True)
parser.add_argument('--train_data_path',
                    help='file path of training data', required=True)
parser.add_argument('--model_name',
                    help='name of the sklearn model e.g. RandomForestClassifier',
                    required=True)
parser.add_argument('--import_module',
                    help='import module for sklearn model e.g. sklearn.ensemble',
                    required=True)
parser.add_argument('--model_params',
                    help='sklearn model params as a JSON dictionary',
                    required=False,
                    default='{}')
args = vars(parser.parse_args())

model_save_path = args['model_save_path']
train_data_path = args['train_data_path']
model_name = args['model_name']
import_module = args['import_module']
model_params = json.loads(args['model_params'])


def train():
    data = pd.read_csv(train_data_path)

    print(f"training data read at {train_data_path}")

    model = Model(
        model_name=model_name,
        import_module=import_module,
        model_params=model_params
    )

    preprocessor = PreProcessor(
        categorical_features=[2, 5, 6],
        numeric_features=[0, 1, 3, 4]
    )

    X, y = preprocessor.x_y_split(data, 'y')

    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

    transformer = preprocessor.create_transformer()

    clf = Pipeline(
        steps=[("preprocessor", transformer),
               ("classifier", model.model)]
    )

    print("training model...")
    clf.fit(X_train, y_train)
    print("model trained")

    pred = clf.predict(X_test)

    print("evaluating model...")
    metrics = model.evaluate(y_test, pred)
    print(f"metrics: {metrics}")

    model.log_metrics(metrics)

    model.mlflow_log_model()

    joblib.dump(clf, model_save_path)

    print(f"model saved at: {model_save_path}")


if __name__ == '__main__':
    train()
