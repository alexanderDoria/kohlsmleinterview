import argparse
import joblib
from model.model import Model
from model.preprocessing import PreProcessor
from sklearn.pipeline import Pipeline
import pandas as pd


parser = argparse.ArgumentParser(description='train a model')
parser.add_argument('--model-save-path',
                    help='file path to save model', required=True)
parser.add_argument('--train-data-path',
                    help='file path of training data', required=True)
parser.add_argument('--model-name',
                    help='name of the sklearn model e.g. RandomForestClassifier',
                    required=True)
parser.add_argument('--import-module',
                    help='import module for sklearn model e.g. sklearn.ensemble',
                    required=True)
parser.add_argument('--model-params',
                    help='sklearn model params as a JSON dictionary',
                    required=False)
args = vars(parser.parse_args())

model_save_path = args['model-save-path']
train_data_path = args['train-data-path']
model_name = args['model-name']
import_module = args['import-module']
model_params = args['model-params']

print("model_save_path: ", model_save_path)
print("train_data_path: ", train_data_path)
print("model_name: ", model_name)
print("import_module: ", import_module)
print("model_params: ", model_params)


def train():
    data = pd.read_csv(train_data_path)

    model = Model(
        model_name=model_name,
        import_module=import_module,
        model_params=model_params
    )

    preprocessor = PreProcessor(
        categorical_features=["x3", "x6", "x7"],
        numeric_features=["x1", "x2", "x4", "x5"]
    )

    X, y = preprocessor.x_y_split(data, 'y')

    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

    transformer = preprocessor.create_transformer()

    clf = Pipeline(
        steps=[("preprocessor", transformer),
               ("classifier", model.model)]
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    metrics = model.evaluate(pred, y_test)

    model.log_metrics(metrics)

    model.save_model()

    joblib.dump(clf, f"{model_save_path}")


if __name__ == '__main__':
    train()
