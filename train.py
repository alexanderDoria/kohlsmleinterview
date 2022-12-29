import argparse
from model.model import Model
from model.preprocessing import PreProcessor
from sklearn.pipeline import Pipeline
import pandas as pd


parser = argparse.ArgumentParser(description='train a model')
parser.add_argument('--model-save-path',
                    help='file path to save model', required=True)
parser.add_argument('--train-data-path',
                    help='file path of training data', required=True)
parser.add_argument('--test-data-path',
                    help='file path to save model', required=True)
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
test_data_path = args['test-data-path']
model_name = args['model-name']
import_module = args['import-module']
model_parms = args['model-params']


def train():

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    model = Model(
        model_name=model_name,
        import_module=import_module,
        model_params=model_parms
    )

    # based on dataset, determine which features are categorical and numeric
    preprocessor = PreProcessor(
        categorical_features=["x3", "x6", "x7"],
        numeric_features=["x1", "x2", "x4", "x5"]
    )

    x_train, y_train = preprocessor.x_y_split(train_data)
    x_test, y_test = preprocessor.x_y_split(test_data)

    clf = Pipeline(
        steps=[("preprocessor", preprocessor.create_preprocessor()),
               ("classifier", model.model)]
    )

    clf.fit(x_train, y_train)


if __name__ == '__main__':
    train()
