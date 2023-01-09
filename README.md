# Deploy ML model

This repo contains instructions, code, and data for creating a training script and deploying a model to a Docker container with a Flask backend.

## Train model

To train the model, use the training script: `train.py`. An example run would be:

`python3 train.py --model_save_path=saved_models/RandomForestClassifier --train_data_path=data/train.csv --model_name=RandomForestClassifier --import_module=sklearn.ensemble --model_params='{"n_estimators": 50}'`

A complete set of arguments can be found towards the top of the file.

## Track metrics

This script uses MLflow to log metrics. To view them (along with artifacts), make sure MLflow is installed, and run `mlflow ui` in the root directory of this project. Some MLflow run metrics have already been saved under `mlruns/`.

## Build Docker container

The Docker build command accepts a parameter which has the location of the saved model. An example of building the container would be: 

`docker build . -t ml-app --build-arg MODEL_LOCATION=saved_models/RandomForestClassifier`

### Send request to running Docker container

The Flask backend only has a `/predict` endpoint. An example post request is:

`curl --location --request POST 'http://localhost:5000/predict' --header 'Content-Type: application/json' --data-raw '[[2.7, 21.59, "Thu", -1.1, 122, "California", "ford"]]'`
