# Deploy ML model

This repo contains instructions, code, and data for creating a training script and deploying a model to a Docker container with a Flask backend.

## Train model

To train the model, use the training script: `train.py`. An example run would be:

`python3 train.py --model_save_path=saved_models/RandomForestClassifier --train_data_path=data/train.csv --model_name=RandomForestClassifier --import_module=sklearn.ensemble --model_params='{"n_estimators": 50}'`

## Build Docker container

`docker build . -t ml-app --build-arg MODEL_LOCATION=saved_models/RandomForestClassifier`

### Send request to running Docker container

`curl --location --request POST 'http://localhost:5000/predict' --header 'Content-Type: application/json' --data-raw '[[2.7, 21.59, "Thu", -1.1, 122, "California", "ford"]]'`
