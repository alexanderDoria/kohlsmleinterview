from flask import Flask, request, jsonify
import joblib
import os
import logging

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json = request.get_json()
        print(json)

        pred = clf.predict(json).tolist()

    return jsonify(pred)


if __name__ == '__main__':

    # load model
    # use build arg for location
    clf = joblib.load(os.environ.get('MODEL_LOCATION', None))

    if not clf:
        logging.error("no model specified")

    app.run(host='0.0.0.0', port=5000)
