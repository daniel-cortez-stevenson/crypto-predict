#!flask/bin/python
from flask import Flask, jsonify, abort, request, Response
from src.CryptoPredict.SavedModel import SavedModel
from src.CryptoPredict.Preprocesser import Preprocesser
from src.data.get_data import retrieve_all_data

import pandas as pd
import os

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict', methods=['GET'])
def get_prediction():

    coin = ''
    try:
        coin = request.args.get('coin')
    except KeyError:
        # Want to put more descriptive errors here.
        abort(404)
    except Exception as e:
        abort(404)
        print(e)

    Tx=72
    Ty=1
    feature_window=72
    target='close'

    data = retrieve_all_data(coin, Tx + feature_window - 1)

    preprocessor = Preprocesser(data, target, Tx=Tx, Ty=Ty, moving_averages=[6, 12, 24, 48, 72],
                                name='CryptoPredict_{}_tx{}_ty{}_flag{}'.format(coin, Tx, Ty, feature_window))
    X, _ = preprocessor.preprocess_predict()

    model_path = 'models/xgboost_{}_tx{}_ty{}_flag{}.pkl'.format(coin, Tx, Ty, feature_window)
    model = SavedModel(model_path)
    model.load()

    prediction = model.predict(X)

    last_target = preprocessor.data[target].iloc[-1]
    last_time = preprocessor.data['timestamp'].iloc[-1]
    predict_times = [last_time + pd.Timedelta(hours=1*(ix+1)) for ix in range(model.Ty)]

    return jsonify({'{}+00:00'.format(predict_times[0]): '{} USD'.format(last_target+prediction[0]/100*last_target)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
