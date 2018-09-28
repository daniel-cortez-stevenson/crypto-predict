#!flask/bin/python
from flask import Flask, jsonify, abort, request, Response
from src.CryptoPredict.SavedModel import SavedModel

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

    model = SavedModel(coin=coin, Tx=72, Ty=1, feature_window=72)
    model.load()
    prediction = model.predict()

    last_close = model.data['close'].iloc[-1]
    last_time = model.data['timestamp'].iloc[-1]
    predict_times = [last_time + pd.Timedelta(hours=1*(ix+1)) for ix in range(model.Ty)]

    # Not a UI guy so here ya go
    return jsonify({'{}+00:00'.format(predict_times[0]): '{} USD'.format(last_close+prediction[0]/100*last_close)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
