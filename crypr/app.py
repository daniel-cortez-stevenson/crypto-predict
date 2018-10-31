#!flask/bin/python
from flask import Flask, jsonify, abort, request, Response
import numpy as np
from crypr.base.models import SavedRegressionModel
from crypr.base.preprocessors import CWTPreprocessor
from crypr.data.cryptocompare import retrieve_all_data

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
    N=28

    target='close'

    data = retrieve_all_data(coin, Tx)

    preprocessor = CWTPreprocessor(True, target, Tx=Tx, Ty=Ty, N=N, wavelet='MORLET',
                                name='CryptoPredict_WavePreprocessor_{}'.format(coin))
    X = preprocessor.fit(data).transform(data)

    if coin == 'ETH':
        prediction = eth_model.predict(X)
    elif coin == 'BTC':
        prediction = btc_model.predict(X)
    else:
        #FIXME: More descriptive error
        abort(404)

    # If a multi-output model, take the last prediction - which will be pct change.
    if len(prediction) == 2:
        prediction=prediction[1]

    last_target = data[target].iloc[-1]
    predicted_price = np.squeeze(last_target+prediction[0]/100*last_target)
    last_time = data['timestamp'].iloc[-1]
    predict_times = [last_time + pd.Timedelta(hours=1*(ix+1)) for ix in range(Ty)]

    return jsonify({'{}+00:00'.format(predict_times[0]): '{} USD'.format(predicted_price)})

if __name__ == '__main__':

    global eth_model
    eth_model = SavedRegressionModel('models/{}_cwt_{}x{}_{}_{}.h5'.format('LSTM_WSAEs', 28, 72, 'MORLET', 'ETH'))

    global btc_model
    btc_model = SavedRegressionModel('models/{}_cwt_{}x{}_{}_{}.h5'.format('LSTM_WSAEs', 28, 72, 'MORLET', 'BTC'))

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)