#!flask/bin/python
from flask import Flask, abort, request
from crypr.base.models import SavedRegressionModel
from crypr.base.preprocessors import DWTSmoothPreprocessor
from crypr.data.cryptocompare import retrieve_all_data
from crypr.api.json import PredictionFormatter
from dotenv import find_dotenv, load_dotenv
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
    # N=28
    wavelet='haar'
    target='close'

    data = retrieve_all_data(coin, Tx+1)

    preprocessor = DWTSmoothPreprocessor(True, target, Tx=Tx, Ty=Ty, wavelet=wavelet,
                                name='CryptoPredict_DWTSmoothPreprocessor_{}'.format(coin))
    X = preprocessor.fit(data).transform(data)

    if coin == 'ETH':
        prediction = eth_model.predict(X)
    elif coin == 'BTC':
        prediction = btc_model.predict(X)
    else:
        #FIXME: More descriptive error
        abort(404)

    # If a multi-output model, take the last prediction - which will be pct change.
    print(prediction)
    if len(prediction) == 2:
        prediction = prediction[1][0]
    else:
        prediction = prediction[0]
    print('pred is {}'.format(prediction))
    formatter = PredictionFormatter(
        prediction=prediction,
        last_value=data[target].iloc[-1],
        last_time=data['timestamp'].iloc[-1]
    )
    print('format of resp is: {}'.format(formatter._format()))
    return formatter.respond()
    # return jsonify({'{}+00:00'.format(predict_times[0]): '{} USD'.format(predicted_price)})

if __name__ == '__main__':
    model_type='ae_lstm'
    wavelet='haar'

    load_dotenv(find_dotenv())
    base_path = os.path.dirname(find_dotenv())

    global eth_model
    eth_model = SavedRegressionModel('{}/models/{}_smooth_{}x{}_{}_{}.h5'.format(base_path, model_type, 1, 72, wavelet, 'ETH'))

    global btc_model
    btc_model = SavedRegressionModel('{}/models/{}_smooth_{}x{}_{}_{}.h5'.format(base_path, model_type, 1, 72, wavelet, 'BTC'))

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
