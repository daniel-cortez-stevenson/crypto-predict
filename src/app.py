#!flask/bin/python
from flask import Flask, jsonify, abort, request, Response
from src.data import get_data
from src.models import load_model
from src.features import build_features
import pandas as pd

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


    # Could alter this to gerneralize the function
    if coin != 'BTC':
        abort(404)

    # Then you would match coins to a model path using a dict.
    # Loading this model here rather than at app start because scalability is not an issue (yet!)
    model = load_model.load_from_pickle(path='models/xgboost_model_btc.pkl')

    Tx=72
    FEATURE_WINDOW=72
    # I would have the app subscribe to the hourly data from cryptocompare and then query the cryptocompare API only hourly
    data = get_data.retrieve_all_data(coin, Tx+FEATURE_WINDOW-1)

    last_close = data['close'].iloc[-1]
    last_time = data['timestamp'].iloc[-1]
    predict_times = [last_time + pd.Timedelta(hours=1*(ix+1)) for ix in range(1)]

    data = build_features.make_features(data)
    data = build_features.series_to_supervised(data, n_in=72, n_out=0, dropnan=True)
    prediction = model.predict(data)
    # Not a UI guy so here ya go
    return jsonify({'{}+00:00'.format(predict_times[0]): '{} USD'.format(last_close+prediction[0]/100*last_close)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
