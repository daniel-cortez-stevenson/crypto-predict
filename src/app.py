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
    model = load_model.load_model_from_path('models/linear_model_btc.pkl')

    data = pd.DataFrame()
    # I would have the app subscribe to the hourly data from cryptocompare and then query the cryptocompare API only hourly
    try :
        data = get_data.retrieve_all_data(coin, 24)
    except:
        abort(404)

    data = build_features.prep_df(data)

    current_time = list(data['time'])[-1]
    predict_times = [current_time + 3600 * ix for ix, x in enumerate(range(6))]
    predict_times = pd.to_datetime(predict_times, unit='s')

    data = build_features.series_to_supervised(data, n_in=24, n_out=0, dropnan=True)

    prediction = model.predict(data)

    # Not a UI guy so here ya go
    return jsonify({str(predict_times[0]): prediction[0][0],
                    str(predict_times[1]): prediction[0][1],
                    str(predict_times[2]): prediction[0][2],
                    str(predict_times[3]): prediction[0][3],
                    str(predict_times[4]): prediction[0][4],
                    str(predict_times[5]): prediction[0][5]
                    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
