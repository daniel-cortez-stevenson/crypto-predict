#!flask/bin/python
from flask import Flask, jsonify, abort, Request
from src.data import get_data
from src.models import load_model
from src.features import build_features
import json

app = Flask(__name__)




@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict', methods=['GET'])
def get_prediction():

    coin = Request.args.get("coin", "")
    if coin != 'BTC':
        abort(404)

    model = load_model.load_model_from_path('/app/models/linear_model_btc.pkl')
    data = get_data.retrieve_all_data(coin, 24)
    data = build_features.prep_df(data)
    data = build_features.series_to_supervised(data, n_in=24, n_out=0, dropnan=True)
    prediction = model.predict(data)
    return jsonify({'predictions': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
