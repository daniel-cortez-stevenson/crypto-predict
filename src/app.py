#!flask/bin/python
from flask import Flask, jsonify, abort

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict/<string:coin>', methods=['GET'])
def get_prediction(coin):
    if coin != 'BTC':
        abort(404)
    return jsonify({'prediction': 'to the moon!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
