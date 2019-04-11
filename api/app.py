"""Run the API by calling this module"""
import connexion
from flask import abort
from os.path import join
import pandas as pd
from crypr.models import SavedKerasTensorflowModel
from crypr.build import make_features, series_to_predict_matrix, make_3d
from crypr.cryptocompare import retrieve_all_data
from crypr.util import get_project_path

models_path = join(get_project_path(), 'models')

model_type = 'lstm_ng'

global btc_model, eth_model
btc_model_filename = '{}_{}.h5'.format(model_type, 'BTC')
btc_model = SavedKerasTensorflowModel(join(models_path, btc_model_filename))
eth_model_filename = '{}_{}.h5'.format(model_type, 'ETH')
eth_model = SavedKerasTensorflowModel(join(models_path, eth_model_filename))


def description():
    return {'message': 'The crypto-predict API'}


def say_hello(name=None):
    return {'message': 'Hello, {}!'.format(name or '')}


def predict(coin=None):
    coin = coin or 'BTC'
    Tx = 72
    target = 'close'

    cryptocompare_data = retrieve_all_data(coin, Tx + 1 + 48)
    preprocessed_data = make_features(cryptocompare_data, target_col=target,
                                      keep_cols=['close', 'high', 'low', 'volumeto', 'volumefrom'],
                                      ma_lags=[6, 12, 24, 48], ma_cols=['close', 'volumefrom', 'volumeto'])
    time_series_data = series_to_predict_matrix(preprocessed_data, Tx)
    n_features = int(time_series_data.shape[1]/Tx)
    model_input_data = make_3d(arr=time_series_data, tx=Tx, num_channels=n_features)

    if coin == 'ETH':
        model = eth_model
    elif coin == 'BTC':
        model = btc_model
    else:
        abort(404)  # FIXME: More descriptive error

    with model.graph.as_default():
        prediction = model.estimator.predict(model_input_data)

    def parse_keras_prediction(keras_prediction):
        """Handles multi-output Keras models"""
        if len(keras_prediction) == 2:
            return keras_prediction[1][0]
        else:
            return keras_prediction[0]
    parsed_prediction = parse_keras_prediction(prediction)

    last_value = cryptocompare_data[target].iloc[-1]
    last_time = cryptocompare_data['timestamp'].iloc[-1]

    prediction_val = [last_value + pred/100*last_value for pred in parsed_prediction]
    time_val = [last_time + pd.Timedelta(hours=ix + 1) for ix in range(len(parsed_prediction))]
    return dict(prediction=prediction_val, time=time_val)


app = connexion.App(__name__)
app.add_api('swagger.yaml')
application = app.app


if __name__ == '__main__':
    app.run(port=5000, server='gevent')
