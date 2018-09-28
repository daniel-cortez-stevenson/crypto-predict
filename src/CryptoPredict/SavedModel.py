from src.tests.unit_decorator import my_logger, my_timer
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.load_model import load_from_pickle
from src.data.get_data import retrieve_all_data
from src.features import build_features

class SavedModel(object):

    @my_logger
    @my_timer
    def __init__(self, coin, Tx, Ty, feature_window):
        self.coin, self.Tx, self.Ty, self.feature_window = coin, Tx, Ty, feature_window


    @my_logger
    @my_timer
    def load(self):
        self.model_path = 'models/xgboost_{}_tx{}_ty{}_flag{}.pkl'.format(self.coin, self.Tx, self.Ty, self.feature_window)
        self.model = load_from_pickle(self.model_path)


    @my_logger
    @my_timer
    def predict(self, X):
        self.X = X
        self.prediction = self.model.predict(self.X)
        return self.prediction
