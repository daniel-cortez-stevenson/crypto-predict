from crypr.tests.unit_decorator import my_logger, my_timer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from keras.models import load_model
from crypr.util.io import load_from_pickle


class Model(BaseEstimator):

    @my_logger
    @my_timer
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    # @my_logger
    # @my_timer
    # def set_parameters(self, parameters):
    #     self.parameters = parameters
    #     self.estimator.set_params(**self.parameters)

    @my_logger
    @my_timer
    def fit(self, X, y=None):
        self.estimator.fit(X, y)

    @my_logger
    @my_timer
    def predict(self, X, y=None):
        return self.estimator.predict(X)

    @my_logger
    @my_timer
    def save_estimator(self, path):
        self.estimator.to_pickle('{}/{}.pkl'.format(path, self.name))


class RegressionModel(Model, RegressorMixin):

    def evaluate(self, y_pred, y_true):
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
        rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
        print("RMSE: {}\nMAE: {}\n".format(rmse, mae))
        return rmse, mae


class SavedRegressionModel(RegressionModel):

    @my_logger
    @my_timer
    def __init__(self, path):
        self.path = path
        self.ext = self.path.split('.')[-1]
        self.name=self.path.split('.')[-2]
        self.load()


    @my_logger
    @my_timer
    def load(self):
        if self.ext == 'pkl':
            self.estimator = load_from_pickle(self.path)
        elif self.ext == 'h5':
            self.estimator = load_model(self.path)
        else:
            print('WARNING: File Extension {} not supported.'.format(self.ext))