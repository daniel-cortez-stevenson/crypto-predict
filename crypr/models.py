"""Model classes base off of SciKit-Learn"""
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from crypr.util import load_from_pickle


class Model(BaseEstimator):
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    def fit(self, X, y=None, **kwargs):
        self.fit=self.estimator.fit(X, y, **kwargs)
        return self.fit

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def save_estimator(self, path):
        ext=path.split('.')[-1]
        if ext=='pkl':
            self.estimator.to_pickle(path)
        elif ext=='h5':
            self.estimator.save(path)


class RegressionModel(Model, RegressorMixin):
    def evaluate(self, y_pred, y_true):
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
        rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
        print("RMSE: {}\nMAE: {}\n".format(rmse, mae))
        return rmse, mae


class SavedRegressionModel(RegressionModel):
    def __init__(self, path):
        self.path = path
        self.ext = self.path.split('.')[-1]
        self.name=self.path.split('.')[-2]
        self.load()

    def load(self):
        if self.ext == 'pkl':
            self.estimator = load_from_pickle(self.path)
        elif self.ext == 'h5':
            self.estimator = load_model(self.path)
        else:
            print('WARNING: File Extension {} not supported.'.format(self.ext))


class SavedKerasTensorflowModel(object):
    def __init__(self, path):
        self.path = path
        self.estimator = None
        self.graph = None
        self.load()

    def load(self):
        self.estimator = load_model(self.path)
        self.graph = tf.get_default_graph()
