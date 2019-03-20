"""Model classes base off of SciKit-Learn"""
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple
from abc import abstractmethod
from crypr.util import load_from_pickle


class Model(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.fit=self.estimator.fit(X, y, **kwargs)
        return self.fit

    def predict(self, X, y=None):
        return self.estimator.predict(X)


class RegressionModel(Model, RegressorMixin):
    def evaluate(self, X_pred, y_true) -> Tuple[float, float]:
        y_pred = self.estimator.predict(X_pred)
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
        rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
        print('RMSE: {}\nMAE: {}\n'.format(rmse, mae))
        return rmse, mae


class SavedModel(Model):
    def __init__(self, path):
        self.path = path
        self.estimator = None
        self.load()
        Model.__init__(self, self.estimator)

    @abstractmethod
    def load(self):
        raise NotImplementedError

    
class SavedPickleRegressionModel(SavedModel):
    def __init__(self, path):
        SavedModel.__init__(self, path=path)

    def load(self) -> None:
        ext = self.path.split('.')[-1]
        if ext == 'pkl':
            self.estimator = load_from_pickle(self.path)
        else:
            raise ValueError('File Extension {} not supported.'.format(ext))


class SavedKerasTensorflowModel(SavedModel):
    def __init__(self, path):
        self.graph = None
        SavedModel.__init__(self, path=path)

    def load(self):
        self.estimator = load_model(self.path)
        self.graph = tf.get_default_graph()
