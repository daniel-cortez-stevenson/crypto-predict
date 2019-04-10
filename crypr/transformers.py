"""Proprocesser classes modeled after SciKit-Learn"""
import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer:
    def __init__(self):
        self.feature_names = None

    def fit(self, X, y=None):
        self.fit_feature_names(X)
        return self

    def fit_feature_names(self, X):
        try:
            self.feature_names = list(X.columns)
        except AttributeError as e:
            print(e)

    def get_feature_names(self):
        return self.feature_names


class MovingAverageTransformer(BaseEstimator, TransformerMixin, BaseTransformer):
    def __init__(self, n: int):
        self.n = n
        BaseTransformer.__init__(self)

    def transform(self, X):
        v = np.ones((self.n,))/float(self.n)
        X = np.apply_along_axis(np.convolve, axis=0, arr=X, v=v, mode='valid')
        na_fill = self.na_fill_array(X.shape[1])
        return np.concatenate([na_fill, X])

    def na_fill_array(self, num_columns):
        na_fill = np.empty((self.n - 1, num_columns))
        na_fill[:] = np.nan
        return na_fill


class PercentChangeTransformer(BaseEstimator, TransformerMixin, BaseTransformer):
    def transform(self, X):
        X = np.divide(np.diff(X, axis=0), X[:-1])*100
        na_fill = self.na_fill_array(X.shape[1])
        return np.concatenate([na_fill, X])

    def na_fill_array(self, num_columns):
        na_fill = np.empty((1, num_columns))
        na_fill[:] = np.nan
        return na_fill


class PassthroughTransformer(BaseEstimator, TransformerMixin, BaseTransformer):
    def transform(self, X):
        return X


class HaarSmoothTransformer(BaseEstimator, TransformerMixin, BaseTransformer):
    def __init__(self, smooth_factor):
        self.smooth_factor = smooth_factor
        BaseTransformer.__init__(self)

    def transform(self, X):
        return np.apply_along_axis(func1d=lambda x: self.discrete_wavelet_transform(x),
                                   axis=-1, arr=X.values)

    def discrete_wavelet_transform(self, x) -> np.ndarray:
        cA, cD = pywt.dwt(x, 'haar')
        cAt = pywt.threshold(cA, self.smoothing_threshold(cA, self.smooth_factor), mode='soft')
        cDt = pywt.threshold(cD, self.smoothing_threshold(cD, self.smooth_factor), mode='soft')
        tx = pywt.idwt(cAt, cDt, 'haar')
        return tx

    @staticmethod
    def smoothing_threshold(x, factor: float = 1.) -> float:
        return np.std(x) * np.sqrt(factor * np.log(x.size))
