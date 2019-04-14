"""Proprocesser classes modeled after SciKit-Learn"""
import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
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

    def transform(self, X):
        return X

    def get_feature_names(self):
        return self.feature_names


PassthroughTransformer = BaseTransformer


class MovingAverageTransformer(BaseTransformer):
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


class PercentChangeTransformer(BaseTransformer):
    def transform(self, X):
        X = np.divide(np.diff(X, axis=0), X[:-1])*100
        na_fill = self.na_fill_array(X.shape[1])
        return np.concatenate([na_fill, X])

    def na_fill_array(self, num_columns):
        na_fill = np.empty((1, num_columns))
        na_fill[:] = np.nan
        return na_fill


class HaarSmoothTransformer(BaseTransformer):
    def __init__(self, smooth_factor, wavelet='haar'):
        self.smooth_factor = smooth_factor
        self.wavelet = wavelet
        BaseTransformer.__init__(self)

    def transform(self, X):
        return np.apply_along_axis(func1d=lambda x: self.discrete_wavelet_transform(x), axis=0, arr=X)

    def discrete_wavelet_transform(self, arr1d: np.ndarray) -> np.ndarray:
        cA, cD = pywt.dwt(arr1d, self.wavelet, mode='symmetric', axis=-1)
        cAt, cDt = self.threshold(cA), self.threshold(cD)
        tx = pywt.idwt(cAt, cDt, self.wavelet, mode='symmetric', axis=-1)
        return tx

    def threshold(self, arr1d: np.ndarray) -> np.ndarray:
        return pywt.threshold(arr1d, value=self.smoothing_threshold(arr1d), mode='soft')

    def smoothing_threshold(self, x) -> float:
        return np.std(x) * np.sqrt(self.smooth_factor * np.log(x.size))
