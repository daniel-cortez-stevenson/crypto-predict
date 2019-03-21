"""Proprocesser classes modeled after SciKit-Learn"""
import numpy as np
from sklearn.base import TransformerMixin
from typing import Tuple, Union
from crypr.build import make_features, make_single_feature, series_to_predict_matrix, data_to_supervised, discrete_wavelet_transform


class Preprocesser(TransformerMixin):
    def __init__(self, production, Tx, Ty, target_col):
        self.production = production
        self.Tx = Tx
        self.Ty = Ty
        self.target_col = target_col
        self.engineered_columns = None

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return X


class SimplePreprocessor(Preprocesser):
    def __init__(self, production, target_col, Tx, Ty, moving_averages):
        Preprocesser.__init__(self, production, Tx, Ty, target_col)
        self.moving_averages = moving_averages

    def transform(self, X):
        fe = make_features(X, self.target_col, self.moving_averages)
        if self.production:
            X = series_to_predict_matrix(fe, n_in=self.Tx, dropnan=True)
            return self._reshape(X)
        else:
            X, y = data_to_supervised(fe, self.Tx, self.Ty)
            return self._reshape(X), y

    def _reshape(self, X):
        self.engineered_columns = X.columns.values
        X = np.expand_dims(X, axis=-1)
        X = np.reshape(a=X, newshape=(X.shape[0], self.Tx, -1))
        return np.swapaxes(a=X, axis1=-1, axis2=-2)


class DWTSmoothPreprocessor(Preprocesser):
    def __init__(self, production, target_col, Tx, Ty, wavelet):
        Preprocesser.__init__(self, production, Tx, Ty, target_col)
        self.wavelet = wavelet

    def transform(self, X):
        fe = make_single_feature(X, self.target_col)
        if self.production:
            X = series_to_predict_matrix(fe['target'], n_in=self.Tx, dropnan=True)
            X = discrete_wavelet_transform(X, wavelet=self.wavelet)
            return self._reshape(X)
        else:
            X, y = data_to_supervised(input_df=fe[['target']], Tx=self.Tx, Ty=self.Ty)
            X = discrete_wavelet_transform(X, wavelet=self.wavelet)
            return self._reshape(X), y

    def _reshape(self, X) -> np.ndarray:
        if len(X.shape) < 3:
            X = np.swapaxes(np.expand_dims(X, axis=-1), axis1=-2, axis2=-1)
        return X
