"""Proprocesser classes modeled after SciKit-Learn"""
import numpy as np
from sklearn.base import TransformerMixin
from crypr.build import make_features, series_to_predict_matrix, data_to_supervised
from crypr.build import continuous_wavelet_transform, make_single_feature, discrete_wavelet_transform


class Preprocesser(TransformerMixin):
    def __init__(self, production, Tx, Ty, target_col, name):
        self.production = production
        self.Tx, self.Ty = Tx, Ty
        self.target_col=target_col
        self.name = name
        self.input_columns = None
        self.engineered_columns = None

    def fit(self, X, y=None):
        self.input_columns = X.columns.values
        return self

    def transform(self, X):
        return X


class SimplePreprocessor(Preprocesser):
    def __init__(self, production, target_col, Tx, Ty, moving_averages,  name):
        Preprocesser.__init__(self, production, Tx, Ty, target_col, name)
        self.moving_averages = moving_averages

    def transform(self, X):
        fe = make_features(X, self.target_col, self.moving_averages)
        if self.production:
            X = series_to_predict_matrix(fe, n_in=self.Tx, dropnan=True)
            X = self._reshape(X)
            return X
        else:
            X, y = data_to_supervised(fe, self.Tx, self.Ty)
            X = self._reshape(X)
            return X, y

    def _reshape(self, X):
        self.engineered_columns = X.columns.values
        X = np.expand_dims(X, axis=-1)
        X = np.reshape(a=X, newshape=(X.shape[0], self.Tx, -1))
        print('int shape {}'.format(X.shape))
        X = np.swapaxes(a=X, axis1=-1, axis2=-2)
        return X


class CWTPreprocessor(Preprocesser):
    def __init__(self, production, target_col, Tx, Ty, N, wavelet, name):
        Preprocesser.__init__(self, production, Tx, Ty, target_col, name)
        self.N = N
        self.wavelet = wavelet

    def transform(self, X):
        fe = make_single_feature(X, self.target_col)
        if self.production:
            X = series_to_predict_matrix(fe.target.tolist(), n_in=self.Tx, dropnan=True)
            X = continuous_wavelet_transform(X, N=self.N, wavelet=self.wavelet)
            return X
        else:
            X, y = data_to_supervised(input_df=fe, Tx=self.Tx, Ty=self.Ty)
            X = continuous_wavelet_transform(X, N=self.N, wavelet=self.wavelet)
            return X, y


class DWTSmoothPreprocessor(Preprocesser):
    def __init__(self, production, target_col, Tx, Ty, wavelet, name):
        Preprocesser.__init__(self, production, Tx, Ty, target_col, name)
        self.wavelet = wavelet

    def transform(self, X):
        fe = make_single_feature(X, self.target_col)
        if self.production:
            X = series_to_predict_matrix(fe['target'].tolist(), n_in=self.Tx, dropnan=True)
            X = discrete_wavelet_transform(X, wavelet=self.wavelet)
            X = self._reshape(X)
            return X
        else:
            X, y = data_to_supervised(input_df=fe[['target']], Tx=self.Tx, Ty=self.Ty)
            X = discrete_wavelet_transform(X, wavelet=self.wavelet)
            X = self._reshape(X)
            return X, y

    def _reshape(self, X):
        if len(X.shape) < 3:
            X = np.swapaxes(np.expand_dims(X, axis=-1), axis1=-2, axis2=-1)
        return X
