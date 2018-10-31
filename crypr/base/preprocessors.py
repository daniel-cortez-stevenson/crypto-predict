from crypr.tests.unit_decorator import my_logger, my_timer
from crypr.features.build import make_features, series_to_predict_matrix, data_to_supervised
from sklearn.base import TransformerMixin
from crypr.features.build import continuous_wavelet_transform, make_single_feature
import numpy as np


class Preprocesser(TransformerMixin):
    @my_logger
    @my_timer
    def __init__(self, production, Tx, Ty, name):
        self.production = production
        self.Tx, self.Ty = Tx, Ty
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class SimplePreprocessor(Preprocesser):
    @my_logger
    @my_timer
    def __init__(self, production, target_col, Tx, Ty, moving_averages,  name):
        self.production = production
        self.target_col = target_col
        self.Tx, self.Ty = Tx, Ty
        self.moving_averages = moving_averages
        self.name = name

    @my_logger
    @my_timer
    def transform(self, X):
        fe = make_features(X, self.target_col, self.moving_averages)
        if self.production:
            X = series_to_predict_matrix(fe, n_in=self.Tx, dropnan=True)
            return X
        else:
            X, y = data_to_supervised(fe, self.Tx, self.Ty)
            return X, y
    # @my_logger
    # @my_timer
    # def save_output(self, path):
    #     if self.X:
    #         self.X.to_csv('{}/X_{}.csv'.format(path, self.name))
    #         print('Feature data saved to: {}/X_{}.csv'.format(path, self.name))
    #     if self.y:
    #         self.y.to_csv('{}/y_{}.csv'.format(path, self.name))
    #         print('Target data saved to: {}/y_{}.csv'.format(path, self.name))




class CWTPreprocessor(Preprocesser):

    @my_logger
    @my_timer
    def __init__(self, production, target_col, Tx, Ty, N, wavelet, name):
        self.production = production
        self.target_col = target_col
        self.Tx, self.Ty = Tx, Ty
        self.N = N
        self.wavelet = wavelet
        self.name = name

    @my_logger
    @my_timer
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

    # @my_logger
    # @my_timer
    # def save_output(self, path):
    #     if self.X:
    #         np.save(self.X, '{}/X_{}_{}x{}.npy'.format(path, self.name, self.Tx, self.N))
    #         print('Feature data saved to: {}/X_CWT_{}_{}x{}.npy'.format(path, self.wavelet, self.Tx, self.N))
    #     if self.y:
    #         np.save(self.y, '{}/y_{}_{}x{}.npy'.format(path, self.name, self.Tx, self.N))
    #         print('Target data saved to: {}/y_CWT_{}_{}x{}.npy'.format(path, self.wavelet, self.Tx, self.N))