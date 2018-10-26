from crypr.tests.unit_decorator import my_logger, my_timer
from crypr.features.build_features import continuous_wavelet_transform, make_single_feature, series_to_predict_matrix, data_to_supervised
import numpy as np


class WavePreprocesser(object):

    @my_logger
    @my_timer
    def __init__(self, data, target, Tx, Ty, resolution, name):
        self.data = data
        self.target = target
        self.Tx, self.Ty = Tx, Ty
        self.resolution = resolution
        self.name = name

    @my_logger
    @my_timer
    def preprocess_train(self, wavelet='RICKER'):
        fe = make_single_feature(self.data, 'close')
        self.X, self.y = data_to_supervised(input_df=fe, Tx=self.Tx, Ty=self.Ty)
        self.X = continuous_wavelet_transform(self.X, N=self.resolution, wavelet=wavelet)
        return self.X, self.y

    @my_logger
    @my_timer
    def preprocess_predict(self, wavelet='RICKER'):
        fe = make_single_feature(self.data, 'close')
        self.X = series_to_predict_matrix(fe.target.tolist(), n_in=self.Tx, dropnan=True)
        self.X = continuous_wavelet_transform(self.X, N=self.resolution, wavelet=wavelet)
        return self.X

    @my_logger
    @my_timer
    def save_output(self, path):
        if self.X:
            np.save(self.X, '{}/X_{}_{}x{}.npy'.format(path, self.name, self.Tx, self.resolution))
            print('Feature data saved to: {}/X_{}_{}x{}.npy'.format(path, self.name, self.Tx, self.resolution))
        if self.y:
            np.save(self.y, '{}/y_{}_{}x{}.npy'.format(path, self.name, self.Tx, self.resolution))
            print('Target data saved to: {}/y_{}_{}x{}.npy'.format(path, self.name, self.Tx, self.resolution))