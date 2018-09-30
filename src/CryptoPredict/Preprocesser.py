from src.tests.unit_decorator import my_logger, my_timer
from src.features.build_features import make_features, series_to_supervised, data_to_supervised


class Preprocesser(object):

    @my_logger
    @my_timer
    def __init__(self, data, target, Tx, Ty, moving_averages,  name):
        self.data = data
        self.target = target
        self.Tx, self.Ty = Tx, Ty
        self.moving_averages = moving_averages
        self.name = name

    @my_logger
    @my_timer
    def preprocess_train(self):
        fe = make_features(self.data, self.target, self.moving_averages)
        self.X, self.y = data_to_supervised(fe, self.Tx, self.Ty)
        return self.X, self.y, fe.columns.size

    @my_logger
    @my_timer
    def preprocess_predict(self):
        fe = make_features(self.data, self.target, self.moving_averages)
        self.X = series_to_supervised(fe, n_in=self.Tx, n_out=0, dropnan=True)
        return self.X, fe.columns.size

    @my_logger
    @my_timer
    def save_output(self, path):
        if self.X:
            self.X.to_csv('{}/X_{}.csv'.format(path, self.name))
            print('Feature data saved to: {}/X_{}.csv'.format(path, self.name))
        if self.y:
            self.y.to_csv('{}/y_{}.csv'.format(path, self.name))
            print('Target data saved to: {}/y_{}.csv'.format(path, self.name))