from crypr.tests.unit_decorator import my_logger, my_timer
from crypr.models.load_model import load_from_pickle
from keras.models import load_model


class SavedModel(object):

    @my_logger
    @my_timer
    def __init__(self, path):
        self.path = path
        self.ext = self.path.split('.')[-1]


    @my_logger
    @my_timer
    def load(self):
        if self.ext == 'pkl':
            self.model = load_from_pickle(self.path)
        elif self.ext == 'h5':
            self.model = load_model(self.path)
        else:
            print('WARNING: File Extension {} not supported.'.format(self.ext))


    @my_logger
    @my_timer
    def predict(self, X):
        self.X = X
        self.prediction = self.model.predict(self.X)
        return self.prediction
