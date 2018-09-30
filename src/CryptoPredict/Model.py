from src.tests.unit_decorator import my_logger, my_timer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Model(object):

    @my_logger
    @my_timer
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    @my_logger
    @my_timer
    def set_parameters(self, parameters):
        self.parameters = parameters
        self.estimator.set_params(**self.parameters)

    @my_logger
    @my_timer
    def fit(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        self.train_y_predicted = self.estimator.predict(X_train)
        self.train_mae = mean_absolute_error(y_pred=self.train_y_predicted, y_true=y_train)
        self.train_rmse = np.sqrt(mean_squared_error(y_pred=self.train_y_predicted, y_true=y_train))
        return self.train_rmse, self.train_mae

    @my_logger
    @my_timer
    def predict(self, X_test, y_test):
        self.test_y_predicted = self.estimator.predict(X_test)
        self.test_mae = mean_absolute_error(y_pred=self.test_y_predicted, y_true=y_test)
        self.test_rmse = np.sqrt(mean_squared_error(y_pred=self.test_y_predicted, y_true=y_test))
        print("Test RMSE for Model:\n {}\n".format(self.test_rmse))
        return self.test_rmse, self.test_mae

    @my_logger
    @my_timer
    def save_estimator(self, path):
        self.estimator.to_pickle('{}/{}.pkl'.format(path, self.name))