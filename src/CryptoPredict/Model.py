from src.tests.unit_decorator import my_logger, my_timer
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Model(object):

    @my_logger
    @my_timer
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    @my_logger
    @my_timer
    def fit(self):
        self.parameters = {
          'objective':'reg:linear',
          'learning_rate': .07,
          'max_depth': 3,
          'min_child_weight': 4,
          'silent': 1,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'n_estimators': 400,
          'early_stopping_rounds':50
        }
        self.regressor = XGBRegressor(
            **self.parameters
        )
        self.regressor.fit(self.X_train, self.y_train)
        self.train_y_predicted = self.regressor.predict(self.X_train)
        self.train_mae = mean_absolute_error(y_pred=self.train_y_predicted, y_true=self.y_train)
        self.train_rmse = np.sqrt(mean_squared_error(y_pred=self.train_y_predicted, y_true=self.y_train))
        return [self.train_rmse, self.train_mae]

    @my_logger
    @my_timer
    def predict(self):
        self.test_y_predicted = self.regressor.predict(self.X_test)
        self.test_mae = mean_absolute_error(y_pred=self.test_y_predicted, y_true=self.y_test)
        self.test_rmse = np.sqrt(mean_squared_error(y_pred=self.test_y_predicted, y_true=self.y_test))
        print("Test RMSE for regressor:\n {}\n".format(self.test_rmse))
        return [self.test_rmse, self.test_mae]
