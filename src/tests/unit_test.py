import unittest
import numpy as np
import datetime
from src.data.get_data import retrieve_all_data
from src.features.build_features import *
from src.CryptoPredict.Model import Model
from xgboost import XGBRegressor

class TestInput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # print('setupClass')
        pass

    @classmethod
    def tearDownClass(cls):
        # print('teardownClass')
        pass

    def setUp(self):
        print('setUp')
        np.random.seed(31337)

        SYM = 'ETH'
        LAST_N_HOURS = 16000
        MOVING_AVERAGE_LAGS = [6, 12, 24, 48, 72]
        TARGET = 'close'
        Tx = 72
        Ty = 1
        TEST_SIZE = 0.05

        data = retrieve_all_data(coin=SYM, num_hours=LAST_N_HOURS, comparison_symbol='USD',
                                 end_time=(np.datetime64(datetime.datetime(2018,6,27)).astype('uint64') / 1e6).astype('uint32'))

        df = data[['open', 'high', 'close', 'low', 'volumeto', 'volumefrom']] \
            .pipe(calc_target, TARGET) \
            .pipe(calc_volume_ma, MOVING_AVERAGE_LAGS) \
            .dropna(how='any', axis=0)

        N_FEATURES = len(df.columns)

        X, y = data_to_supervised(df, Tx, Ty)

        self.X_train, self.X_test, self.y_train, self.y_test = ttsplit_and_trim(X, y, TEST_SIZE, N_FEATURES, Ty)

        self.train_mae = 0.8968620419351427
        self.train_rmse = 1.4206723934489915
        self.test_mae = 0.6681529049518523
        self.test_rmse = 1.0195120681618908


    def tearDown(self):
        # print('tearDown')
        pass

    def test_fit(self):
        np.random.seed(31337)
        self.ta = Model(XGBRegressor(), 'xgboost_regressor')
        self.parameters = {
            'objective': 'reg:linear',
            'learning_rate': .07,
            'max_depth': 10,
            'min_child_weight': 4,
            'silent': 1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'n_estimators': 20
        }
        self.ta.set_parameters(self.parameters)
        self.assertEqual(self.ta.fit(self.X_train, self.y_train), [self.train_rmse, self.train_mae])

    def test_predict(self):
        np.random.seed(31337)
        self.ta = Model(XGBRegressor(), 'xgboost_regressor')
        self.parameters = {
            'objective': 'reg:linear',
            'learning_rate': .07,
            'max_depth': 10,
            'min_child_weight': 4,
            'silent': 1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'n_estimators': 20
        }
        self.ta.set_parameters(self.parameters)
        self.ta.fit(self.X_train, self.y_train)
        self.assertEqual(self.ta.predict(self.X_test, self.y_test), [self.test_rmse, self.test_mae])


if __name__ == '__main__':

    # run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)