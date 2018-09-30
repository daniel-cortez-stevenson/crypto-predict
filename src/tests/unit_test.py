import unittest
import numpy as np
import datetime
from src.data.get_data import retrieve_all_data
from src.features.build_features import *
from src.CryptoPredict.Model import Model
from src.CryptoPredict.Preprocesser import Preprocesser
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

        self.SYM = 'ETH'
        LAST_N_HOURS = 16000
        self.MOVING_AVERAGE_LAGS = [6, 12, 24, 48, 72]
        self.TARGET = 'close'
        self.Tx = 72
        self.Ty = 1
        self.TEST_SIZE = 0.05

        self.data = retrieve_all_data(coin=self.SYM, num_hours=LAST_N_HOURS, comparison_symbol='USD',
                                 end_time=(np.datetime64(datetime.datetime(2018,6,27)).astype('uint64') / 1e6).astype('uint32'))

        self.X_sample = 705.68
        self.y_sample = -0.28191361260253567

        self.X_train_sample = 88.2
        self.y_train_sample = 0.17391304347826875

        self.X_test_sample = 487.58
        self.y_test_sample = 0.9448599618077758

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

        self.train_mae = 0.8968620419351427
        self.train_rmse = 1.4206723934489915
        self.test_mae = 0.6681529049518523
        self.test_rmse = 1.0195120681618908

    def tearDown(self):
        # print('tearDown')
        pass

    def test_preprocess(self):
        np.random.seed(31337)
        preprocessor=Preprocesser(self.data, self.TARGET, self.Tx, self.Ty, self.MOVING_AVERAGE_LAGS, name='Unit_Test')
        X, y, _ = preprocessor.preprocess_train()
        X_sample = X.sample(1, random_state=0).values[0][0]
        y_sample = y.sample(1, random_state=0).values[0][0]
        self.assertEqual((X_sample, y_sample), (self.X_sample, self.y_sample))

    def test_split(self):
        np.random.seed(31337)
        preprocessor=Preprocesser(self.data, self.TARGET, self.Tx, self.Ty, self.MOVING_AVERAGE_LAGS, name='Unit_Test')
        X, y, n_features = preprocessor.preprocess_train()
        X_train, X_test, y_train, y_test = ttsplit_and_trim(X, y, self.TEST_SIZE, n_features, self.Ty)
        X_train_sample=X_train.sample(1, random_state=0).values[0][0]
        X_test_sample=X_test.sample(1, random_state=0).values[0][0]
        y_train_sample=y_train.sample(1, random_state=0).values[0][0]
        y_test_sample=y_test.sample(1, random_state=0).values[0][0]
        self.assertEqual((X_train_sample, X_test_sample, y_train_sample, y_test_sample),
                         (self.X_train_sample, self.X_test_sample, self.y_train_sample, self.y_test_sample))

    def test_fit(self):
        np.random.seed(31337)
        preprocessor = Preprocesser(self.data, self.TARGET, self.Tx, self.Ty, self.MOVING_AVERAGE_LAGS,
                                    name='Unit_Test')
        X, y, n_features = preprocessor.preprocess_train()
        X_train, X_test, y_train, y_test = ttsplit_and_trim(X, y, self.TEST_SIZE, n_features, self.Ty)
        self.ta = Model(XGBRegressor(), 'xgboost_regressor')

        self.ta.set_parameters(self.parameters)
        self.assertEqual(self.ta.fit(X_train, y_train), [self.train_rmse, self.train_mae])

    def test_predict(self):
        np.random.seed(31337)
        preprocessor = Preprocesser(self.data, self.TARGET, self.Tx, self.Ty, self.MOVING_AVERAGE_LAGS,
                                    name='Unit_Test')
        X, y, n_features = preprocessor.preprocess_train()
        X_train, X_test, y_train, y_test = ttsplit_and_trim(X, y, self.TEST_SIZE, n_features, self.Ty)
        self.ta = Model(XGBRegressor(), 'xgboost_regressor')

        self.ta.set_parameters(self.parameters)
        self.ta.fit(X_train, y_train)
        self.assertEqual(self.ta.predict(X_test, y_test), [self.test_rmse, self.test_mae])


if __name__ == '__main__':

    # run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)