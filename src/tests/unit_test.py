import unittest
import numpy as np
import datetime
from src.data.get_data import retrieve_all_data
from src.features.build_features import *
from src.CryptoPredict.Model import Model

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

        self.train_mae = 0.8339931222304684
        self.train_rmse = 1.2384700065887169
        self.test_mae = 0.6743845487304544
        self.test_rmse = 1.0307934086762733

    def tearDown(self):
        # print('tearDown')
        pass

    def test_fit(self):
        np.random.seed(31337)
        self.ta = Model(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertEqual(self.ta.fit(), [self.train_rmse, self.train_mae])

    def test_predict(self):
        np.random.seed(31337)
        self.ta = Model(self.X_train, self.y_train, self.X_test, self.y_test)
        self.ta.fit()
        self.assertEqual(self.ta.predict(), [self.test_rmse, self.test_mae])


if __name__ == '__main__':

    # run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)