''' Check That Preprocessors are Functioning Correctly '''

import unittest
import numpy as np
import datetime
from crypr import cryptocompare


class TestCryptocompare(unittest.TestCase):
    ''' Base class for testing cryptocompare API calls '''

    def setUp(self):
        ''' Define some unique data for validation '''
        self.columns = np.array(['volumeto', 'volumefrom', 'open', 'high', 'close', 'low', 'time', 'timestamp'])
        self.coin = 'BTC'
        self.comparison_sym = 'USD'
        self.end_to_time = (np.datetime64(datetime.datetime(2018, 6, 27)).astype('uint64') / 1e6).astype('uint32')
        self.limit = 2000
        self.exchange = 'CCCAGG'
        self.num_hours = [100, 2000, 4000]
        self.seconds_between_obs = 3600
        self.acceptable_codes = [200]

    def tearDown(self):
        # ''' Destroy unique data '''
        self.columns = None
        self.coin = None
        self.comparison_sym = None
        self.end_to_time = None
        self.limit = None
        self.exchange = None
        self.num_hours = None
        self.seconds_between_obs = None
        self.acceptable_codes = None

    def responseCheck(self, r):
        self.assertTrue(r.status_code in self.acceptable_codes)

    def shapeCheck(self, data, num_hours):
        self.assertEqual(data.shape[0], num_hours)

    def columnsCheck(self, data):
        self.assertTrue((data.columns.values == self.columns).all())

    def ascendingUniqueCheck(self, data):
        self.assertTrue(data.set_index('time').index.is_monotonic_increasing)

    def equalSpacingCheck(self, data):
        time_diffs = data['time'].diff()[1:]
        self.assertTrue((time_diffs == self.seconds_between_obs).all())


class TestRetrieveHourlyData(TestCryptocompare):

    def runTest(self):
        r = cryptocompare.retrieve_hourly_data(coin=self.coin, comparison_symbol=self.comparison_sym,
                                               to_time=self.end_to_time, limit=self.limit, exchange=self.exchange)
        self.responseCheck(r)


class TestRetrieveAllData(TestCryptocompare):

    def runTest(self):
        for hours in self.num_hours:
            data = cryptocompare.retrieve_all_data(coin=self.coin, num_hours=hours,
                                                   comparison_symbol=self.comparison_sym,
                                                   exchange=self.exchange, end_time=self.end_to_time)
            self.shapeCheck(data=data, num_hours=hours)
            self.columnsCheck(data=data)
            self.ascendingUniqueCheck(data=data)
            self.equalSpacingCheck(data=data)
