"""Check That Preprocessors are Functioning Correctly"""
import unittest
from crypr.util import utc_timestamp_now
from crypr.cryptocompare import CryptocompareAPI, retrieve_all_data


class TestCryptocompare(unittest.TestCase):
    """Base class for testing Cryptocompare API calls"""
    def setUp(self):
        self.columns = ['volumeto', 'volumefrom', 'open', 'high', 'close', 'low', 'time', 'timestamp']
        self.coin = 'BTC'
        self.comparison_sym = 'USD'
        self.end_to_time = utc_timestamp_now()
        self.limit = 2000
        self.exchange = 'CCCAGG'
        self.num_hours = [100, 2000, 4000]
        self.seconds_between_obs = 3600
        self.acceptable_codes = [200]

    def tearDown(self):
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
        r = CryptocompareAPI(fsym=self.coin, tsym=self.comparison_sym, toTs=self.end_to_time,
                             limit=self.limit, e=self.exchange).retrieve_hourly()
        self.responseCheck(r)


class TestRetrieveAllData(TestCryptocompare):
    def runTest(self):
        for hours in self.num_hours:
            data = retrieve_all_data(coin=self.coin, num_hours=hours, comparison_symbol=self.comparison_sym,
                                     exchange=self.exchange, end_time=self.end_to_time)
            self.shapeCheck(data=data, num_hours=hours)
            self.columnsCheck(data=data)
            self.ascendingUniqueCheck(data=data)
            self.equalSpacingCheck(data=data)
