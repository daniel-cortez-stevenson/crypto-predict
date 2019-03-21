"""Interacting with CryptoCompare API (retrieving raw data)"""
import requests
from math import ceil
import pandas as pd
from datetime import datetime
from crypr.util import my_logger, utc_timestamp_now


class CryptocompareAPI(object):
    url = 'https://min-api.cryptocompare.com/data/histohour'
    default_params = {
        'fsym': 'BTC',
        'tsym': 'USD',
        'e': 'CCCAGG',
        'limit': 2000,
        'toTs': utc_timestamp_now(),
    }
    valid_param_keys = ['fsym', 'tsym', 'e', 'limit', 'toTs']

    def __init__(self, **kwargs):
        self.params = self.default_params
        self.set_params(**kwargs)

    def set_params(self, **kwargs) -> None:
        self.params.update(**kwargs)

    @my_logger
    def retrieve_hourly(self) -> requests.Response:
        self.verify_params()
        return requests.get(self.url, params=self.params)

    def verify_params(self) -> None:
        for k in self.params.keys():
            self.check_key_is_valid(k)

    def check_key_is_valid(self, k) -> None:
        if k not in self.valid_param_keys:
            msg = '{} is not a valid param. {} are valid.'.format(k, self.valid_param_keys)
            raise ValueError(msg)


@my_logger
def retrieve_all_data(coin, num_hours, comparison_symbol='USD', exchange='CCCAGG',
                      end_time=utc_timestamp_now()) -> pd.DataFrame:
    df = pd.DataFrame()
    api = CryptocompareAPI()

    if num_hours <= 2000:
        num_calls = 1
        limit = num_hours
        last_limit = limit
    else:
        limit = 2000
        num_calls = int(ceil(num_hours/limit))
        last_limit = num_hours % limit if num_hours % limit > 0 else 2000

    print('Will call API {} times'.format(num_calls))
    for i in range(num_calls):

        if i == num_calls - 1:
            limit = last_limit

        api.set_params(fsym=coin, tsym=comparison_symbol, toTs=end_time, limit=limit, e=exchange)
        r = api.retrieve_hourly()
        print('Call # {} with Response code: {}'.format(i + 1, r.status_code))

        r_data = r.json()['Data']
        end_time = r.json()['TimeFrom'] + 3600
        this_df = pd.DataFrame(r_data)

        if num_calls == 1:
            this_df = this_df.iloc[-limit:]

        df = pd.concat([this_df, df])

    df['timestamp'] = [datetime.fromtimestamp(d) for d in df.time]
    df.sort_values('timestamp', inplace=True)
    df = df[['volumeto', 'volumefrom', 'open', 'high', 'close', 'low', 'time', 'timestamp']]
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    return df
