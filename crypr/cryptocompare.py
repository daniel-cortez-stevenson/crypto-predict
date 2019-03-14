"""Interacting with CryptoCompare API (retrieving raw data)"""
import requests
import numpy as np
import pandas as pd
import datetime
from crypr.decorator import my_logger


@my_logger
def retrieve_hourly_data(coin, comparison_symbol='USD',
                         to_time=(np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32'),
                         limit=2000, exchange='CCCAGG'):
    params = {
        'fsym': coin.upper(),
        'tsym': comparison_symbol.upper(),
        'limit': limit,
        'toTs': to_time,
        'e': exchange
    }

    url = "https://min-api.cryptocompare.com/data/histohour"
    r = requests.get(url, params=params)
    return r


@my_logger
def retrieve_all_data(coin, num_hours, comparison_symbol='USD', exchange='CCCAGG',
                      end_time=(np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32')):
    df = pd.DataFrame()

    if num_hours <= 2000:
        num_calls = 1
        limit = num_hours
        last_limit = limit
    else:
        limit = 2000
        num_calls = np.int(np.ceil(num_hours / limit))
        last_limit = num_hours % limit if num_hours % limit > 0 else 2000

    print('Will call API {} times'.format(num_calls))
    for i in range(num_calls):
        if i == num_calls - 1:
            limit = last_limit

        r = retrieve_hourly_data(coin=coin, comparison_symbol=comparison_symbol,
                                 to_time=end_time, limit=limit, exchange=exchange)
        print('Call # {} with Response code: {}'.format(i + 1, r.status_code))
        r_data = r.json()['Data']

        end_time = r.json()['TimeFrom'] + 3600

        this_df = pd.DataFrame(r_data)

        if num_calls == 1:
            this_df = this_df.iloc[-limit:]

        df = pd.concat([this_df, df])

    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]

    df.sort_values('timestamp', inplace=True)

    df = df[['volumeto', 'volumefrom', 'open', 'high', 'close', 'low', 'time', 'timestamp']]

    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    return df
