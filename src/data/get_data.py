import requests
import numpy as np
import pandas as pd
import datetime


def retrieve_hourly_data(coin,
                         comparison_symbol='USD',
                         to_time=(np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32'),
                         limit=2000,
                         exchange='CCCAGG'):
    params = {
        'fsym': coin.upper(),
        'tsym': comparison_symbol.upper(),
        'limit': limit,
        'toTs': to_time,
        'e':exchange
    }

    url = "https://min-api.cryptocompare.com/data/histohour"
    r = requests.get(url, params=params)
    return r


def retrieve_all_data(coin,
                      num_hours,
                      comparison_symbol='USD',
                      exchange='CCCAGG'):
    df = pd.DataFrame()
    end_time = (np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32')

    if num_hours <= 2000:
        num_calls = 1
        limit = num_hours
        last_limit=limit
    else:
        limit = 2000
        num_calls = np.int(np.ceil(num_hours / limit))
        last_limit = num_hours % limit


    for i in range(num_calls):
        if i == num_calls -1:
            limit = last_limit

        r = retrieve_hourly_data(coin=coin, comparison_symbol=comparison_symbol, to_time=end_time, limit=limit, exchange=exchange)

        r_data = r.json()['Data']

        end_time = r.json()['TimeFrom']

        this_df = pd.DataFrame(r_data)

        df = pd.concat([this_df, df])

    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]

    df.sort_values('timestamp', inplace=True)

    df = df[['volumeto', 'volumefrom', 'open', 'high', 'close', 'low', 'time', 'timestamp']]

    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    return df