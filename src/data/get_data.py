import requests
import numpy as np
import pandas as pd
import datetime

def retrieve_hourly_data(coin,
                         to_time=(np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32'),
                         limit=2000):
    params = {
        'fsym': coin,
        'tsym': 'USD',
        'limit': limit,
        'toTs': to_time
    }
    url = "https://min-api.cryptocompare.com/data/histohour"
    r = requests.get(url, params=params)
    return r


def retrieve_all_data(coin, num_hours):
    df = pd.DataFrame()
    end_time = (np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32')

    limit = 0

    if num_hours < 2000:
        num_calls = 1
        limit = num_hours
    else:
        num_calls = np.int(num_hours / limit)
        limit = 2000

    for i in range(num_calls):
        r = retrieve_hourly_data(coin, end_time, limit)
        end_time = r.json()['TimeFrom']
        r_data = r.json()['Data']
        this_df = pd.DataFrame()

        for k in r_data[0].keys():
            k_data = []
            for d in r_data:
                k_data.append(d[k])
            this_df[k] = k_data
        df = pd.concat([df, this_df])
    df.reset_index(inplace=True)
    df.drop_duplicates(inplace=True)
    return df