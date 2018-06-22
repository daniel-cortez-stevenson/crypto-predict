import requests
import numpy as np
import pandas as pd
import datetime

def retrieve_hourly_data(coin, to_time=(np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32')):
    params = {
        'fsym': coin,
        'tsym': 'USD',
        'limit': 2000,
        'toTs': to_time
    }
    url = "https://min-api.cryptocompare.com/data/histohour"
    r = requests.get(url, params=params)
    return r


def retrieve_all_data(coin, num_hours):
    df = pd.DataFrame()
    end_time = (np.datetime64(datetime.datetime.now()).astype('uint64') / 1e6).astype('uint32')
    for i in range(np.int(num_hours / 2000)):
        r = retrieve_hourly_data(coin, end_time)
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