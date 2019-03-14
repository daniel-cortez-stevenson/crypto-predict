"""Functions to build features from raw data"""
import numpy as np
import pandas as pd
from scipy import signal
import pywt


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True) -> pd.DataFrame:
    """Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def series_to_predict_matrix(data, n_in=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in-1, -1, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i + 1)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def truncate(input_df, last_hours_to_keep) -> pd.DataFrame:
    df = input_df.copy()
    return df.iloc[-last_hours_to_keep:, :]


def calc_target(input_df, target) -> pd.DataFrame:
    df = input_df.copy()
    df['target'] = df[target].pct_change()*100
    return df


def calc_volume_ma(input_df, lags) -> pd.DataFrame:
    df = input_df.copy()
    for MA in lags:
        df['vt_ma' + str(MA)] = df.volumeto.rolling(MA).mean()
        df['vf_ma' + str(MA)] = df.volumefrom.rolling(MA).mean()
    return df


def data_to_supervised(input_df, Tx, Ty) -> (pd.DataFrame, pd.Series):
    n_features = input_df.columns.values.size
    X = series_to_supervised(data=input_df, n_in=Tx, n_out=Ty).iloc[:, :-(Ty*n_features)]
    y = series_to_supervised(data=list(input_df['target']), n_in=Tx, n_out=Ty).iloc[:, -Ty:]
    return X, y


def save_preprocessing_output(X_train, X_test, y_train, y_test, sym, Tx, Ty, max_lag):
    X_train.to_csv('../data/processed/X_train_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag), index=False)
    X_test.to_csv('../data/processed/X_test_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag), index=False)
    y_train.to_csv('../data/processed/y_train_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag), index=False)
    y_test.to_csv('../data/processed/y_test_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag), index=False)
    return None


def make_features(input_df, target_col, moving_average_lags, train_on_x_last_hours=None) -> pd.DataFrame:
    df = input_df.copy()
    df = df[['open', 'high', 'close', 'low', 'volumeto', 'volumefrom']]
    if train_on_x_last_hours:
        df = df.pipe(truncate, train_on_x_last_hours)
    return df \
        .pipe(calc_target, target_col) \
        .pipe(calc_volume_ma, moving_average_lags) \
        .dropna(how='any', axis=0)


def make_single_feature(input_df, target_col, train_on_x_last_hours=None):
    df = input_df.copy()
    df = df.loc[:, [target_col]]
    if train_on_x_last_hours:
        df = df.pipe(truncate, train_on_x_last_hours)
    return df\
        .pipe(calc_target, target_col) \
        .dropna(how='any', axis=0)


def continuous_wavelet_transform(input_df, N, wavelet='RICKER'):
    widths = np.arange(1, N + 1)

    if wavelet == 'RICKER':
        wt_transform_fun = lambda x: signal.cwt(x, wavelet=signal.ricker, widths=widths)
    elif wavelet == 'MORLET':
        wt_transform_fun = lambda x: pywt.cwt(x, scales=widths, wavelet='morl')[0]
    else:
        raise NotImplementedError

    X_cwt_coef = np.apply_along_axis(func1d=wt_transform_fun, axis=-1, arr=input_df.values)
    return X_cwt_coef


def discrete_wavelet_transform_smooth(input_df, wavelet):

    def dwt_smooth(x, wavelet):
        cA, cD = pywt.dwt(x, wavelet)

        def make_threshold(x):
            return np.std(x) * np.sqrt(1 * np.log(x.size))

        cAt = pywt.threshold(cA, make_threshold(cA), mode='soft')
        cDt = pywt.threshold(cD, make_threshold(cD), mode='soft')
        tx = pywt.idwt(cAt, cDt, wavelet)
        return tx

    X_smooth = np.apply_along_axis(func1d=lambda x: dwt_smooth(x, wavelet), axis=-1, arr=input_df.values)
    return X_smooth