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
    n_vars = 1 if type(data) in (list, pd.Series) else data.shape[1]
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

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg


def series_to_predict_matrix(data, n_in=1, dropnan=True):
    n_vars = 1 if type(data) in (list, pd.Series) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in - 1, -1, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i + 1)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg


def truncate(input_df, keep_last=None) -> pd.DataFrame:
    if keep_last is not None:
        return input_df.iloc[-keep_last:, :]
    return input_df


def calc_target(input_df, target) -> pd.DataFrame:
    df = input_df.copy()
    pct_change = df[target].pct_change()*100
    df['target'] = pct_change
    return df


def calc_volume_ma(input_df, lags) -> pd.DataFrame:
    df = input_df.copy()
    for ma in lags:
        df['vt_ma' + str(ma)] = df.volumeto.rolling(ma).mean()
        df['vf_ma' + str(ma)] = df.volumefrom.rolling(ma).mean()
    return df


def data_to_supervised(input_df, Tx, Ty) -> (pd.DataFrame, pd.Series):
    n_features = input_df.shape[1]
    X = series_to_supervised(data=input_df, n_in=Tx, n_out=Ty).iloc[:, :-(Ty*n_features)]
    y = series_to_supervised(data=input_df['target'], n_in=Tx, n_out=Ty).iloc[:, -Ty:]
    return X, y


def make_features(input_df, target_col, moving_average_lags, n_samples=None) -> pd.DataFrame:
    return input_df[['open', 'high', 'close', 'low', 'volumeto', 'volumefrom']] \
        .pipe(truncate, n_samples) \
        .pipe(calc_target, target_col) \
        .pipe(calc_volume_ma, moving_average_lags) \
        .dropna(how='any', axis=0)


def make_single_feature(input_df, target_col, n_samples=None) -> pd.DataFrame:
    return input_df.loc[:, [target_col]] \
        .pipe(truncate, n_samples) \
        .pipe(calc_target, target_col) \
        .dropna(how='any', axis=0)


def continuous_wavelet_transform(input_df, N, wavelet='RICKER') -> np.ndarray:
    widths = np.arange(1, N + 1)
    if wavelet == 'RICKER':
        wt_transform_fun = lambda x: signal.cwt(x, wavelet=signal.ricker, widths=widths)
    elif wavelet == 'MORLET':
        wt_transform_fun = lambda x: pywt.cwt(x, scales=widths, wavelet='morl')[0]
    else:
        raise ValueError('{} wavelet is not supported'.format(wavelet))
    return np.apply_along_axis(func1d=wt_transform_fun, axis=-1, arr=input_df.values)


def discrete_wavelet_transform(input_df, wavelet, smooth_factor=1) -> np.ndarray:
    return np.apply_along_axis(func1d=lambda x: dwt_smoother(x, wavelet, smooth_factor),
                               axis=-1, arr=input_df.values)


def dwt_smoother(x, wavelet, smooth_factor=1) -> np.ndarray:
    cA, cD = pywt.dwt(x, wavelet)
    cAt = pywt.threshold(cA, smoothing_threshold(cA, smooth_factor), mode='soft')
    cDt = pywt.threshold(cD, smoothing_threshold(cD, smooth_factor), mode='soft')
    tx = pywt.idwt(cAt, cDt, wavelet)
    return tx


def smoothing_threshold(x, factor=1) -> float:
    return np.std(x) * np.sqrt(factor*np.log(x.size))
