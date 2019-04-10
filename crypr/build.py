"""Functions to build features from raw data"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from typing import Tuple, Union
from crypr.types import ArrayLike
from crypr.transformers import PassthroughTransformer, PercentChangeTransformer, MovingAverageTransformer


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


def series_to_predict_matrix(data, n_in=1, dropnan=True) -> pd.DataFrame:
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


def data_to_supervised(input_df, target_ix: Union[str, int], Tx: int, Ty: int) -> Tuple[pd.DataFrame, pd.Series]:
    n_features = input_df.shape[1]
    X = series_to_supervised(data=input_df, n_in=Tx, n_out=Ty).iloc[:, :-(Ty*n_features)]
    y = series_to_supervised(data=input_df.iloc[:, [target_ix]], n_in=Tx, n_out=Ty).iloc[:, -Ty:]
    return X, y


def make_features(input_df, target_col, keep_cols=None, ma_lags=None, ma_cols=None, n_samples=None) -> pd.DataFrame:
    transformers = list()
    if keep_cols:
        transformers.extend([('passthrough', PassthroughTransformer(), keep_cols)])
    if ma_lags and ma_cols:
        transformers.extend([('ma' + str(n), MovingAverageTransformer(n), ma_cols) for n in ma_lags])
    transformers.extend([('target', PercentChangeTransformer(), [target_col])])
    ct = ColumnTransformer(transformers=transformers, remainder='drop', n_jobs=-1)

    arr = ct.fit_transform(input_df)
    arr = strip_nan_rows(arr)
    if n_samples:
        arr = keep_last_n_rows(arr, n_samples)
    return pd.DataFrame(data=arr, columns=list(ct.get_feature_names()))


def strip_nan_rows(arr: ArrayLike) -> ArrayLike:
    return arr[~np.isnan(arr).any(axis=1)]


def keep_last_n_rows(arr: ArrayLike, n: int) -> ArrayLike:
    return arr[-n:, :]


def make_3d(arr: ArrayLike, tx: int, num_channels: int) -> np.ndarray:
    arr = np.expand_dims(arr, axis=-1)
    return np.reshape(arr, (-1, tx, num_channels))
