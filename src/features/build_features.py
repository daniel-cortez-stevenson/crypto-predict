import pandas as pd
from sklearn.model_selection import train_test_split


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
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


def truncate(input_df, last_hours_to_keep)->pd.DataFrame:
    df=input_df.copy()
    return df.iloc[-last_hours_to_keep:,:]


def calc_target(input_df, target)->pd.DataFrame:
    df=input_df.copy()
    df['target']=df[target].pct_change()*100
    return df


def calc_volume_ma(input_df, lags)->pd.DataFrame:
    df=input_df.copy()
    for MA in lags:
        df['vt_ma' + str(MA)] = df.volumeto.rolling(MA).mean()
        df['vf_ma' + str(MA)] = df.volumefrom.rolling(MA).mean()
    return df


def data_to_supervised(input_df, Tx, Ty)->(pd.DataFrame, pd.Series):
    X = series_to_supervised(data=input_df, n_in=Tx, n_out=Ty)
    y = series_to_supervised(data=list(input_df['target']), n_in=Tx, n_out=Ty)
    return X, y


def ttsplit_and_trim(X, y, test_size, n_features, Ty):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    y_train = y_train.iloc[:,-Ty:]
    y_test = y_test.iloc[:,-Ty:]
    X_train = X_train.iloc[:,:-(Ty*n_features)]
    X_test = X_test.iloc[:,:-(Ty*n_features)]
    return X_train, X_test, y_train, y_test


def save_preprocessing_output(X_train, X_test, y_train, y_test, sym, Tx, Ty, max_lag):
    X_train.to_csv('../data/processed/X_train_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag),index=False)
    X_test.to_csv('../data/processed/X_test_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag),index=False)
    y_train.to_csv('../data/processed/y_train_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag),index=False)
    y_test.to_csv('../data/processed/y_test_{}_tx{}_ty{}_flag{}.csv'.format(sym, Tx, Ty, max_lag),index=False)


def make_features(input_df, target_col, moving_average_lags, train_on_x_last_hours=None) -> pd.DataFrame:
    df=input_df.copy()
    df=df[['open', 'high', 'close', 'low', 'volumeto', 'volumefrom']]
    if train_on_x_last_hours:
        df = df.pipe(truncate, train_on_x_last_hours)
    return df\
        .pipe(calc_target, target_col) \
        .pipe(calc_volume_ma, moving_average_lags) \
        .dropna(how='any', axis=0)