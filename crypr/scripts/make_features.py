"""Main data preprocessing script"""
from os.path import join
from os import makedirs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from crypr.util import get_project_path, my_logger
from crypr.build import make_features, data_to_supervised, make_3d


@my_logger
def main():
    print('Making features from raw data...')

    data_dir = join(get_project_path(), 'data', 'raw')
    output_dir = join(get_project_path(), 'data', 'processed')
    makedirs(output_dir, exist_ok=True)

    coins = ['BTC', 'ETH']
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05

    for SYM in coins:
        raw_data_path = join(data_dir, SYM + '.csv')
        print('Featurizing raw {} data from {}...'.format(SYM, raw_data_path))

        raw_df = pd.read_csv(raw_data_path, index_col=0)

        feature_df = make_features(raw_df, target_col=TARGET,
                                   keep_cols=['close', 'high', 'low', 'volumeto', 'volumefrom'],
                                   ma_lags=[6, 12, 24, 48], ma_cols=['close', 'volumefrom', 'volumeto'])

        X, y = data_to_supervised(feature_df, target_ix=-1, Tx=Tx, Ty=Ty)

        num_features = int(X.shape[1]/Tx)
        X = make_3d(X, tx=Tx, num_channels=num_features)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

        np.save(arr=X_train, file=join(output_dir, 'X_train_{}'.format(SYM)))
        np.save(arr=X_test, file=join(output_dir, 'X_test_{}'.format(SYM)))
        np.save(arr=y_train, file=join(output_dir, 'y_train_{}'.format(SYM)))
        np.save(arr=y_test, file=join(output_dir, 'y_test_{}'.format(SYM)))
