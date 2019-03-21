"""Main data preprocessing script"""
import logging
from os.path import join
from os import makedirs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from crypr.util import get_project_path
from crypr.preprocessors import DWTSmoothPreprocessor


def main():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Making features from raw data...')

    data_dir = join(get_project_path(), 'data', 'raw')
    output_dir = join(get_project_path(), 'data', 'processed')
    makedirs(output_dir, exist_ok=True)

    coins = ['BTC', 'ETH']
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05
    WAVELET = 'haar'

    for SYM in coins:
        raw_data_path = join(data_dir, SYM + '.csv')
        logger.info('Featurizing raw {} data from {}...'.format(SYM, raw_data_path))

        raw_df = pd.read_csv(raw_data_path, index_col=0)

        preprocessor = DWTSmoothPreprocessor(production=False, target_col=TARGET, Tx=Tx, Ty=Ty, wavelet=WAVELET)
        X_smoothed, y = preprocessor.fit_transform(raw_df)

        X_train, X_test, y_train, y_test = train_test_split(X_smoothed, y, test_size=TEST_SIZE, shuffle=False)

        np.save(arr=X_train, file=join(output_dir, 'X_train_{}_{}_smooth_{}'.format(SYM, WAVELET, Tx)))
        np.save(arr=X_test, file=join(output_dir, 'X_test_{}_{}_smooth_{}'.format(SYM, WAVELET, Tx)))
        np.save(arr=y_train, file=join(output_dir, 'y_train_{}_{}_smooth_{}'.format(SYM, WAVELET, Tx)))
        np.save(arr=y_test, file=join(output_dir, 'y_test_{}_{}_smooth_{}'.format(SYM, WAVELET, Tx)))
