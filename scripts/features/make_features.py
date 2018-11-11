# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from crypr.base.preprocessors import DWTSmoothPreprocessor
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    logger = logging.getLogger(__name__)
    logger.info('Making features from raw data for RNN Models ...')

    coins = ['BTC', 'ETH']
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05
    output_path = '{}/data/processed'.format(project_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    WAVELET = 'haar'

    for SYM in coins:
        data_path = '{}/data/raw/{}.csv'.format(project_path, SYM)
        data = pd.read_csv(data_path, index_col=0)
        preprocessor = DWTSmoothPreprocessor(production=False, target_col=TARGET, Tx=Tx, Ty=Ty, wavelet=WAVELET,
                                             name='MakeFeatures_DWTSmoothPreprocessor_{}'.format(SYM))
        X_smoothed, y = preprocessor.fit(data).transform(data)
        X_train, X_test, y_train, y_test = train_test_split(X_smoothed, y, test_size=TEST_SIZE, shuffle=False)

        np.save(arr=X_train, file='{}/X_train_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))
        np.save(arr=X_test, file='{}/X_test_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))
        np.save(arr=y_train, file='{}/y_train_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))
        np.save(arr=y_test, file='{}/y_test_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    main()
