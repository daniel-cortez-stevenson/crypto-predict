# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from crypr.features.build import make_single_feature, data_to_supervised, discrete_wavelet_transform_smooth
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making features from raw data')

    coins = ['BTC', 'ETH']
    LAST_N_HOURS = 16000
    TARGET = 'close'
    Tx = 72
    Ty = 1
    TEST_SIZE = 0.05
    output_path = '{}/data/processed'.format(project_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    WAVELET='haar'

    for SYM in coins:
        data_path = '{}/data/raw/{}.csv'.format(project_path, SYM)
        data = pd.read_csv(data_path, index_col=0)
        df = make_single_feature(input_df=data, target_col=TARGET, train_on_x_last_hours=LAST_N_HOURS)
        X, y = data_to_supervised(input_df=pd.DataFrame(df['target']), Tx=Tx, Ty=Ty)
        X_smoothed = discrete_wavelet_transform_smooth(X, WAVELET)
        X_train, X_test, y_train, y_test = train_test_split(X_smoothed, y, test_size=TEST_SIZE, shuffle=False)

        np.save(arr=X_train, file='{}/X_train_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))
        np.save(arr=X_test, file='{}/X_test_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))
        np.save(arr=y_train, file='{}/y_train_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))
        np.save(arr=y_test, file='{}/y_test_{}_{}_smooth_{}'.format(output_path, SYM, WAVELET, Tx))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    main()
