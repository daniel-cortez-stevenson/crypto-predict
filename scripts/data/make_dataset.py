# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from crypr.data import cryptocompare
import os

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('downloading data from cryptocompare...')

    output_path='{}/data/raw'.format(project_path)
    if not os.path.exists(output_path):
        print('Making output directory...')
        os.makedirs(output_path)

    coins=['BTC', 'ETH']

    for coin in coins:
        coin_data = cryptocompare.retrieve_all_data(coin=coin, num_hours=46000, comparison_symbol='USD')
        coin_output_path='{}/{}.csv'.format(output_path, coin)
        if not os.path.exists(coin_output_path):
            def touch(path):
                print('touching {}'.format(path))
                with open(path, 'a'):
                    os.utime(path, None)
            touch(coin_output_path)
        coin_data.to_csv(coin_output_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    main()
