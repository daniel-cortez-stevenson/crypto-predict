# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from crypr.data import get_data
import os

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    coins=['BTC', 'ETH']

    for coin in coins:
        coin_data = get_data.retrieve_all_data(coin=coin, num_hours=46000, comparison_symbol='USD')
        coin_data.to_csv(project_path + '/data/raw/{}.csv'.format(coin))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    main()
