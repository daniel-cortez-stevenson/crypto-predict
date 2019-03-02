# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
from crypr.data import cryptocompare
import os

@click.command()
@click.option("-h", "--hours", default=46000, type=click.INT,
              help="Number of hours of data to download for each coin.")
def main(hours):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    
    logger = logging.getLogger(__name__)
    logger.info('Downloading data from Cryptocompare ...')

    output_path = '{}/data/raw'.format(project_path)
    if not os.path.exists(output_path):
        print('Making output directory...')
        os.makedirs(output_path)

    coins = ['BTC', 'ETH']

    for coin in coins:
        coin_data = cryptocompare.retrieve_all_data(coin=coin, num_hours=hours, comparison_symbol='USD')
        coin_output_path = '{}/{}.csv'.format(output_path, coin)
        coin_data.to_csv(coin_output_path)
