"""Main raw data download script"""
import logging
import os
import click
from crypr.cryptocompare import retrieve_all_data
from crypr.util import get_project_path


@click.command()
@click.option("-h", "--hours", default=6000, type=click.INT,
              help="Number of hours of data to download for each coin.")
def main(hours):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Downloading data from Cryptocompare ...')

    project_path = get_project_path()

    output_dir = os.path.join(project_path, 'data', 'raw')
    if not os.path.exists(output_dir):
        logger.info('Making output directory {}...'.format(output_dir))
        os.makedirs(output_dir)

    coins = ['BTC', 'ETH']

    for coin in coins:
        logger.info('Retrieving {} coin data from API...'.format(coin))
        raw_df = retrieve_all_data(coin=coin, num_hours=hours, comparison_symbol='USD')
        output_path = os.path.join(output_dir, coin + '.csv')
        raw_df.to_csv(output_path)
