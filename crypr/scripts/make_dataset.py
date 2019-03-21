"""Main raw data download script"""
import logging
from os.path import join
from os import makedirs
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

    output_dir = join(get_project_path(), 'data', 'raw')
    makedirs(output_dir, exist_ok=True)

    coins = ['BTC', 'ETH']

    for coin in coins:
        logger.info('Retrieving {} coin data from API...'.format(coin))
        raw_df = retrieve_all_data(coin=coin, num_hours=hours, comparison_symbol='USD')
        output_path = join(output_dir, coin + '.csv')
        raw_df.to_csv(output_path)
