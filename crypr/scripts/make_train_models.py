"""Main script to train models for the API to serve"""
import logging
import os
import click
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from crypr.models import RegressionModel
from crypr.util import get_project_path
from crypr.zoo import build_ae_lstm


@click.command()
@click.option('-e', '--epochs', default=10, type=click.INT,
              help='Number of epochs to run for each model.')
@click.option('-v', '--verbose', default=1, type=click.INT,
              help='Verbosity of logging output.')
def main(epochs, verbose):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Creating and training models for API...')

    project_path = get_project_path()
    input_dir = os.path.join(project_path, 'data', 'processed')
    output_dir = os.path.join(project_path, 'models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Data params
    coins = ['BTC', 'ETH']
    ty = 1
    tx = 72
    num_channels = 1
    wavelet = 'haar'

    # Model params
    batch_size = 32
    learning_rate = .001
    loss = 'mae'
    beta_1 = 0.9
    beta_2 = 0.999
    decay = 0.01
    model_type = 'ae_lstm'

    for coin in coins:
        logger.info('Loading preprocessed {} data from {}'.format(coin, input_dir))

        X_train = np.load('{}/X_train_{}_{}_smooth_{}.npy'.format(input_dir, coin, wavelet, tx))
        X_test = np.load('{}/X_test_{}_{}_smooth_{}.npy'.format(input_dir, coin, wavelet, tx))
        y_train = np.load('{}/y_train_{}_{}_smooth_{}.npy'.format(input_dir, coin, wavelet, tx))
        y_test = np.load('{}/y_test_{}_{}_smooth_{}.npy'.format(input_dir, coin, wavelet, tx))

        logger.info('Building model {}...'.format(model_type))
        if model_type == 'ae_lstm':
            estimator = build_ae_lstm(num_inputs=X_train.shape[-1], num_channels=num_channels, num_outputs=ty)
            model = RegressionModel(estimator, model_type)
        else:
            raise ValueError('Model type {} is not supported. Exiting.'.format(model_type))
        logger.info(model.estimator.summary())

        tb_log_dir = os.path.join(output_dir, 'logs')
        tensorboard = TensorBoard(log_dir=tb_log_dir,
                                  histogram_freq=0, batch_size=batch_size,
                                  write_graph=True, write_grads=False, write_images=False)

        opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
        model.estimator.compile(loss=loss, optimizer=opt)

        logger.info('Training model for {} epochs ...'.format(epochs))
        logger.info('Track model fit with `tensorboard --logdir {}`'.format(tb_log_dir))

        model.fit(
            X_train, [X_train, y_train],
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, [X_test, y_test]),
            callbacks=[tensorboard],
            verbose=verbose
        )

        model_filename = '{}_smooth_{}x{}_{}_{}.h5'.format(model_type, num_channels, tx, wavelet, coin)
        output_path = os.path.join(output_dir, model_filename)
        logger.info('Saving trained model to {}...'.format(output_path))
        model.estimator.save(output_path)
