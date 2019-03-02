# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
from crypr.base.models import RegressionModel
import os
import numpy as np
from crypr.models.zoo import build_ae_lstm
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


@click.command()
@click.option("-e", "--epochs", default=10, type=click.INT,
              help="Number of epochs to run for each model.")
def main(epochs):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    
    logger = logging.getLogger(__name__)
    logger.info('Creating and Training RNN Models...')

    coins = ['BTC', 'ETH']
    Ty = 1
    Tx = 72
    num_channels = 1
    input_path = '{}/data/processed'.format(project_path)
    output_path = '{}/models'.format(project_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    WAVELET = 'haar'

    batch_size = 32
    learning_rate = .001
    beta_1 = 0.9
    beta_2 = 0.999
    decay = 0.01

    model_type = 'ae_lstm'

    for SYM in coins:
        X_train = np.load('{}/X_train_{}_{}_smooth_{}.npy'.format(input_path, SYM, WAVELET, Tx))
        X_test = np.load('{}/X_test_{}_{}_smooth_{}.npy'.format(input_path, SYM, WAVELET, Tx))
        y_train = np.load('{}/y_train_{}_{}_smooth_{}.npy'.format(input_path, SYM, WAVELET, Tx))
        y_test = np.load('{}/y_test_{}_{}_smooth_{}.npy'.format(input_path, SYM, WAVELET, Tx))

        if model_type == 'ae_lstm':
            estimator = build_ae_lstm(num_inputs=X_train.shape[-1], num_channels=num_channels, num_outputs=Ty)
            model = RegressionModel(estimator, model_type)
        else:
            model = None

        print(model.estimator.summary())

        tb_log_dir = '{}/logs'.format(output_path)
        tensorboard = TensorBoard(log_dir=tb_log_dir,
                                  histogram_freq=0, batch_size=batch_size,
                                  write_graph=True, write_grads=False, write_images=False)
        opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
        model.estimator.compile(loss='mae', optimizer=opt)

        print('Training model for {} epochs ...'.format(epochs))
        print('Track model fit with `tensorboard --logdir {}`'.format(tb_log_dir))
        model.fit(
            X_train, [X_train, y_train],
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, [X_test, y_test]),
            callbacks=[tensorboard],
            verbose=0
        )

        model.save_estimator(path='{}/{}_smooth_{}x{}_{}_{}.h5'.format(
            output_path, model_type, num_channels, Tx, WAVELET, SYM)
        )
