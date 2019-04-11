"""Main script to train models for the API to serve"""
import click
from os.path import join
from os import makedirs
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from crypr.models import RegressionModel
from crypr.zoo import build_ae_lstm, LSTM_triggerNG
from crypr.util import get_project_path, my_logger
from collections import namedtuple

@click.command()
@click.option('-e', '--epochs', default=10, type=click.INT,
              help='Number of epochs to run for each model.')
@click.option('-v', '--verbose', default=1, type=click.INT,
              help='Verbosity of logging output.')
@my_logger
def main(epochs, verbose):
    print('Creating and training models for API...')

    input_dir = join(get_project_path(), 'data', 'processed')
    output_dir = join(get_project_path(), 'models')
    makedirs(output_dir, exist_ok=True)

    # Data params
    coins = ['BTC', 'ETH']

    # Model params
    batch_size = 64
    learning_rate = 1e-3
    loss = 'mse'
    metrics = ['mae']
    beta_1 = 0.9
    beta_2 = 0.999
    decay = 0.01
    model_type = 'lstm_ng'

    for coin in coins:
        print('Loading preprocessed {} data from {}'.format(coin, input_dir))

        X_train = np.load(join(input_dir, 'X_train_{}.npy'.format(coin)))
        X_test = np.load(join(input_dir, 'X_test_{}.npy'.format(coin)))
        y_train = np.load(join(input_dir, 'y_train_{}.npy'.format(coin)))
        y_test = np.load(join(input_dir, 'y_test_{}.npy'.format(coin)))

        Outputs = namedtuple('Outputs', 'train, test')
        print('Building model {}...'.format(model_type))
        tx, num_channels = X_train.shape[1:]
        ty = y_train.shape[1]
        if model_type == 'ae_lstm':
            estimator = build_ae_lstm(tx=tx, num_channels=num_channels, num_outputs=ty)
            model = RegressionModel(estimator)
            outputs = Outputs([X_train, y_train], [X_test, y_test])
        elif model_type == 'lstm_ng':
            estimator = LSTM_triggerNG(tx=tx, num_channels=num_channels, num_outputs=ty)
            model = RegressionModel(estimator)
            outputs = Outputs([y_train], [y_test])
        else:
            raise ValueError('Model type {} is not supported. Exiting.'.format(model_type))
        print(model.estimator.summary())

        tb_log_dir = join(output_dir, 'logs')
        tensorboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, batch_size=batch_size,
                                  write_graph=True, write_grads=False, write_images=False)

        opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
        model.estimator.compile(loss=loss, metrics=metrics, optimizer=opt)

        print('Training model for {} epochs ...'.format(epochs))
        print('Track model fit with `tensorboard --logdir {}`'.format(tb_log_dir))

        model.fit(
            X_train, outputs.train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, outputs.test),
            callbacks=[tensorboard],
            verbose=verbose,
        )

        model_filename = '{}_{}.h5'.format(model_type, coin)
        output_path = join(output_dir, model_filename)
        print('Saving trained model to {}...'.format(output_path))
        model.estimator.save(output_path)
