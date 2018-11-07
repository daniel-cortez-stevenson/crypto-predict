# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from crypr.features.build import make_single_feature, data_to_supervised, discrete_wavelet_transform_smooth
import os
import numpy as np
from crypr.models.zoo import build_ae_lstm
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from time import time

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    coins = ['BTC', 'ETH']
    Ty = 1
    Tx = 72
    num_channels = 1
    input_path='{}/data/processed'.format(project_path)
    output_path = '{}/models'.format(project_path)
    WAVELET='haar'

    epochs = 10
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

        if len(X_train.shape) < 3:
            model_X_train = np.swapaxes(np.expand_dims(X_train, axis=-1), axis1=-2, axis2=-1)
            model_X_test = np.swapaxes(np.expand_dims(X_test, axis=-1), axis1=-2, axis2=-1)
        else:
            model_X_train = X_train
            model_X_test = X_test


        if model_type == 'ae_lstm':
            model = build_ae_lstm(num_inputs=model_X_train.shape[-1], num_channels=num_channels, num_outputs=Ty)
        else:
            model = None

        print(model.summary())

        tb_log_dir='{}/logs'.format(output_path)
        tensorboard = TensorBoard(log_dir=tb_log_dir,
                                  histogram_freq=0, batch_size=batch_size,
                                  write_graph=True, write_grads=False, write_images=False)
        opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
        model.compile(loss='mae', optimizer=opt)

        print('Fitting model ...')
        print('Track model fit with `tensorboard --logdir {}`'.format(tb_log_dir))
        fit = model.fit(model_X_train, [model_X_train, y_train],
                        shuffle=False,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(model_X_test, [model_X_test, y_test]),
                        callbacks=[tensorboard]
                        )

        model.save(filepath='{}/{}_smooth_{}x{}_{}_{}.h5'.format(output_path, model_type, num_channels, Tx, WAVELET, SYM))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    project_path = os.path.dirname(find_dotenv())
    main()
