"""
Welcome to the Zoo. All the different models - for your viewing!
"""
import numpy as np
from keras.layers import Input, LSTM, GRU, BatchNormalization, Dense, Conv1D, TimeDistributed, Activation, Dropout, Reshape
from keras import Model
from keras.regularizers import l1


def LSTM_triggerNG(num_inputs, num_channels, num_outputs, kernel_init='normal', bias_init='zeros'):
    """
    Modified from Andrew Ng's Deeplearning.ai Sequence Modeling course on Coursera.
    Originally used for trigger word detection.
    Notes:
        - Switched GRU cells to LSTM cells
        - BiDirectional???
        - Custom, guesswork kernel_size and strides in Conv1D
    """

    model_input = Input(shape=(num_channels, num_inputs), dtype='float32')

    X = Conv1D(196, kernel_size=4, strides=2, kernel_initializer=kernel_init, bias_initializer=bias_init)(model_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = LSTM(units=128, return_sequences=True, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization(axis=-1)(X)

    X = LSTM(units=128, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization(axis=-1)(X)
    X = Dropout(0.8)(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear')(X)

    model = Model(inputs=model_input, outputs=X)
    return model


def LSTM_WSAEs(num_inputs, num_channels=1, num_outputs=1, encoding_dim=10, kernel_init='normal', bias_init='zeros'):
    """
    WORK IN PROGRESS
    TODO:
        ?num_delays = 5
        ?stateful

    From Paper:
        "A deep learning framework for financial time series using stacked autoencoders and long-short term memory"
        Authors:
            Wei Bao,
            Jun Yue,
            Yulei Rao
    Notes:
        - Added BatchNormalization Layer after LSTM
        - Increased LSTM hidden unit layer to 64
        - Made up hidden_dim (only specified encoding_dim=10

    """

    # Encoder/Decoder hidden unit size halfway between num_inputs and encoding_dim
    hidden_dim = num_inputs - np.int((num_inputs - encoding_dim) / 2)

    # Autoencoder
    X_input = Input(shape=(num_channels, num_inputs), dtype='float32')
    encoder = Dense(hidden_dim, activation='relu', name='encoder')(X_input)
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(encoder)
    decoder = Dense(hidden_dim, activation='relu', name='decoder')(encoded)
    decoded = Dense(num_inputs, activation='linear', name='decoded')(decoder)

    # LSTM
    X = Reshape(target_shape=(encoding_dim, num_channels), name='rs_0')(encoded)
    X = LSTM(units=64, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init, name='lstm_0')(X)
    X = BatchNormalization(axis=-1, name='bn_0')(X)

    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear', name='dense_0')(X)

    model = Model(inputs=[X_input], outputs=[decoded, X])
    return model


def build_ae_lstm(num_inputs, num_channels=1, num_outputs=1, kernel_init='normal', bias_init='zeros') -> Model:
    """
    Own implementation of a stacked autoencoder with LSTM for smoothed time-series data.

    :param num_inputs:
    :param num_channels:
    :param num_outputs:
    :param kernel_init:
    :param bias_init:
    :return model: Keras Model instance

    Notes:
        - encoded layer is a sparse encoder (l1 regularized)
    """
    hidden_dim = np.int(num_inputs / 2)
    encoding_dim = np.int(hidden_dim / 2)

    X_input = Input(shape=(num_channels, num_inputs), dtype='float32', name='input_0')

    # Autoencoder
    encoder = Dense(units=hidden_dim, activation='relu', name='encoder')(X_input)
    encoded = Dense(units=encoding_dim, activation='relu', activity_regularizer=l1(1e-6), name='encoded')(encoder)
    decoder = Dense(units=hidden_dim, activation='relu', name='decoder')(encoded)
    decoded = Dense(units=num_inputs, activation='linear', name='decoded')(decoder)

    # Dual LSTM
    X = Reshape(target_shape=(encoding_dim, num_channels), name='rs_0')(encoded)
    X = LSTM(units=64, return_sequences=True, kernel_initializer=kernel_init, bias_initializer=bias_init,
             name='lstm_0')(X)
    X = Dropout(0.2, name='dr_0')(X)
    X = BatchNormalization(axis=-1, name='bn_0')(X)

    X = LSTM(units=64, return_sequences=False, kernel_initializer=kernel_init, bias_initializer=bias_init,
             name='lstm_1')(X)
    X = Dropout(0.2, name='dr_1')(X)
    X = BatchNormalization(axis=-1, name='bn_1')(X)
    X = Dropout(0.2, name='dr_2')(X)

    # Dense Activation
    X = Dense(num_outputs, kernel_initializer=kernel_init, bias_initializer=bias_init, activation='linear',
              name='dense_0')(X)

    model = Model(inputs=[X_input], outputs=[decoded, X])
    return model